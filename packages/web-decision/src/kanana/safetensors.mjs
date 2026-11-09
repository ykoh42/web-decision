import {WebDecisionError} from '../kev.js';

export const SAFETENSORS_MAX_HEADER_BYTES = 16 * 1024 * 1024;

const DTYPE_BYTES = Object.freeze({
  BOOL: 1,
  U8: 1,
  I8: 1,
  F16: 2,
  BF16: 2,
  I16: 2,
  U16: 2,
  F32: 4,
  I32: 4,
  U32: 4,
  F64: 8,
  I64: 8,
  U64: 8
});

function invalid(message) {
  return new WebDecisionError('INVALID_ASSET', message);
}

function bytesView(value, label) {
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (!ArrayBuffer.isView(value) || value instanceof DataView) throw invalid(`${label} must be byte data.`);
  return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function tensorByteLength(shape, elementBytes) {
  const elements = shape.reduce((total, dimension) => total * dimension, 1);
  const bytes = elements * elementBytes;
  if (!Number.isSafeInteger(elements) || !Number.isSafeInteger(bytes)) throw invalid('Safetensors shape is too large.');
  return bytes;
}

function parseTensor(name, descriptor, dataBytes) {
  if (!name || name === '__metadata__' || !isRecord(descriptor)) throw invalid('Invalid Safetensors tensor entry.');
  const elementBytes = DTYPE_BYTES[descriptor.dtype];
  if (!elementBytes) throw invalid(`Unsupported Safetensors dtype for ${name}.`);
  if (
    !Array.isArray(descriptor.shape) ||
    descriptor.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 0)
  )
    throw invalid(`Invalid Safetensors shape for ${name}.`);
  if (
    !Array.isArray(descriptor.data_offsets) ||
    descriptor.data_offsets.length !== 2 ||
    !descriptor.data_offsets.every((offset) => Number.isSafeInteger(offset) && offset >= 0)
  )
    throw invalid(`Invalid Safetensors data offsets for ${name}.`);

  const [start, end] = descriptor.data_offsets;
  const length = tensorByteLength(descriptor.shape, elementBytes);
  if (start > end || end > dataBytes || end - start !== length)
    throw invalid(`Safetensors byte range does not match ${name}'s shape and dtype.`);
  return {
    name,
    dtype: descriptor.dtype,
    shape: [...descriptor.shape],
    dataStart: start,
    dataEnd: end,
    length
  };
}

/** Parse the 8-byte length prefix and complete JSON header without reading tensor data. */
export function parseSafetensorsHeader(
  headerInput,
  totalFileBytes,
  {maxHeaderBytes = SAFETENSORS_MAX_HEADER_BYTES} = {}
) {
  const bytes = bytesView(headerInput, 'Safetensors header');
  if (
    !Number.isSafeInteger(totalFileBytes) ||
    totalFileBytes < 8 ||
    !Number.isSafeInteger(maxHeaderBytes) ||
    maxHeaderBytes < 2 ||
    bytes.byteLength < 8
  )
    throw invalid('Invalid Safetensors header bounds.');

  const headerLengthBig = new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true);
  if (headerLengthBig > BigInt(Number.MAX_SAFE_INTEGER)) throw invalid('Safetensors header length exceeds safe range.');
  const headerLength = Number(headerLengthBig);
  if (
    headerLength < 2 ||
    headerLength > maxHeaderBytes ||
    headerLength > totalFileBytes - 8 ||
    bytes.byteLength !== 8 + headerLength
  )
    throw invalid('Safetensors header length is invalid or incomplete.');

  let header;
  try {
    const json = new TextDecoder('utf-8', {fatal: true}).decode(bytes.subarray(8));
    header = JSON.parse(json);
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Safetensors header is not valid UTF-8 JSON.', {cause});
  }
  if (!isRecord(header)) throw invalid('Safetensors header must be an object.');
  if (Object.hasOwn(header, '__metadata__')) {
    const metadata = header.__metadata__;
    if (!isRecord(metadata) || Object.values(metadata).some((value) => typeof value !== 'string'))
      throw invalid('Safetensors metadata values must be strings.');
  }

  const dataOffset = 8 + headerLength;
  const dataBytes = totalFileBytes - dataOffset;
  const entries = Object.entries(header).filter(([name]) => name !== '__metadata__');
  if (!entries.length) throw invalid('Safetensors header contains no tensors.');
  const tensors = entries.map(([name, descriptor]) => parseTensor(name, descriptor, dataBytes));
  const ordered = [...tensors].sort((left, right) => left.dataStart - right.dataStart || left.dataEnd - right.dataEnd);
  let nextOffset = 0;
  for (const tensor of ordered) {
    if (tensor.dataStart !== nextOffset) throw invalid('Safetensors tensor ranges overlap or leave a gap.');
    nextOffset = tensor.dataEnd;
  }
  if (nextOffset !== dataBytes) throw invalid('Safetensors tensor ranges do not cover the data section.');

  return {
    headerLength,
    dataOffset,
    dataBytes,
    tensorCount: tensors.length,
    metadata: Object.hasOwn(header, '__metadata__') ? {...header.__metadata__} : {},
    tensors: tensors.map(({name, dtype, shape, dataStart, length}) => ({
      name,
      dtype,
      shape,
      offset: dataOffset + dataStart,
      length
    }))
  };
}

function failBf16Conversion(message, byteOffset, reason) {
  const error = invalid(`${message} At tensor byte offset ${byteOffset}.`);
  error.byteOffset = byteOffset;
  error.reason = reason;
  throw error;
}

function roundToNearestEven(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  return fraction > 0.5 || (fraction === 0.5 && (lower & 1) === 1) ? lower + 1 : lower;
}

function roundBf16ToFp16Bits(bfloat16Bits, byteOffset) {
  const sign = bfloat16Bits & 0x8000;
  const exponent = (bfloat16Bits >>> 7) & 0xff;
  const mantissa = bfloat16Bits & 0x7f;
  if (exponent === 0xff)
    failBf16Conversion('BF16 NaN and infinity are not valid finite FP16 weights.', byteOffset, 'non-finite');
  if (exponent === 0) {
    return sign;
  }

  const unbiasedExponent = exponent - 127;
  if (unbiasedExponent > 15) failBf16Conversion('BF16 value overflows finite FP16.', byteOffset, 'overflow');
  if (unbiasedExponent >= -14) return sign | ((unbiasedExponent + 15) << 10) | (mantissa << 3);

  const significand = 128 + mantissa;
  const shift = unbiasedExponent + 17;
  const subnormalMantissa = shift >= 0 ? significand * 2 ** shift : roundToNearestEven(significand / 2 ** -shift);
  return sign | subnormalMantissa;
}

/** Convert little-endian BF16 values to finite FP16 with IEEE round-to-nearest, ties-to-even. */
export function convertBf16ToFp16(input) {
  const bytes = bytesView(input, 'BF16 tensor');
  if (bytes.byteLength % 2) throw invalid('BF16 tensor byte length must be even.');
  const source = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const result = new Uint8Array(bytes.byteLength);
  const target = new DataView(result.buffer);
  for (let offset = 0; offset < bytes.byteLength; offset += 2) {
    const bits = roundBf16ToFp16Bits(source.getUint16(offset, true), offset);
    target.setUint16(offset, bits, true);
  }
  return result;
}

/** Inspect every BF16 value using the same finite FP16 round-to-nearest rules without retaining output. */
export function inspectBf16ToFp16(input) {
  const bytes = bytesView(input, 'BF16 tensor');
  if (bytes.byteLength % 2) throw invalid('BF16 tensor byte length must be even.');
  const source = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const result = {
    invalidValueCount: 0,
    overflowValueCount: 0,
    nonFiniteValueCount: 0,
    firstFailure: undefined
  };
  for (let offset = 0; offset < bytes.byteLength; offset += 2) {
    try {
      roundBf16ToFp16Bits(source.getUint16(offset, true), offset);
    } catch (error) {
      result.invalidValueCount += 1;
      if (error.reason === 'overflow') result.overflowValueCount += 1;
      if (error.reason === 'non-finite') result.nonFiniteValueCount += 1;
      if (!result.firstFailure)
        result.firstFailure = {
          byteOffset: offset,
          reason: error.reason,
          message: error.message
        };
    }
  }
  return Object.freeze(result);
}
