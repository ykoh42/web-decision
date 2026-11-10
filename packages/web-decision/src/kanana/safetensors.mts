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

type SafetensorsDtype = keyof typeof DTYPE_BYTES;
type TensorRecord = Readonly<{
  name: string;
  dtype: SafetensorsDtype;
  shape: number[];
  dataStart: number;
  dataEnd: number;
  length: number;
}>;
type ParsedTensorRecord = Readonly<{
  name: string;
  dtype: SafetensorsDtype;
  shape: number[];
  offset: number;
  length: number;
}>;
type ParsedSafetensorsHeader = Readonly<{
  headerLength: number;
  dataOffset: number;
  dataBytes: number;
  tensorCount: number;
  metadata: Record<string, string>;
  tensors: ParsedTensorRecord[];
}>;
type ConversionFailure = Readonly<{byteOffset: number; reason: 'overflow' | 'non-finite'; message: string}>;
type ConversionStatistics = {
  invalidValueCount: number;
  overflowValueCount: number;
  nonFiniteValueCount: number;
  firstFailure: ConversionFailure | undefined;
};

function invalid(message: string) {
  return new WebDecisionError('INVALID_ASSET', message);
}

function bytesView(value: unknown, label: string): Uint8Array {
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (!ArrayBuffer.isView(value) || value instanceof DataView) throw invalid(`${label} must be byte data.`);
  return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function tensorByteLength(shape: number[], elementBytes: number): number {
  const elements = shape.reduce((total, dimension) => total * dimension, 1);
  const bytes = elements * elementBytes;
  if (!Number.isSafeInteger(elements) || !Number.isSafeInteger(bytes)) throw invalid('Safetensors shape is too large.');
  return bytes;
}

function parseTensor(name: string, descriptor: unknown, dataBytes: number): TensorRecord {
  if (!name || name === '__metadata__' || !isRecord(descriptor)) throw invalid('Invalid Safetensors tensor entry.');
  if (typeof descriptor.dtype !== 'string' || !Object.hasOwn(DTYPE_BYTES, descriptor.dtype))
    throw invalid(`Unsupported Safetensors dtype for ${name}.`);
  const dtype = descriptor.dtype as SafetensorsDtype;
  const elementBytes = DTYPE_BYTES[dtype];
  if (!elementBytes) throw invalid(`Unsupported Safetensors dtype for ${name}.`);
  if (
    !Array.isArray(descriptor.shape) ||
    !(descriptor.shape as unknown[]).every((dimension) => Number.isSafeInteger(dimension) && (dimension as number) >= 0)
  )
    throw invalid(`Invalid Safetensors shape for ${name}.`);
  if (
    !Array.isArray(descriptor.data_offsets) ||
    descriptor.data_offsets.length !== 2 ||
    !(descriptor.data_offsets as unknown[]).every((offset) => Number.isSafeInteger(offset) && (offset as number) >= 0)
  )
    throw invalid(`Invalid Safetensors data offsets for ${name}.`);

  const [start, end] = descriptor.data_offsets as [number, number];
  const shape = descriptor.shape as number[];
  const length = tensorByteLength(shape, elementBytes);
  if (start > end || end > dataBytes || end - start !== length)
    throw invalid(`Safetensors byte range does not match ${name}'s shape and dtype.`);
  return {
    name,
    dtype,
    shape: [...shape],
    dataStart: start,
    dataEnd: end,
    length
  };
}

/** Parse the 8-byte length prefix and complete JSON header without reading tensor data. */
export function parseSafetensorsHeader(
  headerInput: unknown,
  totalFileBytes: number,
  {maxHeaderBytes = SAFETENSORS_MAX_HEADER_BYTES}: {maxHeaderBytes?: number} = {}
): ParsedSafetensorsHeader {
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

  let parsedHeader: unknown;
  try {
    const json = new TextDecoder('utf-8', {fatal: true}).decode(bytes.subarray(8));
    parsedHeader = JSON.parse(json) as unknown;
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Safetensors header is not valid UTF-8 JSON.', {cause});
  }
  if (!isRecord(parsedHeader)) throw invalid('Safetensors header must be an object.');
  const header = parsedHeader;
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

  const metadata = Object.hasOwn(header, '__metadata__') ? header.__metadata__ : {};
  return {
    headerLength,
    dataOffset,
    dataBytes,
    tensorCount: tensors.length,
    metadata: metadata as Record<string, string>,
    tensors: tensors.map(({name, dtype, shape, dataStart, length}) => ({
      name,
      dtype,
      shape,
      offset: dataOffset + dataStart,
      length
    }))
  };
}

function failBf16Conversion(message: string, byteOffset: number, reason: ConversionFailure['reason']): never {
  const error = Object.assign(invalid(`${message} At tensor byte offset ${byteOffset}.`), {byteOffset, reason});
  throw error;
}

function roundToNearestEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  return fraction > 0.5 || (fraction === 0.5 && (lower & 1) === 1) ? lower + 1 : lower;
}

function roundBf16ToFp16Bits(bfloat16Bits: number, byteOffset: number): number {
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
export function convertBf16ToFp16(input: unknown): Uint8Array {
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

/** Convert finite little-endian FP16 values to the FP32 bytes required by the TVM vision graph. */
export function convertFp16ToFp32(input: unknown): Uint8Array {
  const bytes = bytesView(input, 'FP16 tensor');
  if (bytes.byteLength % 2) throw invalid('FP16 tensor byte length must be even.');
  const source = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = new Float32Array(bytes.byteLength / 2);
  for (let offset = 0; offset < bytes.byteLength; offset += 2) {
    const bits = source.getUint16(offset, true);
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >>> 10) & 0x1f;
    const fraction = bits & 0x03ff;
    if (exponent === 0x1f)
      throw invalid(`FP16 NaN and infinity are not valid TVM vision weights. At byte offset ${offset}.`);
    values[offset / 2] =
      exponent === 0 ? sign * 2 ** -14 * (fraction / 1024) : sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
  }
  return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
}

function recordBf16InspectionFailure(error: unknown, byteOffset: number, statistics: ConversionStatistics): void {
  if (typeof error !== 'object' || error === null || !('reason' in error)) throw error;
  const failure = error as ConversionFailure;
  statistics.invalidValueCount += 1;
  if (failure.reason === 'overflow') statistics.overflowValueCount += 1;
  if (failure.reason === 'non-finite') statistics.nonFiniteValueCount += 1;
  if (!statistics.firstFailure) {
    statistics.firstFailure = {
      byteOffset,
      reason: failure.reason,
      message: failure.message
    };
  }
}

/** Inspect every BF16 value using the same finite FP16 round-to-nearest rules without retaining output. */
export function inspectBf16ToFp16(input: unknown): Readonly<ConversionStatistics> {
  const bytes = bytesView(input, 'BF16 tensor');
  if (bytes.byteLength % 2) throw invalid('BF16 tensor byte length must be even.');
  const source = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const result = {
    invalidValueCount: 0,
    overflowValueCount: 0,
    nonFiniteValueCount: 0,
    firstFailure: undefined as ConversionFailure | undefined
  };
  for (let offset = 0; offset < bytes.byteLength; offset += 2) {
    try {
      roundBf16ToFp16Bits(source.getUint16(offset, true), offset);
    } catch (error: unknown) {
      recordBf16InspectionFailure(error, offset, result);
    }
  }
  return Object.freeze(result);
}
