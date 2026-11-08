const RGB_CHANNELS = 3;
const PILLOW_MAX_IMAGE_PIXELS = 89_478_485n;
const PILLOW_ERROR_IMAGE_PIXELS = PILLOW_MAX_IMAGE_PIXELS * 2n;
const WHITESPACE = new Set([0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x20]);
const ASCII_DECODER = new TextDecoder('utf-8', {fatal: true});

function asBytes(input) {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  throw new TypeError('PNM input must be bytes.');
}

function skipComment(bytes, cursor) {
  while (cursor.offset < bytes.length && bytes[cursor.offset] !== 0x0a && bytes[cursor.offset] !== 0x0d)
    cursor.offset++;
  if (cursor.offset === bytes.length) throw new TypeError('PNM comment has no line ending.');
  if (bytes[cursor.offset] === 0x0d && bytes[cursor.offset + 1] === 0x0a) cursor.offset += 2;
  else cursor.offset++;
}

function skipSeparators(bytes, cursor) {
  while (cursor.offset < bytes.length) {
    if (WHITESPACE.has(bytes[cursor.offset])) cursor.offset++;
    else if (bytes[cursor.offset] === 0x23) skipComment(bytes, cursor);
    else break;
  }
}

function readToken(bytes, cursor, label) {
  skipSeparators(bytes, cursor);
  const start = cursor.offset;
  while (cursor.offset < bytes.length && !WHITESPACE.has(bytes[cursor.offset]) && bytes[cursor.offset] !== 0x23) {
    if (bytes[cursor.offset] > 0x7f) throw new TypeError(`PNM ${label} must be ASCII.`);
    cursor.offset++;
  }
  const length = cursor.offset - start;
  if (length === 0) throw new TypeError(`PNM ${label} is missing.`);
  return ASCII_DECODER.decode(bytes.subarray(start, cursor.offset));
}

function consumeRasterDelimiter(bytes, cursor) {
  if (cursor.offset >= bytes.length) throw new TypeError('PNM raster is missing.');
  if (bytes[cursor.offset] === 0x23) {
    skipComment(bytes, cursor);
    return;
  }
  if (!WHITESPACE.has(bytes[cursor.offset])) throw new TypeError('PNM header has no raster delimiter.');
  const delimiter = bytes[cursor.offset++];
  if (delimiter === 0x0d && bytes[cursor.offset] === 0x0a) cursor.offset++;
}

function unsignedInteger(token, label) {
  if (!/^\d+$/.test(token)) throw new TypeError(`PNM ${label} must be an unsigned integer.`);
  const value = Number(token);
  if (!Number.isSafeInteger(value)) throw new RangeError(`PNM ${label} exceeds the safe integer range.`);
  return value;
}

function checkPillowPixelLimit(pixels) {
  if (pixels > PILLOW_ERROR_IMAGE_PIXELS) {
    throw new RangeError(
      `Kanana image size (${pixels} pixels) exceeds Pillow 12.1.1's decompression-bomb limit of ${PILLOW_ERROR_IMAGE_PIXELS} pixels.`
    );
  }
  if (pixels > PILLOW_MAX_IMAGE_PIXELS) {
    globalThis.console?.warn?.(
      `Kanana image size (${pixels} pixels) exceeds Pillow 12.1.1's warning limit of ${PILLOW_MAX_IMAGE_PIXELS} pixels.`
    );
  }
}

function imageSize(widthToken, heightToken) {
  const width = unsignedInteger(widthToken, 'width');
  const height = unsignedInteger(heightToken, 'height');
  if (width < 1 || height < 1) throw new RangeError('PNM dimensions must be positive.');

  checkPillowPixelLimit(BigInt(width) * BigInt(height));

  const pixels = width * height;
  const rgbLength = pixels * RGB_CHANNELS;
  if (!Number.isSafeInteger(pixels) || !Number.isSafeInteger(rgbLength))
    throw new RangeError('PNM dimensions exceed the browser pixel representation.');
  return {width, height, pixels, rgbLength};
}

function roundToEven(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

function clampByte(value) {
  return Math.max(0, Math.min(255, value));
}

function pnmByte(sample, maxval, grayscale, integerGray, direct16BitGray) {
  if (grayscale && integerGray) {
    const integerValue = direct16BitGray ? sample : roundToEven((sample / maxval) * 65535);
    return clampByte(integerValue);
  }
  return clampByte(roundToEven((sample / maxval) * 255));
}

function writeGray(rgb, pixel, value) {
  const offset = pixel * RGB_CHANNELS;
  rgb[offset] = value;
  rgb[offset + 1] = value;
  rgb[offset + 2] = value;
}

function writePnmSample(rgb, sampleIndex, value, grayscale) {
  if (grayscale) writeGray(rgb, sampleIndex, value);
  else rgb[sampleIndex] = value;
}

function decodedPnmImage(rgb, size) {
  return {rgb, width: size.width, height: size.height};
}

function decodePlainPnm(bytes, cursor, size, maxval, grayscale, integerGray, sampleCount) {
  if (sampleCount > bytes.length - cursor.offset) throw new TypeError('PNM raster is truncated.');

  const rgb = new Uint8Array(size.rgbLength);
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
    const sample = unsignedInteger(readToken(bytes, cursor, 'sample'), 'sample');
    if (sample > maxval) throw new RangeError(`PNM sample ${sample} exceeds maxval ${maxval}.`);
    const value = pnmByte(sample, maxval, grayscale, integerGray, false);
    writePnmSample(rgb, sampleIndex, value, grayscale);
  }
  return decodedPnmImage(rgb, size);
}

function rawPnmSample(bytes, offset, bytesPerSample) {
  if (bytesPerSample === 1) return bytes[offset];
  return (bytes[offset] << 8) | bytes[offset + 1];
}

function decodeRawPnm(bytes, cursor, size, maxval, grayscale, integerGray, direct16BitGray, sampleCount) {
  consumeRasterDelimiter(bytes, cursor);
  const bytesPerSample = maxval < 256 ? 1 : 2;
  const rasterLength = sampleCount * bytesPerSample;
  if (!Number.isSafeInteger(rasterLength) || cursor.offset + rasterLength > bytes.length)
    throw new TypeError('PNM raster is truncated.');

  const rgb = new Uint8Array(size.rgbLength);
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
    const offset = cursor.offset + sampleIndex * bytesPerSample;
    const sample = rawPnmSample(bytes, offset, bytesPerSample);
    if (sample > maxval) throw new RangeError(`PNM sample ${sample} exceeds maxval ${maxval}.`);
    const value = pnmByte(sample, maxval, grayscale, integerGray, direct16BitGray);
    writePnmSample(rgb, sampleIndex, value, grayscale);
  }
  return decodedPnmImage(rgb, size);
}

function decodePnm(bytes, cursor, magic, size) {
  const grayscale = magic === 'P2' || magic === 'P5';
  const plain = magic === 'P2' || magic === 'P3';
  const channels = grayscale ? 1 : RGB_CHANNELS;
  const maxval = unsignedInteger(readToken(bytes, cursor, 'maxval'), 'maxval');
  if (maxval < 1 || maxval > 65535) throw new RangeError('PNM maxval must be between 1 and 65535.');

  const integerGray = grayscale && maxval > 255;
  const direct16BitGray = grayscale && maxval === 65535 && !plain;
  const sampleCount = size.pixels * channels;
  if (!Number.isSafeInteger(sampleCount)) throw new RangeError('PNM sample count exceeds the safe integer range.');
  if (plain) return decodePlainPnm(bytes, cursor, size, maxval, grayscale, integerGray, sampleCount);
  return decodeRawPnm(bytes, cursor, size, maxval, grayscale, integerGray, direct16BitGray, sampleCount);
}

function decodePlainPbm(bytes, cursor, size) {
  if (size.pixels > bytes.length - cursor.offset) throw new TypeError('PBM raster is truncated.');

  const rgb = new Uint8Array(size.rgbLength);
  for (let pixel = 0; pixel < size.pixels; pixel++) {
    const sample = readPbmBit(bytes, cursor);
    writeGray(rgb, pixel, sample === 0 ? 255 : 0);
  }
  return decodedPnmImage(rgb, size);
}

function readPbmBit(bytes, cursor) {
  skipSeparators(bytes, cursor);
  const sample = bytes[cursor.offset++];
  if (sample === 0x30) return 0;
  if (sample === 0x31) return 1;
  if (sample === undefined) throw new TypeError('PNM sample is missing.');
  throw new TypeError('PBM sample must be 0 or 1.');
}

function decodeRawPbm(bytes, cursor, size) {
  consumeRasterDelimiter(bytes, cursor);
  const bytesPerRow = Math.ceil(size.width / 8);
  const rasterLength = bytesPerRow * size.height;
  if (!Number.isSafeInteger(rasterLength) || cursor.offset + rasterLength > bytes.length)
    throw new TypeError('PBM raster is truncated.');

  const rgb = new Uint8Array(size.rgbLength);
  for (let y = 0; y < size.height; y++) {
    for (let x = 0; x < size.width; x++) {
      const value = bytes[cursor.offset + y * bytesPerRow + Math.floor(x / 8)];
      const black = (value & (0x80 >> (x % 8))) !== 0;
      writeGray(rgb, y * size.width + x, black ? 0 : 255);
    }
  }
  return decodedPnmImage(rgb, size);
}

function decodePbm(bytes, cursor, magic, size) {
  if (magic === 'P1') return decodePlainPbm(bytes, cursor, size);
  return decodeRawPbm(bytes, cursor, size);
}

function pfmScale(token) {
  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(token))
    throw new TypeError('PFM scale must be a finite, non-zero number.');
  const scale = Number(token);
  if (!Number.isFinite(scale) || scale === 0) throw new TypeError('PFM scale must be a finite, non-zero number.');
  return scale;
}

function decodePfm(bytes, cursor, size) {
  const scale = pfmScale(readToken(bytes, cursor, 'scale'));
  consumeRasterDelimiter(bytes, cursor);
  const sampleCount = size.pixels;
  const rasterLength = sampleCount * 4;
  if (!Number.isSafeInteger(rasterLength) || cursor.offset + rasterLength > bytes.length)
    throw new TypeError('PFM raster is truncated.');

  const littleEndian = scale < 0;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const rgb = new Uint8Array(size.rgbLength);
  for (let y = 0; y < size.height; y++) {
    const sourceY = size.height - y - 1;
    for (let x = 0; x < size.width; x++) {
      const source = cursor.offset + (sourceY * size.width + x) * 4;
      const sample = view.getFloat32(source, littleEndian);
      if (!Number.isFinite(sample)) throw new TypeError('PFM samples must be finite.');
      writeGray(rgb, y * size.width + x, clampByte(Math.trunc(sample)));
    }
  }
  return {rgb, width: size.width, height: size.height};
}

/** Decode Pillow-compatible grayscale or RGB PNM/PFM samples into RGB bytes. */
export function decodeKananaPnmImage(input) {
  const bytes = asBytes(input);
  if (bytes.length < 3 || bytes[0] !== 0x50 || !WHITESPACE.has(bytes[2]))
    throw new TypeError('Input is not a supported PNM/PFM image.');

  const magic = String.fromCharCode(bytes[0], bytes[1]);
  if (!['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'Pf'].includes(magic))
    throw new TypeError(`Unsupported PNM/PFM format ${magic}.`);

  const cursor = {offset: 2};
  const size = imageSize(readToken(bytes, cursor, 'width'), readToken(bytes, cursor, 'height'));
  if (magic === 'Pf') return decodePfm(bytes, cursor, size);
  if (magic === 'P1' || magic === 'P4') return decodePbm(bytes, cursor, magic, size);
  return decodePnm(bytes, cursor, magic, size);
}
