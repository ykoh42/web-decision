const TIFF_SIGNATURES = [
  [73, 73, 42, 0],
  [77, 77, 0, 42]
];
const PILLOW_MAX_IMAGE_PIXELS = 89_478_485n;
const PILLOW_ERROR_IMAGE_PIXELS = PILLOW_MAX_IMAGE_PIXELS * 2n;
const RGB_CHANNELS = 3;
const ALLOWED_TAGS = new Set([256, 257, 258, 259, 262, 273, 274, 277, 278, 279, 284, 317, 339]);

function startsWith(bytes, signature) {
  return signature.every((value, index) => bytes[index] === value);
}

function isTiff(bytes) {
  return TIFF_SIGNATURES.some((signature) => startsWith(bytes, signature));
}

function tiffOrder(bytes) {
  if (bytes[0] === 73 && bytes[1] === 73) return true;
  if (bytes[0] === 77 && bytes[1] === 77) return false;
  return null;
}

function readInteger(view, offset, type, littleEndian) {
  if (type === 1) return view.getUint8(offset);
  if (type === 3) return view.getUint16(offset, littleEndian);
  if (type === 4) return view.getUint32(offset, littleEndian);
  return null;
}

function integerSize(type) {
  if (type === 1) return 1;
  if (type === 3) return 2;
  if (type === 4) return 4;
  return 0;
}

function readTagValues(view, entryOffset, type, count, littleEndian) {
  const size = integerSize(type);
  const bytes = size * count;
  if (!size || !Number.isSafeInteger(bytes)) return null;

  const valueOffset = bytes <= 4 ? entryOffset + 8 : view.getUint32(entryOffset + 8, littleEndian);
  if (!Number.isSafeInteger(valueOffset + bytes) || valueOffset + bytes > view.byteLength) return null;

  const values = [];
  for (let index = 0; index < count; index++) {
    values.push(readInteger(view, valueOffset + index * size, type, littleEndian));
  }
  return values;
}

function arrayDescriptor(view, entryOffset, type, count, littleEndian) {
  const size = integerSize(type);
  const byteLength = size * count;
  if (!size || !Number.isSafeInteger(byteLength)) return null;

  const offset = byteLength <= 4 ? entryOffset + 8 : view.getUint32(entryOffset + 8, littleEndian);
  if (!Number.isSafeInteger(offset + byteLength) || offset + byteLength > view.byteLength) return null;
  return {type, count, offset};
}

function validTagValueCount(tag, count) {
  if (count < 1) return false;
  if (tag === 273 || tag === 279) return true;
  return count <= (tag === 258 || tag === 339 ? RGB_CHANNELS : 1);
}

function readDirectoryTag(view, entryOffset, littleEndian, tags) {
  const tag = view.getUint16(entryOffset, littleEndian);
  if (!ALLOWED_TAGS.has(tag)) return true;

  const type = view.getUint16(entryOffset + 2, littleEndian);
  const count = view.getUint32(entryOffset + 4, littleEndian);
  if (tags.has(tag) || !validTagValueCount(tag, count)) return false;

  const values =
    tag === 273 || tag === 279
      ? arrayDescriptor(view, entryOffset, type, count, littleEndian)
      : readTagValues(view, entryOffset, type, count, littleEndian);
  if (values === null) return false;
  tags.set(tag, values);
  return true;
}

function readFirstDirectory(view, littleEndian) {
  if (view.byteLength < 8 || view.getUint16(2, littleEndian) !== 42) return null;

  const offset = view.getUint32(4, littleEndian);
  if (offset < 8 || offset + 2 > view.byteLength) return null;
  const count = view.getUint16(offset, littleEndian);
  const entriesEnd = offset + 2 + count * 12;
  if (!Number.isSafeInteger(entriesEnd + 4) || entriesEnd + 4 > view.byteLength) return null;

  const tags = new Map();
  for (let index = 0; index < count; index++) {
    const entryOffset = offset + 2 + index * 12;
    if (!readDirectoryTag(view, entryOffset, littleEndian, tags)) return null;
  }

  return {tags};
}

function scalar(tags, tag, fallback = null) {
  const values = tags.get(tag);
  if (values === undefined) return fallback;
  return values.length === 1 ? values[0] : null;
}

function pillowPixelCount(width, height) {
  const pixels = BigInt(width) * BigInt(height);
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

  const samples = pixels * BigInt(RGB_CHANNELS);
  const byteLength = samples * 2n;
  if (pixels < 1n || samples > BigInt(Number.MAX_SAFE_INTEGER) || byteLength > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new RangeError('Kanana TIFF dimensions exceed the browser pixel representation.');
  }
  return Number(pixels);
}

function stripList(tags, tag, expectedCount, view, littleEndian) {
  const descriptor = tags.get(tag);
  if (!descriptor || descriptor.count !== expectedCount) return null;
  const {type, offset} = descriptor;
  const size = integerSize(type);
  const values = [];
  for (let index = 0; index < expectedCount; index++) {
    values.push(readInteger(view, offset + index * size, type, littleEndian));
  }
  if (tag === 273 && values.some((value) => !Number.isSafeInteger(value) || value < 0 || value >= view.byteLength))
    return null;
  return values;
}

function supportedSampleMode(tags) {
  const width = scalar(tags, 256);
  const height = scalar(tags, 257);
  const channels = scalar(tags, 277, 1);
  const bits = tags.get(258);
  const photometric = scalar(tags, 262);
  const sampleFormat = tags.get(339) ?? Array(channels).fill(1);

  if (!Number.isSafeInteger(width) || width < 1 || !Number.isSafeInteger(height) || height < 1) return null;
  if (channels !== 1 && channels !== RGB_CHANNELS) return null;
  if (!bits || bits.length !== channels || bits.some((value) => value !== 16)) return null;
  if (channels === 1 && photometric !== 1) return null;
  if (channels === RGB_CHANNELS && photometric !== 2) return null;
  if (sampleFormat.length !== channels || sampleFormat.some((value) => value !== 1)) return null;
  pillowPixelCount(width, height);
  return {width, height, channels};
}

function supportedEncoding(tags, height) {
  const compression = scalar(tags, 259, 1);
  const rowsPerStrip = scalar(tags, 278, height);
  const planar = scalar(tags, 284, 1);
  const predictor = scalar(tags, 317, 1);
  const orientation = scalar(tags, 274, 1);

  if (orientation !== 1 || planar !== 1 || predictor !== 1) return null;
  if (![1, 5, 8].includes(compression)) return null;
  if (!Number.isSafeInteger(rowsPerStrip) || rowsPerStrip < 1) return null;
  return {compression, rowsPerStrip};
}

function inspectStrips(tags, view, littleEndian, mode, encoding) {
  const {width, height, channels} = mode;
  const {compression, rowsPerStrip} = encoding;
  const stripCount = Math.ceil(height / rowsPerStrip);
  const offsets = stripList(tags, 273, stripCount, view, littleEndian);
  const byteCounts = stripList(tags, 279, stripCount, view, littleEndian);
  if (!offsets || !byteCounts) return null;

  const stripLengths = [];
  for (let index = 0; index < stripCount; index++) {
    const rows = Math.min(rowsPerStrip, height - index * rowsPerStrip);
    const expected = rows * width * channels * 2;
    const end = offsets[index] + byteCounts[index];
    if (!Number.isSafeInteger(expected) || end > view.byteLength) return null;
    if (compression === 1 && byteCounts[index] !== expected) return null;
    stripLengths.push(expected);
  }

  return {view, width, height, channels, compression, littleEndian, offsets, byteCounts, stripLengths};
}

function inspectTiff16(bytes) {
  if (!isTiff(bytes)) return null;

  const littleEndian = tiffOrder(bytes);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const directory = readFirstDirectory(view, littleEndian);
  if (!directory) return null;

  if (!directory.tags.get(258)?.includes(16)) return null;

  const mode = supportedSampleMode(directory.tags);
  if (!mode) throw new TypeError('Kanana 16-bit TIFF layout is not covered by the Pillow parity decoder.');
  const encoding = supportedEncoding(directory.tags, mode.height);
  if (!encoding) throw new TypeError('Kanana 16-bit TIFF layout is not covered by the Pillow parity decoder.');
  const image = inspectStrips(directory.tags, view, littleEndian, mode, encoding);
  if (!image) throw new TypeError('Kanana 16-bit TIFF layout is not covered by the Pillow parity decoder.');
  return image;
}

function writeGrayPixel(rgb, target, sample) {
  const gray = Math.min(sample, 255);
  rgb[target] = gray;
  rgb[target + 1] = gray;
  rgb[target + 2] = gray;
}

function writeRgbPixel(rgb, target, red, green, blue) {
  rgb[target] = red >>> 8;
  rgb[target + 1] = green >>> 8;
  rgb[target + 2] = blue >>> 8;
}

function nextLzwCode(bytes, state) {
  if (state.bitOffset + state.bitLength > bytes.length * 8) return null;
  let code = 0;
  for (let bit = 0; bit < state.bitLength; bit++) {
    const offset = state.bitOffset + bit;
    code = (code << 1) | ((bytes[offset >> 3] >> (7 - (offset & 7))) & 1);
  }
  state.bitOffset += state.bitLength;
  return code;
}

function resetLzwTable(table, state) {
  for (let value = 0; value < 256; value++) table[value] = Uint8Array.of(value);
  state.tableLength = 258;
  state.bitLength = 9;
}

function lzwEntry(table, state, old, code) {
  if (code < state.tableLength) return table[code];
  if (code !== state.tableLength || old === null) return null;
  const entry = new Uint8Array(old.length + 1);
  entry.set(old);
  entry[old.length] = old[0];
  return entry;
}

function writeLzwEntry(entry, writer, outputOffset, expectedLength) {
  if (outputOffset + entry.length > expectedLength) {
    throw new TypeError('Kanana TIFF LZW data exceeds its declared strip size.');
  }
  writer.write(entry);
  return outputOffset + entry.length;
}

function addLzwEntry(table, state, old, entry) {
  if (old === null || state.tableLength >= 4096) return;
  const next = new Uint8Array(old.length + 1);
  next.set(old);
  next[old.length] = entry[0];
  table[state.tableLength++] = next;
  const widthLimit = 2 ** state.bitLength - 1;
  if (state.tableLength === widthLimit && state.bitLength < 12) state.bitLength++;
}

function createStripWriter(image, rgb, rowOffset, expectedLength) {
  const {width, channels, littleEndian} = image;
  let byteLength = 0;
  let firstByte = null;
  let channel = 0;
  let pixel = 0;
  let red = 0;
  let green = 0;
  let blue = 0;
  const rows = expectedLength / (width * channels * 2);
  const pixelCount = rows * width;

  function writeSample(sample) {
    const target = (rowOffset * width + pixel) * RGB_CHANNELS;
    if (channels === 1) {
      writeGrayPixel(rgb, target, sample);
      pixel++;
      return;
    }

    if (channel === 0) red = sample;
    else if (channel === 1) green = sample;
    else {
      blue = sample;
      writeRgbPixel(rgb, target, red, green, blue);
      pixel++;
    }
    channel = (channel + 1) % RGB_CHANNELS;
  }

  return {
    write(bytes) {
      if (byteLength + bytes.length > expectedLength) {
        throw new TypeError('Kanana TIFF compressed data exceeds its declared strip size.');
      }
      byteLength += bytes.length;
      for (const byte of bytes) {
        if (firstByte === null) firstByte = byte;
        else {
          writeSample(littleEndian ? firstByte | (byte << 8) : (firstByte << 8) | byte);
          firstByte = null;
        }
      }
    },
    finish() {
      if (byteLength !== expectedLength || firstByte !== null || channel !== 0 || pixel !== pixelCount) {
        throw new TypeError('Kanana TIFF compressed strip is incomplete.');
      }
    }
  };
}

function decodeLzwStrip(bytes, expectedLength, writer, signal) {
  const table = new Array(4096);
  const state = {bitOffset: 0, bitLength: 9, tableLength: 258};
  let outputOffset = 0;
  let old = null;
  let ended = false;
  resetLzwTable(table, state);

  while (true) {
    if ((outputOffset & 0xffff) === 0) signal?.throwIfAborted();
    const code = nextLzwCode(bytes, state);
    if (code === null) break;
    if (code === 257) {
      ended = true;
      break;
    }
    if (code === 256) {
      resetLzwTable(table, state);
      old = null;
      continue;
    }

    const entry = lzwEntry(table, state, old, code);
    if (!entry) throw new TypeError('Kanana TIFF contains invalid LZW data.');
    outputOffset = writeLzwEntry(entry, writer, outputOffset, expectedLength);
    addLzwEntry(table, state, old, entry);
    old = entry;
  }

  if (!ended || outputOffset !== expectedLength) throw new TypeError('Kanana TIFF LZW strip is incomplete.');
}

async function inflateStrip(bytes, expectedLength, writer, signal) {
  if (typeof globalThis.DecompressionStream !== 'function' || typeof globalThis.ReadableStream !== 'function') {
    throw new TypeError('Kanana TIFF Deflate requires browser DecompressionStream support.');
  }

  const source = new ReadableStream({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    }
  });
  const stream = source.pipeThrough(new DecompressionStream('deflate'));
  const reader = stream.getReader();
  let outputLength = 0;
  const abort = () => reader.cancel(signal.reason).catch(() => {});
  signal?.addEventListener('abort', abort, {once: true});

  try {
    while (true) {
      signal?.throwIfAborted();
      const {done, value} = await reader.read();
      if (done) break;
      if (outputLength + value.length > expectedLength) {
        throw new TypeError('Kanana TIFF Deflate data exceeds its declared strip size.');
      }
      writer.write(value);
      outputLength += value.length;
    }
  } finally {
    signal?.removeEventListener('abort', abort);
    reader.releaseLock();
  }

  signal?.throwIfAborted();
  if (outputLength !== expectedLength) throw new TypeError('Kanana TIFF Deflate strip is incomplete.');
}

async function stripPixels(image, signal) {
  const {width, height, channels, offsets, byteCounts, stripLengths, view, compression} = image;
  const rgb = new Uint8Array(width * height * RGB_CHANNELS);
  const samplesPerRow = width * channels;
  let rowOffset = 0;

  for (let strip = 0; strip < offsets.length; strip++) {
    signal?.throwIfAborted();
    const start = view.byteOffset + offsets[strip];
    const compressed = new Uint8Array(view.buffer, start, byteCounts[strip]);
    const writer = createStripWriter(image, rgb, rowOffset, stripLengths[strip]);
    if (compression === 1) writer.write(compressed);
    else if (compression === 5) decodeLzwStrip(compressed, stripLengths[strip], writer, signal);
    else await inflateStrip(compressed, stripLengths[strip], writer, signal);
    writer.finish();
    const rows = stripLengths[strip] / (samplesPerRow * 2);
    rowOffset += rows;
  }

  return {rgb, width, height};
}

/** Decode the tested classic TIFF 16-bit grayscale and chunky RGB subset. */
export async function decodeKananaTiff16Image(source, options = {}) {
  const signal = options.signal;
  signal?.throwIfAborted();
  const bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
  const image = inspectTiff16(bytes);
  if (!image) return null;
  signal?.throwIfAborted();
  return stripPixels(image, signal);
}
