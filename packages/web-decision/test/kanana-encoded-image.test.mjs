import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFile} from 'node:fs/promises';
import {test, before} from 'node:test';
import {fileURLToPath} from 'node:url';
import {init as initJpeg} from '@jsquash/jpeg/decode.js';
import {encodedImage} from './kanana-core-modules.mjs';

const {decodeKananaEncodedImageSource, preflightKananaImageHeader} = encodedImage;

const fixtureDirectory = new URL('./fixtures/kanana-image/', import.meta.url);
const require = createRequire(import.meta.url);
const jpegWasmPath = require.resolve('@jsquash/jpeg/codec/dec/mozjpeg_dec.wasm');
const magickWasmPath = fileURLToPath(import.meta.resolve('@imagemagick/magick-wasm/magick.wasm'));
const magickWasm = await readFile(magickWasmPath);

async function fixture(name) {
  return readFile(new URL(name, fixtureDirectory));
}

async function expectedRgb(name) {
  return Uint8Array.from(await fixture(name));
}

function writeUint32(bytes, offset, value) {
  for (let index = 0; index < 4; index++) {
    bytes[offset + index] = Math.floor(value / 2 ** (24 - index * 8)) & 0xff;
  }
}

function setClassicTiffTag(bytes, tag, value) {
  const littleEndian = bytes[0] === 73;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const directory = view.getUint32(4, littleEndian);
  const entryCount = view.getUint16(directory, littleEndian);

  for (let index = 0; index < entryCount; index++) {
    const entry = directory + 2 + index * 12;
    if (view.getUint16(entry, littleEndian) !== tag) continue;
    const type = view.getUint16(entry + 2, littleEndian);
    if (type === 4) view.setUint32(entry + 8, value, littleEndian);
    else view.setUint16(entry + 8, value, littleEndian);
    return;
  }
  throw new Error(`Missing TIFF tag ${tag}`);
}

function pngHeader(width, height) {
  const bytes = new Uint8Array(24);
  bytes.set([137, 80, 78, 71, 13, 10, 26, 10]);
  bytes.set([0, 0, 0, 13, 73, 72, 68, 82], 8);
  writeUint32(bytes, 16, width);
  writeUint32(bytes, 20, height);
  return bytes;
}

function jpegHeader(width, height) {
  return Uint8Array.from([
    0xff,
    0xd8,
    0xff,
    0xc0,
    0,
    11,
    8,
    height >> 8,
    height & 0xff,
    width >> 8,
    width & 0xff,
    1,
    1,
    0x11,
    0
  ]);
}

async function assertPixels(source, golden, width, height, options) {
  if (source instanceof Uint8Array) source = new Blob([source]);
  const result = await decodeKananaEncodedImageSource(source, options);
  assert.equal(result.width, width);
  assert.equal(result.height, height);
  assert.deepEqual(result.rgb, await expectedRgb(golden));
}

before(async () => {
  const wasmBytes = await readFile(jpegWasmPath);
  await initJpeg(new WebAssembly.Module(wasmBytes));
});

test('PNG IHDR and JPEG SOF headers reject dimensions beyond Pillow 12.1.1 limits before decoding', () => {
  assert.throws(() => preflightKananaImageHeader(pngHeader(178_956_971, 1)), {
    name: 'RangeError',
    message: /Pillow 12\.1\.1's decompression-bomb limit of 178956970 pixels/
  });
  assert.throws(() => preflightKananaImageHeader(jpegHeader(65_535, 2_732)), {
    name: 'RangeError',
    message: /Pillow 12\.1\.1's decompression-bomb limit of 178956970 pixels/
  });
});

test('encoded PNG entry point rejects an oversized IHDR before invoking the PNG decoder', async () => {
  await assert.rejects(decodeKananaEncodedImageSource(new Blob([pngHeader(178_956_971, 1)])), {
    name: 'RangeError',
    message: /Pillow 12\.1\.1's decompression-bomb limit of 178956970 pixels/
  });
});

test('Pillow warning-range PNG and JPEG headers pass preflight', () => {
  const originalWarn = console.warn;
  const warnings = [];
  console.warn = (message) => warnings.push(message);

  try {
    assert.deepEqual(preflightKananaImageHeader(pngHeader(89_478_486, 1)), {
      format: 'png',
      width: 89_478_486,
      height: 1
    });
    assert.deepEqual(preflightKananaImageHeader(jpegHeader(65_535, 1_366)), {
      format: 'jpeg',
      width: 65_535,
      height: 1_366,
      components: 1
    });
  } finally {
    console.warn = originalWarn;
  }

  assert.equal(warnings.length, 2);
  assert.match(warnings[0], /exceeds Pillow 12\.1\.1's warning limit of 89478485 pixels/);
  assert.match(warnings[1], /exceeds Pillow 12\.1\.1's warning limit of 89478485 pixels/);
});

test('PNG RGBA uses Pillow white compositing and preserves byte rounding', async () => {
  const encoded = await fixture('rgba-alpha.png');
  await assertPixels(new Blob([encoded]), 'rgba-alpha.rgb', 6, 1);
});

test('PNG palette transparency does not composite palette colors', async () => {
  await assertPixels(await fixture('palette-trns.png'), 'palette-trns.rgb', 3, 1);
});

test('PNG RGB tRNS does not change RGB samples', async () => {
  await assertPixels(await fixture('rgb-trns.png'), 'rgb-trns.rgb', 3, 1);
});

test('PNG LA conversion repeats luminance and ignores alpha', async () => {
  await assertPixels(await fixture('la.png'), 'la.rgb', 3, 1);
});

test('grayscale PNG converts 1, 2, 4, 8, and 16-bit samples', async () => {
  for (const depth of [1, 2, 4, 8, 16]) {
    const width = depth === 16 ? 3 : 5;
    await assertPixels(await fixture(`gray-${depth}bit.png`), `gray-${depth}bit.rgb`, width, 2);
  }
});

test('indexed PNG converts 1, 2, 4, and 8-bit palette samples', async () => {
  for (const depth of [1, 2, 4, 8]) {
    await assertPixels(await fixture(`palette-${depth}bit.png`), `palette-${depth}bit.rgb`, 5, 2);
  }
});

test('16-bit RGB and RGBA PNG conversion matches Pillow byte-for-byte', async () => {
  await assertPixels(await fixture('rgb-16bit.png'), 'rgb-16bit.rgb', 3, 2);
  await assertPixels(await fixture('rgba-16bit.png'), 'rgba-16bit.rgb', 3, 2);
});

test('RGB JPEG preserves raw raster orientation when EXIF says rotate', async () => {
  await assertPixels(await fixture('rgb-exif6.jpg'), 'rgb-exif6.rgb', 64, 32);
});

test('grayscale JPEG converts to RGB like the Pillow golden', async () => {
  await assertPixels(await fixture('gray.jpg'), 'gray.rgb', 64, 32);
});

test('URL object input uses the encoded decoder path', async () => {
  const url = new URL('https://images.example/image.png');
  const encoded = await fixture('rgba-alpha.png');
  await assertPixels(url, 'rgba-alpha.rgb', 6, 1, {
    fetch: async () => new Response(new Blob([encoded]), {status: 200})
  });
});

test('URL string input uses the encoded decoder path', async () => {
  const encoded = await fixture('rgb-exif6.jpg');
  const result = await decodeKananaEncodedImageSource('https://images.example/image.jpg', {
    fetch: async () => new Response(new Blob([encoded]), {status: 200})
  });
  assert.equal(result.width, 64);
  assert.equal(result.height, 32);
  assert.deepEqual(result.rgb, await expectedRgb('rgb-exif6.rgb'));
});

test('HTMLImageElement input fetches currentSrc and reports unreadable CORS', async () => {
  const previousImageElement = globalThis.HTMLImageElement;
  class MockImageElement {}
  globalThis.HTMLImageElement = MockImageElement;
  const image = Object.assign(new MockImageElement(), {
    currentSrc: 'https://images.example/current.png',
    src: 'https://images.example/fallback.png'
  });
  const encoded = await fixture('rgba-alpha.png');
  let fetchedUrl;

  try {
    await assertPixels(image, 'rgba-alpha.rgb', 6, 1, {
      fetch: async (url) => {
        fetchedUrl = url;
        return new Response(new Blob([encoded]), {status: 200});
      }
    });
    assert.equal(fetchedUrl, image.currentSrc);
    await assert.rejects(
      decodeKananaEncodedImageSource(image, {
        fetch: async () => {
          throw new TypeError('blocked');
        }
      }),
      /readable CORS response/
    );
  } finally {
    if (previousImageElement === undefined) delete globalThis.HTMLImageElement;
    else globalThis.HTMLImageElement = previousImageElement;
  }
});

test('CMYK JPEG matches the pinned Pillow RGB golden through the encoded entry point', async () => {
  const encoded = await fixture('cmyk.jpg');
  await assertPixels(encoded, 'cmyk.rgb', 16, 8, {magickWasm});
});

test('16-bit grayscale and RGB TIFF use Pillow-compatible sample conversion', async () => {
  const variants = [
    {name: 'gray-16bit-le', width: 4, height: 2},
    {name: 'gray-16bit-lzw', width: 4, height: 2},
    {name: 'gray-16bit-lzw-wide', width: 64, height: 16},
    {name: 'gray-16bit-be-deflate', width: 4, height: 2},
    {name: 'rgb-16bit-le', width: 4, height: 2},
    {name: 'rgb-16bit-be-deflate', width: 4, height: 2},
    {name: 'rgb-16bit-be-deflate-strips', width: 4, height: 2}
  ];
  for (const {name, width, height} of variants) {
    await assertPixels(await fixture(`${name}.tif`), `${name}.rgb`, width, height);
  }
});

test('unsupported 16-bit TIFF layouts do not fall back to lossy Q8 decoding', async () => {
  const encoded = new Uint8Array(await fixture('rgb-16bit-le.tif'));
  setClassicTiffTag(encoded, 259, 32773);

  await assert.rejects(
    decodeKananaEncodedImageSource(new Blob([encoded]), {magickWasm}),
    /16-bit TIFF layout is not covered by the Pillow parity decoder/
  );
});

test('unsupported encoded formats fail explicitly', async () => {
  await assert.rejects(
    decodeKananaEncodedImageSource(new Blob(['not an image'])),
    /supports BMP, GIF, WebP, TIFF, and CMYK JPEG/
  );
});

test('PNG with a corrupt chunk CRC is rejected', async () => {
  const encoded = await fixture('rgba-alpha.png');
  encoded[encoded.length - 1] ^= 1;
  await assert.rejects(decodeKananaEncodedImageSource(new Blob([encoded])), /CRC mismatch/);
});

test('data URL input is resolved through fetch', async () => {
  const encoded = await fixture('la.png');
  const dataUrl = `data:image/png;base64,${encoded.toString('base64')}`;
  const result = await decodeKananaEncodedImageSource(dataUrl, {
    fetch: async () => new Response(new Blob([encoded]), {status: 200})
  });
  assert.equal(result.width, 3);
  assert.equal(result.height, 1);
  assert.deepEqual(result.rgb, await expectedRgb('la.rgb'));
});

test('abort after URL fetch prevents decoding response bytes', async () => {
  const controller = new AbortController();
  const encoded = await fixture('la.png');
  await assert.rejects(
    decodeKananaEncodedImageSource('https://images.example/image.png', {
      signal: controller.signal,
      fetch: async (_url, options) => {
        assert.equal(options.signal, controller.signal);
        controller.abort();
        return new Response(new Blob([encoded]), {status: 200});
      }
    }),
    {name: 'AbortError'}
  );
});
