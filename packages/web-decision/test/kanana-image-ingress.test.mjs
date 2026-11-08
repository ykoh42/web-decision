import assert from 'node:assert/strict';
import {File} from 'node:buffer';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {decodeKananaImageSource} from '../src/kanana/browser-image.mjs';

const fixtureDirectory = new URL('./fixtures/kanana-image/', import.meta.url);

async function fixture(name) {
  return readFile(new URL(name, fixtureDirectory));
}

async function rgbFixture(name) {
  return Uint8Array.from(await fixture(name));
}

async function decode(source, options) {
  return decodeKananaImageSource(source, {
    createImageBitmap: () => assert.fail('encoded inputs must not use native decoding'),
    createCanvas: () => assert.fail('encoded inputs must not use canvas readback'),
    ...options
  });
}

test('Blob and File inputs preserve encoded PNG mode conversion', async () => {
  const encoded = await fixture('rgba-alpha.png');
  const expected = await rgbFixture('rgba-alpha.rgb');
  const blob = new Blob([encoded], {type: 'image/png'});
  const file = new File([encoded], 'transparent.png', {type: 'image/png'});

  const fromBlob = await decode(blob);
  const fromFile = await decode(file);

  assert.deepEqual(fromBlob, {rgb: expected, width: 6, height: 1});
  assert.deepEqual(fromFile, fromBlob);
});

test('URL objects, URL strings, and image data URLs fetch original bytes', async () => {
  const encoded = await fixture('palette-trns.png');
  const expected = await rgbFixture('palette-trns.rgb');
  const addresses = [];
  const fetch = async (address) => {
    addresses.push(address);
    return new Response(new Blob([encoded], {type: 'image/png'}), {status: 200});
  };

  const sources = [
    new URL('https://example.test/photo.png'),
    'https://example.test/photo.png',
    'data:image/png;base64,AA=='
  ];
  for (const source of sources) {
    const decoded = await decode(source, {fetch});
    assert.deepEqual(decoded.rgb, expected);
    assert.equal(decoded.width, 3);
    assert.equal(decoded.height, 1);
  }

  assert.deepEqual(addresses, [
    'https://example.test/photo.png',
    'https://example.test/photo.png',
    'data:image/png;base64,AA=='
  ]);
});

test('HTMLImageElement fetches currentSrc to recover the original bytes', async () => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'HTMLImageElement');
  Object.defineProperty(globalThis, 'HTMLImageElement', {
    value: class BrowserImageElement {
      constructor() {
        this.currentSrc = 'https://example.test/current.png';
        this.src = 'https://example.test/fallback.png';
      }
    },
    configurable: true,
    writable: true
  });

  try {
    const source = new globalThis.HTMLImageElement();
    const encoded = await fixture('rgb-trns.png');
    const expected = await rgbFixture('rgb-trns.rgb');
    const addresses = [];
    const result = await decode(source, {
      fetch: async (address) => {
        addresses.push(address);
        return new Response(new Blob([encoded]), {status: 200});
      }
    });

    assert.deepEqual(addresses, [source.currentSrc]);
    assert.deepEqual(result.rgb, expected);
  } finally {
    if (previous) Object.defineProperty(globalThis, 'HTMLImageElement', previous);
    else delete globalThis.HTMLImageElement;
  }
});

test('HTMLImageElement CORS failures are reported as unreadable source bytes', async () => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'HTMLImageElement');
  Object.defineProperty(globalThis, 'HTMLImageElement', {
    value: class BrowserImageElement {
      currentSrc = 'https://foreign.example/photo.png';
    },
    configurable: true,
    writable: true
  });

  try {
    await assert.rejects(
      decode(new globalThis.HTMLImageElement(), {
        fetch: async () => {
          throw new TypeError('blocked by CORS');
        }
      }),
      /readable CORS response/
    );
  } finally {
    if (previous) Object.defineProperty(globalThis, 'HTMLImageElement', previous);
    else delete globalThis.HTMLImageElement;
  }
});

test('already-decoded ImageBitmap and canvas inputs are rejected explicitly', async () => {
  const names = ['ImageBitmap', 'HTMLCanvasElement', 'OffscreenCanvas'];
  const previous = new Map(names.map((name) => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
  for (const name of names) {
    Object.defineProperty(globalThis, name, {
      value: class DecodedImageSource {},
      configurable: true,
      writable: true
    });
  }

  try {
    for (const name of names) {
      await assert.rejects(decode(new globalThis[name]()), /original image mode and orientation/);
    }
  } finally {
    for (const name of names) {
      const descriptor = previous.get(name);
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
  }
});

test('unsupported encoded bytes fail without native decoder fallback', async () => {
  const unsupported = new Blob(['not an image'], {type: 'image/webp'});
  await assert.rejects(decode(unsupported), /supports BMP, GIF, WebP, TIFF, and CMYK JPEG/);
});

test('URL reads preserve cancellation and reject opaque or failed responses', async () => {
  const signal = new AbortController().signal;
  let receivedOptions;
  const input = 'https://example.test/photo.png';
  const encoded = await fixture('rgba-alpha.png');

  const result = await decode(input, {
    signal,
    fetch: async (_address, options) => {
      receivedOptions = options;
      return new Response(new Blob([encoded]), {status: 200});
    }
  });
  assert.strictEqual(receivedOptions.signal, signal);
  assert.deepEqual(result.rgb, await rgbFixture('rgba-alpha.rgb'));

  await assert.rejects(
    decode(input, {fetch: async () => ({ok: false, status: 404})}),
    /request failed with status 404/
  );
  await assert.rejects(decode(input, {fetch: async () => ({ok: true, type: 'opaque'})}), /readable CORS response/);
});

test('pre-aborted signals stop URL, Blob, and drawable inputs before work starts', async () => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'HTMLCanvasElement');
  Object.defineProperty(globalThis, 'HTMLCanvasElement', {
    value: class BrowserCanvas {},
    configurable: true,
    writable: true
  });

  try {
    const controller = new AbortController();
    controller.abort();
    const sources = [
      new URL('https://example.test/photo.png'),
      new Blob(['image']),
      new globalThis.HTMLCanvasElement()
    ];

    for (const source of sources) {
      await assert.rejects(
        decodeKananaImageSource(source, {
          signal: controller.signal,
          fetch: async () => assert.fail('pre-aborted input must not be fetched'),
          createImageBitmap: async () => assert.fail('native decoding must not run'),
          createCanvas: () => assert.fail('canvas readback must not run')
        }),
        {name: 'AbortError'}
      );
    }
  } finally {
    if (previous) Object.defineProperty(globalThis, 'HTMLCanvasElement', previous);
    else delete globalThis.HTMLCanvasElement;
  }
});

test('abort during URL fetch skips decoding response bytes', async () => {
  const controller = new AbortController();
  let startFetch;
  let finishFetch;
  const fetchingStarted = new Promise((resolve) => {
    startFetch = resolve;
  });
  const task = decodeKananaImageSource('https://example.test/photo.png', {
    signal: controller.signal,
    fetch: () =>
      new Promise((resolve) => {
        finishFetch = resolve;
        startFetch();
      }),
    createImageBitmap: async () => assert.fail('native decoding must not run'),
    createCanvas: () => assert.fail('canvas readback must not run')
  });

  await fetchingStarted;
  controller.abort();
  finishFetch(new Response(new Blob([await fixture('rgba-alpha.png')]), {status: 200}));

  await assert.rejects(task, {name: 'AbortError'});
});

test('unsupported URL schemes and non-image data URLs fail explicitly', async () => {
  await assert.rejects(decodeKananaImageSource('data:text/plain,hello', {fetch: async () => assert.fail()}), {
    name: 'TypeError'
  });
  await assert.rejects(decodeKananaImageSource('javascript:alert(1)', {fetch: async () => assert.fail()}), {
    name: 'TypeError'
  });
  await assert.rejects(decodeKananaImageSource('/relative/image.png', {fetch: async () => assert.fail()}), {
    name: 'TypeError'
  });
});
