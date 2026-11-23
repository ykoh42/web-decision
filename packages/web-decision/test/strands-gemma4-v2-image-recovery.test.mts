import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {test} from 'node:test';
import {init as initJpeg} from '@jsquash/jpeg/decode.js';
import {prepareGemma4V2EncodedImage} from '../src/strands/gemma4-v2-image.mts';
import {createGemma4V2Session} from '../src/strands/gemma4-v2-session.mts';

const fixtures = new URL('./fixtures/qwen35-image-input/', import.meta.url);
const require = createRequire(import.meta.url);
const request = (image: string) => ({state: 's', images: [image], questions: {q: {instructions: 'yes?'}}});
async function encoded(name: string) {
  return (await readFile(new URL(name, fixtures))).toString('base64');
}
async function initCodec() {
  await initJpeg({wasmBinary: await readFile(require.resolve('@jsquash/jpeg/codec/dec/mozjpeg_dec.wasm'))});
}

function create(codecs: {prepareJpeg(): Promise<void>; loadMagick(): Promise<Uint8Array>}) {
  let calls = 0;
  let disposed = false;
  const session = createGemma4V2Session({
    tokenizer: {
      encode(text) {
        return {ids: Array.from(new TextEncoder().encode(text))};
      }
    },
    runner: {
      async run() {
        calls++;
        return [0, 0];
      },
      releaseCachedMemory() {},
      dispose() {
        disposed = true;
      }
    },
    async prepareImage(image, signal) {
      const value = await prepareGemma4V2EncodedImage(image, signal === undefined ? {codecs} : {codecs, signal});
      return {tokens: value.tokens, value};
    },
    modelName: 'e2b-v2',
    temperature: 1.5456,
    softcap: 30
  });
  return {session, calls: () => calls, disposed: () => disposed};
}

test('invalid data and decoder initialization failure do not poison the session', async () => {
  let attempts = 0;
  const {session, calls} = create({
    async prepareJpeg() {
      if (++attempts === 1) throw new Error('jpeg initialization failed');
      await initCodec();
    },
    async loadMagick() {
      throw new Error('Unexpected ImageMagick load');
    }
  });
  const jpeg = await encoded('jpeg-orientation-6.jpg');
  const png = await encoded('single-pixel.png');
  try {
    await assert.rejects(session.systemOne(request('%%%')), /base64/i);
    await assert.rejects(session.systemOne(request(jpeg)), /jpeg initialization failed/i);
    assert.equal(calls(), 0);
    await session.systemOne(request(png));
    await session.systemOne(request(jpeg));
    assert.equal(attempts, 2);
    assert.equal(calls(), 2);
  } finally {
    await session.dispose();
  }
});

test('cancellation while JPEG initializes permits retry and orderly disposal', async () => {
  let started!: () => void;
  let release!: () => void;
  const began = new Promise<void>((resolve) => {
    started = resolve;
  });
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const initialization = gate.then(initCodec);
  let attempts = 0;
  const {session, calls, disposed} = create({
    prepareJpeg() {
      attempts++;
      started();
      return initialization;
    },
    async loadMagick() {
      throw new Error('Unexpected ImageMagick load');
    }
  });
  const jpeg = await encoded('jpeg-orientation-6.jpg');
  const controller = new AbortController();
  const first = session.systemOne(request(jpeg), {signal: controller.signal});
  await began;
  controller.abort();
  await assert.rejects(first, {name: 'AbortError'});
  const retry = session.systemOne(request(jpeg));
  release();
  await retry;
  assert.equal(calls(), 1);
  assert.equal(attempts, 2);
  assert.equal(disposed(), false);
  await session.dispose();
  assert.equal(disposed(), true);
  await assert.rejects(session.systemOne(request(jpeg)), /disposed|closed/i);
});

test('disposing while an image decoder is active waits for the request to finish', async () => {
  let started!: () => void;
  let release!: () => void;
  const began = new Promise<void>((resolve) => {
    started = resolve;
  });
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const {session, calls, disposed} = create({
    async prepareJpeg() {
      started();
      await gate;
      await initCodec();
    },
    async loadMagick() {
      throw new Error('Unexpected ImageMagick load');
    }
  });
  const jpeg = await encoded('jpeg-orientation-6.jpg');
  const running = session.systemOne(request(jpeg));
  await began;
  const closing = session.dispose();
  assert.equal(disposed(), false);
  release();
  await running;
  await closing;
  assert.equal(calls(), 1);
  assert.equal(disposed(), true);
  await assert.rejects(session.systemOne(request(jpeg)), /disposed|closed/i);
});
