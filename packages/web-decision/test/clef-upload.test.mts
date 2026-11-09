import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {uploadFakeClefParameters} from '../src/clef/upload.mjs';

type FakeUploadOptions = Parameters<typeof uploadFakeClefParameters>[1];

interface FakeClefEntry {
  position: number;
  target: string;
  dtype: 'uint32' | 'float16' | 'float32';
  shape: number[];
  bytes: number;
  chunks: Array<{offset: number; length: number; sha256: string}>;
}

const hash = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex');
const entry = (data: Uint8Array, dtype: FakeClefEntry['dtype'] = 'float16'): FakeClefEntry => ({
  position: 17,
  target: 'x',
  dtype,
  shape: [data.length / (dtype === 'float16' ? 2 : 4)],
  bytes: data.length,
  chunks: [{offset: 0, length: data.length, sha256: hash(data)}]
});

test('authenticates and pads an FP16 two-byte tail before an aligned write', async () => {
  const bytes = Uint8Array.of(0, 60, 0, 64, 0, 66),
    writes: Array<[number, number[]]> = [];
  const result = await uploadFakeClefParameters([entry(bytes)], {
    fake: true,
    createTensor: () => ({}),
    readRange: (() => bytes) as unknown as FakeUploadOptions['readRange'],
    writeRange: (_tensor, offset, data) => writes.push([offset, [...data]])
  });
  assert.equal(result.length, 1);
  assert.deepEqual(writes, [[0, [0, 60, 0, 64, 0, 66, 0, 0]]]);
});

test('rejects bad hashes, nonfinite values, and malformed ranges', async () => {
  const bytes = Uint8Array.of(0, 60, 0, 64),
    metadata = entry(bytes),
    options: FakeUploadOptions = {
      fake: true,
      createTensor: () => ({}),
      readRange: (() => Uint8Array.of(1, 60, 0, 64)) as unknown as FakeUploadOptions['readRange'],
      writeRange: () => assert.fail('must not write')
    };
  await assert.rejects(uploadFakeClefParameters([metadata], options), /hash mismatch/);
  const nonfinite = Uint8Array.of(0, 124, 0, 64);
  await assert.rejects(
    uploadFakeClefParameters([entry(nonfinite)], {
      ...options,
      readRange: (() => nonfinite) as unknown as FakeUploadOptions['readRange']
    }),
    /Nonfinite/
  );
  const bad = structuredClone(metadata);
  bad.chunks[0]!.offset = 4;
  await assert.rejects(uploadFakeClefParameters([bad], options), /range/);
});

test('waits for an issued write to settle after cancellation', async () => {
  const bytes = Uint8Array.of(0, 60, 0, 64),
    controller = new AbortController();
  let release!: () => void, started!: () => void;
  const entered = new Promise<void>((resolve) => {
    started = resolve;
  });
  const pending = new Promise<void>((resolve) => {
    release = resolve;
  });
  let settled = false;
  const operation = uploadFakeClefParameters([entry(bytes)], {
    fake: true,
    signal: controller.signal,
    createTensor: () => ({}),
    readRange: (() => bytes) as unknown as FakeUploadOptions['readRange'],
    writeRange: async () => {
      started();
      await pending;
    }
  }).finally(() => {
    settled = true;
  });
  await entered;
  controller.abort();
  await Promise.resolve();
  assert.equal(settled, false);
  release();
  await assert.rejects(operation, {name: 'AbortError'});
});

test('abort during parameter hashing returns promptly and handles a late rejection', async (t) => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  let started!: () => void, rejectHash!: (reason?: unknown) => void;
  const entered = new Promise<void>((resolve) => {
    started = resolve;
  });
  const pending = new Promise<ArrayBuffer>((_, reject) => {
    rejectHash = reject;
  });
  Object.defineProperty(globalThis, 'crypto', {
    configurable: true,
    value: {
      subtle: {
        digest() {
          started();
          return pending;
        }
      }
    }
  });
  t.after(() =>
    original ? Object.defineProperty(globalThis, 'crypto', original) : delete (globalThis as {crypto?: Crypto}).crypto
  );
  const bytes = Uint8Array.of(0, 60, 0, 64),
    controller = new AbortController(),
    reason = Error('hash cancelled');
  let writes = 0;
  const operation = uploadFakeClefParameters([entry(bytes)], {
    fake: true,
    signal: controller.signal,
    createTensor: () => ({}),
    readRange: (() => bytes) as unknown as FakeUploadOptions['readRange'],
    writeRange: () => {
      writes++;
    }
  });
  const outcome = operation.then(
    () => 'resolved',
    (error) => error
  );
  await entered;
  controller.abort(reason);
  const observed = await Promise.race([outcome, new Promise((resolve) => setTimeout(() => resolve('pending'), 50))]);
  rejectHash(Error('late hash failure'));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(observed, reason);
  assert.equal(writes, 0);
});

test('abort after the digest settles takes precedence over a mismatching hash', async (t) => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  let started!: () => void, resolveHash!: (value: ArrayBuffer) => void;
  const entered = new Promise<void>((resolve) => {
    started = resolve;
  });
  const pending = new Promise<ArrayBuffer>((resolve) => {
    resolveHash = resolve;
  });
  Object.defineProperty(globalThis, 'crypto', {
    configurable: true,
    value: {
      subtle: {
        digest() {
          started();
          return pending;
        }
      }
    }
  });
  t.after(() =>
    original ? Object.defineProperty(globalThis, 'crypto', original) : delete (globalThis as {crypto?: Crypto}).crypto
  );
  const bytes = Uint8Array.of(0, 60, 0, 64),
    controller = new AbortController(),
    reason = Error('hash cancelled');
  let writes = 0;
  const operation = uploadFakeClefParameters([entry(bytes)], {
    fake: true,
    signal: controller.signal,
    createTensor: () => ({}),
    readRange: (() => bytes) as unknown as FakeUploadOptions['readRange'],
    writeRange: () => {
      writes++;
    }
  });
  await entered;
  resolveHash(new Uint8Array(32).buffer);
  queueMicrotask(() => queueMicrotask(() => controller.abort(reason)));
  await assert.rejects(operation, (error) => error === reason);
  assert.equal(writes, 0);
});
