import assert from 'node:assert/strict';
import test from 'node:test';
import type {DLDevice, Instance, Tensor} from '../src/runtime/tvm-types.js';
import {uploadMapika2bTextParameters} from '../src/mapika2b/text-runtime.mts';

test('uploads pinned parameter order and reports the verified payload bytes', async () => {
  const uploads: number[] = [];
  const progress: number[] = [];
  const runtime = {
    empty() {
      return {
        copyFromRawBytes(bytes: Uint8Array) {
          uploads.push(bytes[0]!);
          return this;
        }
      } as unknown as Tensor;
    }
  } as unknown as Instance;
  const device = {async sync() {}} as DLDevice;
  const bindings = Array.from({length: 319}, (_, index) => ({name: String(index), shape: [1]}));
  const answers = new Uint16Array(255 * 2048);
  const parameters = await uploadMapika2bTextParameters(
    runtime,
    device,
    bindings,
    answers,
    async ({name}) => new Uint8Array([Number(name) % 256, 0, 0, 0]),
    undefined,
    (bytes) => progress.push(bytes)
  );
  assert.equal(parameters.length, 320);
  assert.deepEqual(
    uploads.slice(1),
    bindings.map(({name}) => Number(name) % 256)
  );
  assert.equal(progress[0], answers.byteLength);
  assert.deepEqual(progress.slice(1), Array<number>(319).fill(4));
});

test('aborting a parameter read stops before uploading that tensor', async () => {
  let uploads = 0;
  const controller = new AbortController();
  const reason = new Error('cancelled');
  const runtime = {
    empty() {
      return {
        copyFromRawBytes() {
          uploads += 1;
          return this;
        }
      } as unknown as Tensor;
    }
  } as unknown as Instance;
  const device = {async sync() {}} as DLDevice;
  const bindings = Array.from({length: 319}, (_, index) => ({name: String(index), shape: [1]}));
  await assert.rejects(
    uploadMapika2bTextParameters(
      runtime,
      device,
      bindings,
      new Uint16Array(255 * 2048),
      async () => {
        controller.abort(reason);
        return new Uint8Array(4);
      },
      controller.signal
    ),
    (error) => error === reason
  );
  assert.equal(uploads, 1);
});

test('prefetches a bounded group while preserving GPU argument order', async () => {
  let active = 0;
  let maximumActive = 0;
  const uploads: number[] = [];
  const runtime = {
    empty() {
      return {
        copyFromRawBytes(bytes: Uint8Array) {
          uploads.push(bytes[0]!);
          return this;
        }
      } as unknown as Tensor;
    }
  } as unknown as Instance;
  const device = {async sync() {}} as DLDevice;
  const bindings = Array.from({length: 319}, (_, index) => ({name: String(index), shape: [1]}));
  await uploadMapika2bTextParameters(runtime, device, bindings, new Uint16Array(255 * 2048), async ({name}) => {
    active += 1;
    maximumActive = Math.max(active, maximumActive);
    await new Promise((resolve) => setTimeout(resolve, 1));
    active -= 1;
    return new Uint8Array([Number(name) % 256, 0, 0, 0]);
  });
  assert.equal(maximumActive, 4);
  assert.deepEqual(
    uploads.slice(1),
    bindings.map(({name}) => Number(name) % 256)
  );
});
