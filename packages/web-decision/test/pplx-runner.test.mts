import assert from 'node:assert/strict';
import test from 'node:test';
import {FlashTvmRunner} from '../src/pplx/runner.mjs';

interface FakeTensor {
  value?: Float32Array | number[];
}

function fixture() {
  const calls: FakeTensor[][] = [],
    writes: {bytes: number[]; length: number}[] = [],
    events: string[] = [];
  const context = {
    pendingGPUToCPUCopy: null as Promise<void> | null,
    copyRawBytesToBuffer(data: Uint8Array, _ptr: number, _offset: number, length: number) {
      writes.push({bytes: [...data], length});
    }
  };
  const device = {
    async sync() {
      const pending = context.pendingGPUToCPUCopy;
      context.pendingGPUToCPUCopy = null;
      if (pending) await pending;
    }
  };
  const runtime = {
    lib: {webGPUContext: context},
    beginScope() {
      events.push('begin');
    },
    endScope() {
      events.push('end');
    },
    dispose() {
      events.push('dispose');
    },
    cpu() {
      return 'cpu';
    },
    empty(shape: number[], dtype: string, where: unknown) {
      return {
        shape,
        dtype,
        byteOffset: 0,
        getDataPtr: () => 1,
        copyFrom(this: FakeTensor, data: ArrayLike<number>) {
          if (where === 'cpu') {
            context.pendingGPUToCPUCopy = Promise.resolve();
            this.value = Float32Array.from({length: 510}, (_, i) => i);
          } else this.value = Array.from(data);
          return this;
        },
        toArray(this: FakeTensor) {
          return this.value;
        }
      };
    }
  };
  const gpu = {
    pushErrorScope() {},
    async popErrorScope() {
      return null;
    }
  };
  const runner = new FlashTvmRunner({
    runtime,
    device,
    gpu,
    fn: (...args: FakeTensor[]) => {
      calls.push(args);
      return {dtype: 'float32', shape: [1, 2, 255]};
    },
    weights: Array.from({length: 682}, () => ({})),
    lifecycle: {loss: null, disposed: false},
    disposalGuard: {blocked: false, retain() {}},
    uncaptured: [],
    teardownObserver: {async beforeDispose() {}, async afterDispose() {}},
    teardownTimeoutMs: 10
  });
  return {runner, calls, writes, events};
}

test('runs one graph with both markers and reads the corresponding score rows', async () => {
  const f = fixture();
  const rows = await f.runner.run({ids: [4, 248062, 5, 248062, 6], groups: [{index: 1}, {index: 3}]});
  assert.equal(f.calls.length, 1);
  assert.deepEqual(f.calls[0]![4]!.value, [1, 3]);
  assert.equal(f.calls[0]!.length, 687);
  assert.equal(rows.length, 2);
  assert.equal(rows[0]![254], 254);
  assert.equal(rows[1]![0], 255);
  assert.ok(f.writes[0]!.bytes.every((value) => value === 0));
  assert.deepEqual(f.events, ['begin', 'end']);
  await f.runner.dispose();
});

test('rejects mismatched or duplicate markers before submitting GPU work', async () => {
  const f = fixture();
  for (const packed of [
    {ids: [4, 248062, 5], groups: [{index: 0}]},
    {ids: [4, 248062, 5], groups: [{index: 1}, {index: 1}]},
    {ids: [248062, 248062, 5], groups: [{index: 0}]}
  ])
    await assert.rejects(f.runner.run(packed), {code: 'INVALID_INPUT'});
  assert.equal(f.calls.length, 0);
  await f.runner.dispose();
});
