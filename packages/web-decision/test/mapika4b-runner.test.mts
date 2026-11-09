import assert from 'node:assert/strict';
import test from 'node:test';
import {Mapika4TvmRunner} from '../src/mapika4b/runner.mjs';

interface FakeTensor {
  shape: number[];
  dtype: string;
  byteOffset: number;
  value: unknown;
  getDataPtr(): number;
  copyFrom(data: unknown): FakeTensor;
  toArray(): unknown;
}

interface FakeGpuContext {
  pendingGPUToCPUCopy: Promise<void> | null;
  copyRawBytesToBuffer(data: Uint8Array, pointer: unknown, offset: number, length: number): void;
}

function fixture() {
  const calls: unknown[][] = [],
    writes: Array<{bytes: number[]; length: number}> = [],
    events: string[] = [];
  const context: FakeGpuContext = {
    pendingGPUToCPUCopy: null,
    copyRawBytesToBuffer(data, _ptr, _offset, length) {
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
    empty(shape: number[], dtype: string, where: unknown): FakeTensor {
      return {
        shape,
        dtype,
        byteOffset: 0,
        getDataPtr: () => 1,
        copyFrom(this: FakeTensor, data: unknown) {
          if (where === 'cpu') {
            context.pendingGPUToCPUCopy = Promise.resolve();
            this.value = Float32Array.from({length: 510}, (_, i) => i);
          } else this.value = Array.from(data as ArrayLike<number>);
          return this;
        },
        toArray(this: FakeTensor) {
          return this.value;
        }
      } as FakeTensor;
    }
  };
  const gpu = {
    pushErrorScope() {},
    async popErrorScope() {
      return null;
    }
  };
  const runner = new Mapika4TvmRunner({
    runtime,
    device,
    gpu,
    fn: (...args: unknown[]) => {
      calls.push(args);
      return {dtype: 'float32', shape: [1, 2, 255]};
    },
    weights: Array.from({length: 682}, () => ({})),
    lifecycle: {loss: null, disposed: false},
    disposalGuard: {blocked: false, retain() {}},
    uncaptured: [],
    teardownObserver: {async beforeDispose() {}, async afterDispose() {}},
    teardownTimeoutMs: 10
  } as unknown as ConstructorParameters<typeof Mapika4TvmRunner>[0]);
  return {runner, calls, writes, events};
}

test('runs one graph with both markers and reads the corresponding score rows', async () => {
  const f = fixture();
  const rows = await f.runner.run({ids: [4, 5, 6, 7, 8], groups: [{slot: 1}, {slot: 3}]});
  assert.equal(f.calls.length, 1);
  assert.deepEqual((f.calls[0]![4] as FakeTensor).value, [1, 3]);
  assert.equal(f.calls[0]!.length, 687);
  assert.equal(rows.length, 2);
  assert.equal(rows[0]![254], 254);
  assert.equal(rows[1]![0], 255);
  assert.deepEqual(f.writes[0]!.bytes.slice(0, 6), [0, 0, 255, 251, 255, 251]);
  assert.deepEqual(f.events, ['begin', 'end']);
  await f.runner.dispose();
});

test('rejects mismatched or duplicate markers before submitting GPU work', async () => {
  const f = fixture();
  for (const packed of [
    {ids: [4, 5, 6], groups: [{slot: 3}]},
    {ids: [4, 5, 6], groups: [{slot: 1}, {slot: 1}]},
    {ids: [248320, 5], groups: [{slot: 0}]}
  ])
    await assert.rejects(f.runner.run(packed), {code: 'INVALID_INPUT'});
  assert.equal(f.calls.length, 0);
  await f.runner.dispose();
});
