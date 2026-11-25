import assert from 'node:assert/strict';
import test from 'node:test';
import type {DLDevice, Instance} from '../src/runtime/tvm-types.js';
import {createGemma4V1TvmStages} from '../src/strands/gemma4-v1-runtime.mjs';

function observeStages(failSyncAt?: number) {
  const copies: number[] = [];
  const calls: DemoTensor[][] = [];
  const scopes: DemoTensor[][] = [];
  class DemoTensor {
    readonly dtype = 'float32';
    value = 0;
    disposed = false;

    constructor(readonly shape: number[]) {}

    copyFrom(input: DemoTensor | Float32Array) {
      this.value = input instanceof DemoTensor ? input.value : input[0]!;
      if (input instanceof Float32Array) {
        copies.push(this.value);
        if (this.value < 0) throw Error('copy failed');
      }
      return this;
    }

    toArray() {
      assert.equal(this.disposed, false);
      return Float32Array.of(this.value);
    }
  }
  let syncs = 0;
  const device = {
    sync: async () => {
      if (++syncs === failSyncAt) throw Error('sync failed');
    }
  } as unknown as DLDevice;
  function empty(shape: number[]) {
    const tensor = new DemoTensor(shape);
    scopes.at(-1)!.push(tensor);
    return tensor;
  }
  const runtime = {
    beginScope() {
      scopes.push([]);
    },
    endScope() {
      for (const tensor of scopes.pop()!) tensor.disposed = true;
    },
    createVirtualMachine() {
      return {
        getFunction(name: string) {
          return (...inputs: DemoTensor[]) => {
            assert.equal(
              inputs.some((tensor) => tensor.disposed),
              false
            );
            calls.push(inputs);
            if (name === 'broken') throw Error('graph failed');
            const result = empty([1]);
            result.value = inputs.reduce((sum, input) => sum + input.value, 0);
            return result;
          };
        }
      };
    },
    empty,
    cpu: () => device,
    dispose: () => assert.deepEqual(scopes, [])
  } as unknown as Instance;
  const source = {tensor: async () => assert.fail('No model weights should be read')};
  return {copies, calls, scopes, stages: createGemma4V1TvmStages(runtime, device, source)};
}

const input = (value: number, reuseWithinLayer = false) => ({
  shape: [1],
  data: Float32Array.of(value),
  reuseWithinLayer
});

test('reuses immutable K/V only within one layer and releases all of the layer inputs', async () => {
  const {copies, calls, scopes, stages} = observeStages();
  const key = input(1, true);
  const value = input(2, true);
  const query = input(4);
  const signal = new AbortController().signal;
  const result = await stages.withWeights({}, signal, async (call) => {
    const first = await call('attention', [query, key, value]);
    assert.equal(calls[0]![0]!.disposed, true);
    assert.equal(calls[0]![1]!.disposed, false);
    const second = await call('attention', [input(8), key, value], true);
    assert.equal(calls[1]![1]!.disposed, true);
    assert.equal(calls[1]![2]!.disposed, true);
    assert.equal(scopes.length, 2);
    await call('tail', [input(6)]);
    return [first[0]!.data[0], second[0]!.data[0]];
  });
  assert.deepEqual(result, [7, 11]);
  assert.deepEqual(copies, [1, 2, 4, 8, 6]);
  assert.equal(calls[0]![1], calls[1]![1]);
  assert.equal(calls[0]![2], calls[1]![2]);
  assert.ok(calls.flat().every(({disposed}) => disposed));
  await stages.withWeights({}, signal, async (call) => call('attention', [query, key, value]));
  assert.notEqual(calls[0]![1], calls[3]![1]);
  assert.deepEqual(copies, [1, 2, 4, 8, 6, 1, 2, 4]);
  assert.equal(scopes.length, 1);
  await stages.dispose();
});

test('a failed resident upload or caller cancellation leaves the next layer reusable', async () => {
  const {copies, calls, scopes, stages} = observeStages();
  const key = input(1, true);
  const controller = new AbortController();
  await assert.rejects(
    stages.withWeights({}, controller.signal, async (call) => call('attention', [key, input(-2, true)])),
    /copy failed/
  );
  assert.deepEqual(copies, [1, -2]);
  assert.equal(scopes.length, 1);
  await assert.rejects(
    stages.withWeights({}, controller.signal, async (call) => {
      await call('attention', [key, input(2, true)]);
      controller.abort(Error('Cancelled'));
      return call('attention', [key, input(4, true)]);
    }),
    /Cancelled/
  );
  assert.deepEqual(copies, [1, -2, 1, 2]);
  assert.ok(calls.flat().every(({disposed}) => disposed));
  await stages.withWeights({}, new AbortController().signal, async (call) => call('attention', [key]));
  assert.deepEqual(copies, [1, -2, 1, 2, 1]);
  assert.equal(scopes.length, 1);
  await stages.dispose();
});

test('releases resident inputs even when the last graph call fails', async () => {
  const {scopes, calls, stages} = observeStages();
  const key = input(1, true);
  const value = input(2, true);
  const signal = new AbortController().signal;
  await assert.rejects(
    stages.withWeights({}, signal, async (call) => {
      await call('attention', [key, value]);
      return call('broken', [key, value], true);
    }),
    /graph failed/
  );
  assert.equal(scopes.length, 1);
  assert.ok(calls.flat().every(({disposed}) => disposed));
  await stages.dispose();
});

test('a failed call synchronization unwinds all three scopes in order', async () => {
  const {scopes, calls, stages} = observeStages(2);
  const key = input(5, true);
  await assert.rejects(
    stages.withWeights({}, new AbortController().signal, async (call) => call('attention', [key], true)),
    /sync failed/
  );
  assert.equal(scopes.length, 1);
  assert.equal(calls[0]![0]!.disposed, true);
  await stages.withWeights({}, new AbortController().signal, async (call) => call('attention', [key], true));
  await stages.dispose();
});
