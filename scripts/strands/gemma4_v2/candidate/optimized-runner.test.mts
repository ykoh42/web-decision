import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createGemma4V2Runner} from '../../../../packages/web-decision/src/strands/gemma4-v2-runner.mjs';
import {transposedSplitGraphs} from './split-adapter.mjs';

function fixture(failAt = 0, failReadAt = 0, abortAt = 0) {
  const scopes: Array<Set<FakeTensor>> = [];
  const all: FakeTensor[] = [];
  const calls: string[] = [];
  const error = new Error('device lost');
  let syncCount = 0;
  let readCount = 0;
  let releaseCount = 0;
  const controller = new AbortController();
  const device = {
    async sync() {
      if (++syncCount === failAt) throw error;
      if (syncCount === abortAt) controller.abort(error);
    }
  };
  class FakeTensor {
    disposed = false;
    constructor(
      readonly shape: number[],
      readonly dtype = 'float32'
    ) {
      assert.ok(scopes.length);
      scopes.at(-1)!.add(this);
      all.push(this);
    }
    copyFrom(_value: unknown) {
      return this;
    }
    toArray() {
      return new Float32Array(255).fill(0.25);
    }
    dispose() {
      assert.equal(this.disposed, false);
      this.disposed = true;
    }
  }
  const runtime = {
    beginScope() {
      scopes.push(new Set<FakeTensor>());
    },
    endScope() {
      for (const value of scopes.pop()!) value.dispose();
    },
    detachFromCurrentScope(value: FakeTensor) {
      assert.ok(scopes.at(-1)!.delete(value));
      return value;
    },
    empty(shape: number[], dtype: string) {
      return new FakeTensor(shape, dtype);
    },
    cpu() {
      return device;
    }
  };
  const tuple = (values: FakeTensor[]) => ({size: () => values.length, get: (index: number) => values[index]!});
  const vm = {
    getFunction(name: string) {
      calls.push(name);
      if (name === 'word_packed_main') return () => new FakeTensor([1, 7, 1536]);
      if (name === 'ple_packed_main') return () => new FakeTensor([1, 7, 256]);
      if (name.startsWith('rope_')) return () => new FakeTensor([1, 7, 256]);
      if (/^split(?:0|4)attn_packed_main$/u.test(name))
        return () => tuple([new FakeTensor([1, 7, 1536]), new FakeTensor([1]), new FakeTensor([1])]);
      if (/^split(?:15|19)attn_packed_main$/u.test(name)) return () => new FakeTensor([1, 7, 1536]);
      if (/^split(?:0|4|15|19)unpack_t_main$/u.test(name))
        return () => tuple(Array.from({length: 8}, () => new FakeTensor([1])));
      if (name === 'split_slice_main')
        return (_hidden: unknown, _per: unknown, _offset: unknown, template: FakeTensor) =>
          tuple([new FakeTensor([1, template.shape[1]!, 1536]), new FakeTensor([1, template.shape[1]!, 256])]);
      if (/^split(?:0|4|15|19)optimized_tail_main$/u.test(name))
        return (hidden: FakeTensor) => new FakeTensor(hidden.shape);
      if (name === 'split_concat_main')
        return (left: FakeTensor, right: FakeTensor) => new FakeTensor([1, left.shape[1]! + right.shape[1]!, 1536]);
      if (name === 'readout_packed_main') return () => new FakeTensor([1, 255]);
      throw new Error(`unexpected graph ${name}`);
    }
  };
  const source = {
    async readEmbeddingRows() {
      return new Uint32Array(1);
    },
    async readTensorRange() {
      return new Uint8Array(4);
    },
    async readTensor(name: string) {
      if (++readCount === failReadAt) throw error;
      return new Uint8Array(name.endsWith('layer_scalar') ? 2 : 4);
    },
    releaseCachedMemory() {
      releaseCount++;
    }
  };
  const deps = {
    runtime,
    device,
    vm,
    source,
    readout: new Uint8Array(255 * 1536 * 2),
    localFrequency: new Float32Array(128),
    globalFrequency: new Float32Array(256),
    splitExecution: {graphs: transposedSplitGraphs(), chunkTokens: 4},
    async insertImages(word: FakeTensor) {
      return word;
    },
    dispose() {}
  };
  return {
    deps,
    calls,
    scopes,
    all,
    error,
    controller,
    get releaseCount() {
      return releaseCount;
    }
  };
}

const input = {stateIds: [1, 2, 3, 4], questionIds: [5, 6, 7], images: [], imageSpans: []};

test('refuses to enable split execution when the pinned graph lacks a required function', async () => {
  const state = fixture();
  const getFunction = state.deps.vm.getFunction;
  state.deps.vm.getFunction = (name: string) =>
    name === 'split19optimized_tail_main' ? (undefined as never) : getFunction(name);
  await assert.rejects(createGemma4V2Runner(state.deps as never), /split functions/i);
  assert.equal(state.scopes.length, 0);
  assert.equal(state.all.length, 0);
});

test('rejects an invalid internal chunk size before graph lookup', async () => {
  const state = fixture();
  state.deps.splitExecution.chunkTokens = 0;
  await assert.rejects(createGemma4V2Runner(state.deps as never), /chunk size/i);
  assert.equal(state.calls.length, 0);
});

test('optimized test runner executes every layer and releases all GPU tensor handles', async () => {
  const state = fixture();
  const runner = await createGemma4V2Runner(state.deps as never);
  state.calls.length = 0;
  assert.deepEqual(await runner.run(input), new Float32Array(255).fill(0.25));
  for (const family of [0, 4, 15, 19]) assert.ok(state.calls.includes(`split${family}attn_packed_main`));
  assert.equal(state.calls.filter((name) => name.endsWith('attn_packed_main')).length, 35);
  assert.equal(state.scopes.length, 0);
  assert.ok(state.all.every((value) => value.disposed));
  await runner.releaseCachedMemory();
  assert.equal(state.releaseCount, 1);
  await runner.dispose();
});

test('optimized test runner releases its scoped and retained tensors on device loss', async () => {
  const state = fixture(90);
  const runner = await createGemma4V2Runner(state.deps as never);
  state.calls.length = 0;
  await assert.rejects(runner.run(input), (error) => error === state.error);
  assert.ok(state.calls.includes('split15attn_packed_main'));
  assert.equal(state.scopes.length, 0);
  assert.ok(state.all.every((value) => value.disposed));
  await runner.dispose();
});

test('source failure after KV reuse releases the layer scope and retained tensors', async () => {
  const state = fixture(0, 280);
  const runner = await createGemma4V2Runner(state.deps as never);
  state.calls.length = 0;
  await assert.rejects(runner.run(input), (error) => error === state.error);
  assert.ok(state.calls.includes('split15attn_packed_main'));
  assert.equal(state.scopes.length, 0);
  assert.ok(state.all.every((value) => value.disposed));
  await runner.dispose();
});

test('abort after KV reuse preserves the abort reason and releases all handles', async () => {
  const state = fixture(0, 0, 90);
  const runner = await createGemma4V2Runner(state.deps as never);
  state.calls.length = 0;
  await assert.rejects(runner.run(input, state.controller.signal), (error) => error === state.error);
  assert.ok(state.calls.includes('split15attn_packed_main'));
  assert.equal(state.scopes.length, 0);
  assert.ok(state.all.every((value) => value.disposed));
  await runner.dispose();
});
