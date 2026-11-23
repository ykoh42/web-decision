import assert from 'node:assert/strict';
import {test} from 'node:test';
import {runGemma4V2ChunkedDecoder} from '../src/strands/gemma4-v2-chunked-decoder.mjs';

function setup(options: {queryTiles?: boolean; shared?: boolean; abortAtSync?: number; failAtSync?: number} = {}) {
  const scopes: Array<Set<FakeTensor>> = [];
  const tensors: FakeTensor[] = [];
  const calls: Array<{name: string; args: unknown[]}> = [];
  const offsets: number[] = [];
  const abort = new AbortController();
  let syncs = 0;
  const lost = new Error('GPU device lost');
  const device = {
    deviceType: 15,
    deviceId: 0,
    async sync() {
      syncs++;
      if (syncs === options.abortAtSync) abort.abort();
      if (syncs === options.failAtSync) throw lost;
    }
  };
  class FakeTensor {
    readonly device = device;
    disposed = false;
    data: unknown;
    constructor(
      readonly shape: number[],
      readonly dtype = 'float32',
      scoped = true
    ) {
      tensors.push(this);
      if (scoped) {
        assert.ok(scopes.length);
        scopes.at(-1)!.add(this);
      }
    }
    copyFrom(data: unknown) {
      this.data = data;
      return this;
    }
    dispose() {
      assert.equal(this.disposed, false);
      this.disposed = true;
    }
  }
  const tensor = (shape: number[], dtype = 'float32') => new FakeTensor(shape, dtype, false);
  const runtime = {
    beginScope() {
      scopes.push(new Set<FakeTensor>());
    },
    endScope() {
      for (const value of scopes.pop()!) value.dispose();
    },
    detachFromCurrentScope(value: FakeTensor) {
      assert.ok(scopes.at(-1)?.delete(value));
      return value;
    },
    empty(shape: number[], dtype: string) {
      return new FakeTensor(shape, dtype);
    }
  };
  const tuple = (values: FakeTensor[]) => ({size: () => values.length, get: (index: number) => values[index]!});
  const vm = {
    getFunction(name: string) {
      return (...args: unknown[]) => {
        calls.push({name, args});
        if (name === 'attention') {
          const hidden = new FakeTensor([1, 7, 1536]);
          return options.shared
            ? hidden
            : tuple([hidden, new FakeTensor([1, 1, 7, 256]), new FakeTensor([1, 1, 7, 256])]);
        }
        if (name === 'unpack') return tuple(Array.from({length: 8}, (_, index) => new FakeTensor([index + 1])));
        if (name === 'queryWeights')
          return tuple(Array.from({length: options.shared ? 5 : 8}, (_, index) => new FakeTensor([index + 10])));
        if (name === 'queryKv') return tuple([new FakeTensor([1, 1, 7, 256]), new FakeTensor([1, 1, 7, 256])]);
        if (name === 'queryRope') {
          const length = (args[3] as FakeTensor).shape[1]!;
          return tuple([new FakeTensor([1, length, 256]), new FakeTensor([1, length, 256])]);
        }
        if (name === 'queryChunk') return new FakeTensor((args[0] as FakeTensor).shape.slice());
        if (name === 'slice') {
          const offset = (args[2] as FakeTensor).data as Int32Array;
          offsets.push(offset[0]!);
          const length = (args[3] as FakeTensor).shape[1]!;
          return tuple([new FakeTensor([1, length, 1536]), new FakeTensor([1, length, 256])]);
        }
        if (name === 'tail') return new FakeTensor((args[0] as FakeTensor).shape.slice());
        if (name === 'concat')
          return new FakeTensor([1, (args[0] as FakeTensor).shape[1]! + (args[1] as FakeTensor).shape[1]!, 1536]);
        throw new Error(`unexpected graph ${name}`);
      };
    }
  };
  const context = {
    runtime,
    device,
    vm,
    graphs: {
      attention: 'attention',
      unpack: 'unpack',
      tail: 'tail',
      slice: 'slice',
      concat: 'concat',
      ...(options.queryTiles
        ? {
            query: {
              weights: 'queryWeights',
              chunk: 'queryChunk',
              rope: 'queryRope',
              ...(options.shared ? {} : {kv: 'queryKv'})
            }
          }
        : {})
    },
    signal: abort.signal
  };
  const cache = options.shared ? [tensor([1, 1, 7, 256]), tensor([1, 1, 7, 256])] : [];
  const input = {
    hidden: tensor([1, 7, 1536]),
    perLayer: tensor([1, 7, 256]),
    cosine: tensor([1]),
    sine: tensor([1]),
    split: tensor([1], 'int32'),
    prefixTokens: 4,
    valid: tensor([1, 7], 'int32'),
    scale: tensor([1]),
    cache,
    attentionWeights: [tensor([1])],
    tailWeights: Array.from({length: 8}, () => tensor([1], 'uint32')),
    retainKeyValues: !options.shared,
    plan: {tokens: 7, chunkTokens: 3}
  };
  return {
    context,
    input,
    scopes,
    tensors,
    calls,
    offsets,
    abort,
    lost,
    get syncs() {
      return syncs;
    }
  };
}

test('preserves every token, computes attention and unpack once, and retains requested KV', async () => {
  const state = setup();
  const output = await runGemma4V2ChunkedDecoder(state.context as never, state.input as never);
  assert.deepEqual(state.offsets, [0, 3, 6]);
  assert.equal(state.calls.filter((call) => call.name === 'attention').length, 1);
  assert.equal(state.calls.filter((call) => call.name === 'unpack').length, 1);
  assert.equal(state.calls.filter((call) => call.name === 'tail').length, 3);
  assert.equal(state.calls.filter((call) => call.name === 'concat').length, 2);
  assert.deepEqual(output.hidden.shape, [1, 7, 1536]);
  assert.deepEqual(output.key?.shape, [1, 1, 7, 256]);
  assert.deepEqual(output.value?.shape, [1, 1, 7, 256]);
  assert.equal(state.scopes.length, 0);
  const live = state.tensors.filter((value) => !value.disposed && !Object.values(state.input).includes(value as never));
  assert.ok(live.includes(output.hidden as never));
  output.hidden.dispose();
  output.key!.dispose();
  output.value!.dispose();
});

test('shared KV is passed intact and output returns no new KV', async () => {
  const state = setup({shared: true});
  const output = await runGemma4V2ChunkedDecoder(state.context as never, state.input as never);
  const args = state.calls.find((call) => call.name === 'attention')!.args;
  assert.equal(args[5], state.input.cache[0]);
  assert.equal(args[6], state.input.cache[1]);
  assert.equal(output.key, undefined);
  assert.equal(output.value, undefined);
  output.hidden.dispose();
  assert.equal(state.scopes.length, 0);
});

test('malformed plans fail before any GPU work', async () => {
  const state = setup();
  await assert.rejects(
    runGemma4V2ChunkedDecoder(
      state.context as never,
      {...state.input, plan: {...state.input.plan, chunkTokens: 0}} as never
    ),
    /plan/i
  );
  await assert.rejects(
    runGemma4V2ChunkedDecoder(
      state.context as never,
      {...state.input, plan: {...state.input.plan, tokens: 6}} as never
    ),
    /match/i
  );
  assert.equal(state.calls.length, 0);
  assert.equal(state.syncs, 0);
  assert.equal(state.scopes.length, 0);
});

test('aborting after queued work drains the GPU and releases every temporary', async () => {
  const state = setup({abortAtSync: 1});
  await assert.rejects(runGemma4V2ChunkedDecoder(state.context as never, state.input as never), {name: 'AbortError'});
  assert.ok(state.syncs >= 2);
  assert.equal(state.scopes.length, 0);
  assert.ok(
    state.tensors
      .filter((tensor) => !Object.values(state.input).includes(tensor as never) && tensor.shape[0] === 1)
      .some((tensor) => tensor.disposed)
  );
  const created = state.tensors.slice(state.tensors.indexOf(state.input.tailWeights.at(-1)!) + 1);
  assert.ok(created.length > 0);
  assert.ok(created.every((tensor) => tensor.disposed));
});

test('device loss propagates while scopes and detached chunks are released', async () => {
  const state = setup({failAtSync: 2});
  await assert.rejects(
    runGemma4V2ChunkedDecoder(state.context as never, state.input as never),
    (error) => error === state.lost
  );
  assert.equal(state.scopes.length, 0);
  const created = state.tensors.slice(state.tensors.indexOf(state.input.tailWeights.at(-1)!) + 1);
  assert.ok(created.length > 0);
  assert.ok(created.every((tensor) => tensor.disposed));
});

test('aborting during final synchronization disposes would-be returned tensors', async () => {
  const state = setup({abortAtSync: 6});
  await assert.rejects(runGemma4V2ChunkedDecoder(state.context as never, state.input as never), {name: 'AbortError'});
  assert.equal(state.scopes.length, 0);
  const created = state.tensors.slice(state.tensors.indexOf(state.input.tailWeights.at(-1)!) + 1);
  assert.ok(created.length > 0);
  assert.ok(created.every((tensor) => tensor.disposed));
});

test('one-token and full-length chunks preserve every requested token', async () => {
  for (const chunkTokens of [1, 7]) {
    const state = setup({shared: true});
    const output = await runGemma4V2ChunkedDecoder(
      state.context as never,
      {
        ...state.input,
        plan: {...state.input.plan, chunkTokens}
      } as never
    );
    assert.deepEqual(
      state.offsets,
      Array.from({length: Math.ceil(7 / chunkTokens)}, (_, index) => index * chunkTokens)
    );
    assert.deepEqual(output.hidden.shape, [1, 7, 1536]);
    output.hidden.dispose();
  }
});

test('query tiles retain full K/V, use absolute offsets and reuse the decoded weights', async () => {
  for (const shared of [false, true]) {
    const state = setup({queryTiles: true, shared});
    const output = await runGemma4V2ChunkedDecoder(state.context as never, state.input as never);
    const named = (name: string) => state.calls.filter((call) => call.name === name);
    assert.equal(named('attention').length, 0);
    assert.equal(named('queryWeights').length, 1);
    assert.equal(named('queryKv').length, shared ? 0 : 1);
    assert.equal(named('queryRope').length, 3);
    assert.equal(named('unpack').length, 1);
    assert.deepEqual(state.offsets, [0, 3, 6]);
    const calls = named('queryChunk');
    assert.deepEqual(
      calls.map(({args}) => (args[0] as {shape: number[]}).shape[1]),
      [3, 3, 1]
    );
    assert.deepEqual(
      calls.map(({args}) => [...(args[4] as {data: Int32Array}).data]),
      [
        [4, 0],
        [4, 3],
        [4, 6]
      ]
    );
    for (const {args} of calls) {
      assert.equal(args[3], state.input.valid);
      assert.equal(args[5], shared ? state.input.cache[0] : output.key);
      assert.equal(args[6], shared ? state.input.cache[1] : output.value);
      assert.deepEqual(args.slice(7), calls[0]!.args.slice(7));
    }
    assert.equal(state.scopes.length, 0);
    assert.equal(
      shared ? state.input.cache.every((tensor) => !tensor.disposed) : Boolean(output.key && output.value),
      true
    );
    output.hidden.dispose();
    output.key?.dispose();
    output.value?.dispose();
  }
});
