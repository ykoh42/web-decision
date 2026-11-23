import assert from 'node:assert/strict';
import {test} from 'node:test';
import {loadPinnedGemma4V2Bindings} from '../../../../packages/web-decision/src/strands/gemma4-v2-bindings.mjs';
import {gemma4V2LayerFamily, runSplitAdapter, transposedSplitGraphs} from './split-adapter.mjs';

const graphs = transposedSplitGraphs();
const tensor = (name: string) => ({name, dispose() {}});
const base = {
  hidden: tensor('hidden'),
  perLayer: tensor('perLayer'),
  cosine: tensor('cosine'),
  sine: tensor('sine'),
  split: tensor('split'),
  valid: tensor('valid'),
  scale: tensor('scale'),
  plan: {tokens: 7, chunkTokens: 3}
};

test('selects each family, orders pinned weights, and preserves shared KV for all 35 layers', async () => {
  const bindings = await loadPinnedGemma4V2Bindings();
  const key = tensor('key');
  const value = tensor('value');
  for (let index = 0; index < 35; index++) {
    const layer = bindings.layer(index);
    const weights = layer.parameters.map(({name}) => ({name, value: tensor(name)})).reverse();
    const family = gemma4V2LayerFamily(index);
    assert.equal(family, layer.family);
    await runSplitAdapter(
      {graphs} as never,
      {...base, index, weights, ...(index >= 15 ? {sharedCache: [key, value]} : {})} as never,
      async (context, input) => {
        assert.equal(context.graphs, graphs[family]);
        assert.equal(input.attentionWeights.length, index < 15 ? 8 : 5);
        assert.equal(input.tailWeights.length, 8);
        const prefix = `model.language_model.layers.${index}.`;
        const suffixes = (values: readonly unknown[]) =>
          values.map((item) => (item as {name: string}).name.slice(prefix.length));
        assert.deepEqual(suffixes(input.attentionWeights), [
          'self_attn.q_proj.weight',
          'self_attn.q_norm.weight',
          ...(index < 15 ? ['self_attn.k_norm.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight'] : []),
          'self_attn.o_proj.weight',
          'input_layernorm.weight',
          'post_attention_layernorm.weight'
        ]);
        assert.deepEqual(suffixes(input.tailWeights), [
          'pre_feedforward_layernorm.weight',
          'mlp.gate_proj.weight',
          'mlp.up_proj.weight',
          'mlp.down_proj.weight',
          'post_feedforward_layernorm.weight',
          'per_layer_input_gate.weight',
          'per_layer_projection.weight',
          'post_per_layer_input_norm.weight'
        ]);
        assert.deepEqual(input.cache, index >= 15 ? [key, value] : []);
        assert.equal(input.retainKeyValues, index === 13 || index === 14);
        assert.equal(input.plan.tokens, 7);
        return {hidden: tensor('result')} as never;
      }
    );
  }
});

test('rejects invalid family, missing weights, and invalid cache before dispatch', async () => {
  const bindings = await loadPinnedGemma4V2Bindings();
  const weights = bindings.layer(15).parameters.map(({name}) => ({name, value: tensor(name)}));
  let calls = 0;
  const run = async () => {
    calls++;
    return {hidden: tensor('result')} as never;
  };
  assert.throws(() => gemma4V2LayerFamily(35), /invalid/i);
  await assert.rejects(runSplitAdapter({graphs} as never, {...base, index: 15, weights} as never, run), /KV/i);
  await assert.rejects(
    runSplitAdapter(
      {graphs} as never,
      {...base, index: 15, weights: weights.slice(1), sharedCache: [tensor('k'), tensor('v')]} as never,
      run
    ),
    /weight/i
  );
  await assert.rejects(
    runSplitAdapter(
      {graphs} as never,
      {
        ...base,
        index: 15,
        weights: [...weights.slice(0, -1), weights[0]],
        sharedCache: [tensor('k'), tensor('v')]
      } as never,
      run
    ),
    /weight/i
  );
  await assert.rejects(
    runSplitAdapter(
      {graphs: {...graphs, 15: {...graphs[15], tail: ''}}} as never,
      {...base, index: 15, weights, sharedCache: [tensor('k'), tensor('v')]} as never,
      run
    ),
    /graph/i
  );
  assert.equal(calls, 0);
});

test('abort after dispatch releases returned tensors and keeps the original abort reason', async () => {
  const controller = new AbortController();
  const reason = new Error('device lost');
  const bindings = await loadPinnedGemma4V2Bindings();
  const weights = bindings.layer(13).parameters.map(({name}) => ({name, value: tensor(name)}));
  const disposed: string[] = [];
  const owned = (name: string) => ({
    dispose() {
      disposed.push(name);
    }
  });
  await assert.rejects(
    runSplitAdapter({graphs, signal: controller.signal} as never, {...base, index: 13, weights} as never, async () => {
      controller.abort(reason);
      return {hidden: owned('hidden'), key: owned('key'), value: owned('value')} as never;
    }),
    (error) => error === reason
  );
  assert.deepEqual(disposed, ['hidden', 'key', 'value']);
});

test('real chunk executor uses the selected graph names and cleans up a lost device', async () => {
  const bindings = await loadPinnedGemma4V2Bindings();
  const scopes: Array<Set<FakeTensor>> = [];
  const calls: string[] = [];
  const created: FakeTensor[] = [];
  let syncs = 0;
  const lost = new Error('device lost');
  class FakeTensor {
    readonly dtype = 'float32';
    disposed = false;
    constructor(
      readonly shape: number[],
      scoped = true
    ) {
      created.push(this);
      if (scoped) scopes.at(-1)!.add(this);
    }
    copyFrom() {
      return this;
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
    empty(shape: number[]) {
      return new FakeTensor(shape);
    }
  };
  const device = {
    async sync() {
      if (++syncs === 2) throw lost;
    }
  };
  const tuple = (values: FakeTensor[]) => ({size: () => values.length, get: (index: number) => values[index]!});
  const vm = {
    getFunction(name: string) {
      calls.push(name);
      if (name === graphs[15].attention) return () => new FakeTensor([1, 7, 1536]);
      if (name === graphs[15].unpack) return () => tuple(Array.from({length: 8}, () => new FakeTensor([1])));
      if (name === graphs[15].slice)
        return (_hidden: unknown, _per: unknown, _offset: unknown, template: FakeTensor) =>
          tuple([new FakeTensor([1, template.shape[1]!, 1536]), new FakeTensor([1, template.shape[1]!, 256])]);
      if (name === graphs[15].tail) return (hidden: FakeTensor) => new FakeTensor(hidden.shape);
      throw new Error(`unexpected graph ${name}`);
    }
  };
  const outside = (shape: number[]) => new FakeTensor(shape, false);
  const weights = bindings.layer(15).parameters.map(({name}) => ({name, value: outside([1])}));
  const before = created.length;
  const input = {
    ...base,
    index: 15,
    weights,
    hidden: outside([1, 7, 1536]),
    perLayer: outside([1, 7, 256]),
    sharedCache: [outside([1]), outside([1])]
  };
  const firstTemporary = created.length;
  await assert.rejects(
    runSplitAdapter({runtime, device, vm, graphs} as never, input as never),
    (error) => error === lost
  );
  assert.ok(calls.includes(graphs[15].attention));
  assert.ok(calls.includes(graphs[15].unpack));
  assert.ok(calls.includes(graphs[15].tail));
  assert.equal(scopes.length, 0);
  assert.ok(created.slice(firstTemporary).every((value) => value.disposed));
  assert.ok(created.slice(0, before).every((value) => !value.disposed));
});
