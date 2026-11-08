import assert from 'node:assert/strict';
import test from 'node:test';
import {createSourceLoader} from '../src/strands/source-loader-internal.mjs';
import {selectModelFamily} from '../src/model-family.js';

function fixture() {
  const calls = [];
  const adapter = {
    features: new Set(['shader-f16']),
    limits: {maxBufferSize: 1017118720, maxStorageBufferBindingSize: 1017118720, maxStorageBuffersPerShaderStage: 8}
  };
  const gpuApi = {
    async requestAdapter() {
      calls.push('adapter');
      return adapter;
    }
  };
  const load = createSourceLoader({
    async openCache() {
      calls.push('cache');
      return {};
    },
    async openSource() {
      calls.push('source');
      return {bytes: {}, transaction: {read() {}, commit() {}}};
    },
    async loadBytes(_options, dependencies) {
      calls.push('load');
      assert.equal(dependencies.preparedAdapter, adapter);
      return 'fake';
    }
  });
  return {calls, load, gpuApi};
}

test('Strands prefetch rejects unsupported values before adapter, cache or source', async () => {
  for (const value of [1, 1024, -1, NaN, Infinity, 0.5, Number.MAX_SAFE_INTEGER + 1, null, '0']) {
    const {calls, load, gpuApi} = fixture();
    await assert.rejects(load({assetsUrl: 'https://example.test/assets/', gpuApi, maxPrefetchBytes: value}), {
      code: 'INVALID_INPUT'
    });
    assert.deepEqual(calls, []);
  }
});

test('Strands accepts omitted, undefined and zero prefetch budgets', async () => {
  for (const extra of [{}, {maxPrefetchBytes: undefined}, {maxPrefetchBytes: 0}]) {
    const {calls, load, gpuApi} = fixture();
    assert.equal(await load({assetsUrl: 'https://example.test/assets/', gpuApi, ...extra}), 'fake');
    assert.deepEqual(calls, ['adapter', 'cache', 'source', 'load']);
  }
});

test('Strands is routed without changing the existing model families or default', () => {
  for (const [model, family] of [
    [undefined, 'kev'],
    ['kev-0.6b-q4f16', 'kev'],
    ['kev-0.6b-reflexes-q4f16', 'kev'],
    ['open-jev-deberta-v3-large-q4f16', 'deberta'],
    ['mapika-0.8b-f16', 'mapika'],
    ['mapika-0.8b-q4f16', 'mapika-q4'],
    ['kev-0.8b-f16', 'kev08'],
    ['experimental-strands-qwen-2b-fp16', 'strands'],
    ['experimental-strands-gemma4-e2b-fp16', 'strands-gemma4']
  ]) {
    assert.equal(selectModelFamily({model}), family);
  }
  assert.throws(() => selectModelFamily({model: 'unrecognized'}), {code: 'INVALID_INPUT'});
});
