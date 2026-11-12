import assert from 'node:assert/strict';
import {test} from 'node:test';
import type {ModelId} from '../src/catalog/types.generated.js';
import {readLoadOptions} from '../src/curated/load-options.ts';

const kananaId = 'kanana-2-v-decide-3b-2611';
const isModelId = (value: unknown): value is ModelId => value === kananaId;

test('accepts a curated model ID and copies supported options without changing the caller', () => {
  const input = {modelId: kananaId, offline: true, cache: true};
  const parsed = readLoadOptions(input, isModelId);
  assert.equal(parsed.modelId, kananaId);
  assert.equal(Object.hasOwn(parsed.options, 'model'), false);
  assert.equal(parsed.options.offline, true);
  assert.equal(parsed.options.cache, true);
  assert.equal(Object.getPrototypeOf(parsed.options), null);
  assert.deepEqual(input, {modelId: kananaId, offline: true, cache: true});
});

test('rejects unknown IDs, omitted IDs and legacy model routes', () => {
  for (const input of [undefined, {}, {modelId: 'unknown'}, {model: kananaId}, {modelId: kananaId, model: kananaId}]) {
    assert.throws(() => readLoadOptions(input, isModelId), {code: 'INVALID_INPUT'});
  }
});

test('rejects caller-specified sources and dependency injection', () => {
  for (const input of [
    {modelId: kananaId, assetsUrl: 'https://example.org/'},
    {modelId: kananaId, manifest: {}},
    {modelId: kananaId, runtime: {}},
    {modelId: kananaId, [Symbol('source')]: 'other'}
  ]) {
    assert.throws(() => readLoadOptions(input, isModelId), {code: 'INVALID_INPUT'});
  }
});

test('rejects Kev-only load controls for Kanana without turning them into request caps', () => {
  for (const key of ['maxPrefetchBytes', 'maxInputTokens', 'maxInputCharacters']) {
    assert.throws(() => readLoadOptions({modelId: kananaId, [key]: 128}, isModelId), {code: 'INVALID_INPUT'});
  }
  const parsed = readLoadOptions({modelId: kananaId, cache: false}, isModelId);
  assert.equal(parsed.modelId, kananaId);
  assert.equal(parsed.options.cache, false);
});

test('rejects accessors without invoking them', () => {
  let invoked = false;
  const input = {modelId: kananaId};
  Object.defineProperty(input, 'offline', {get: () => ((invoked = true), true)});
  assert.throws(() => readLoadOptions(input, isModelId), {code: 'INVALID_INPUT'});
  assert.equal(invoked, false);
});
