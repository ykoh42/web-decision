import assert from 'node:assert/strict';
import {test} from 'node:test';
import type {ModelId} from '../src/catalog/types.generated.js';
import {readLoadOptions} from '../src/curated/load-options.ts';

const kevId = 'kev-0.6b-q4f16';
const kananaId = 'kanana-2-v-decide-3b-2611';
const isModelId = (value: unknown): value is ModelId => value === kevId || value === kananaId;

test('accepts a curated model ID and copies supported options without changing the caller', () => {
  const input = {modelId: kevId, offline: true, maxInputTokens: 4096};
  const parsed = readLoadOptions(input, isModelId);
  assert.equal(parsed.modelId, kevId);
  assert.equal(Object.hasOwn(parsed.options, 'model'), false);
  assert.equal(parsed.options.offline, true);
  assert.equal(parsed.options.maxInputTokens, 4096);
  assert.equal(Object.getPrototypeOf(parsed.options), null);
  assert.deepEqual(input, {modelId: kevId, offline: true, maxInputTokens: 4096});
});

test('rejects unknown IDs, omitted IDs and legacy model routes', () => {
  for (const input of [undefined, {}, {modelId: 'unknown'}, {model: kevId}, {modelId: kevId, model: kevId}]) {
    assert.throws(() => readLoadOptions(input, isModelId), {code: 'INVALID_INPUT'});
  }
});

test('rejects caller-specified sources and dependency injection', () => {
  for (const input of [
    {modelId: kevId, assetsUrl: 'https://example.org/'},
    {modelId: kevId, manifest: {}},
    {modelId: kevId, runtime: {}},
    {modelId: kevId, [Symbol('source')]: 'other'}
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
  const input = {modelId: kevId};
  Object.defineProperty(input, 'offline', {get: () => ((invoked = true), true)});
  assert.throws(() => readLoadOptions(input, isModelId), {code: 'INVALID_INPUT'});
  assert.equal(invoked, false);
});
