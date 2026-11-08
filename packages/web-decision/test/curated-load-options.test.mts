import assert from 'node:assert/strict';
import {test} from 'node:test';
import type {ModelId} from '../src/catalog/types.generated.js';
import {readLoadOptions} from '../src/curated/load-options.ts';

const id = 'kev-0.6b-q4f16';
const isModelId = (value: unknown): value is ModelId => value === id;

test('accepts a curated model ID and copies supported options without changing the caller', () => {
  const input = {modelId: id, offline: true};
  const parsed = readLoadOptions(input, isModelId);
  assert.equal(parsed.modelId, id);
  assert.equal(Object.hasOwn(parsed.options, 'model'), false);
  assert.equal(parsed.options.offline, true);
  assert.equal(Object.getPrototypeOf(parsed.options), null);
  assert.deepEqual(input, {modelId: id, offline: true});
});

test('rejects unknown IDs, omitted IDs and legacy model routes', () => {
  for (const input of [undefined, {}, {modelId: 'unknown'}, {model: id}, {modelId: id, model: id}]) {
    assert.throws(() => readLoadOptions(input, isModelId), {code: 'INVALID_INPUT'});
  }
});

test('rejects caller-specified sources and dependency injection', () => {
  for (const input of [
    {modelId: id, assetsUrl: 'https://example.org/'},
    {modelId: id, manifest: {}},
    {modelId: id, runtime: {}},
    {modelId: id, [Symbol('source')]: 'other'}
  ]) {
    assert.throws(() => readLoadOptions(input, isModelId), {code: 'INVALID_INPUT'});
  }
});

test('rejects accessors without invoking them', () => {
  let invoked = false;
  const input = {modelId: id};
  Object.defineProperty(input, 'offline', {get: () => ((invoked = true), true)});
  assert.throws(() => readLoadOptions(input, isModelId), {code: 'INVALID_INPUT'});
  assert.equal(invoked, false);
});
