import assert from 'node:assert/strict';
import {test} from 'node:test';
import {loadOptions} from './model-options.ts';

const models = [{modelId: 'kev-0.6b-q4f16'}] as const;

test('selects only an offered model ID and forwards progress', () => {
  const onProgress = () => {};
  assert.deepEqual(loadOptions(models, 'kev-0.6b-q4f16', onProgress), {modelId: 'kev-0.6b-q4f16', onProgress});
  assert.deepEqual(loadOptions(models, 'kev-0.6b-q4f16'), {modelId: 'kev-0.6b-q4f16'});
});

test('rejects unknown and unavailable models', () => {
  assert.throws(() => loadOptions(models, 'unknown'), /supported model/);
  assert.throws(() => loadOptions([], 'kev-0.6b-q4f16'), /supported model/);
});
