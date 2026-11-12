import assert from 'node:assert/strict';
import {test} from 'node:test';
import {loadOptions} from './model-options.ts';

const models = [{modelId: 'kanana-2-v-decide-3b-2611'}] as const;

test('selects only an offered model ID and forwards progress', () => {
  const onProgress = () => {};
  assert.deepEqual(loadOptions(models, 'kanana-2-v-decide-3b-2611', onProgress), {
    modelId: 'kanana-2-v-decide-3b-2611',
    onProgress
  });
  assert.deepEqual(loadOptions(models, 'kanana-2-v-decide-3b-2611'), {modelId: 'kanana-2-v-decide-3b-2611'});
});

test('rejects unknown and unavailable models', () => {
  assert.throws(() => loadOptions(models, 'unknown'), /supported model/);
  assert.throws(() => loadOptions([], 'kanana-2-v-decide-3b-2611'), /supported model/);
});
