import assert from 'node:assert/strict';
import test from 'node:test';
import {createModelArtifactLock, publicLoadOptionsForAdapter} from '../scripts/build-model-catalog.mts';

test('Mapika catalog recipe pins native checkpoints and packaged TVM artifacts', async () => {
  const model = {
    modelId: 'mapika-decider-2b-v12',
    description: {
      adapter: 'mapika2b',
      checkpoint: {repository: 'Mapika/decider-2b', revision: '7614a6c7ea5a047aa7d7bcfb1a0015469381c3e3'}
    }
  } as Parameters<typeof createModelArtifactLock>[0];
  const lock = await createModelArtifactLock(model);
  assert.equal(lock.manifest.sourcePath, 'assets/mapika2b/manifest.json');
  assert.equal(lock.manifest.rawSha256, 'da2f6c24bfa545d92b7658081913e47b39ad7cfaee5880c84f411fed0b7de561');
  assert.equal(
    lock.manifest.files['text.wasm']?.sha256,
    'da20aa80a0dd0cb8b4e1c58ce6a0f313ffb5f82675e610be18dcc5805cfe70d4'
  );
  assert.deepEqual(
    lock.upstreams?.map(({role}) => role),
    ['text', 'vision']
  );
  assert.deepEqual(
    lock.bundledRuntimeFiles.map(({manifestName}) => manifestName),
    [
      'text.wasm',
      'text-bindings.json',
      'source-inventory.json',
      'vision/fixture.json',
      'vision/weight-bindings.json',
      'vision/qwen35-dynamic-vision.webgpu.wasm'
    ]
  );
  assert.deepEqual(publicLoadOptionsForAdapter('mapika2b').slice(-3), ['temperature', 'temperatureByType', 'limits']);
  await assert.rejects(
    createModelArtifactLock({
      ...model,
      description: {...model.description, checkpoint: {...model.description.checkpoint, revision: 'other'}}
    }),
    /differs from its immutable pin/u
  );
});
