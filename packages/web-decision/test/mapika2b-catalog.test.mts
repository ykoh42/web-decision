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
  assert.equal(lock.manifest.rawSha256, '90095c0d11e21cfa6c5cc9c82a27ea272dac6164b63e258c243e07f5a0494b03');
  assert.equal(
    lock.manifest.files['text.wasm']?.sha256,
    '95e6bea454ee9d5dbbe40e2e84523d53077f7f1fa285f609c0e5422146b40bb2'
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
