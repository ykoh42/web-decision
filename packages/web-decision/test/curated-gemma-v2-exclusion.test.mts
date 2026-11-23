import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import {APPROVED_ADAPTER_LOADERS} from '../src/catalog/public-adapters.generated.ts';
import modelProvenance from '../src/catalog/model-provenance.json' with {type: 'json'};
import artifactLock from '../src/catalog/artifacts.lock.json' with {type: 'json'};
import {describeModel, listModels, loadDecisionModel} from '../src/curated/index.ts';
import type {ModelId} from '../src/curated/public-types.js';

const gemmaId = 'strands-decider-e2b-gemma4-v2-2611';
const publicId = gemmaId as ModelId;

test('Gemma catalog, loader, and package files follow committed provenance', async () => {
  const approved = modelProvenance.models.find(({modelId}) => modelId === gemmaId)?.status === 'verified';
  assert.equal(
    listModels().some(({modelId}) => modelId === publicId),
    approved
  );
  assert.equal(Object.hasOwn(APPROVED_ADAPTER_LOADERS, 'strands-gemma4-v2'), approved);
  const {files} = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8')) as {files: string[]};
  const modelFiles = files.filter((path) => path.startsWith('assets/strands-gemma4-v2/')).sort();
  if (!approved) {
    assert.deepEqual(modelFiles, []);
    await assert.rejects(loadDecisionModel({modelId: publicId}), /unsupported decision model/i);
    assert.throws(() => describeModel(publicId), /unsupported decision model/i);
    return;
  }
  const lock = artifactLock.byModelId[gemmaId];
  assert.deepEqual(
    modelFiles,
    [lock.manifest.sourcePath, ...lock.bundledRuntimeFiles.map(({sourcePath}) => sourcePath)].sort()
  );
  assert.equal(describeModel(publicId).assets.manifestSha256, lock.manifest.canonicalSha256);
});
