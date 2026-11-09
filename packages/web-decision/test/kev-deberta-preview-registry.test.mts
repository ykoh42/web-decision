import assert from 'node:assert/strict';
import test from 'node:test';
import {describeModel, isDecisionModel, listDecisionModels} from '../src/kev-deberta-preview-registry.js';

const KEV = 'kev-0.6b-q4f16';
const DEBERTA = 'open-jev-deberta-v3-large-q4f16';

test('two-model preview lists only the validated Kev and DeBERTa routes', () => {
  assert.deepEqual(listDecisionModels(), [KEV, DEBERTA]);
  assert.equal(isDecisionModel(KEV), true);
  assert.equal(isDecisionModel(DEBERTA), true);
  assert.equal(isDecisionModel('experimental-tinydecide-10m-q4'), false);
});

test('two-model preview describes pinned assets and native limits', () => {
  const kev = describeModel(KEV);
  const deberta = describeModel(DEBERTA);

  assert.equal(kev.assets.integrity, 'pinned-manifest');
  assert.equal(deberta.assets.integrity, 'pinned-files');
  assert.equal(deberta.checkpoint.revision, '7c79f25b5ac496089f448a969c801872ad59d31c');
  assert.deepEqual(deberta.backend.requiredFeatures, ['shader-f16']);
  assert.equal(deberta.features.textChoice.limits[0].max, 512);
});

test('two-model preview rejects unknown descriptors', () => {
  assert.throws(() => describeModel('experimental-strands-qwen-2b-fp16'), {code: 'INVALID_INPUT'});
});
