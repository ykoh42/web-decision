import assert from 'node:assert/strict';
import test from 'node:test';
import {describeModel, isDecisionModel, listDecisionModels} from '../src/kev-deberta-reflexes-preview-registry.js';

const KEV = 'kev-0.6b-q4f16';
const DEBERTA = 'open-jev-deberta-v3-large-q4f16';
const REFLEXES = 'kev-0.6b-reflexes-q4f16';

test('three-model preview lists only the pinned Kev, DeBERTa and Reflexes routes', () => {
  assert.deepEqual(listDecisionModels(), [KEV, DEBERTA, REFLEXES]);
  assert.equal(isDecisionModel(KEV), true);
  assert.equal(isDecisionModel(DEBERTA), true);
  assert.equal(isDecisionModel(REFLEXES), true);
  assert.equal(isDecisionModel('experimental-tinydecide-10m-q4'), false);
});

test('three-model preview describes pinned assets and native limits', () => {
  const kev = describeModel(KEV);
  const deberta = describeModel(DEBERTA);
  const reflexes = describeModel(REFLEXES);

  assert.equal(kev.verification.currentBundleGpu, 'not-revalidated');
  assert.deepEqual(kev.features.typedTextQuestionTypes, ['choice', 'noul', 'score']);
  assert.match(kev.verification.priorEvidence, /Preview\.39 has not completed GPU validation/);
  assert.equal(kev.assets.integrity, 'pinned-manifest');
  assert.equal(deberta.assets.integrity, 'pinned-files');
  assert.equal(deberta.checkpoint.revision, '7c79f25b5ac496089f448a969c801872ad59d31c');
  assert.deepEqual(deberta.backend.requiredFeatures, ['shader-f16']);
  assert.equal(deberta.features.textChoice.limits[0].max, 512);
  assert.equal(deberta.verification.currentBundleGpu, 'not-revalidated');
  assert.match(deberta.verification.priorEvidence, /Preview\.36/);
  assert.equal(reflexes.assets.integrity, 'pinned-manifest');
  assert.equal(reflexes.assets.manifestSha256, '1cd07a9326e4d8273d8972fdfb4ba9ab23cf3ff1f5e5b99cbea263116eebee07');
  assert.equal(reflexes.checkpoint.revision, '243bcef054df85dc6ed4e4becb68590c65a9f36e');
  assert.deepEqual(reflexes.backend.requiredFeatures, ['shader-f16']);
  assert.equal(reflexes.features.textChoice.limits[0].max, 8191);
  assert.equal(reflexes.features.textChoice.limits[1].max, 8192);
  assert.equal(reflexes.features.textChoice.limits[2].min, 1);
  assert.equal(reflexes.features.textChoice.limits[2].max, 255);
  assert.equal(reflexes.features.textChoice.abstention, 'not-implemented');
  assert.deepEqual(reflexes.features.typedTextQuestionTypes, ['choice', 'noul', 'score']);
  assert.equal(reflexes.verification.currentBundleGpu, 'not-revalidated');
  assert.match(reflexes.verification.priorEvidence, /Preview\.37 did not complete/);
});

test('three-model preview rejects unknown descriptors', () => {
  assert.throws(() => describeModel('experimental-strands-qwen-2b-fp16'), {code: 'INVALID_INPUT'});
});
