import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {
  mapikaReleaseCases,
  type MapikaAnswerReference,
  type MapikaReference
} from '../../tests/browser/mapika2b-evidence.mts';
import {archiveIdentity, sha256} from './curated-gpu-evidence.mts';
import {assertMapikaReleaseRecord} from './mapika-release-evidence.mts';
import type {ArchiveInfo} from './verify-release-evidence.ts';

const fixtures = new URL('../mapika/', import.meta.url);
const prepared = readFileSync(new URL('reference-fp32-prepared.json', fixtures));
const answers = readFileSync(new URL('reference-fp32-answers.json', fixtures));
const source = JSON.parse(answers.toString('utf8')) as MapikaReference<MapikaAnswerReference>;
const archive = {
  archiveSha256: 'mock-archive',
  packageTreeSha256: 'mock-tree',
  bundleSha256: 'mock-bundle',
  packageName: 'web-decision',
  packageVersion: '0.6.0'
} as ArchiveInfo;

test('Mapika release evidence is bound to live assets, exact package, and offline parity', () => {
  const results = mapikaReleaseCases.map((id) => source.cases.find((item) => item.id === id)?.result);
  const limits = {maxBufferSize: 1_073_741_824};
  const report = {
    schemaVersion: 1,
    modelId: 'mapika-decider-2b-v12',
    source: 'pinned-upstream',
    archive: archiveIdentity(archive),
    references: {preparedSha256: sha256(prepared), answersSha256: sha256(answers)},
    cases: mapikaReleaseCases,
    browser: {name: 'chromium'},
    hardware: {status: 'Hardware accelerated'},
    adapter: {isFallbackAdapter: false, ...limits},
    progressive: {
      fullCheckpointBytes: 4_426_558_200,
      unboundedWeightRequests: [],
      cacheStatusAfterLoad: 'incomplete',
      cacheStatusBeforePrepare: 'incomplete',
      afterLoad: rangeCounts(100, 0),
      afterFirstText: rangeCounts(200, 0),
      afterFirstImage: rangeCounts(200, 300),
      beforePrepare: rangeCounts(200, 300),
      afterPrepare: rangeCounts(500, 300)
    },
    results,
    modelCacheStatus: 'complete',
    offline: {cacheStatus: 'complete', blockedRequests: [], results}
  };
  assert.doesNotThrow(() => assertMapikaReleaseRecord(report, archive, source, limits));
  const rejected = [
    {change: {source: 'local-snapshot'}, error: /live pinned Hugging Face/u},
    {change: {archive: {...report.archive, sha256: 'other'}}, error: /different npm archive/u},
    {change: {progressive: {...report.progressive, afterLoad: rangeCounts(100, 10)}}, error: /vision weights before/u},
    {
      change: {progressive: {...report.progressive, afterPrepare: report.progressive.beforePrepare}},
      error: /separate offline preparation/u
    },
    {change: {results: Array(results.length).fill(results[0])}, error: /result differs/u},
    {change: {offline: {...report.offline, blockedRequests: ['unexpected']}}, error: /requested model assets/u}
  ];
  for (const {change, error} of rejected)
    assert.throws(() => assertMapikaReleaseRecord({...report, ...change}, archive, source, limits), error);
});

function rangeCounts(textBytes: number, visionBytes: number) {
  return {
    textBytes,
    visionBytes,
    requestedBytes: textBytes + visionBytes,
    textRanges: 1,
    visionRanges: Number(visionBytes > 0)
  };
}
