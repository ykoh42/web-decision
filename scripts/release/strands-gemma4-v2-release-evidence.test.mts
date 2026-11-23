import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {archiveIdentity} from './curated-gpu-evidence.mts';
import {assertGemma4V2ReleaseRecord} from './strands-gemma4-v2-release-evidence.mts';
import type {ArchiveInfo} from './verify-release-evidence.ts';
import {
  assertGemmaV2LongReferences,
  gemmaV2CpuRequest,
  validateGemmaV2CpuReferences
} from './strands-gemma4-v2-source-reference.mts';

import {
  gemmaV2PublicCodeRow,
  gemmaV2PublicRunAttachment,
  gemmaV2PublicTitle,
  gemmaV2PublicVectorsAttachment
} from '../../tests/browser/gemma-v2-public-evidence.mjs';
import {readGemmaV2PlaywrightRun} from './strands-gemma4-v2-release-browser.mjs';
import {readPinnedGemma4V2Subset} from './strands-gemma4-v2-conformance.mjs';

// Synthetic records test rejection logic only; they are never release evidence.
const archive = {
  archiveSha256: 'archive',
  packageTreeSha256: 'tree',
  bundleSha256: 'bundle',
  packageName: 'web-decision',
  packageVersion: '0.0.0-test'
} as ArchiveInfo;
const ids = ['text-1024', 'text-32768', 'noul', 'score', 'image-single', 'image-multiple'];
const cases = Object.fromEntries(
  ids.map((id) => [
    id,
    {
      requestSha256: 'a'.repeat(64),
      inputTokens: id === 'text-32768' ? 32768 : id === 'text-1024' ? 1024 : 17,
      outputTokens: id === 'noul' || id === 'score' ? 2 : 1,
      imageCount: id === 'image-single' ? 1 : id === 'image-multiple' ? 2 : 0,
      result:
        id === 'noul'
          ? {type: 'noul', noul: 0.8}
          : id === 'score'
            ? {type: 'score', score: 0.8}
            : {type: 'choice', choice: 'A', probabilities: {A: 0.8, B: 0.2}}
    }
  ])
);
const reference = {status: 'pinned-upstream-cpu', cases};
const report = {
  schemaVersion: 1,
  modelId: 'strands-decider-e2b-gemma4-v2-2611',
  source: 'pinned-upstream',
  archive: archiveIdentity(archive),
  browser: {name: 'chromium', version: 'test'},
  hardware: {status: 'Hardware accelerated'},
  adapter: {isFallbackAdapter: false},
  cases,
  offline: {cacheStatus: 'complete', blockedRequests: [], result: cases['image-single']!.result},
  cleanup: 'passed'
};

test('collected CPU results bind all six exact upstream requests without approving release', () => {
  const bytes = readFileSync(
    new URL('../../tests/fixtures/strands-gemma4-v2-upstream-cpu-reference.json', import.meta.url)
  );
  const cpu = validateGemmaV2CpuReferences(bytes);
  assert.equal(cpu['text-1024']!.inputTokens, 1024);
  assert.equal(cpu['text-32768']!.inputTokens, 32768);
  assert.equal(cpu['image-single']!.inputTokens, 337);
  assert.equal(cpu['image-multiple']!.inputTokens, 598);
  assert.equal(cpu['noul']!.inputTokens, 227);
  assert.equal(cpu['score']!.inputTokens, 227);
  assert.equal(cpu['noul']!.requestSha256, cpu['score']!.requestSha256);
  assert.deepEqual(cpu['noul']!.result, {type: 'noul', noul: 0.9986});
  assert.equal((cpu['score']!.result as {score: number}).score, 1.6342);
  assert.equal((cpu['image-single']!.result as {choice: string}).choice, 'red');
  assert.equal((cpu['image-multiple']!.result as {choice: string}).choice, 'green');
  assert.equal(JSON.parse(gemmaV2CpuRequest(955)).state.length, 955 * ' apple'.length);
  assertGemmaV2LongReferences({cases: cpu}, cpu);
  assert.throws(() => validateGemmaV2CpuReferences(Buffer.concat([bytes, Buffer.from(' ')])), /fixture differs/u);
  const changed = structuredClone(cpu);
  (changed['text-32768']!.result as {choice: string}).choice = 'yes';
  assert.throws(() => assertGemmaV2LongReferences({cases: changed}, cpu), /result differs/u);
  assert.throws(
    () => assertGemmaV2LongReferences({cases: {'text-1024': cpu['text-1024']}}, cpu),
    /Missing text-32768/u
  );
});

test('Gemma release records require the exact archive, coverage, reference, offline, and a non-fallback adapter', () => {
  const measured = assertGemma4V2ReleaseRecord(report, archive, reference);
  assert.ok(measured.length > 0 && measured.every((difference) => difference.absolute === 0));
  const shifted = {
    ...report.cases,
    'text-1024': {
      ...report.cases['text-1024'],
      result: {type: 'choice', choice: 'A', probabilities: {A: 0.79, B: 0.21}}
    }
  };
  const differences = assertGemma4V2ReleaseRecord({...report, cases: shifted}, archive, reference);
  assert.ok(
    differences.some((difference) => difference.path === 'text-1024.probabilities.A' && difference.absolute > 0)
  );
  assert.throws(
    () => assertGemma4V2ReleaseRecord(report, archive, {...reference, status: 'pending'}),
    /requires pinned upstream CPU cases/u
  );
  assert.throws(
    () => assertGemma4V2ReleaseRecord({...report, source: 'local-fixture'}, archive, reference),
    /actual pinned upstream/u
  );
  assert.throws(
    () => assertGemma4V2ReleaseRecord({...report, archive: {...report.archive, sha256: 'wrong'}}, archive, reference),
    /different npm archive/u
  );
  assert.throws(
    () => assertGemma4V2ReleaseRecord({...report, adapter: {isFallbackAdapter: true}}, archive, reference),
    /non-fallback WebGPU/u
  );
  const missingLong = {...report.cases};
  delete missingLong['text-32768'];
  assert.throws(() => assertGemma4V2ReleaseRecord({...report, cases: missingLong}, archive, reference));
  assert.throws(
    () =>
      assertGemma4V2ReleaseRecord(
        {...report, offline: {...report.offline, blockedRequests: ['https://example.com']}},
        archive,
        reference
      ),
    /network request/u
  );
  assert.throws(() => assertGemma4V2ReleaseRecord({...report, cleanup: 'failed'}, archive, reference), /cleanup/u);
  const bad = {
    ...report.cases,
    'text-1024': {...report.cases['text-1024'], result: {type: 'choice', choice: 'B', probabilities: {A: 0.8, B: 0.2}}}
  };
  assert.throws(() => assertGemma4V2ReleaseRecord({...report, cases: bad}, archive, reference), /differs/u);
});

test('the exact official selection excludes rows 300 through 429', () => {
  const bytes = readFileSync(new URL('../../tests/fixtures/strands-gemma4-v2-official-subset.json', import.meta.url));
  const rows = readPinnedGemma4V2Subset(bytes);
  assert.deepEqual(rows, [
    ...Array.from({length: 300}, (_, index) => index),
    ...Array.from({length: 60}, (_, index) => 430 + index)
  ]);
  assert.throws(() => readPinnedGemma4V2Subset(Buffer.concat([bytes, Buffer.from(' ')])), /subset bytes differ/u);
});

test('public Gemma vectors preserve ordered labels and count all question tokens', () => {
  const golden = {
    variant: 'strands-chat',
    state_ids: [10],
    request: {
      questions: {
        topic: {type: 'choice' as const, criteria: {red: 'red', blue: 'blue'}},
        accepted: {type: 'noul' as const, criteria: null},
        urgency: {type: 'score' as const, criteria: ['low', 'high']}
      }
    },
    questions: {
      topic: {codes: ['A', 'B'], question_ids: [100, 101]},
      accepted: {codes: ['A', 'B'], question_ids: [200]},
      urgency: {codes: ['A', 'B'], question_ids: [300, 301, 302]}
    }
  };
  const observed = {
    model: 'strands-decider-e2b-gemma4-v2-2611',
    usage: {input_tokens: 7, output_tokens: 3},
    answers: {
      topic: {type: 'choice' as const, probabilities: {red: 0.8, blue: 0.2}},
      accepted: {type: 'noul' as const, noul: 0.9999},
      urgency: {type: 'score' as const, probabilities: {'0': 0.6, '1': 0.4}}
    }
  };
  assert.deepEqual(gemmaV2PublicCodeRow(3, golden, observed), {
    line: 3,
    questions: {
      topic: {codes: ['A', 'B'], probabilities: [0.8, 0.2]},
      accepted: {codes: ['A', 'B'], probabilities: [0.0001, 0.9999]},
      urgency: {codes: ['A', 'B'], probabilities: [0.6, 0.4]}
    }
  });
  assert.throws(() => gemmaV2PublicCodeRow(3, golden, {...observed, usage: {...observed.usage, input_tokens: 6}}));
  assert.throws(
    () =>
      gemmaV2PublicCodeRow(3, golden, {
        ...observed,
        answers: {...observed.answers, topic: {type: 'choice', probabilities: {blue: 0.2, red: 0.8}}}
      }),
    /labels or order differ/u
  );
});

test('Gemma browser evidence requires passing Chromium results and both embedded files', () => {
  const tmpRoot = '/tmp/synthetic-gemma-browser-run';
  const runBytes = Buffer.from(JSON.stringify({testFixture: true}));
  const vectors = Buffer.from('{}\n');
  const attachments = [
    {name: gemmaV2PublicRunAttachment, contentType: 'application/json', body: runBytes.toString('base64')},
    {name: gemmaV2PublicVectorsAttachment, contentType: 'application/x-ndjson', body: vectors.toString('base64')}
  ];
  const result = {status: 'passed', duration: 75, retry: 0, errors: [], attachments};
  const spec = {
    file: 'gemma-v2-public-real-webgpu.spec.ts',
    title: gemmaV2PublicTitle,
    ok: true,
    tests: [{projectName: 'chromium', status: 'expected', expectedStatus: 'passed', results: [result]}]
  };
  const playwright = {
    config: {rootDir: tmpRoot + '/tests/browser'},
    stats: {expected: 1, unexpected: 0, flaky: 0, skipped: 0, startTime: '2026-11-23T00:00:00Z'},
    errors: [],
    suites: [{specs: [spec]}]
  };
  assert.deepEqual(readGemmaV2PlaywrightRun(playwright, tmpRoot), {runBytes, vectors});
  assert.throws(() => readGemmaV2PlaywrightRun(report, tmpRoot), /Missing Playwright configuration/u);
  const changedReport = structuredClone(playwright);
  changedReport.suites[0]!.specs[0]!.tests[0]!.results[0]!.attachments.pop();
  assert.throws(() => readGemmaV2PlaywrightRun(changedReport, tmpRoot), /Both browser-run files/u);
  const retriedReport = structuredClone(playwright);
  retriedReport.suites[0]!.specs[0]!.tests[0]!.results[0]!.retry = 1;
  assert.throws(() => readGemmaV2PlaywrightRun(retriedReport, tmpRoot));
  const wrongSpec = structuredClone(playwright);
  wrongSpec.suites[0]!.specs[0]!.file = 'private-candidate.spec.ts';
  assert.throws(() => readGemmaV2PlaywrightRun(wrongSpec, tmpRoot));
});
