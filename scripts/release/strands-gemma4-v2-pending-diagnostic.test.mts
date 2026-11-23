import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {buildPendingGemmaDiagnostic, buildPendingGemmaReference} from './strands-gemma4-v2-pending-diagnostic.mts';
import {assertGemma4V2ReleaseRecord} from './strands-gemma4-v2-release-evidence.mts';
import type {ArchiveInfo} from './verify-release-evidence.ts';
import {validateGemmaV2CpuReferences} from './strands-gemma4-v2-source-reference.mts';

const fixtureBytes = readFileSync(
  new URL('../../tests/fixtures/strands-gemma4-v2-upstream-cpu-reference.json', import.meta.url)
);
const source = validateGemmaV2CpuReferences(fixtureBytes);

test('long-ramp diagnostics retain raw differences without inferring request binding or approval', () => {
  const raw = {
    graphSha256: 'a'.repeat(64),
    tokens: 32768,
    split: {answer: {type: 'choice', choice: 'no', probabilities: {yes: 0.3917, no: 0.6083}, confidence: 0.2166}}
  };
  const log = Buffer.from(`some output\nLONG_RAMP_RESULT ${JSON.stringify(raw)}\n`);
  const actual = buildPendingGemmaDiagnostic(source, log);
  assert.equal(actual.status, 'pending-review');
  assert.equal(actual.releaseApproved, false);
  assert.equal(actual.numericalAcceptance, 'not-established');
  assert.equal(actual.graphSha256, raw.graphSha256);
  assert.equal(actual.observations[0]!.requestBinding, 'unavailable');
  const differences = actual.observations[0]!.differences;
  assert.ok(Math.abs(differences.find(({path}) => path.endsWith('probabilities.yes'))!.absolute! - 0.0131) < 1e-10);
  assert.ok(Math.abs(differences.find(({path}) => path.endsWith('confidence'))!.absolute! - 0.0263) < 1e-10);
  assert.deepEqual(
    buildPendingGemmaDiagnostic(source, Buffer.from(JSON.stringify(raw))).observations,
    actual.observations
  );
  assert.throws(() => buildPendingGemmaDiagnostic(source, Buffer.from(`${log}${log}`)), /exactly one/u);
});

test('case reports preserve wrong input, result, or request rather than approving them', () => {
  const original = source['noul']!;
  const raw = {cases: {noul: {...original, requestSha256: 'wrong', inputTokens: 4, result: {type: 'choice'}}}};
  const diagnostic = buildPendingGemmaDiagnostic(source, Buffer.from(JSON.stringify(raw)));
  assert.ok(diagnostic.observations);
  const item = diagnostic.observations[0]!;
  assert.equal(item.requestBinding, 'differs');
  assert.equal(item.inputTokens, 4);
  assert.equal(item.expectedInputTokens, 227);
  assert.ok(item.differences.some(({path}) => path === 'noul.type'));
  assert.ok(item.differences.some(({path}) => path === 'noul.noul'));
  assert.throws(() => buildPendingGemmaDiagnostic(source, Buffer.from('{"cases":{"unknown":{}}}')), /No pinned/u);
});

test('source-only draft contains all six cases and cannot pass release verification', () => {
  const draft = buildPendingGemmaReference(source);
  assert.equal(draft.status, 'pending');
  assert.deepEqual(Object.keys(draft.cases).sort(), [
    'image-multiple',
    'image-single',
    'noul',
    'score',
    'text-1024',
    'text-32768'
  ]);
  assert.throws(() => assertGemma4V2ReleaseRecord({}, {} as ArchiveInfo, draft), /requires pinned upstream CPU cases/u);
});

test('installed raw logs bind the exact CPU request and report all answer and usage differences', () => {
  const fixture = JSON.parse(fixtureBytes.toString('utf8'));
  const entry = fixture.capturedCases['multi-kind'];
  const reference = JSON.parse(entry.resultJson);
  const actual = structuredClone(reference);
  actual.answers.route.extra = 1;
  actual.answers.refund.noul -= 0.01;
  delete actual.answers.urgency.confidence;
  const raw = {
    requestSha256: entry.requestSha256,
    referenceSha256: entry.resultSha256,
    archiveSha256: 'recorded-only',
    bundleSha256: 'also-recorded-only',
    reference,
    actual
  };
  const log = Buffer.from(`testing\nINSTALLED_REFERENCE_RAW ${JSON.stringify(raw)}\n`);
  const diagnostic = buildPendingGemmaDiagnostic(source, log, fixtureBytes);
  assert.ok('case' in diagnostic);
  assert.equal(diagnostic.case, 'multi-kind');
  assert.equal(diagnostic.requestBinding, 'matches-pinned-sha-in-log');
  assert.equal(diagnostic.referenceBinding, 'matches-pinned-sha-in-log');
  assert.equal(diagnostic.archiveBinding, 'not-independently-verified');
  assert.equal(diagnostic.releaseApproved, false);
  assert.ok(diagnostic.differences?.some(({path}) => path === 'answers.route.extra'));
  assert.ok(diagnostic.differences?.some(({path}) => path === 'answers.refund.noul'));
  assert.ok(diagnostic.differences?.some(({path}) => path === 'answers.urgency.confidence'));
  assert.throws(() => buildPendingGemmaDiagnostic(source, log), /fixture bytes/u);
  assert.throws(
    () =>
      buildPendingGemmaDiagnostic(source, Buffer.from(JSON.stringify({...raw, requestSha256: 'wrong'})), fixtureBytes),
    /request SHA/u
  );
  const shiftedReference = {...reference, model: 'other'};
  const shifted = buildPendingGemmaDiagnostic(
    source,
    Buffer.from(JSON.stringify({...raw, reference: shiftedReference})),
    fixtureBytes
  );
  assert.ok('case' in shifted);
  assert.ok(shifted.embeddedReferenceDifferences?.some(({path}) => path === 'reference.model'));
  const otherReference = buildPendingGemmaDiagnostic(
    source,
    Buffer.from(JSON.stringify({...raw, referenceSha256: 'other'})),
    fixtureBytes
  );
  assert.ok('case' in otherReference);
  assert.equal(otherReference.referenceBinding, 'differs-from-pinned-sha');
});
