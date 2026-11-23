import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {
  GEMMA_V2_CPU_REFERENCE_PATH,
  GEMMA_V2_CPU_REFERENCE_SHA256,
  validateGemmaV2CpuReferences
} from './strands-gemma4-v2-source-reference.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const sha256 = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');
type SourceCases = ReturnType<typeof validateGemmaV2CpuReferences>;
type Difference = {path: string; actual: unknown; expected: unknown; absolute?: number};

function compareValues(actual: unknown, expected: unknown, path: string, differences: Difference[]): void {
  if (typeof actual === 'number' && typeof expected === 'number') {
    assert.ok(Number.isFinite(actual) && Number.isFinite(expected), `${path} must be finite.`);
    differences.push({path, actual, expected, absolute: Math.abs(actual - expected)});
    return;
  }
  if (actual && expected && typeof actual === 'object' && typeof expected === 'object') {
    const a = actual as Record<string, unknown>;
    const e = expected as Record<string, unknown>;
    for (const key of new Set([...Object.keys(a), ...Object.keys(e)])) {
      compareValues(a[key], e[key], `${path}.${key}`, differences);
    }
    return;
  }
  if (actual !== expected) differences.push({path, actual: actual ?? null, expected: expected ?? null});
}

function parseInput(bytes: Uint8Array): Record<string, unknown> {
  const text = new TextDecoder().decode(bytes).trim();
  if (text.startsWith('{')) return JSON.parse(text) as Record<string, unknown>;
  const lines = text
    .split(/\r?\n/u)
    .filter((line) => line.startsWith('LONG_RAMP_RESULT ') || line.startsWith('INSTALLED_REFERENCE_RAW '));
  assert.equal(lines.length, 1, 'Expected exactly one supported raw result in the log.');
  return JSON.parse(lines[0]!.slice(lines[0]!.indexOf(' ') + 1)) as Record<string, unknown>;
}

function extractCases(input: Record<string, unknown>): Record<string, unknown> {
  if (input.cases && typeof input.cases === 'object' && !Array.isArray(input.cases)) {
    return input.cases as Record<string, unknown>;
  }
  const phase = (input.split ?? input.baseline) as {answer?: unknown} | null | undefined;
  assert.ok(phase && phase.answer && typeof phase.answer === 'object', 'Missing long-ramp answer.');
  assert.ok(input.tokens === 1024 || input.tokens === 32768, 'No pinned source reference for this token count.');
  return {[`text-${input.tokens}`]: {inputTokens: input.tokens, result: phase.answer}};
}

type SourceCapture = {requestSha256: string; resultSha256: string; resultJson: string};
type SourceFixture = {cases: Record<string, SourceCapture>; capturedCases: Record<string, SourceCapture>};

function installedObservation(input: Record<string, unknown>, fixtureBytes: Uint8Array) {
  validateGemmaV2CpuReferences(fixtureBytes);
  const fixture = JSON.parse(new TextDecoder().decode(fixtureBytes)) as SourceFixture;
  const captures = {...fixture.cases, ...fixture.capturedCases};
  const matches = Object.entries(captures).filter(([, value]) => value.requestSha256 === input.requestSha256);
  assert.equal(matches.length, 1, 'Installed request SHA does not match exactly one pinned CPU input.');
  const [name, capture] = matches[0]!;
  const sourceResult = JSON.parse(capture.resultJson) as {answers: Record<string, unknown>; usage: unknown};
  const actual = input.actual as {answers?: unknown; usage?: unknown};
  assert.ok(actual && typeof actual === 'object' && actual.answers && actual.usage, 'Missing installed result.');
  const differences: Difference[] = [];
  compareValues(actual.answers, sourceResult.answers, 'answers', differences);
  compareValues(actual.usage, sourceResult.usage, 'usage', differences);
  const embeddedReferenceDifferences: Difference[] = [];
  compareValues(input.reference, sourceResult, 'reference', embeddedReferenceDifferences);
  return {
    status: 'pending-review',
    releaseApproved: false,
    numericalAcceptance: 'not-established',
    sourceFixturePath: GEMMA_V2_CPU_REFERENCE_PATH,
    sourceFixtureSha256: GEMMA_V2_CPU_REFERENCE_SHA256,
    case: name,
    requestBinding: 'matches-pinned-sha-in-log',
    referenceBinding:
      input.referenceSha256 === capture.resultSha256 ? 'matches-pinned-sha-in-log' : 'differs-from-pinned-sha',
    embeddedReferenceDifferences,
    expectedReferenceSha256: capture.resultSha256,
    observedReferenceSha256: input.referenceSha256 ?? null,
    recordedArchiveSha256: input.archiveSha256 ?? null,
    recordedBundleSha256: input.bundleSha256 ?? null,
    archiveBinding: 'not-independently-verified',
    inputTokens:
      actual.usage && typeof actual.usage === 'object'
        ? ((actual.usage as {input_tokens?: unknown}).input_tokens ?? null)
        : null,
    differences
  };
}

/** Assemble a source-only draft; release review must not treat it as verified. */
export function buildPendingGemmaReference(cases: SourceCases) {
  const draft = Object.fromEntries(
    Object.entries(cases).map(([name, {requestSha256, inputTokens, result}]) => [
      name,
      {requestSha256, inputTokens, result}
    ])
  );
  return {status: 'pending', sourceFixtureSha256: GEMMA_V2_CPU_REFERENCE_SHA256, cases: draft};
}

/** Collect differences for review only. Missing request binding or a numeric difference is never an approval. */
export function buildPendingGemmaDiagnostic(reference: SourceCases, bytes: Uint8Array, fixtureBytes?: Uint8Array) {
  const input = parseInput(bytes);
  if (input.actual !== undefined && input.reference !== undefined) {
    assert.ok(fixtureBytes, 'Pinned CPU fixture bytes are required for installed-result diagnostics.');
    return {...installedObservation(input, fixtureBytes), inputSha256: sha256(bytes)};
  }
  const cases = extractCases(input);
  const observations = Object.entries(cases).map(([name, value]) => {
    const source = reference[name];
    assert.ok(source, `No pinned CPU reference for ${name}.`);
    assert.ok(value && typeof value === 'object' && !Array.isArray(value), `Invalid ${name} observation.`);
    const actual = value as {requestSha256?: unknown; inputTokens?: unknown; result?: unknown};
    assert.ok(actual.result && typeof actual.result === 'object', `Missing ${name} result.`);
    const differences: Difference[] = [];
    compareValues(actual.result, source.result, name, differences);
    return {
      case: name,
      requestBinding:
        actual.requestSha256 === undefined
          ? 'unavailable'
          : actual.requestSha256 === source.requestSha256
            ? 'matches'
            : 'differs',
      inputTokens: actual.inputTokens ?? null,
      expectedInputTokens: source.inputTokens,
      differences
    };
  });
  assert.ok(observations.length > 0, 'No observations to compare.');
  return {
    status: 'pending-review',
    releaseApproved: false,
    numericalAcceptance: 'not-established',
    sourceFixturePath: GEMMA_V2_CPU_REFERENCE_PATH,
    sourceFixtureSha256: GEMMA_V2_CPU_REFERENCE_SHA256,
    inputSha256: sha256(bytes),
    graphSha256: input.graphSha256 ?? null,
    observations
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [inputPath, ...extra] = process.argv.slice(2);
  assert.ok(
    inputPath && extra.length === 0,
    'Expected --reference-draft, a JSON evidence file, or a supported raw result log.'
  );
  const fixtureBytes = await readFile(resolve(root, GEMMA_V2_CPU_REFERENCE_PATH));
  const source = validateGemmaV2CpuReferences(fixtureBytes);
  const output =
    inputPath === '--reference-draft'
      ? buildPendingGemmaReference(source)
      : buildPendingGemmaDiagnostic(source, await readFile(resolve(inputPath)), fixtureBytes);
  console.log(JSON.stringify(output, null, 2));
}
