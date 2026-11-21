import {createHash} from 'node:crypto';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {dirname} from 'node:path';
import {isDeepStrictEqual} from 'node:util';
import {expect, test} from '@playwright/test';
import {
  MAPIKA2B_DECIDER_REVISION,
  MAPIKA2B_REPOSITORY,
  MAPIKA2B_REVISION
} from '../../packages/web-decision/src/mapika2b/contract.mts';
import {
  summarizeMapikaAnswerParity,
  summarizeMapikaLogits,
  mapikaReferenceMismatches,
  type AcceptedMapikaCase,
  type MapikaAnswerReference,
  type MapikaBrowserResult,
  type MapikaReference,
  type MapikaReferenceCase
} from './mapika2b-evidence.mts';
import {readAdapter, readHardwareReport} from './mapika2b-hardware.mts';
import {routeMapikaSnapshot} from './mapika2b-snapshot.mts';
import {verifyInstalledMapikaCandidate} from './mapika2b-installed.mts';

const oracle = new URL('../../scripts/mapika/', import.meta.url);
const rejectionPatterns = new Map<string, RegExp>([
  ['choice-over-limit', /2 to 255 options/u],
  ['score-over-limit', /2 to 10 levels/u],
  ['row-over-limit', /scoring row limit/u],
  ['image-over-limit', /image reference limit/u],
  ['image-unreferenced', /not referenced in the state/u]
]);

async function readOracle<T>(file: string | URL) {
  const text = await readFile(file, 'utf8');
  const data = JSON.parse(text) as MapikaReference<T>;
  expect(data.source).toMatchObject({
    repository: MAPIKA2B_REPOSITORY,
    revision: MAPIKA2B_REVISION,
    deciderRevision: MAPIKA2B_DECIDER_REVISION
  });
  return {...data, sha256: createHash('sha256').update(text).digest('hex')};
}

function selectCases(cases: readonly MapikaReferenceCase[]): readonly MapikaReferenceCase[] {
  const selection = process.env.WEB_DECISION_MAPIKA2B_CASES;
  if (!selection) return cases;
  const available = new Map(cases.map((item) => [item.id, item]));
  return [...new Set(selection.split(',').map((name) => name.trim()))].map((name) => {
    const item = available.get(name);
    if (!item) throw new Error(`Unknown Mapika fixture: ${name}.`);
    return item;
  });
}

function assertAnswerShape(item: AcceptedMapikaCase, {result}: Extract<MapikaBrowserResult, {status: 'accepted'}>) {
  for (const name of item.questionIndex.map(([question]) => question)) {
    const expected = item.renderedQuestions[name]!;
    const answer = result.answers[name];
    expect.soft(answer?.type, `${item.id}/${name}: kind`).toBe(expected.type);
    if (answer?.type === 'choice' || answer?.type === 'score') {
      expect
        .soft(Object.keys(answer.probabilities).sort(), `${item.id}/${name}: labels`)
        .toEqual(expected.names.map(String).sort());
    }
  }
}

function assertFixture(item: MapikaReferenceCase, observed: MapikaBrowserResult) {
  expect.soft(observed.status, item.id).toBe(item.status);
  if (item.status === 'rejected' && observed.status === 'rejected') {
    expect.soft(observed.code, item.id).toBe('INVALID_INPUT');
    const pattern = rejectionPatterns.get(item.id);
    if (pattern) expect.soft(observed.message, item.id).toMatch(pattern);
  }
  if (item.status !== 'accepted' || observed.status !== 'accepted') return;
  expect
    .soft(observed.rows, `${item.id}: forward inputs`)
    .toEqual(
      item.rows.map(({inputIds, slots, optionCounts, temperatures}) => ({inputIds, slots, optionCounts, temperatures}))
    );
  expect.soft(observed.questionNames, `${item.id}: question order`).toEqual(item.questionIndex.map(([name]) => name));
  expect.soft(observed.result.usage, `${item.id}: usage`).toEqual(item.usage);
  assertAnswerShape(item, observed);
  expect
    .soft(
      observed.logits.every((row) => row.every((slot) => slot.every(Number.isFinite))),
      item.id
    )
    .toBe(true);
}

function recordCase(item: MapikaReferenceCase, observed: MapikaBrowserResult, official?: MapikaAnswerReference) {
  assertFixture(item, observed);
  if (observed.status === 'rejected') return {...observed, id: item.id, expectedStatus: item.status};
  const exactAnswer = official ? isDeepStrictEqual(observed.result, official.result) : null;
  const logitDelta = official
    ? summarizeMapikaLogits(
        observed.logits,
        official.rows.map(({logits}) => logits)
      )
    : null;
  if (official) {
    expect
      .soft(mapikaReferenceMismatches(observed.result, official.result), `${item.id}: reference outputs`)
      .toEqual([]);
    expect.soft(logitDelta?.maxAbs, `${item.id}: maximum FP32 logit error`).toBeLessThanOrEqual(0.001);
  }
  return {
    id: item.id,
    status: observed.status,
    tokensByRow: observed.rows.map(({inputIds}) => inputIds.length),
    result: observed.result,
    referenceResult: official?.result,
    exactAnswer,
    answerParity: official ? summarizeMapikaAnswerParity(observed.result, official.result) : null,
    logits: observed.logits,
    logitDelta
  };
}

test.skip(
  process.env.WEB_DECISION_MAPIKA2B_GPU !== '1' || process.env.WEB_DECISION_REAL_GPU !== '1',
  'Set WEB_DECISION_MAPIKA2B_GPU=1 and WEB_DECISION_REAL_GPU=1 with staged Mapika text artifacts.'
);
test.skip(
  ({browserName}) => browserName !== 'chromium',
  'This candidate harness verifies physical GPU through Chromium.'
);

test('Mapika 2B v12 native TVM WebGPU matches the pinned decision oracle', async ({page}, testInfo) => {
  test.setTimeout(30 * 60 * 1000);
  const archive = await verifyInstalledMapikaCandidate();
  if (!archive && !process.env.WEB_DECISION_PUBLIC_ASSET_ROOT)
    throw new Error('WEB_DECISION_PUBLIC_ASSET_ROOT must contain the Mapika text artifacts.');
  const prepared = process.env.WEB_DECISION_MAPIKA2B_PREPARED_ORACLE ?? new URL('reference-prepared.json', oracle);
  const answerReference =
    process.env.WEB_DECISION_MAPIKA2B_ANSWER_ORACLE ?? new URL('reference-fp32-answers.json', oracle);
  const [fixture, answers] = await Promise.all([
    readOracle<MapikaReferenceCase>(prepared),
    readOracle<MapikaAnswerReference>(answerReference)
  ]);
  const cases = selectCases(fixture.cases);
  const official = new Map(answers.cases.map((item) => [item.id, item]));
  const snapshot = process.env.WEB_DECISION_MAPIKA2B_SNAPSHOT;
  const source = snapshot ? 'local snapshot' : 'pinned Hugging Face URLs';
  const events: string[] = [];
  const log = (message: string) => {
    const event = `${new Date().toISOString()} ${message}`;
    events.push(event);
    console.log(event);
  };
  const evidence = {
    schema: 'mapika2b-v12-candidate-webgpu-v1',
    checkpoint: fixture.source,
    source,
    candidatePackage: archive,
    oracleSha256: {prepared: fixture.sha256, answers: answers.sha256},
    plannedCases: cases.map(({id}) => id),
    hardware: null as Awaited<ReturnType<typeof readHardwareReport>> | null,
    adapter: null as Awaited<ReturnType<typeof readAdapter>>,
    artifacts: null as {readonly wasmSha256: string} | null,
    cases: [] as ReturnType<typeof recordCase>[],
    events
  };
  const output = testInfo.outputPath('mapika2b-webgpu.json');
  await mkdir(dirname(output), {recursive: true});
  const save = () => writeFile(output, JSON.stringify(evidence, null, 2));

  let started = false;
  try {
    log('Checking Chromium WebGPU hardware acceleration.');
    evidence.hardware = await readHardwareReport(page);
    expect(evidence.hardware.status).toBe('Hardware accelerated');
    page.on('pageerror', (error) => log(`Browser error: ${error.message}`));
    page.on('console', (message) => {
      if (message.type() === 'error' || message.type() === 'warning')
        log(`Browser ${message.type()}: ${message.text()}`);
    });
    if (snapshot) await routeMapikaSnapshot(page, snapshot, log);
    await page.goto('/mapika2b-harness.html');
    await page.waitForFunction(() => !!window.mapika2bGpuHarness);
    evidence.adapter = await readAdapter(page);
    expect(evidence.adapter?.isFallbackAdapter, 'WebGPU adapter uses physical hardware').toBe(false);
    log(`Loading ${MAPIKA2B_REPOSITORY} from ${source}.`);
    await save();
    evidence.artifacts = await page.evaluate(
      (installedArchive) => window.mapika2bGpuHarness!.initialize(installedArchive),
      archive !== null
    );
    started = true;
    log(`Loaded text Wasm SHA-256 ${evidence.artifacts.wasmSha256}.`);
    await save();

    for (const item of cases) {
      await test.step(item.id, async () => {
        log(`Running ${item.id}.`);
        await save();
        const observed = await page.evaluate((current) => window.mapika2bGpuHarness!.evaluate(current), item);
        evidence.cases.push(recordCase(item, observed, official.get(item.id)));
        const detail = observed.status === 'rejected' ? ` (${observed.code ?? 'error'}: ${observed.message})` : '';
        log(`${item.id}: ${observed.status}${detail}.`);
        await save();
      });
    }
  } finally {
    if (started && !page.isClosed()) {
      try {
        await page.evaluate(() => window.mapika2bGpuHarness!.dispose());
      } catch (cause) {
        log(`Candidate cleanup failed: ${String(cause)}.`);
        expect.soft(String(cause), 'candidate cleanup').toBe('');
      }
    }
    await save();
    await testInfo.attach('mapika2b-webgpu.json', {path: output, contentType: 'application/json'});
  }
});
