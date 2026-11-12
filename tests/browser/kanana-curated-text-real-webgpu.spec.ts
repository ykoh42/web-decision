import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {open, readFile, stat} from 'node:fs/promises';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';
import {createKananaReleaseEvidence, KANANA_RELEASE_EVIDENCE_PREFIX} from './kanana-candidate-evidence.ts';
import type {KananaDecisionModel} from '../../packages/web-decision/src/curated/types.d.ts';
import {
  kananaCheckpointSha256,
  kananaModelId,
  kananaRevision,
  kananaSnapshotDirectory,
  kananaUpstreamBase,
  verifyPinnedKananaSnapshot
} from './kanana-snapshot.ts';
const localBuildMode = process.env.WEB_DECISION_KANANA_LOCAL_BUILD_GPU === '1';
const enabled =
  process.env.WEB_DECISION_KANANA_CURATED_REAL_GPU === '1' ||
  process.env.WEB_DECISION_KANANA_LOCAL_GPU === '1' ||
  localBuildMode;
const upstreamPath = new URL(kananaUpstreamBase).pathname;
const probabilityTolerance = 0.001;
const scoreReferenceTolerance = Object.freeze({
  reference: 'Python/PyTorch BF16',
  candidate: 'TVM WebGPU FP32 compute',
  observedMaxAbsoluteDelta: Object.freeze({
    probability: 0.004926208300126356,
    score: 0.0019322554634566558,
    confidence: 0.007389312450189478
  }),
  probabilityTolerance: 0.006,
  scoreTolerance: 0.005,
  confidenceTolerance: 0.01
});
const fixtureBytes = readFileSync(new URL('../fixtures/kanana-decision-python-reference.json', import.meta.url));
const fixtureSha256 = createHash('sha256').update(fixtureBytes).digest('hex');
const publicTextFixtureBytes = readFileSync(
  new URL('../fixtures/kanana-public-text-python-reference.json', import.meta.url)
);
const publicTextFixtureSha256 = createHash('sha256').update(publicTextFixtureBytes).digest('hex');

type ReferenceFixture = Readonly<{
  modelId: string;
  revision: string;
  checkpointSha256: string;
  source: Readonly<{runtime: string; parameterDtype: string; mode: string}>;
  request: Readonly<{
    state: string;
    state_format: 'text';
    temperature: number;
    questions: Readonly<Record<string, ChoiceQuestion>>;
  }>;
  result: Readonly<{
    answers: Readonly<
      Record<string, Readonly<{choice: string; confidence: number; probabilities: Record<string, number>}>>
    >;
  }>;
}>;

type ChoiceQuestion = Readonly<{
  type: 'choice';
  instructions: string;
  criteria: Readonly<Record<string, string>>;
}>;

type NoulQuestion = Readonly<{
  type: 'noul';
  instructions: string;
  criteria: Readonly<{false: string; true: string}>;
}>;

type ScoreQuestion = Readonly<{
  type: 'score';
  instructions: string;
  criteria: readonly string[];
}>;

type PublicTextReferenceFixture = Readonly<{
  modelId: string;
  revision: string;
  checkpointSha256: string;
  source: ReferenceFixture['source'];
  request: Readonly<{
    state: string;
    state_format: 'text';
    temperature: number;
    questions: Readonly<Record<string, NoulQuestion | ScoreQuestion>>;
  }>;
  result: Readonly<{
    answers: Readonly<{
      needs_immediate_page: Readonly<{type: 'noul'; noul: number}>;
      delay_severity: Readonly<{
        type: 'score';
        score: number;
        confidence: number;
        legend: Record<string, string>;
        probabilities: Record<string, number>;
      }>;
    }>;
  }>;
}>;

type DecisionQuestion = ChoiceQuestion | NoulQuestion | ScoreQuestion;

const reference = JSON.parse(fixtureBytes.toString('utf8')) as ReferenceFixture;
const publicTextReference = JSON.parse(publicTextFixtureBytes.toString('utf8')) as PublicTextReferenceFixture;
type BrowserRequest = Readonly<{
  state: string;
  stateFormat: 'text';
  questions: Readonly<Record<string, DecisionQuestion>>;
  temperature: number;
}>;

async function readRange(file: Awaited<ReturnType<typeof open>>, start: number, end: number) {
  const bytes = Buffer.allocUnsafe(end - start + 1);
  let offset = 0;
  while (offset < bytes.byteLength) {
    const {bytesRead} = await file.read(bytes, offset, bytes.byteLength - offset, start + offset);
    if (!bytesRead) throw new Error(`Unexpected end of local Kanana checkpoint at ${start + offset}.`);
    offset += bytesRead;
  }
  return bytes;
}

test('the Kanana candidate matches the pinned text-only PyTorch choice reference on WebGPU', async ({
  page,
  browserName,
  browser
}, testInfo) => {
  test.skip(!enabled, 'Set WEB_DECISION_KANANA_CURATED_REAL_GPU=1 to run the Kanana candidate GPU check.');
  test.setTimeout(45 * 60 * 1000);

  await verifyPinnedKananaSnapshot();
  expect(fixtureSha256).toBe('822156454d468153cef23fde6fad8c88b59821d9da1b86897733fa906abfd420');
  expect(publicTextFixtureSha256).toBe('4c2b8763ef2d11201723d6e37479fefe5a7d3068823f2eca9e03c9ae7610c5e6');
  expect(reference.modelId).toBe(`kakaocorp/${kananaModelId}`);
  expect(reference.revision).toBe(kananaRevision);
  expect(reference.checkpointSha256).toBe(kananaCheckpointSha256);
  expect(reference.source).toEqual({
    runtime: 'Python/PyTorch system_one',
    parameterDtype: 'bfloat16',
    mode: 'checkpoint_full'
  });
  expect(reference.request.state_format).toBe('text');
  expect(Object.keys(reference.request.questions)).toEqual(['delivery_status']);
  expect(publicTextReference.modelId).toBe(reference.modelId);
  expect(publicTextReference.revision).toBe(reference.revision);
  expect(publicTextReference.checkpointSha256).toBe(reference.checkpointSha256);
  expect(publicTextReference.source).toEqual(reference.source);
  expect(publicTextReference.request.state).toBe(reference.request.state);
  expect(publicTextReference.request.state_format).toBe(reference.request.state_format);
  expect(publicTextReference.request.temperature).toBe(reference.request.temperature);
  expect(Object.keys(publicTextReference.request.questions)).toEqual(['needs_immediate_page', 'delay_severity']);

  const modelPath = resolve(kananaSnapshotDirectory, 'model.safetensors');
  const modelFile = await open(modelPath, 'r');
  const modelSize = (await stat(modelPath)).size;
  const fetched: {name: string; bytes: number; range?: string}[] = [];
  const visionWasmRequests: string[] = [];

  try {
    page.on('request', (request) => {
      if (new URL(request.url()).pathname.includes('kanana-vision-model')) visionWasmRequests.push(request.url());
    });
    await page.route(`${kananaUpstreamBase}**`, async (route) => {
      const request = route.request();
      const name = new URL(request.url()).pathname.slice(upstreamPath.length);
      const cors = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Range',
        'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        'Access-Control-Expose-Headers': 'Accept-Ranges, Content-Length, Content-Range'
      };

      if (request.method() === 'OPTIONS') {
        await route.fulfill({status: 204, headers: cors});
        return;
      }

      if (name === 'model.safetensors') {
        const range = request.headers().range;
        const match = range && /^bytes=(\d+)-(\d+)$/u.exec(range);
        if (!match) {
          await route.fulfill({status: 416, headers: {...cors, 'Content-Range': `bytes */${modelSize}`}});
          return;
        }
        const start = Number(match[1]);
        const end = Number(match[2]);
        if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || end >= modelSize) {
          await route.fulfill({status: 416, headers: {...cors, 'Content-Range': `bytes */${modelSize}`}});
          return;
        }
        const bytes = await readRange(modelFile, start, end);
        fetched.push({name, bytes: bytes.byteLength, range});
        await route.fulfill({
          status: 206,
          headers: {
            ...cors,
            'Accept-Ranges': 'bytes',
            'Content-Type': 'application/octet-stream',
            'Content-Length': String(bytes.byteLength),
            'Content-Range': `bytes ${start}-${end}/${modelSize}`
          },
          body: bytes
        });
        return;
      }

      if (!['config.json', 'tokenizer_config.json', 'tokenizer.json'].includes(name)) {
        await route.fulfill({status: 404, headers: cors});
        return;
      }

      const bytes = await readFile(resolve(kananaSnapshotDirectory, name));
      fetched.push({name, bytes: bytes.byteLength});
      await route.fulfill({status: 200, headers: {...cors, 'Content-Length': String(bytes.byteLength)}, body: bytes});
    });

    const browserErrors: string[] = [];
    page.on('pageerror', (error) => browserErrors.push(error.message));
    await page.goto('/');
    await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');
    const request: BrowserRequest = {
      state: reference.request.state,
      stateFormat: reference.request.state_format,
      questions: {
        ...reference.request.questions,
        ...publicTextReference.request.questions
      },
      temperature: reference.request.temperature
    };
    const {adapterInfo, result, scoreOnly, truncationProbe} = await page.evaluate(async (input: BrowserRequest) => {
      const adapter = await navigator.gpu?.requestAdapter({powerPreference: 'high-performance'});
      if (!adapter) throw new Error('Chromium did not provide a WebGPU adapter.');
      const adapterWithFallback = adapter as GPUAdapter & {readonly isFallbackAdapter?: boolean};
      const isFallback = adapterWithFallback.isFallbackAdapter ?? adapter.info?.isFallbackAdapter ?? null;
      if (isFallback !== false) throw new Error(`Expected a hardware WebGPU adapter; got ${String(isFallback)}.`);

      const model = (await window.webDecision
        .loadDecisionModel({modelId: 'kanana-2-v-decide-3b-2611', cache: false})
        .catch((cause) => {
          const detail = cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause);
          throw new Error(`Kanana model loading failed: ${detail}`, {cause});
        })) as unknown as KananaDecisionModel;
      try {
        const modelMaxLength = model.description.features.textChoice.limits.find(
          ({resource, scope}) => resource === 'tokens' && scope === 'question'
        )?.max;
        if (modelMaxLength !== 20_480) throw new Error(`Kanana model limit changed: ${String(modelMaxLength)}.`);

        const scoreOnlyOutput = await model
          .systemOne({...input, questions: {delay_severity: input.questions.delay_severity!}})
          .catch((cause) => {
            const detail = cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause);
            throw new Error(`Kanana score-only inference failed: ${detail}`, {cause});
          });
        const scoreOnlyAnswer = scoreOnlyOutput.answers.delay_severity;
        if (!scoreOnlyAnswer || scoreOnlyAnswer.type !== 'score')
          throw new Error('Kanana returned no score-only result.');
        const output = await model.systemOne(input).catch((cause) => {
          const detail = cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause);
          throw new Error(`Kanana systemOne inference failed: ${detail}`, {cause});
        });
        const answer = output.answers.delivery_status;
        if (!answer || answer.type !== 'choice') throw new Error('Kanana returned no choice result.');
        const noulAnswer = output.answers.needs_immediate_page;
        if (!noulAnswer || noulAnswer.type !== 'noul') throw new Error('Kanana returned no noul result.');
        const scoreAnswer = output.answers.delay_severity;
        if (!scoreAnswer || scoreAnswer.type !== 'score') throw new Error('Kanana returned no score result.');
        const requestedMaxLength = 128;
        const truncationProbeOutput = await model
          .systemOne(
            {
              state: 'x '.repeat(256),
              stateFormat: 'text',
              questions: {
                truncation_probe: {
                  type: 'choice',
                  instructions: 'Choose the label for this truncation test.',
                  criteria: {first: 'First option', second: 'Second option'}
                }
              }
            },
            {maxLength: requestedMaxLength}
          )
          .catch((cause) => {
            const detail = cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause);
            throw new Error(`Kanana truncation-probe inference failed: ${detail}`, {cause});
          });
        const truncationProbeAnswer = truncationProbeOutput.answers.truncation_probe;
        if (!truncationProbeAnswer || truncationProbeAnswer.type !== 'choice')
          throw new Error('Kanana returned no choice for the truncation probe.');
        if (!truncationProbeOutput.inputTruncation)
          throw new Error('Kanana did not report truncation for the truncation probe.');
        const checkpointSha256 = model.description.checkpoint.weightSha256;
        if (!checkpointSha256) throw new Error('Kanana descriptor omitted the checkpoint digest.');
        return {
          adapterInfo: {
            vendor: adapter.info?.vendor ?? '',
            architecture: adapter.info?.architecture ?? '',
            device: adapter.info?.device ?? '',
            isFallback,
            isFallbackAdapter: isFallback,
            features: Array.from(adapter.features),
            limits: {
              maxBufferSize: adapter.limits.maxBufferSize,
              maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
              maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
              maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage
            }
          },
          truncationProbe: {
            modelMaxLength,
            requestedMaxLength,
            answer: truncationProbeAnswer.choice,
            probabilities: truncationProbeAnswer.probabilities,
            inputTruncation: truncationProbeOutput.inputTruncation,
            usage: truncationProbeOutput.usage
          },
          result: {
            modelId: model.modelId,
            descriptionModelId: model.description.modelId,
            checkpointRevision: model.description.checkpoint.revision,
            checkpointSha256,
            backend: model.description.backend.name,
            assetManifestSha256: model.description.assets.manifestSha256,
            choice: answer.choice,
            confidence: answer.confidence,
            probabilities: answer.probabilities,
            probabilityEntries: answer.probabilityEntries,
            answerOrder: Object.keys(output.answers),
            additionalAnswers: {
              needs_immediate_page: {
                type: noulAnswer.type,
                noul: noulAnswer.noul,
                fields: Object.keys(noulAnswer)
              },
              delay_severity: {
                type: scoreAnswer.type,
                score: scoreAnswer.score,
                confidence: scoreAnswer.confidence,
                probabilities: scoreAnswer.probabilities,
                probabilityEntries: scoreAnswer.probabilityEntries,
                legend: scoreAnswer.legend
              }
            },
            usage: output.usage
          },
          scoreOnly: {
            probabilities: scoreOnlyAnswer.probabilities,
            score: scoreOnlyAnswer.score,
            confidence: scoreOnlyAnswer.confidence
          }
        };
      } finally {
        await model.dispose();
      }
    }, request);

    const expected = reference.result.answers.delivery_status;
    if (!expected) throw new Error('Kanana Python reference is missing delivery_status.');
    expect(scoreOnly.probabilities).toEqual(result.additionalAnswers.delay_severity.probabilities);
    expect(scoreOnly.score).toBe(result.additionalAnswers.delay_severity.score);
    expect(scoreOnly.confidence).toBe(result.additionalAnswers.delay_severity.confidence);
    expect(adapterInfo.isFallback).toBe(false);
    expect(result.modelId).toBe(kananaModelId);
    expect(result.descriptionModelId).toBe(kananaModelId);
    expect(result.checkpointRevision).toBe(kananaRevision);
    expect(result.checkpointSha256).toBe(kananaCheckpointSha256);
    expect(result.backend).toBe('tvm-webgpu');
    expect(result.answerOrder).toEqual(Object.keys(request.questions));
    expect(result.choice).toBe(expected.choice);
    expect(result.probabilityEntries.map(([key]) => key)).toEqual(Object.keys(expected.probabilities));
    for (const [key, expectedProbability] of Object.entries(expected.probabilities)) {
      const actualProbability = result.probabilities[key];
      if (actualProbability === undefined) throw new Error(`Kanana omitted probability ${key}.`);
      expect(Number.isFinite(actualProbability), key).toBe(true);
      expect(Math.abs(actualProbability - expectedProbability), key).toBeLessThan(probabilityTolerance);
    }
    expect(Math.abs(result.confidence - expected.confidence)).toBeLessThan(probabilityTolerance);
    const noulAnswer = result.additionalAnswers.needs_immediate_page;
    const expectedNoul = publicTextReference.result.answers.needs_immediate_page;
    expect(noulAnswer.type).toBe('noul');
    expect(noulAnswer.fields).toEqual(['type', 'noul']);
    expect(Number.isFinite(noulAnswer.noul)).toBe(true);
    expect(noulAnswer.noul).toBeGreaterThanOrEqual(0);
    expect(noulAnswer.noul).toBeLessThanOrEqual(1);
    expect(Math.abs(noulAnswer.noul - expectedNoul.noul)).toBeLessThan(probabilityTolerance);

    const scoreAnswer = result.additionalAnswers.delay_severity;
    const scoreQuestion = publicTextReference.request.questions.delay_severity;
    const expectedScore = publicTextReference.result.answers.delay_severity;
    if (!scoreQuestion || scoreQuestion.type !== 'score') throw new Error('The GPU request has no score question.');
    const scoreCriteria = scoreQuestion.criteria;
    const scoreEntries = scoreAnswer.probabilityEntries;
    expect(scoreAnswer.type).toBe('score');
    const expectedScoreKeys = Object.keys(expectedScore.probabilities);
    expect(scoreEntries.map(([key]) => key)).toEqual(expectedScoreKeys);
    expect(Object.keys(scoreAnswer.probabilities)).toEqual(expectedScoreKeys);
    expect(Object.keys(scoreAnswer.legend)).toEqual(Object.keys(expectedScore.legend));
    expect(scoreAnswer.legend).toEqual(expectedScore.legend);
    expect(Object.values(scoreAnswer.legend)).toEqual(scoreCriteria);
    expect(
      scoreEntries.every(([, probability]) => Number.isFinite(probability) && probability >= 0 && probability <= 1)
    ).toBe(true);
    expect(Math.abs(scoreEntries.reduce((sum, [, probability]) => sum + probability, 0) - 1)).toBeLessThan(1e-6);
    for (const [index, key] of expectedScoreKeys.entries()) {
      const actualProbability = scoreAnswer.probabilities[key];
      if (actualProbability === undefined) throw new Error(`Kanana omitted score probability ${key}.`);
      expect(Math.abs(actualProbability - expectedScore.probabilities[key]!), key).toBeLessThan(
        scoreReferenceTolerance.probabilityTolerance
      );
      expect(Math.abs(scoreEntries[index]![1] - expectedScore.probabilities[key]!), key).toBeLessThan(
        scoreReferenceTolerance.probabilityTolerance
      );
    }
    expect(Math.abs(scoreAnswer.score - expectedScore.score)).toBeLessThan(scoreReferenceTolerance.scoreTolerance);
    expect(Math.abs(scoreAnswer.confidence - expectedScore.confidence)).toBeLessThan(
      scoreReferenceTolerance.confidenceTolerance
    );
    const computedScore = scoreEntries.reduce((sum, [, probability], index) => sum + index * probability, 0);
    expect(Math.abs(scoreAnswer.score - computedScore)).toBeLessThan(1e-9);
    const highestProbability = Math.max(...scoreEntries.map(([, probability]) => probability));
    const remainingProbabilityMean =
      (scoreEntries.reduce((sum, [, probability]) => sum + probability, 0) - highestProbability) /
      (scoreEntries.length - 1);
    const expectedScoreConfidence = highestProbability - remainingProbabilityMean;
    expect(Math.abs(scoreAnswer.confidence - expectedScoreConfidence)).toBeLessThan(1e-9);
    expect(truncationProbe.modelMaxLength).toBe(20_480);
    expect(truncationProbe.requestedMaxLength).toBe(128);
    expect(truncationProbe.inputTruncation).toEqual({
      truncated: true,
      maxLength: 128,
      queryQuestionIds: ['truncation_probe'],
      candidateDocuments: []
    });
    expect(truncationProbe.usage.input_tokens).toBeGreaterThan(truncationProbe.requestedMaxLength);
    expect(Object.keys(truncationProbe.probabilities)).toEqual(['first', 'second']);
    expect(Object.values(truncationProbe.probabilities).every(Number.isFinite)).toBe(true);
    expect(visionWasmRequests).toEqual([]);
    expect(fetched.some(({name}) => name === 'model.safetensors')).toBe(true);
    expect(browserErrors).toEqual([]);
    console.log(
      `KANANA_TRUNCATION_PROBE_REAL_GPU ${JSON.stringify({browserName, version: browser.version(), truncationProbe})}`
    );
    console.log(
      `KANANA_CURATED_TEXT_REAL_GPU ${JSON.stringify({browserName, version: browser.version(), adapterInfo, result, probabilityTolerance, scoreReferenceTolerance, fetchedRanges: fetched.length, fetchedBytes: fetched.reduce((sum, item) => sum + item.bytes, 0)})}`
    );
    const userAgent = await page.evaluate(() => navigator.userAgent);
    const evidence = await createKananaReleaseEvidence({
      kind: 'text',
      testInfo,
      browserName,
      browserVersion: browser.version(),
      userAgent,
      model: {
        requestedId: kananaModelId,
        modelId: result.modelId,
        descriptionModelId: result.descriptionModelId,
        checkpointRevision: result.checkpointRevision,
        checkpointSha256: result.checkpointSha256,
        assetManifestSha256: result.assetManifestSha256,
        backend: result.backend
      },
      adapter: adapterInfo,
      observation: {
        result,
        truncationProbe,
        probabilityTolerance,
        scoreReferenceTolerance,
        referenceFixtureSha256: fixtureSha256,
        publicTextFixtureSha256
      }
    });
    if (evidence) console.log(`${KANANA_RELEASE_EVIDENCE_PREFIX}${JSON.stringify(evidence)}`);
  } finally {
    await modelFile.close();
  }
});
