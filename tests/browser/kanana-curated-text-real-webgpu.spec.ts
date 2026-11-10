import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {open, readFile, stat} from 'node:fs/promises';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';
import {
  kananaCheckpointSha256,
  kananaModelId,
  kananaRevision,
  kananaSnapshotDirectory,
  kananaUpstreamBase,
  verifyPinnedKananaSnapshot
} from './kanana-snapshot.ts';
import {
  collectInstalledKananaReleaseInputs,
  createKananaReleaseEvidenceEnvelope,
  KANANA_RELEASE_EVIDENCE_PREFIX,
  localKananaCandidateMode
} from './kanana-release-evidence.ts';

const enabled = process.env.WEB_DECISION_KANANA_CURATED_REAL_GPU === '1' || localKananaCandidateMode;
const upstreamPath = new URL(kananaUpstreamBase).pathname;
const probabilityTolerance = 0.001;
const fixtureBytes = readFileSync(new URL('../fixtures/kanana-decision-python-reference.json', import.meta.url));
const fixtureSha256 = createHash('sha256').update(fixtureBytes).digest('hex');

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

type DecisionQuestion = ChoiceQuestion | NoulQuestion | ScoreQuestion;

const reference = JSON.parse(fixtureBytes.toString('utf8')) as ReferenceFixture;
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

test('the public Kanana package loader matches the pinned text-only PyTorch choice reference on WebGPU', async ({
  page,
  browserName,
  browser
}) => {
  test.skip(!enabled, 'Set WEB_DECISION_KANANA_CURATED_REAL_GPU=1 to run the full package consumer GPU check.');
  test.setTimeout(45 * 60 * 1000);

  const releaseInputs = await collectInstalledKananaReleaseInputs();
  await verifyPinnedKananaSnapshot();
  expect(fixtureSha256).toBe('822156454d468153cef23fde6fad8c88b59821d9da1b86897733fa906abfd420');
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

  const modelPath = resolve(kananaSnapshotDirectory, 'model.safetensors');
  const modelFile = await open(modelPath, 'r');
  const modelSize = (await stat(modelPath)).size;
  const fetched: {name: string; bytes: number; range?: string}[] = [];

  try {
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
        needs_immediate_page: {
          type: 'noul',
          instructions: 'Did the parcel arrive late?',
          criteria: {
            false: 'The parcel arrived on or before the promised date.',
            true: 'The parcel arrived after the promised date.'
          }
        },
        delay_severity: {
          type: 'score',
          instructions: 'How severe was the delivery delay?',
          criteria: ['No delay', 'Small delay', 'Significant delay']
        }
      },
      temperature: reference.request.temperature
    };
    const {adapterInfo, isFallbackAdapter, adapterFeatures, adapterLimits, userAgent, result, maximumContext} =
      await page.evaluate(async (input: BrowserRequest) => {
        const adapter = await navigator.gpu?.requestAdapter({powerPreference: 'high-performance'});
        if (!adapter) throw new Error('Chromium did not provide a WebGPU adapter.');
        const adapterWithFallback = adapter as GPUAdapter & {readonly isFallbackAdapter?: boolean};
        const isFallback = adapterWithFallback.isFallbackAdapter ?? adapter.info?.isFallbackAdapter ?? null;
        if (isFallback !== false) throw new Error(`Expected a hardware WebGPU adapter; got ${String(isFallback)}.`);

        const model = await window.webDecision.loadDecisionModel({modelId: 'kanana-2-v-decide-3b-2611', cache: false});
        try {
          const output = await model.systemOne(input);
          const answer = output.answers.delivery_status;
          if (!answer || answer.type !== 'choice') throw new Error('Kanana returned no choice result.');
          const noulAnswer = output.answers.needs_immediate_page;
          if (!noulAnswer || noulAnswer.type !== 'noul') throw new Error('Kanana returned no noul result.');
          const scoreAnswer = output.answers.delay_severity;
          if (!scoreAnswer || scoreAnswer.type !== 'score') throw new Error('Kanana returned no score result.');
          const maximumContextOutput = await model.systemOne(
            {
              state: 'x '.repeat(30_000),
              stateFormat: 'text',
              questions: {
                maximum_context: {
                  type: 'choice',
                  instructions: 'Choose the label for this boundary test.',
                  criteria: {first: 'First option', second: 'Second option'}
                }
              }
            },
            {maxLength: 20_480}
          );
          const maximumContextAnswer = maximumContextOutput.answers.maximum_context;
          if (!maximumContextAnswer || maximumContextAnswer.type !== 'choice')
            throw new Error('Kanana returned no choice for the maximum context input.');
          if (!maximumContextOutput.inputTruncation)
            throw new Error('Kanana did not report truncation for the maximum context input.');
          const checkpointSha256 = model.description.checkpoint.weightSha256;
          if (!checkpointSha256) throw new Error('Kanana descriptor omitted the checkpoint digest.');
          return {
            adapterInfo: {
              vendor: adapter.info?.vendor ?? '',
              architecture: adapter.info?.architecture ?? '',
              isFallback,
              isFallbackAdapter: isFallback
            },
            isFallbackAdapter: isFallback,
            adapterFeatures: [...adapter.features],
            adapterLimits: {
              maxBufferSize: adapter.limits.maxBufferSize,
              maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
              maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
              maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage
            },
            userAgent: navigator.userAgent,
            maximumContext: {
              answer: maximumContextAnswer.choice,
              probabilities: maximumContextAnswer.probabilities,
              inputTruncation: maximumContextOutput.inputTruncation,
              usage: maximumContextOutput.usage
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
            }
          };
        } finally {
          await model.dispose();
        }
      }, request);

    const expected = reference.result.answers.delivery_status;
    if (!expected) throw new Error('Kanana Python reference is missing delivery_status.');
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
    expect(noulAnswer.type).toBe('noul');
    expect(noulAnswer.fields).toEqual(['type', 'noul']);
    expect(Number.isFinite(noulAnswer.noul)).toBe(true);
    expect(noulAnswer.noul).toBeGreaterThanOrEqual(0);
    expect(noulAnswer.noul).toBeLessThanOrEqual(1);

    const scoreAnswer = result.additionalAnswers.delay_severity;
    const scoreQuestion = request.questions.delay_severity;
    if (!scoreQuestion || scoreQuestion.type !== 'score') throw new Error('The GPU request has no score question.');
    const scoreCriteria = scoreQuestion.criteria;
    const scoreEntries = scoreAnswer.probabilityEntries;
    expect(scoreAnswer.type).toBe('score');
    expect(scoreEntries.map(([key]) => key)).toEqual(['0', '1', '2']);
    expect(Object.keys(scoreAnswer.probabilities)).toEqual(['0', '1', '2']);
    expect(Object.keys(scoreAnswer.legend)).toEqual(['0', '1', '2']);
    expect(Object.values(scoreAnswer.legend)).toEqual(scoreCriteria);
    expect(
      scoreEntries.every(([, probability]) => Number.isFinite(probability) && probability >= 0 && probability <= 1)
    ).toBe(true);
    expect(Math.abs(scoreEntries.reduce((sum, [, probability]) => sum + probability, 0) - 1)).toBeLessThan(1e-6);
    const expectedScore = scoreEntries.reduce((sum, [, probability], index) => sum + index * probability, 0);
    expect(Math.abs(scoreAnswer.score - expectedScore)).toBeLessThan(1e-9);
    const highestProbability = Math.max(...scoreEntries.map(([, probability]) => probability));
    const remainingProbabilityMean =
      (scoreEntries.reduce((sum, [, probability]) => sum + probability, 0) - highestProbability) /
      (scoreEntries.length - 1);
    const expectedScoreConfidence = highestProbability - remainingProbabilityMean;
    expect(Math.abs(scoreAnswer.confidence - expectedScoreConfidence)).toBeLessThan(1e-9);
    expect(maximumContext.inputTruncation).toEqual({
      truncated: true,
      maxLength: 20_480,
      queryQuestionIds: ['maximum_context'],
      candidateDocuments: []
    });
    expect(maximumContext.usage.input_tokens).toBeGreaterThan(20_480);
    expect(Object.keys(maximumContext.probabilities)).toEqual(['first', 'second']);
    expect(Object.values(maximumContext.probabilities).every(Number.isFinite)).toBe(true);
    expect(fetched.some(({name}) => name === 'model.safetensors')).toBe(true);
    expect(browserErrors).toEqual([]);
    console.log(
      `KANANA_MAX_CONTEXT_REAL_GPU ${JSON.stringify({browserName, version: browser.version(), maximumContext})}`
    );
    console.log(
      `KANANA_CURATED_TEXT_REAL_GPU ${JSON.stringify({browserName, version: browser.version(), adapterInfo, result, probabilityTolerance, fetchedRanges: fetched.length, fetchedBytes: fetched.reduce((sum, item) => sum + item.bytes, 0)})}`
    );
    if (localKananaCandidateMode && process.env.WEB_DECISION_KANANA_EMIT_EVIDENCE !== '1') return;
    const evidence = createKananaReleaseEvidenceEnvelope({
      inputs: releaseInputs,
      kind: 'text-choice',
      runtimeArtifactNames: releaseInputs.policy.tests.text.runtimeArtifacts,
      browserName,
      browserVersion: browser.version(),
      userAgent,
      model: {
        loadedId: result.modelId,
        describedId: result.descriptionModelId,
        checkpointRevision: result.checkpointRevision,
        checkpointSha256: result.checkpointSha256
      },
      gpu: {
        isFallbackAdapter,
        adapterInfo,
        features: adapterFeatures,
        limits: adapterLimits
      }
    });
    console.log(
      `${KANANA_RELEASE_EVIDENCE_PREFIX}${JSON.stringify({
        ...evidence,
        referenceFixtureSha256: fixtureSha256,
        probabilityTolerance,
        inference: result
      })}`
    );
  } finally {
    await modelFile.close();
  }
});
