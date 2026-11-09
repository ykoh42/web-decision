import {createHash} from 'node:crypto';
import {readFile, writeFile} from 'node:fs/promises';
import os from 'node:os';
import {resolve} from 'node:path';
import {expect, test, type Page} from '@playwright/test';
import type {DecisionResult, LoadedDecisionModel} from '../../packages/web-decision/src/index.d.ts';

const enabled = process.env.WEB_DECISION_STRANDS_REAL_GPU === '1';
const modelId = 'experimental-strands-qwen-2b-fp16';
const pins = {
  manifest: '74eb7f501ab64d5906a498dac20e161044616cfb67607ddd5c24030887132209',
  wasm: '020720bae0d99b592b3985130daac08a2c41adadad76143ab649f4afe109bc6e',
  tokenizer: 'a2cdd2e108566b09079afa8d266e9e65e7c280f27b771218237a06de5ba9cd86',
  tokenizerConfig: '8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45'
};
const assetRoot = resolve(process.env.WEB_DECISION_PUBLIC_ASSET_ROOT ?? 'models/strands-integration-20261030');
const smallFiles = [
  ['fused-ordered-manifest-4096-candidate-v1.json', pins.manifest],
  ['full-fused-tvm-4096-candidate-v1/model.wasm', pins.wasm],
  ['checkpoint/tokenizer.json', pins.tokenizer],
  ['checkpoint/tokenizer_config.json', pins.tokenizerConfig]
] as const;
const references = [
  {
    state: 'A red parcel goes to the first desk.',
    instructions: 'Which color matches?',
    options: ['red', 'blue'],
    selected: 'red',
    tokenCount: 59
  },
  {
    state: 'A red parcel goes to the first desk.',
    instructions: 'Which color matches?',
    options: ['blue', 'red'],
    selected: 'red',
    tokenCount: 59
  },
  {
    state: 'A billing ticket must go to finance; delivery issues go to shipping. This ticket concerns billing.',
    instructions: 'Which team should handle it?',
    options: ['finance', 'shipping', 'technical support'],
    selected: 'finance',
    tokenCount: 76
  },
  {
    state: '고객이 결제 취소를 요청했습니다.',
    instructions: '문의 유형은 무엇인가요?',
    options: ['결제 취소', '배송 조회', '계정 잠금'],
    selected: '결제 취소',
    tokenCount: 68
  },
  {
    state: 'Please cancel the order before it ships.',
    instructions: 'What is the request?',
    options: ['cancel', 'return', 'delivery status'],
    selected: 'cancel',
    tokenCount: 64
  }
];

function sha256(bytes: Uint8Array) {
  return createHash('sha256').update(bytes).digest('hex');
}

async function verifyAssets() {
  const assets: Record<string, {bytes: number; sha256: string}> = {};
  for (const [path, expected] of smallFiles) {
    const bytes = await readFile(resolve(assetRoot, path));
    expect(sha256(bytes), path).toBe(expected);
    assets[path] = {bytes: bytes.byteLength, sha256: expected};
  }
  const manifest = JSON.parse(await readFile(resolve(assetRoot, smallFiles[0][0]), 'utf8')) as {
    records: Array<{chunks: unknown[]}>;
    totalBytes: number;
  };
  expect(manifest.records).toHaveLength(328);
  expect(manifest.totalBytes).toBe(3_776_255_872);
  return {
    files: assets,
    weights: {
      records: manifest.records.length,
      chunks: manifest.records.reduce((count, record) => count + record.chunks.length, 0),
      bytes: manifest.totalBytes
    }
  };
}

async function readChromeGpuReport(page: Page) {
  await page.goto('chrome://gpu');
  const report = await page.evaluate(() => {
    const parts: string[] = [];
    function visit(root: Document | ShadowRoot) {
      if (root?.textContent) parts.push(root.textContent);
      for (const element of root?.querySelectorAll?.('*') ?? []) if (element.shadowRoot) visit(element.shadowRoot);
    }
    visit(document);
    return parts.join('\n');
  });
  return {
    webGpuStatus: report.match(/WebGPU:\s*(Hardware accelerated|Software only|Disabled|Blocklisted)/i)?.[1] ?? null,
    device: report.match(/ANGLE Metal Renderer:\s*(Apple M4 Pro)\b/i)?.[1] ?? null
  };
}

function assertDecision(result: DecisionResult, expected: (typeof references)[number]) {
  expect(result.answers).toHaveLength(1);
  const answer = result.answers[0]!;
  expect(result.tokenCount).toBe(expected.tokenCount);
  expect(answer.probabilities.map(({option}) => option)).toEqual(expected.options);
  expect(answer.selected).toBe(expected.selected);
  expect(answer.selectedIndex).toBe(expected.options.indexOf(expected.selected));
  expect(
    answer.probabilities.every(({logit, probability}) => Number.isFinite(logit) && Number.isFinite(probability))
  ).toBe(true);
  expect(answer.probabilities.reduce((sum, {probability}) => sum + probability, 0)).toBeCloseTo(1, 12);
}

test('Strands Qwen 2B runs pinned local assets on M4 Pro WebGPU', async ({page, browser, browserName}) => {
  test.skip(!enabled, 'Set WEB_DECISION_STRANDS_REAL_GPU=1 to run the pinned local Strands model on WebGPU.');
  test.skip(process.platform !== 'darwin', 'This evidence test currently targets the M4 Pro macOS host.');
  test.skip(browserName !== 'chromium', 'The hardware evidence path uses Chromium chrome://gpu.');
  if (!process.env.WEB_DECISION_INSTALLED_BUNDLE) {
    throw new Error(
      'The legacy Strands browser test requires WEB_DECISION_INSTALLED_BUNDLE to point to a compatible legacy bundle.'
    );
  }
  test.setTimeout(20 * 60 * 1000);

  const assets = await verifyAssets();
  expect(assets.weights.chunks).toBe(635);
  const bundleSha256 = sha256(await readFile(resolve(process.env.WEB_DECISION_INSTALLED_BUNDLE)));
  await page.goto('/');
  await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');

  const inference = await page.evaluate(
    async ({id, cases}) => {
      if (!navigator.gpu) throw new Error('WebGPU is unavailable.');
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('WebGPU did not provide an adapter.');
      if (
        (adapter as GPUAdapter & {isFallbackAdapter?: boolean}).isFallbackAdapter ||
        (adapter.info as GPUAdapterInfo & {isFallbackAdapter?: boolean})?.isFallbackAdapter
      )
        throw new Error('A hardware WebGPU adapter is required.');
      if (!adapter.features.has('shader-f16')) throw new Error('The adapter does not support shader-f16.');

      let progress!: {loadedBytes: number; totalBytes: number; fraction: number};
      const loadStarted = performance.now();
      const legacy = window.webDecision as unknown as typeof import('../../packages/web-decision/src/index.d.ts');
      const model = (await legacy.loadDecisionModel({
        model: id as 'experimental-strands-qwen-2b-fp16',
        assetsUrl: '/',
        cache: false,
        onProgress: (value: unknown) => {
          progress = value as typeof progress;
        }
      })) as unknown as LoadedDecisionModel & {
        tokenizer: {encode(text: string, options: {add_special_tokens: boolean}): {ids: number[]}};
      };
      const loadElapsedMs = performance.now() - loadStarted;

      try {
        const timed = async <T>(run: () => Promise<T>) => {
          const start = performance.now();
          const result = await run();
          return {elapsedMs: performance.now() - start, result};
        };
        const outputs = [];
        for (const item of cases) {
          const measured = await timed(() =>
            model.decide({state: item.state, questions: [{instructions: item.instructions, options: item.options}]})
          );
          outputs.push({...item, ...measured});
        }

        const multi = await model.decide({
          state: cases[0]!.state,
          questions: [
            {instructions: cases[0]!.instructions, options: cases[0]!.options},
            {instructions: cases[1]!.instructions, options: cases[1]!.options}
          ]
        });
        const options = Array.from({length: 255}, (_, index) => `option-${index}`);
        const maxOptions = await timed(() =>
          model.decide({state: 'Choose the matching id.', questions: [{instructions: 'Select one.', options}]})
        );
        let overOptions;
        try {
          await model.decide({
            state: 'Choose the matching id.',
            questions: [{instructions: 'Select one.', options: [...options, 'option-255']}]
          });
          overOptions = {code: 'UNEXPECTED_SUCCESS'};
        } catch (error) {
          const failure = error as {code?: string; name: string; message: string};
          overOptions = {code: failure.code ?? failure.name, message: failure.message};
        }

        const maxTokens = 4096;
        const stateText = '<state>\n\n</state>\n';
        const questionText = (instructions: string) =>
          `<question type="choice">\nSelect exactly one option.\n${instructions}\n<options>\n1. no\n2. yes\n</options>\n</question>\n<answer>`;
        const tokenCount = (instructions: string) =>
          model.tokenizer.encode(stateText, {add_special_tokens: true}).ids.length +
          model.tokenizer.encode(questionText(instructions), {add_special_tokens: false}).ids.length;
        const baseInstructions = 'Choose exactly one option.';
        const baseTokens = tokenCount(baseInstructions);
        const maxInstructions = `${baseInstructions}${' x'.repeat(maxTokens - baseTokens)}`;
        if (tokenCount(maxInstructions) !== maxTokens)
          throw new Error(`Could not build an exact ${maxTokens}-token input.`);
        const overInstructions = `${maxInstructions} x`;
        if (tokenCount(overInstructions) !== maxTokens + 1)
          throw new Error(`Could not build an exact ${maxTokens + 1}-token input.`);
        const maxContext = await timed(() =>
          model.decide({state: '', questions: [{instructions: maxInstructions, options: ['no', 'yes']}]})
        );
        let overContext;
        try {
          await model.decide({
            state: '',
            questions: [{instructions: overInstructions, options: ['no', 'yes']}]
          });
          overContext = {code: 'UNEXPECTED_SUCCESS'};
        } catch (error) {
          const failure = error as {code?: string; name: string; message: string};
          overContext = {code: failure.code ?? failure.name, message: failure.message};
        }

        return {
          model: model.description.model,
          sameRegistration:
            model.description === legacy.describeModel(id as Parameters<typeof legacy.describeModel>[0]),
          descriptionFrozen: Object.isFrozen(model.description),
          backend: model.description.backend?.name,
          verification: model.description.verification,
          adapterInfo: {
            vendor: adapter.info?.vendor,
            architecture: adapter.info?.architecture,
            isFallbackAdapter:
              (adapter as GPUAdapter & {isFallbackAdapter?: boolean}).isFallbackAdapter ??
              (adapter.info as GPUAdapterInfo & {isFallbackAdapter?: boolean})?.isFallbackAdapter ??
              false
          },
          progress,
          loadElapsedMs,
          outputs,
          multi,
          maxOptions: {elapsedMs: maxOptions.elapsedMs, result: maxOptions.result, count: options.length},
          overOptions,
          maxContext: {elapsedMs: maxContext.elapsedMs, result: maxContext.result, expectedTokens: maxTokens},
          overContext
        };
      } finally {
        await model.dispose();
      }
    },
    {id: modelId, cases: references}
  );

  expect(inference.model).toBe(modelId);
  expect(inference.sameRegistration).toBe(true);
  expect(inference.descriptionFrozen).toBe(true);
  expect(inference.backend).toBe('tvm-webgpu');
  expect(inference.verification.currentBundleGpu).toBe('verified');
  expect(inference.adapterInfo.vendor).toBe('apple');
  expect(inference.adapterInfo.isFallbackAdapter).toBe(false);
  expect(inference.progress.loadedBytes).toBe(inference.progress.totalBytes);
  expect(inference.progress.fraction).toBe(1);
  for (let index = 0; index < references.length; index++)
    assertDecision(inference.outputs[index]!.result, references[index]!);
  expect(inference.multi.tokenCount).toBe(references[0]!.tokenCount + references[1]!.tokenCount);
  expect(inference.multi.answers.map(({selected}) => selected)).toEqual(['red', 'red']);
  expect(inference.multi.answers.map(({probabilities}) => probabilities.map(({option}) => option))).toEqual([
    ['red', 'blue'],
    ['blue', 'red']
  ]);
  expect(inference.maxOptions.count).toBe(255);
  expect(inference.maxOptions.result.answers[0]!.probabilities.map(({option}) => option)).toEqual(
    Array.from({length: 255}, (_, index) => `option-${index}`)
  );
  expect(
    inference.maxOptions.result.answers[0]!.probabilities.reduce((sum, item) => sum + item.probability, 0)
  ).toBeCloseTo(1, 12);
  const maxOptionAnswer = inference.maxOptions.result.answers[0]!;
  expect(maxOptionAnswer.selected).toBe(maxOptionAnswer.probabilities[maxOptionAnswer.selectedIndex]!.option);
  expect(inference.overOptions.code).toBe('INVALID_INPUT');
  expect(inference.maxContext.result.tokenCount).toBe(4096);
  expect(inference.maxContext.result.answers[0]!.probabilities.map(({option}) => option)).toEqual(['no', 'yes']);
  expect(inference.overContext.code).toBe('INVALID_INPUT');

  const gpuPage = await page.context().newPage();
  const chromeGpu = await readChromeGpuReport(gpuPage);
  await gpuPage.close();
  expect(chromeGpu.webGpuStatus).toMatch(/hardware accelerated/i);
  expect(chromeGpu.device).toMatch(/Apple M4 Pro/i);

  const report = {
    state: 'completed',
    date: new Date().toISOString(),
    host: {
      platform: process.platform,
      arch: process.arch,
      totalMemoryBytes: os.totalmem(),
      freeMemoryAfterRunBytes: os.freemem()
    },
    browserVersion: browser.version(),
    browserName,
    chromeGpu,
    package: {name: 'web-decision', version: '0.0.0-preview.34', bundleSha256},
    model: {
      id: modelId,
      repository: 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610',
      revision: '216d1ccb46a9e9b036eb9187b248db98480f77d4',
      license: 'Apache-2.0 (publisher-declared; training-data source terms vary)',
      assets: assets.files,
      weightManifest: assets.weights
    },
    modelLoadElapsedMs: inference.loadElapsedMs,
    adapterInfo: inference.adapterInfo,
    progress: inference.progress,
    syntheticReferenceCases: inference.outputs.map(({state, instructions, options, result, elapsedMs}) => ({
      state,
      instructions,
      options,
      tokenCount: result.tokenCount,
      selected: result.answers[0]!.selected,
      selectedIndex: result.answers[0]!.selectedIndex,
      probabilities: result.answers[0]!.probabilities,
      elapsedMs
    })),
    multipleQuestionRows: inference.multi,
    maximumOptions: {
      count: inference.maxOptions.count,
      elapsedMs: inference.maxOptions.elapsedMs,
      selected: maxOptionAnswer.selected,
      selectedIndex: maxOptionAnswer.selectedIndex,
      outputCount: maxOptionAnswer.probabilities.length,
      firstOption: maxOptionAnswer.probabilities[0]!.option,
      lastOption: maxOptionAnswer.probabilities.at(-1)!.option,
      probabilitySum: maxOptionAnswer.probabilities.reduce((sum, item) => sum + item.probability, 0)
    },
    overMaximumOptions: inference.overOptions,
    maximumContext: inference.maxContext,
    overContext: inference.overContext
  };
  await writeFile(
    resolve(process.env.WEB_DECISION_STRANDS_REPORT ?? 'reports/strands-qwen2-m4pro-verified-real-webgpu.json'),
    `${JSON.stringify(report, null, 2)}\n`
  );
});
