import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {mkdir, readFile, stat, writeFile} from 'node:fs/promises';
import {dirname, resolve} from 'node:path';
import os from 'node:os';
import {expect, test} from '@playwright/test';

const enabled = process.env.WEB_DECISION_KEV06_REAL_GPU === '1';
const manifestSha256 = '8577804e638051cb056af51fee89eb18ea5de85ccec4f196594d830c16fdc767';
const assetsRoot = resolve(process.env.WEB_DECISION_PUBLIC_ASSET_ROOT ?? 'models/web-decision-assets');
const modelAssets = resolve(assetsRoot, 'kev-0.6b-q4f16');

function sortKeys(value) {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, sortKeys(value[key])])
  );
}

function sha256(path) {
  return new Promise((resolveHash, reject) => {
    const hash = createHash('sha256');
    const stream = createReadStream(path);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolveHash(hash.digest('hex')));
  });
}

async function verifyAssets() {
  const manifest = JSON.parse(await readFile(resolve(modelAssets, 'manifest.json'), 'utf8'));
  const actualManifestHash = createHash('sha256')
    .update(`${JSON.stringify(sortKeys(manifest))}\n`)
    .digest('hex');
  expect(actualManifestHash).toBe(manifestSha256);
  for (const [name, expected] of Object.entries(manifest.files)) {
    const path = resolve(modelAssets, name);
    expect((await stat(path)).size, name).toBe(expected.bytes);
    expect(await sha256(path), name).toBe(expected.sha256);
  }
  return manifest;
}

test('Kev 0.6B Q4F16 runs exact-package WebGPU boundary cases', async ({page, browserName, browser}) => {
  test.skip(!enabled, 'Set WEB_DECISION_KEV06_REAL_GPU=1 to run the pinned local Kev model on WebGPU.');
  test.setTimeout(10 * 60 * 1000);

  const manifest = await verifyAssets();
  const bundleSha256 = await sha256(resolve('packages/web-decision/dist/index.js'));
  await page.goto('/');
  await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');

  const inference = await page.evaluate(async () => {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser.');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU did not provide an adapter.');
    if (!adapter.features.has('shader-f16')) throw new Error('The adapter does not support shader-f16.');

    const adapterInfo = adapter.info ?? {};
    const limits = Object.fromEntries(
      ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupStorageSize'].map((name) => [
        name,
        adapter.limits[name]
      ])
    );
    const model = await window.webDecision.loadDecisionModel({
      modelId: 'kev-0.6b-q4f16',
      assetsUrl: '/kev-0.6b-q4f16/'
    });

    try {
      const timed = async (run) => {
        const start = performance.now();
        const result = await run();
        return {elapsedMs: performance.now() - start, result};
      };
      const tokenizer = model.tokenizer;
      const tokenLength = (text) => tokenizer.encode(text, {add_special_tokens: false}).ids.length;
      const exactTokenText = (target) => {
        const pieces = [' the', ' a', ' to', ' of', ' is', ' x', ' q', ' 0', ' yes', ' no', '. ', '! ', '\n'];
        for (const piece of pieces) {
          const candidate = piece.repeat(target);
          if (tokenLength(candidate) === target) return candidate;
        }
        let text = '';
        let count = tokenLength(text);
        while (count < target) {
          let nextText;
          for (const piece of pieces) {
            const candidate = text + piece;
            if (tokenLength(candidate) === count + 1) {
              nextText = candidate;
              break;
            }
          }
          if (nextText === undefined) throw new Error(`Could not construct exactly ${target} tokenizer tokens.`);
          text = nextText;
          count++;
        }
        return text;
      };
      const minimum = await timed(() =>
        model.decide({state: '', questions: [{instructions: 'Choose.', options: ['only']}]})
      );
      const base = await timed(() =>
        model.decide({state: '', questions: [{instructions: 'Choose.', options: ['no', 'yes']}]})
      );
      const fixedTokens = base.result.tokenCount - tokenLength('Choose.');
      const attentionBytes = Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize);
      const attentionTokenLimit = Math.floor(Math.sqrt(attentionBytes / 32));
      const sequenceTokenLimit = Math.min(8192, attentionTokenLimit);
      const lengths = [];
      const targetLengths = [...new Set([32, 69, 256, 1024, 2048, 2049, sequenceTokenLimit])]
        .filter((length) => length <= sequenceTokenLimit)
        .sort((left, right) => left - right);
      for (const targetTokens of targetLengths) {
        const instructions = exactTokenText(targetTokens - fixedTokens);
        const measured = await timed(() =>
          model.decide({state: '', questions: [{instructions, options: ['no', 'yes']}]})
        );
        lengths.push({targetTokens, ...measured});
      }

      const overBoundaryInstructions = exactTokenText(sequenceTokenLimit + 1 - fixedTokens);
      let overBoundary;
      try {
        await model.decide({
          state: '',
          questions: [{instructions: overBoundaryInstructions, options: ['no', 'yes']}]
        });
        overBoundary = {code: 'UNEXPECTED_SUCCESS'};
      } catch (error) {
        overBoundary = {code: error.code ?? error.name, message: error.message};
      }

      const choices = Object.fromEntries(Array.from({length: 255}, (_, index) => [`option-${index}`, null]));
      const typed = await timed(() =>
        model.systemOne('A customer asks for a refund.', {
          route: {type: 'choice', instructions: 'Choose the appropriate action.', criteria: choices},
          refund: {type: 'noul', instructions: 'Is the customer asking for a refund?'},
          priority: {type: 'score', instructions: 'Rate the urgency.', criteria: ['low', 'medium', 'high']}
        })
      );

      return {
        modelId: 'kev-0.6b-q4f16',
        model: model.description.model,
        backend: model.description.backend.name,
        currentBundleGpu: model.description.verification.currentBundleGpu,
        adapterInfo,
        adapterIsFallback: adapter.isFallbackAdapter ?? null,
        adapterFeatures: [...adapter.features],
        limits,
        attentionTokenLimit,
        sequenceTokenLimit,
        minimum,
        base,
        lengths,
        overBoundary,
        typed
      };
    } finally {
      await model.dispose();
    }
  });

  expect(inference.model).toBe('kev-0.6b-q4f16');
  expect(inference.backend).toBe('tvm-webgpu');
  expect(inference.currentBundleGpu).toBe('verified');
  expect(inference.minimum.result.answers[0].selected).toBe('only');
  expect(inference.minimum.result.answers[0].probabilities.map(({option}) => option)).toEqual(['only']);
  expect(inference.base.result.answers[0].probabilities.map(({option}) => option)).toEqual(['no', 'yes']);
  for (const sample of inference.lengths) expect(sample.result.tokenCount).toBe(sample.targetTokens);
  expect(inference.overBoundary.code).toBe(
    inference.sequenceTokenLimit < 8192 ? 'DEVICE_RESOURCE_LIMIT' : 'INVALID_INPUT'
  );
  expect(Object.keys(inference.typed.result.answers.route.probabilities)).toEqual(
    Array.from({length: 255}, (_, index) => `option-${index}`)
  );
  expect(inference.typed.result.answers.refund.noul).toBeGreaterThanOrEqual(0);
  expect(inference.typed.result.answers.refund.noul).toBeLessThanOrEqual(1);
  expect(inference.typed.result.answers.priority.score).toBeGreaterThanOrEqual(0);
  expect(inference.typed.result.answers.priority.score).toBeLessThanOrEqual(2);

  let webGpuFeatureStatus = null;
  let gpuDeviceDescription = null;
  if (browserName === 'chromium') {
    const gpuPage = await page.context().newPage();
    await gpuPage.goto('chrome://gpu');
    const gpuReport = await gpuPage.evaluate(() => {
      const texts = [];
      function visit(root) {
        if (root?.textContent) texts.push(root.textContent);
        for (const element of root?.querySelectorAll?.('*') ?? []) if (element.shadowRoot) visit(element.shadowRoot);
      }
      visit(document);
      return texts.join('\n');
    });
    webGpuFeatureStatus =
      gpuReport.match(/WebGPU:\s*(Hardware accelerated|Software only|Disabled|Blocklisted)/i)?.[1] ?? null;
    gpuDeviceDescription =
      gpuReport.match(/<Integrated GPU>\s*([^\n]+)/i)?.[1]?.trim() ??
      gpuReport.match(/GPU\d+\s+VENDOR=[^\n]*?\bDEVICE=[^\n]*?\[([^\]]+)\]/i)?.[1]?.trim() ??
      gpuReport.match(/GL_RENDERER\s+([^\n]+)/i)?.[1]?.trim() ??
      null;
    await gpuPage.close();
    expect(webGpuFeatureStatus).toMatch(/hardware accelerated/i);
    expect(gpuDeviceDescription).not.toBeNull();
  }

  const summarizeDecision = ({targetTokens, elapsedMs, result}) => ({
    ...(targetTokens === undefined ? {} : {targetTokens}),
    elapsedMs,
    tokenCount: result.tokenCount,
    selected: result.answers[0].selected,
    optionOrder: result.answers[0].probabilities.map(({option}) => option),
    probabilityTotal: result.answers[0].probabilities.reduce((sum, answer) => sum + answer.probability, 0)
  });
  const report = {
    date: new Date().toISOString(),
    host: {
      platform: process.platform,
      release: os.release(),
      arch: os.arch(),
      totalMemoryBytes: os.totalmem(),
      freeMemoryBytes: os.freemem()
    },
    browser: {
      name: browserName,
      version: browser.version(),
      userAgent: await page.evaluate(() => navigator.userAgent),
      webGpuFeatureStatus,
      gpuDeviceDescription
    },
    package: {version: '0.0.0-preview.34', bundleSha256},
    assets: {manifestSha256, files: Object.keys(manifest.files)},
    inference: {
      modelId: inference.modelId,
      model: inference.model,
      backend: inference.backend,
      currentBundleGpu: inference.currentBundleGpu,
      adapterInfo: inference.adapterInfo,
      adapterIsFallback: inference.adapterIsFallback,
      adapterFeatures: inference.adapterFeatures,
      limits: inference.limits,
      attentionTokenLimit: inference.attentionTokenLimit,
      sequenceTokenLimit: inference.sequenceTokenLimit,
      minimum: summarizeDecision(inference.minimum),
      base: summarizeDecision(inference.base),
      lengths: inference.lengths.map(summarizeDecision),
      overBoundary: inference.overBoundary,
      typed: {
        elapsedMs: inference.typed.elapsedMs,
        tokenCount: inference.typed.result.tokenCount,
        choice: inference.typed.result.answers.route.choice,
        choiceOptionCount: Object.keys(inference.typed.result.answers.route.probabilities).length,
        choiceProbabilityTotal: Object.values(inference.typed.result.answers.route.probabilities).reduce(
          (sum, probability) => sum + probability,
          0
        ),
        noul: inference.typed.result.answers.refund.noul,
        score: inference.typed.result.answers.priority.score
      }
    }
  };
  console.log(`Kev 0.6B real WebGPU report: ${JSON.stringify(report)}`);
  if (process.env.WEB_DECISION_KEV06_REPORT) {
    const path = resolve(process.env.WEB_DECISION_KEV06_REPORT);
    await mkdir(dirname(path), {recursive: true});
    await writeFile(path, `${JSON.stringify(report, null, 2)}\n`);
  }
});
