import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {mkdir, readFile, rename, stat, writeFile} from 'node:fs/promises';
import {dirname, resolve} from 'node:path';
import os from 'node:os';
import {expect, test} from '@playwright/test';

const enabled = process.env.WEB_DECISION_KEV06_REAL_GPU === '1';
const manifestSha256 = '8577804e638051cb056af51fee89eb18ea5de85ccec4f196594d830c16fdc767';
const assetsRoot = resolve(process.env.WEB_DECISION_PUBLIC_ASSET_ROOT ?? 'models/web-decision-assets');
const modelAssets = resolve(assetsRoot, 'kev-0.6b-q4f16');
const servedBundle = resolve(process.env.WEB_DECISION_INSTALLED_BUNDLE ?? 'packages/web-decision/dist/index.js');
const progressPrefix = 'KEV06_PROGRESS ';

async function trackAsyncStage(reportProgress, stage, run) {
  const startedAt = Date.now();
  await reportProgress({stage, state: 'started'});
  try {
    const result = await run();
    await reportProgress({stage, state: 'completed', elapsedMs: Date.now() - startedAt});
    return result;
  } catch (error) {
    await reportProgress({
      stage,
      state: 'failed',
      elapsedMs: Date.now() - startedAt,
      error: error instanceof Error ? error.message : String(error)
    });
    throw error;
  }
}

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

async function verifyAssets(reportProgress, onManifest) {
  const manifestText = await trackAsyncStage(reportProgress, 'assets.manifest.read', () =>
    readFile(resolve(modelAssets, 'manifest.json'), 'utf8')
  );
  const manifest = JSON.parse(manifestText);
  onManifest?.(manifest);
  await reportProgress({stage: 'assets.manifest.loaded', state: 'verified', files: Object.keys(manifest.files)});
  const actualManifestHash = createHash('sha256')
    .update(`${JSON.stringify(sortKeys(manifest))}\n`)
    .digest('hex');
  expect(actualManifestHash).toBe(manifestSha256);
  for (const [name, expected] of Object.entries(manifest.files)) {
    const path = resolve(modelAssets, name);
    const info = await trackAsyncStage(reportProgress, `assets.stat:${name}`, () => stat(path));
    expect(info.size, name).toBe(expected.bytes);
    const actualSha256 = await trackAsyncStage(reportProgress, `assets.sha256:${name}`, () => sha256(path));
    expect(actualSha256, name).toBe(expected.sha256);
  }
  return manifest;
}

test('Kev 0.6B Q4F16 runs pinned-asset WebGPU boundary cases on the served bundle', async ({
  page,
  browserName,
  browser
}) => {
  test.skip(!enabled, 'Set WEB_DECISION_KEV06_REAL_GPU=1 to run the pinned local Kev model on WebGPU.');
  test.setTimeout(10 * 60 * 1000);

  const progress = [];
  const reportPath = process.env.WEB_DECISION_KEV06_REPORT ? resolve(process.env.WEB_DECISION_KEV06_REPORT) : null;
  const report = {
    status: 'partial',
    date: new Date().toISOString(),
    host: {
      platform: process.platform,
      release: os.release(),
      arch: os.arch(),
      totalMemoryBytes: os.totalmem(),
      freeMemoryBytes: os.freemem()
    },
    browser: {name: browserName, version: browser.version(), userAgent: null},
    package: {name: null, version: null, bundlePath: servedBundle, bundleSha256: null},
    assets: {manifestSha256, model: null, checkpoint: null, files: []},
    inference: {modelId: 'kev-0.6b-q4f16', status: 'pending', progress: []},
    progress
  };
  let reportWriteQueue = Promise.resolve();
  const persistReport = () => {
    if (!reportPath) return Promise.resolve();
    const snapshot = `${JSON.stringify(report, null, 2)}\n`;
    const temporaryPath = `${reportPath}.${process.pid}.tmp`;
    reportWriteQueue = reportWriteQueue.then(async () => {
      await writeFile(temporaryPath, snapshot);
      await rename(temporaryPath, reportPath);
    });
    return reportWriteQueue;
  };
  const reportProgress = async (event, source = 'node') => {
    const recorded = {...event, recordedAt: new Date().toISOString()};
    if (source === 'node' && event.state === 'failed') report.status = 'failed';
    progress.push(recorded);
    if (source === 'browser') report.inference.progress.push(recorded);
    console.log(`${progressPrefix}${JSON.stringify(recorded)}`);
    await persistReport();
  };
  if (reportPath) {
    await mkdir(dirname(reportPath), {recursive: true});
    await reportProgress({stage: 'report.initialized', state: 'started'});
  }

  const packageJsonPath = resolve(dirname(servedBundle), '..', 'package.json');
  const packageJsonText = await trackAsyncStage(reportProgress, 'package.metadata.read', () =>
    readFile(packageJsonPath, 'utf8')
  );
  const packageMetadata = JSON.parse(packageJsonText);
  expect(packageMetadata.name).toBe('web-decision');
  report.package.name = packageMetadata.name;
  report.package.version = packageMetadata.version;
  await reportProgress({stage: 'package.identity', state: 'verified', version: packageMetadata.version});
  const bundleSha256 = await trackAsyncStage(reportProgress, 'package.bundle.sha256', () => sha256(servedBundle));
  report.package.bundleSha256 = bundleSha256;
  await reportProgress({stage: 'package.bundle.identity', state: 'verified'});
  await verifyAssets(reportProgress, (verifiedManifest) => {
    report.assets.model = verifiedManifest.model;
    report.assets.checkpoint = verifiedManifest.checkpoint;
    report.assets.files = Object.keys(verifiedManifest.files);
  });
  await reportProgress({stage: 'assets.verified', state: 'completed', manifestSha256});
  await trackAsyncStage(reportProgress, 'browser.progress-binding', () =>
    page.exposeFunction('__webDecisionKev06Progress', (event) => reportProgress(event, 'browser'))
  );
  await trackAsyncStage(reportProgress, 'browser.navigate', () => page.goto('/'));
  await trackAsyncStage(reportProgress, 'browser.harness.ready', () =>
    page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function')
  );

  report.inference.status = 'running';
  await reportProgress({stage: 'inference.started', state: 'started'});
  const inference = await trackAsyncStage(reportProgress, 'browser.kev-inference', () =>
    page.evaluate(async () => {
      const browserProgress = [];
      const reportProgress = async (stage, state, details = {}) => {
        const event = {at: new Date().toISOString(), stage, state, ...details};
        browserProgress.push(event);
        console.info(`KEV06_PROGRESS ${JSON.stringify(event)}`);
        await window.__webDecisionKev06Progress(event);
      };
      const runAsyncStage = async (stage, run) => {
        await reportProgress(stage, 'started');
        const startedAt = performance.now();
        try {
          const result = await run();
          await reportProgress(stage, 'completed', {elapsedMs: performance.now() - startedAt});
          return result;
        } catch (error) {
          await reportProgress(stage, 'failed', {
            elapsedMs: performance.now() - startedAt,
            error: error instanceof Error ? error.message : String(error)
          });
          throw error;
        }
      };
      const runSyncStage = async (stage, run) => {
        await reportProgress(stage, 'started');
        const startedAt = performance.now();
        try {
          const result = run();
          await reportProgress(stage, 'completed', {elapsedMs: performance.now() - startedAt});
          return result;
        } catch (error) {
          await reportProgress(stage, 'failed', {
            elapsedMs: performance.now() - startedAt,
            error: error instanceof Error ? error.message : String(error)
          });
          throw error;
        }
      };

      if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser.');
      const adapter = await runAsyncStage('adapter.request', () => navigator.gpu.requestAdapter());
      if (!adapter) throw new Error('WebGPU did not provide an adapter.');
      if (!adapter.features.has('shader-f16')) throw new Error('The adapter does not support shader-f16.');
      const adapterIsFallback = adapter.isFallbackAdapter ?? adapter.info?.isFallbackAdapter ?? null;
      await runSyncStage('adapter.verify', () => {
        if (adapterIsFallback !== false) {
          throw new Error(`Expected a non-fallback WebGPU adapter; got ${String(adapterIsFallback)}.`);
        }
      });

      const adapterInfo = adapter.info ?? {};
      const limits = Object.fromEntries(
        ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupStorageSize'].map((name) => [
          name,
          adapter.limits[name]
        ])
      );
      const model = await runAsyncStage('model.load', () =>
        window.webDecision.loadDecisionModel({
          model: 'kev-0.6b-q4f16',
          assetsUrl: '/kev-0.6b-q4f16/'
        })
      );

      try {
        const timed = async (stage, run) => {
          const start = performance.now();
          const result = await runAsyncStage(stage, run);
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
        const minimum = await timed('inference.minimum', () =>
          model.decide({state: '', questions: [{instructions: 'Choose.', options: ['only']}]})
        );
        const base = await timed('inference.base', () =>
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
          const instructions = await runSyncStage(`input.construct:${targetTokens}`, () =>
            exactTokenText(targetTokens - fixedTokens)
          );
          const measured = await timed(`inference.tokens:${targetTokens}`, () =>
            model.decide({state: '', questions: [{instructions, options: ['no', 'yes']}]})
          );
          lengths.push({targetTokens, ...measured});
        }

        const overBoundaryInstructions = await runSyncStage('input.construct:over-boundary', () =>
          exactTokenText(sequenceTokenLimit + 1 - fixedTokens)
        );
        let overBoundary;
        try {
          await runAsyncStage('inference.over-boundary', () =>
            model.decide({
              state: '',
              questions: [{instructions: overBoundaryInstructions, options: ['no', 'yes']}]
            })
          );
          overBoundary = {code: 'UNEXPECTED_SUCCESS'};
        } catch (error) {
          overBoundary = {code: error.code ?? error.name, message: error.message};
        }

        const choices = Object.fromEntries(Array.from({length: 255}, (_, index) => [`option-${index}`, null]));
        const typed = await timed('inference.typed-255-options', () =>
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
          adapterIsFallback,
          adapterFeatures: [...adapter.features],
          limits,
          attentionTokenLimit,
          sequenceTokenLimit,
          minimum,
          base,
          lengths,
          overBoundary,
          typed,
          progress: browserProgress
        };
      } finally {
        await runAsyncStage('model.dispose', () => model.dispose());
      }
    })
  );

  report.inference = {
    status: 'results-received',
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
    progress: inference.progress
  };
  await reportProgress({stage: 'inference.results-received', state: 'completed'});

  expect(inference.model).toBe('kev-0.6b-q4f16');
  expect(inference.backend).toBe('tvm-webgpu');
  expect(inference.currentBundleGpu).toBe('verified');
  expect(inference.adapterIsFallback).toBe(false);
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
    const gpuPage = await trackAsyncStage(reportProgress, 'chromium.gpu-page.open', () => page.context().newPage());
    await trackAsyncStage(reportProgress, 'chromium.gpu-page.navigate', () => gpuPage.goto('chrome://gpu'));
    const gpuReport = await trackAsyncStage(reportProgress, 'chromium.gpu-report.read', () =>
      gpuPage.evaluate(() => {
        const texts = [];
        function visit(root) {
          if (root?.textContent) texts.push(root.textContent);
          for (const element of root?.querySelectorAll?.('*') ?? []) if (element.shadowRoot) visit(element.shadowRoot);
        }
        visit(document);
        return texts.join('\n');
      })
    );
    webGpuFeatureStatus =
      gpuReport.match(/WebGPU:\s*(Hardware accelerated|Software only|Disabled|Blocklisted)/i)?.[1] ?? null;
    gpuDeviceDescription =
      gpuReport.match(/<Integrated GPU>\s*([^\n]+)/i)?.[1]?.trim() ??
      gpuReport.match(/GPU\d+\s+VENDOR=[^\n]*?\bDEVICE=[^\n]*?\[([^\]]+)\]/i)?.[1]?.trim() ??
      gpuReport.match(/GL_RENDERER\s+([^\n]+)/i)?.[1]?.trim() ??
      null;
    await trackAsyncStage(reportProgress, 'chromium.gpu-page.close', () => gpuPage.close());
    expect(webGpuFeatureStatus).toMatch(/hardware accelerated/i);
    expect(gpuDeviceDescription).not.toBeNull();
  }
  report.browser.webGpuFeatureStatus = webGpuFeatureStatus;
  report.browser.gpuDeviceDescription = gpuDeviceDescription;
  await reportProgress({stage: 'browser.gpu-metadata', state: 'completed'});

  const summarizeDecision = ({targetTokens, elapsedMs, result}) => ({
    ...(targetTokens === undefined ? {} : {targetTokens}),
    elapsedMs,
    tokenCount: result.tokenCount,
    selected: result.answers[0].selected,
    optionOrder: result.answers[0].probabilities.map(({option}) => option),
    probabilityTotal: result.answers[0].probabilities.reduce((sum, answer) => sum + answer.probability, 0)
  });
  report.browser.userAgent = await trackAsyncStage(reportProgress, 'browser.user-agent.read', () =>
    page.evaluate(() => navigator.userAgent)
  );
  report.inference = {
    ...report.inference,
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
  };
  report.host.freeMemoryBytes = os.freemem();
  report.status = 'complete';
  report.inference.status = 'complete';
  await reportProgress({stage: 'report.finalized', state: 'completed'});
  console.log(`Kev 0.6B real WebGPU report: ${JSON.stringify(report)}`);
});
