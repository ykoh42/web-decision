import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {mkdtemp, open, readFile, rm, stat} from 'node:fs/promises';
import {join, resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {expect, test, type BrowserContext} from '@playwright/test';
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

const localBuildMode = process.env.WEB_DECISION_KANANA_LOCAL_BUILD_GPU === '1';
const enabled = process.env.WEB_DECISION_KANANA_CURATED_REAL_GPU === '1' || localKananaCandidateMode || localBuildMode;
const upstreamPath = new URL(kananaUpstreamBase).pathname;
const probabilityTolerance = 0.001;
const fixtureBytes = readFileSync(new URL('../fixtures/kanana-decision-python-reference.json', import.meta.url));
const fixtureSha256 = createHash('sha256').update(fixtureBytes).digest('hex');
const reference = JSON.parse(fixtureBytes.toString('utf8')) as {
  modelId: string;
  revision: string;
  checkpointSha256: string;
  request: {
    state: string;
    state_format: 'text';
    temperature: number;
    questions: Record<string, {type: 'choice'; instructions: string; criteria: Record<string, string>}>;
  };
  result: {
    answers: Record<string, {choice: string; confidence: number; probabilities: Record<string, number>}>;
  };
};

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

test('the public Kanana package loader reuses cached text weights with network requests blocked', async ({
  browserName,
  browser
}) => {
  test.skip(!enabled, 'Set WEB_DECISION_KANANA_CURATED_REAL_GPU=1 to run the full package consumer GPU check.');
  test.setTimeout(45 * 60 * 1000);

  const releaseInputs = localBuildMode ? null : await collectInstalledKananaReleaseInputs();
  expect(fixtureSha256).toBe(
    releaseInputs?.policy.referenceFixture.sha256 ?? '822156454d468153cef23fde6fad8c88b59821d9da1b86897733fa906abfd420'
  );
  expect(reference.modelId).toBe(releaseInputs?.policy.model.repository ?? `kakaocorp/${kananaModelId}`);
  expect(reference.revision).toBe(kananaRevision);
  expect(reference.checkpointSha256).toBe(kananaCheckpointSha256);
  await verifyPinnedKananaSnapshot();

  const modelPath = resolve(kananaSnapshotDirectory, 'model.safetensors');
  const modelFile = await open(modelPath, 'r');
  const modelSize = (await stat(modelPath)).size;
  const fetched: {name: string; bytes: number; range?: string}[] = [];
  const networkAttempts: string[] = [];
  const visionWasmRequests: string[] = [];
  let blockNetwork = false;
  let localOrigin: string | undefined;
  let profileDirectory: string | undefined;
  let persistentContext: BrowserContext | undefined;

  try {
    expect(browserName).toBe('chromium');
    profileDirectory = await mkdtemp(join(tmpdir(), 'web-decision-kanana-offline-'));
    const gpuMarker = process.env.WEB_DECISION_GPU_BROWSER_MARKER;
    persistentContext = await browser.browserType().launchPersistentContext(profileDirectory, {
      baseURL: `http://127.0.0.1:${Number(process.env.WEB_DECISION_BROWSER_PORT || 41773)}`,
      headless: false,
      channel: process.env.WEB_DECISION_BROWSER_CHANNEL || 'chromium',
      args: ['--enable-unsafe-webgpu', ...(gpuMarker ? [gpuMarker] : [])],
      ignoreDefaultArgs: ['--enable-unsafe-swiftshader']
    });
    const page = persistentContext.pages()[0] ?? (await persistentContext.newPage());
    page.on('request', (request) => {
      if (new URL(request.url()).pathname.includes('kanana-vision-model')) visionWasmRequests.push(request.url());
    });
    await page.route('**/*', async (route) => {
      if (!blockNetwork) return route.fallback();
      if (localOrigin && new URL(route.request().url()).origin === localOrigin) return route.fallback();
      networkAttempts.push(route.request().url());
      return route.abort();
    });
    await page.route(`${kananaUpstreamBase}**`, async (route) => {
      const request = route.request();
      if (blockNetwork) {
        networkAttempts.push(request.url());
        return route.abort();
      }
      const name = new URL(request.url()).pathname.slice(upstreamPath.length);
      const cors = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Range',
        'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        'Access-Control-Expose-Headers': 'Accept-Ranges, Content-Length, Content-Range'
      };
      if (request.method() === 'OPTIONS') return route.fulfill({status: 204, headers: cors});
      if (name === 'model.safetensors') {
        const range = request.headers().range;
        const match = range && /^bytes=(\d+)-(\d+)$/u.exec(range);
        if (!match) return route.fulfill({status: 416, headers: {...cors, 'Content-Range': `bytes */${modelSize}`}});
        const start = Number(match[1]);
        const end = Number(match[2]);
        if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || end >= modelSize) {
          return route.fulfill({status: 416, headers: {...cors, 'Content-Range': `bytes */${modelSize}`}});
        }
        const bytes = await readRange(modelFile, start, end);
        fetched.push({name, bytes: bytes.byteLength, range});
        return route.fulfill({
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
      }
      if (!['config.json', 'tokenizer_config.json', 'tokenizer.json'].includes(name)) {
        return route.fulfill({status: 404, headers: cors});
      }
      const bytes = await readFile(resolve(kananaSnapshotDirectory, name));
      fetched.push({name, bytes: bytes.byteLength});
      return route.fulfill({status: 200, headers: {...cors, 'Content-Length': String(bytes.byteLength)}, body: bytes});
    });

    const browserErrors: string[] = [];
    page.on('pageerror', (error) => browserErrors.push(error.message));
    await page.goto('/');
    localOrigin = new URL(page.url()).origin;
    await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');
    await page.evaluate(async () => {
      if (!globalThis.caches) throw new Error('CacheStorage is required for the Kanana offline release test.');
      await caches.delete('web-decision-assets-v1');
    });
    const request = {
      state: reference.request.state,
      stateFormat: reference.request.state_format,
      questions: reference.request.questions,
      temperature: reference.request.temperature
    };
    const onlineRun = await page.evaluate(async (input) => {
      const adapter = await navigator.gpu?.requestAdapter({powerPreference: 'high-performance'});
      if (!adapter) throw new Error('Chromium did not provide a WebGPU adapter.');
      const adapterWithFallback = adapter as GPUAdapter & {readonly isFallbackAdapter?: boolean};
      const isFallback = adapterWithFallback.isFallbackAdapter ?? adapter.info?.isFallbackAdapter ?? null;
      if (isFallback !== false) throw new Error(`Expected a hardware WebGPU adapter; got ${String(isFallback)}.`);
      const model = await window.webDecision.loadDecisionModel({modelId: 'kanana-2-v-decide-3b-2611', cache: true});
      try {
        const output = await model.systemOne(input);
        const answer = output.answers.delivery_status;
        if (!answer || answer.type !== 'choice') throw new Error('Kanana returned no cached online choice result.');
        const checkpointSha256 = model.description.checkpoint.weightSha256;
        if (!checkpointSha256) throw new Error('Kanana descriptor omitted the checkpoint digest.');
        return {
          gpu: {
            isFallbackAdapter: isFallback,
            adapterInfo: {
              vendor: adapter.info?.vendor ?? '',
              architecture: adapter.info?.architecture ?? '',
              isFallbackAdapter: isFallback
            },
            features: [...adapter.features],
            limits: {
              maxBufferSize: adapter.limits.maxBufferSize,
              maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
              maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
              maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage
            }
          },
          userAgent: navigator.userAgent,
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
            usage: output.usage
          }
        };
      } finally {
        await model.dispose();
      }
    }, request);

    const expected = reference.result.answers.delivery_status;
    if (!expected) throw new Error('Kanana Python reference is missing delivery_status.');
    expect(onlineRun.result.modelId).toBe(kananaModelId);
    expect(onlineRun.result.checkpointRevision).toBe(kananaRevision);
    expect(onlineRun.result.checkpointSha256).toBe(kananaCheckpointSha256);
    expect(onlineRun.result.backend).toBe('tvm-webgpu');
    expect(onlineRun.result.choice).toBe(expected.choice);
    expect(visionWasmRequests).toEqual([]);
    for (const [key, probability] of Object.entries(expected.probabilities)) {
      expect(Math.abs(onlineRun.result.probabilities[key]! - probability), key).toBeLessThan(probabilityTolerance);
    }
    expect(fetched.some(({name}) => name === 'model.safetensors')).toBe(true);
    const fetchedRanges = fetched.filter(({name}) => name === 'model.safetensors');

    const hasCompletionMarker = await page.evaluate(async () => {
      const cacheName = 'web-decision-assets-v1';
      const cacheNames = await caches.keys();
      if (!cacheNames.includes(cacheName)) return false;
      const cache = await caches.open(cacheName);
      return (await cache.keys()).some((key) =>
        new URL(key.url).pathname.endsWith('/kanana-text-offline-complete-v1.json')
      );
    });
    expect(hasCompletionMarker).toBe(true);

    blockNetwork = true;
    const offline = await page.evaluate(async (input) => {
      const model = await window.webDecision.loadDecisionModel({modelId: 'kanana-2-v-decide-3b-2611', offline: true});
      try {
        const output = await model.systemOne(input);
        const answer = output.answers.delivery_status;
        if (!answer || answer.type !== 'choice') throw new Error('Kanana returned no offline choice result.');
        const checkpointSha256 = model.description.checkpoint.weightSha256;
        if (!checkpointSha256) throw new Error('Kanana descriptor omitted the checkpoint digest.');
        return {
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
          usage: output.usage
        };
      } finally {
        await model.dispose();
      }
    }, request);

    expect(networkAttempts).toEqual([]);
    expect(offline.modelId).toBe(onlineRun.result.modelId);
    expect(offline.backend).toBe('tvm-webgpu');
    expect(offline.assetManifestSha256).toBe(onlineRun.result.assetManifestSha256);
    expect(offline.choice).toBe(onlineRun.result.choice);
    expect(offline.confidence).toBe(onlineRun.result.confidence);
    expect(offline.probabilities).toEqual(onlineRun.result.probabilities);
    expect(offline.probabilityEntries).toEqual(onlineRun.result.probabilityEntries);
    expect(visionWasmRequests).toEqual([]);
    expect(browserErrors).toEqual([]);

    console.log(
      `KANANA_CURATED_OFFLINE_TEXT_REAL_GPU ${JSON.stringify({browserName, version: browser.version(), online: onlineRun.result, offline, fetchedRanges: fetchedRanges.length, fetchedBytes: fetchedRanges.reduce((sum, item) => sum + item.bytes, 0), networkAttempts})}`
    );
    if (releaseInputs === null) return;
    if (localKananaCandidateMode && process.env.WEB_DECISION_KANANA_EMIT_EVIDENCE !== '1') return;
    const evidence = createKananaReleaseEvidenceEnvelope({
      inputs: releaseInputs,
      kind: 'cached-offline-text-choice',
      runtimeArtifactNames: releaseInputs.policy.tests.offline.runtimeArtifacts,
      browserName,
      browserVersion: browser.version(),
      userAgent: onlineRun.userAgent,
      model: {
        loadedId: offline.modelId,
        describedId: offline.descriptionModelId,
        checkpointRevision: offline.checkpointRevision,
        checkpointSha256: offline.checkpointSha256
      },
      gpu: onlineRun.gpu
    });
    console.log(
      `${KANANA_RELEASE_EVIDENCE_PREFIX}${JSON.stringify({
        ...evidence,
        referenceFixtureSha256: fixtureSha256,
        probabilityTolerance,
        cache: {
          primedOnline: true,
          requestedOffline: true,
          blockedNetwork: true,
          fetchedRanges: fetchedRanges.length,
          fetchedBytes: fetchedRanges.reduce((sum, item) => sum + item.bytes, 0)
        },
        networkAttempts,
        inference: {online: onlineRun.result, offline}
      })}`
    );
  } finally {
    await persistentContext?.close();
    if (profileDirectory) await rm(profileDirectory, {recursive: true, force: true});
    await modelFile.close();
  }
});
