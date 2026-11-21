import {createHash} from 'node:crypto';
import {mkdtemp, readFile, realpath, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium, expect, test as base, type BrowserContext, type Page} from '@playwright/test';
import sourceInventory from '../../packages/web-decision/assets/mapika2b/source-inventory.json' with {type: 'json'};
import {verifyArchiveInstall} from '../../scripts/release/verify-release-evidence.ts';
import {archiveIdentity, cleanSourceCommit} from '../../scripts/release/curated-gpu-evidence.mts';
import type {Mapika2bResult} from '../../packages/web-decision/src/mapika2b/readout.mts';
import {
  MAPIKA2B_DECIDER_REVISION,
  MAPIKA2B_REPOSITORY,
  MAPIKA2B_REVISION
} from '../../packages/web-decision/src/mapika2b/contract.mts';
import {
  mapikaReferenceMismatches,
  mapikaReleaseCases,
  type AcceptedMapikaCase,
  type MapikaAnswerReference,
  type MapikaReference
} from './mapika2b-evidence.mts';
import {readAdapter, readHardwareReport} from './mapika2b-hardware.mts';
import {routeMapikaSnapshot} from './mapika2b-snapshot.mts';

const modelId = 'mapika-decider-2b-v12';
const test = base.extend({
  context: async ({browserName}, use, testInfo) => {
    const profile = await mkdtemp(join(tmpdir(), `web-decision-mapika-public-${browserName}-`));
    const {baseURL, channel, headless, launchOptions} = testInfo.project.use;
    let context: BrowserContext | undefined;
    try {
      context = await chromium.launchPersistentContext(profile, {
        ...(baseURL ? {baseURL} : {}),
        ...(channel ? {channel} : {}),
        ...(typeof headless === 'boolean' ? {headless} : {}),
        ...launchOptions
      });
      await use(context);
    } finally {
      try {
        await context?.close();
      } finally {
        await rm(profile, {recursive: true, force: true});
      }
    }
  }
});

type LoadMilestone = 'afterLoad' | 'afterFirstText' | 'afterFirstImage' | 'beforePrepare' | 'afterPrepare';
type ModelProgress = {phase?: string; loadedBytes: number; totalBytes: number};

async function runPublicModel({
  id,
  inputs,
  offline,
  prepare = 'none'
}: {
  id: string;
  inputs: AcceptedMapikaCase[];
  offline: boolean;
  prepare?: 'none' | 'complete' | 'recover';
}) {
  const api = window.webDecision as unknown as {
    listModels(): readonly {modelId: string}[];
    describeModel(id: string): {modelId: string};
    loadDecisionModel(options: {
      modelId: string;
      cache: boolean;
      offline: boolean;
      onProgress: (progress: ModelProgress) => void;
    }): Promise<{
      modelId: string;
      cacheStatus: 'complete' | 'incomplete' | 'disabled' | 'unavailable';
      prepareOffline(options?: {onProgress?: (progress: ModelProgress) => void}): Promise<void>;
      systemOne(request: unknown): Promise<Mapika2bResult>;
      dispose(): Promise<void>;
    }>;
  };
  if (!api.listModels().some((item) => item.modelId === id) || api.describeModel(id).modelId !== id)
    throw new Error('The public catalog does not expose Mapika.');
  const progress: Record<string, {loadedBytes: number; totalBytes: number}> = {};
  const onProgress = (event: ModelProgress) => {
    progress[event.phase ?? 'unknown'] = {loadedBytes: event.loadedBytes, totalBytes: event.totalBytes};
  };
  const {mapikaLoadMilestone} = window as Window & {mapikaLoadMilestone?: (stage: LoadMilestone) => Promise<void>};
  const start = performance.now();
  const model = await api.loadDecisionModel({modelId: id, cache: true, offline, onProgress});
  const loadMs = performance.now() - start;
  const cacheStatusAfterLoad = model.cacheStatus;

  function referenceRequest(input: AcceptedMapikaCase) {
    const questions = new Map(
      input.questionIndex.map(([name]) => {
        const question = {...input.request.questions[name]!};
        const criteria = question.criteria;
        const rendered = input.renderedQuestions[name]!;
        if (rendered.type === 'choice' && criteria && typeof criteria === 'object' && !Array.isArray(criteria)) {
          const values = criteria as Record<string, unknown>;
          question.criteria = new Map(rendered.names.map((option) => [String(option), values[String(option)]]));
        }
        return [name, question];
      })
    );
    return {...input.request, questions};
  }

  async function runCase(input: AcceptedMapikaCase) {
    const startedInference = performance.now();
    const response = await model.systemOne(referenceRequest(input));
    const elapsedMs = performance.now() - startedInference;
    return {result: JSON.parse(JSON.stringify(response)) as Mapika2bResult, elapsedMs};
  }

  async function prepareCache() {
    const startedPreparation = performance.now();
    try {
      await model.prepareOffline({onProgress});
      return {elapsedMs: performance.now() - startedPreparation, failure: null};
    } catch (cause) {
      if (prepare === 'complete') throw cause;
      return {
        elapsedMs: performance.now() - startedPreparation,
        failure: {code: (cause as {code?: unknown})?.code, message: String(cause)}
      };
    }
  }

  try {
    await mapikaLoadMilestone?.('afterLoad');
    const results = [];
    const inferenceMs = [];
    let firstTextMs: number | null = null;
    const firstImageIndex = inputs.findIndex(({request}) => Object.keys(request.assets ?? {}).length > 0);
    for (const [index, input] of inputs.entries()) {
      const result = await runCase(input);
      results.push(result.result);
      inferenceMs.push(result.elapsedMs);
      if (index === 0 && index !== firstImageIndex) {
        firstTextMs = performance.now() - start;
        await mapikaLoadMilestone?.('afterFirstText');
      }
      if (index === firstImageIndex) await mapikaLoadMilestone?.('afterFirstImage');
    }
    const cacheStatusBeforePrepare = model.cacheStatus;
    await mapikaLoadMilestone?.('beforePrepare');
    const preparation = prepare === 'none' ? null : await prepareCache();
    await mapikaLoadMilestone?.('afterPrepare');
    const recovery = prepare === 'recover' && inputs[0] ? await runCase(inputs[0]) : null;
    return {
      modelId: model.modelId,
      cacheStatus: model.cacheStatus,
      cacheStatusAfterLoad,
      cacheStatusBeforePrepare,
      results,
      progress,
      preparation,
      recovery,
      timings: {
        loadMs,
        firstTextMs,
        firstImageMs: inferenceMs[firstImageIndex] ?? null,
        prepareOfflineMs: preparation?.elapsedMs ?? null,
        inferenceMs
      }
    };
  } finally {
    await model.dispose();
  }
}

type WeightDownloads = {
  textRanges: number;
  visionRanges: number;
  textBytes: number;
  visionBytes: number;
  requestedBytes: number;
};

async function trackWeightDownloads(page: Page) {
  const prefix = new URL(`https://huggingface.co/${MAPIKA2B_REPOSITORY}/resolve/${MAPIKA2B_REVISION}/`);
  const totals = {textRanges: 0, visionRanges: 0, textBytes: 0, visionBytes: 0, requestedBytes: 0};
  const milestones: Partial<Record<LoadMilestone, WeightDownloads>> = {};
  const unboundedWeightRequests: string[] = [];
  page.on('request', (request) => {
    const url = new URL(request.url());
    if (request.method() !== 'GET' || !url.href.startsWith(prefix.href)) return;
    const file = url.pathname.slice(prefix.pathname.length);
    const {text, vision} = sourceInventory.weights;
    if (file !== text.file && file !== vision.file) return;
    const range = /^bytes=(\d+)-(\d+)$/u.exec(request.headers().range ?? '');
    if (!range || Number(range[2]) < Number(range[1])) {
      unboundedWeightRequests.push(file);
      return;
    }
    const bytes = Number(range[2]) - Number(range[1]) + 1;
    if (file === text.file) {
      totals.textRanges++;
      totals.textBytes += bytes;
    } else {
      totals.visionRanges++;
      totals.visionBytes += bytes;
    }
    totals.requestedBytes += bytes;
  });
  await page.exposeFunction('mapikaLoadMilestone', (stage: LoadMilestone) => {
    milestones[stage] = {...totals};
  });
  return {milestones, unboundedWeightRequests};
}

test('the installed public package loads Mapika by model ID and matches the source', async ({
  page,
  context
}, testInfo) => {
  test.skip(process.env.WEB_DECISION_MAPIKA2B_PUBLIC_GPU !== '1', 'Enable the installed public package GPU test.');
  test.skip(testInfo.project.name !== 'chromium', 'This test requires Chromium WebGPU.');
  test.setTimeout(60 * 60 * 1000);
  const archivePath = process.env.WEB_DECISION_RELEASE_ARCHIVE;
  const packagePath = process.env.WEB_DECISION_RELEASE_PACKAGE_DIR;
  if (!archivePath || !packagePath || !process.env.WEB_DECISION_INSTALLED_BUNDLE)
    throw new Error('The public GPU test needs the candidate archive, installed package and bundle paths.');
  const archive = await verifyArchiveInstall(archivePath, packagePath);
  expect(archive.bundlePath).toBe(await realpath(process.env.WEB_DECISION_INSTALLED_BUNDLE));
  const root = new URL('../../scripts/mapika/', import.meta.url);
  const preparedBytes = await readFile(new URL('reference-fp32-prepared.json', root));
  const answerBytes = await readFile(new URL('reference-fp32-answers.json', root));
  const prepared = JSON.parse(preparedBytes.toString('utf8')) as MapikaReference<AcceptedMapikaCase>;
  const answers = JSON.parse(answerBytes.toString('utf8')) as MapikaReference<MapikaAnswerReference>;
  for (const reference of [prepared, answers])
    expect(reference.source).toMatchObject({
      repository: MAPIKA2B_REPOSITORY,
      revision: MAPIKA2B_REVISION,
      deciderRevision: MAPIKA2B_DECIDER_REVISION
    });
  const cases = mapikaReleaseCases.map((id) => {
    const input = prepared.cases.find((item) => item.id === id);
    const expected = answers.cases.find((item) => item.id === id);
    if (!input || !expected) throw new Error('Missing public Mapika reference case: ' + id);
    return {id, input, expected: expected.result};
  });
  const snapshot = process.env.WEB_DECISION_MAPIKA2B_SNAPSHOT;
  const release = process.env.WEB_DECISION_MAPIKA2B_RELEASE_GPU === '1';
  if (release && snapshot) throw new Error('Release GPU evidence requires the live pinned checkpoint.');
  const sourceRoot = fileURLToPath(new URL('../../', import.meta.url));
  const sourceCommit = release ? cleanSourceCommit(sourceRoot) : null;
  if (snapshot) await routeMapikaSnapshot(page, snapshot, () => {});
  await page.addInitScript(() => {
    const cacheWindow = window as Window & {mapikaCacheErrors?: string[]};
    cacheWindow.mapikaCacheErrors = [];
    const put = Cache.prototype.put;
    Cache.prototype.put = async function (request, response) {
      try {
        return await put.call(this, request, response);
      } catch (cause) {
        if ((cacheWindow.mapikaCacheErrors?.length ?? 0) < 3) cacheWindow.mapikaCacheErrors?.push(String(cause));
        throw cause;
      }
    };
  });
  const hardware = await readHardwareReport(page);
  expect(hardware.status).toBe('Hardware accelerated');
  const {milestones, unboundedWeightRequests} = await trackWeightDownloads(page);
  await page.goto('/');
  await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');
  const adapter = await readAdapter(page);
  expect(adapter?.isFallbackAdapter).toBe(false);
  const inputs = cases.map(({input}) => input);
  const observed = await page.evaluate(runPublicModel, {
    id: modelId,
    inputs,
    offline: false,
    prepare: 'complete' as const
  });
  const fullCheckpointBytes = sourceInventory.weights.text.bytes + sourceInventory.weights.vision.bytes;
  const progressive = {
    ...milestones,
    unboundedWeightRequests,
    cacheStatusAfterLoad: observed.cacheStatusAfterLoad,
    cacheStatusBeforePrepare: observed.cacheStatusBeforePrepare,
    fullCheckpointBytes
  };
  expect(progressive.unboundedWeightRequests).toEqual([]);
  expect(progressive.afterLoad?.visionRanges).toBe(0);
  expect(progressive.afterLoad?.textRanges).toBeGreaterThan(0);
  expect(progressive.afterFirstText?.visionRanges).toBe(0);
  expect(progressive.afterFirstText?.requestedBytes).toBeLessThan(sourceInventory.weights.text.bytes);
  expect(progressive.afterFirstImage?.visionRanges).toBeGreaterThan(0);
  expect(progressive.afterPrepare?.requestedBytes).toBeGreaterThan(progressive.beforePrepare?.requestedBytes ?? 0);
  expect(observed.cacheStatusAfterLoad).toBe('incomplete');
  expect(observed.cacheStatusBeforePrepare).toBe('incomplete');
  expect(observed.modelId).toBe(modelId);
  expect(observed.cacheStatus).toBe('complete');
  expect(observed.progress['upstream-source']).toEqual({
    loadedBytes: fullCheckpointBytes,
    totalBytes: fullCheckpointBytes
  });
  for (const [index, item] of cases.entries())
    expect(mapikaReferenceMismatches(observed.results[index], item.expected), item.id).toEqual([]);
  const cacheStatus = await page.evaluate(async () => {
    const names = await caches.keys();
    const entries = [];
    for (const name of names) {
      const cache = await caches.open(name);
      const keys = await cache.keys();
      entries.push({
        name,
        count: keys.length,
        marker: keys.some(({url}) => url.includes('mapika-2b-v12-offline-complete'))
      });
    }
    return {
      storage: await navigator.storage.estimate(),
      entries,
      errors: (window as Window & {mapikaCacheErrors?: string[]}).mapikaCacheErrors
    };
  });
  console.log('Mapika public cache status:', JSON.stringify({cacheStatus, progress: observed.progress}));
  const fresh = await context.newPage();
  const blocked: string[] = [];
  const bootstrapModules: string[] = [];
  let offline;
  try {
    await fresh.goto('/');
    await fresh.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');
    const origin = new URL(fresh.url()).origin;
    await fresh.route('**/*', async (route) => {
      const url = new URL(route.request().url());
      if (url.origin === origin && /\.(?:m?js)$/u.test(url.pathname)) {
        bootstrapModules.push(url.pathname);
        return route.continue();
      }
      blocked.push(url.origin + url.pathname);
      return route.abort();
    });
    offline = await fresh.evaluate(runPublicModel, {id: modelId, inputs, offline: true});
    expect(blocked).toEqual([]);
    expect(offline.modelId).toBe(observed.modelId);
    expect(offline.cacheStatus).toBe('complete');
    expect(offline.results).toEqual(observed.results);
    const warm = await fresh.evaluate(runPublicModel, {id: modelId, inputs: inputs.slice(0, 1), offline: false});
    expect(warm.cacheStatusAfterLoad).toBe('complete');
    expect(warm.results).toEqual(observed.results.slice(0, 1));
    expect(blocked).toEqual([]);
  } finally {
    await fresh.close();
  }
  if (release) expect(cleanSourceCommit(sourceRoot)).toBe(sourceCommit);
  const evidence = JSON.stringify({
    schemaVersion: 1,
    sourceCommit,
    source: snapshot ? 'local-snapshot' : 'pinned-upstream',
    archive: archiveIdentity(archive),
    references: {
      preparedSha256: createHash('sha256').update(preparedBytes).digest('hex'),
      answersSha256: createHash('sha256').update(answerBytes).digest('hex')
    },
    browser: {name: testInfo.project.name, userAgent: await page.evaluate(() => navigator.userAgent)},
    hardware,
    adapter,
    modelId,
    cases: cases.map(({id}) => id),
    results: observed.results,
    modelCacheStatus: observed.cacheStatus,
    cacheStatus,
    progressive,
    progress: observed.progress,
    timings: observed.timings,
    offline: {cacheStatus: offline.cacheStatus, results: offline.results, blockedRequests: blocked, bootstrapModules}
  });
  if (process.env.WEB_DECISION_MAPIKA2B_PUBLIC_REPORT)
    await writeFile(process.env.WEB_DECISION_MAPIKA2B_PUBLIC_REPORT, evidence + '\n');
  await testInfo.attach('mapika-public-evidence.json', {
    body: evidence,
    contentType: 'application/json'
  });
});

test('the public package reports incomplete cache when storage rejects writes', async ({page}, testInfo) => {
  test.skip(process.env.WEB_DECISION_MAPIKA2B_QUOTA_GPU !== '1', 'Enable the public storage quota test.');
  test.skip(testInfo.project.name !== 'chromium', 'This test requires Chromium WebGPU.');
  test.setTimeout(60 * 60 * 1000);
  const archivePath = process.env.WEB_DECISION_RELEASE_ARCHIVE;
  const packagePath = process.env.WEB_DECISION_RELEASE_PACKAGE_DIR;
  if (!archivePath || !packagePath || !process.env.WEB_DECISION_INSTALLED_BUNDLE)
    throw new Error('The public GPU test needs the candidate archive, installed package and bundle paths.');
  const archive = await verifyArchiveInstall(archivePath, packagePath);
  expect(archive.bundlePath).toBe(await realpath(process.env.WEB_DECISION_INSTALLED_BUNDLE));
  const root = new URL('../../scripts/mapika/', import.meta.url);
  const prepared = JSON.parse(
    await readFile(new URL('reference-fp32-prepared.json', root), 'utf8')
  ) as MapikaReference<AcceptedMapikaCase>;
  const answers = JSON.parse(
    await readFile(new URL('reference-fp32-answers.json', root), 'utf8')
  ) as MapikaReference<MapikaAnswerReference>;
  const input = prepared.cases.find(({id}) => id === 'typed-packed');
  const expected = answers.cases.find(({id}) => id === 'typed-packed');
  if (!input || !expected) throw new Error('Missing the Mapika quota reference case.');
  const snapshot = process.env.WEB_DECISION_MAPIKA2B_SNAPSHOT;
  if (snapshot) await routeMapikaSnapshot(page, snapshot, () => {});
  await page.addInitScript(() => {
    if (typeof Cache !== 'undefined')
      Cache.prototype.put = async () => {
        throw new DOMException('Quota exceeded', 'QuotaExceededError');
      };
  });
  expect((await readHardwareReport(page)).status).toBe('Hardware accelerated');
  await page.goto('/');
  await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');
  expect((await readAdapter(page))?.isFallbackAdapter).toBe(false);
  const online = await page.evaluate(runPublicModel, {
    id: modelId,
    inputs: [input],
    offline: false,
    prepare: 'recover' as const
  });
  expect(online.cacheStatusAfterLoad).toBe('incomplete');
  expect(online.cacheStatus).toBe('incomplete');
  expect(online.preparation?.failure?.code).toBe('ASSET_LOAD_FAILED');
  expect(online.recovery?.result).toEqual(online.results[0]);
  expect(mapikaReferenceMismatches(online.results[0], expected.result)).toEqual([]);
  const failure = await page.evaluate(async (id) => {
    try {
      const load = window.webDecision.loadDecisionModel as unknown as (options: {
        modelId: string;
        cache: boolean;
        offline: boolean;
      }) => Promise<{dispose(): Promise<void>}>;
      const model = await load({modelId: id, cache: true, offline: true});
      await model.dispose();
      return null;
    } catch (cause) {
      return {code: (cause as {code?: unknown}).code, message: String(cause)};
    }
  }, modelId);
  expect(failure?.code).toBe('ASSET_LOAD_FAILED');
  const evidence = JSON.stringify({
    scenario: 'simulated-cache-quota-failure',
    archiveSha256: archive.archiveSha256,
    status: online.cacheStatus,
    result: online.results[0],
    preparationFailure: online.preparation?.failure,
    onlineRecoveryResult: online.recovery?.result,
    offlineFailure: failure
  });
  if (process.env.WEB_DECISION_MAPIKA2B_QUOTA_REPORT)
    await writeFile(process.env.WEB_DECISION_MAPIKA2B_QUOTA_REPORT, evidence + '\n');
  await testInfo.attach('mapika-quota-evidence.json', {
    body: evidence,
    contentType: 'application/json'
  });
});
