import {mkdir, mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import {chromium, expect, test as base, type BrowserContext, type Page, type Request} from '@playwright/test';
import {MAPIKA2B_REVISION} from '../../packages/web-decision/src/mapika2b/contract.mts';
import {CACHE_NAME} from '../../packages/web-decision/src/assets.js';
import {
  MAPIKA2B_TEST_CACHE,
  type AcceptedMapikaCase,
  type MapikaBrowserResult,
  type MapikaReference
} from './mapika2b-evidence.mts';
import {readAdapter, readHardwareReport} from './mapika2b-hardware.mts';
import {verifyInstalledMapikaCandidate} from './mapika2b-installed.mts';
import {routeMapikaSnapshot} from './mapika2b-snapshot.mts';

const test = base.extend({
  context: async ({browserName}, use, testInfo) => {
    const profile = await mkdtemp(join(tmpdir(), `web-decision-mapika2b-offline-${browserName}-`));
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

test.skip(
  process.env.WEB_DECISION_REAL_GPU !== '1' ||
    process.env.WEB_DECISION_MAPIKA2B_GPU !== '1' ||
    process.env.WEB_DECISION_MAPIKA2B_OFFLINE_GPU !== '1',
  'Enable WEB_DECISION_REAL_GPU, WEB_DECISION_MAPIKA2B_GPU and WEB_DECISION_MAPIKA2B_OFFLINE_GPU for the full cache proof.'
);
test.skip(
  ({browserName}) => browserName !== 'chromium',
  'This candidate harness verifies physical GPU through Chromium.'
);

async function readCases() {
  const prepared = new URL('../../scripts/mapika/reference-prepared.json', import.meta.url);
  const fixtures = JSON.parse(await readFile(prepared, 'utf8')) as MapikaReference<AcceptedMapikaCase>;
  expect(fixtures.source.revision).toBe(MAPIKA2B_REVISION);
  return ['typed-packed', 'image-single'].map((id) => {
    const item = fixtures.cases.find((fixture) => fixture.id === id);
    if (!item || item.status !== 'accepted') throw new Error(`Missing Mapika offline fixture: ${id}.`);
    return item;
  });
}

type CaseResult = Readonly<{id: string; observed: MapikaBrowserResult}>;
type BootstrapRequest = Readonly<{url: string; resourceType: string}>;

async function scoreCases(
  page: Page,
  cases: readonly AcceptedMapikaCase[],
  results: CaseResult[],
  save: () => Promise<void>,
  log: (message: string) => void
) {
  for (const item of cases) {
    log(`Scoring ${item.id}.`);
    const observed = await page.evaluate((fixture) => window.mapika2bGpuHarness!.evaluate(fixture), item);
    results.push({id: item.id, observed});
    await save();
    if (observed.status !== 'accepted') throw new Error(`${item.id} failed: ${observed.message}`);
  }
}

async function startPage(page: Page) {
  await page.goto('/mapika2b-harness.html');
  await page.waitForFunction(() => !!window.mapika2bGpuHarness);
}

async function cacheEvidence(page: Page) {
  return page.evaluate(async (cacheName) => {
    const cache = await caches.open(cacheName);
    const keys = await cache.keys();
    const marker = keys.find(({url}) => new URL(url).pathname.endsWith('/mapika-2b-v12-offline-complete-v1.json'));
    const completion = marker ? await (await cache.match(marker))?.json() : null;
    const storage = await navigator.storage.estimate();
    return {entries: keys.length, completion, storage};
  }, MAPIKA2B_TEST_CACHE);
}

test('Mapika 2B v12 builds an online cache and scores on a fresh offline page', async ({page, context}, testInfo) => {
  test.setTimeout(30 * 60 * 1000);
  const archive = await verifyInstalledMapikaCandidate();
  if (!archive && !process.env.WEB_DECISION_PUBLIC_ASSET_ROOT)
    throw new Error('WEB_DECISION_PUBLIC_ASSET_ROOT must contain the Mapika text artifacts.');
  const cases = await readCases();
  const snapshot = process.env.WEB_DECISION_MAPIKA2B_SNAPSHOT;
  const installedArchive = archive !== null;
  const evidence = {
    schema: 'mapika2b-v12-offline-webgpu-v1',
    candidatePackage: archive,
    sourceRevision: MAPIKA2B_REVISION,
    source: snapshot ? 'local snapshot' : 'pinned Hugging Face URLs',
    hardware: null as Awaited<ReturnType<typeof readHardwareReport>> | null,
    adapter: null as Awaited<ReturnType<typeof readAdapter>>,
    cache: null as Awaited<ReturnType<typeof cacheEvidence>> | null,
    storageBeforeFill: null as StorageEstimate | null,
    wasmSha256: null as string | null,
    online: [] as CaseResult[],
    offline: [] as CaseResult[],
    bootstrapRequests: [] as BootstrapRequest[],
    requestsDuringOffline: [] as string[]
  };
  const output = testInfo.outputPath('mapika2b-offline-webgpu.json');
  await mkdir(dirname(output), {recursive: true});
  const save = () => writeFile(output, JSON.stringify(evidence, null, 2));
  const log = (message: string) => console.log(`${new Date().toISOString()} ${message}`);

  let offlinePage: Page | undefined;
  try {
    evidence.hardware = await readHardwareReport(page);
    expect(evidence.hardware.status).toBe('Hardware accelerated');
    if (snapshot) await routeMapikaSnapshot(page, snapshot, log);
    await startPage(page);
    evidence.adapter = await readAdapter(page);
    expect(evidence.adapter?.isFallbackAdapter).toBe(false);
    evidence.storageBeforeFill = await page.evaluate(() => navigator.storage.estimate());
    await save();

    log('Filling the complete verified WebGPU checkpoint cache.');
    const {wasmSha256} = await page.evaluate(
      (installed) => window.mapika2bGpuHarness!.initialize(installed, 'fill'),
      installedArchive
    );
    evidence.wasmSha256 = wasmSha256;
    evidence.cache = await cacheEvidence(page);
    log(`Cache populated with ${evidence.cache.entries} entries.`);
    await save();
    expect(evidence.cache.completion).toMatchObject({
      format: 1,
      markerId: expect.stringMatching(/^mapika-2b-v12:[a-f0-9]{64}$/)
    });
    await scoreCases(page, cases, evidence.online, save, log);
    await page.evaluate(() => window.mapika2bGpuHarness!.dispose());
    await page.close();
    await save();

    offlinePage = await context.newPage();
    const recordBootstrap = (request: Request) =>
      evidence.bootstrapRequests.push({url: request.url(), resourceType: request.resourceType()});
    offlinePage.on('request', recordBootstrap);
    await startPage(offlinePage);
    await offlinePage.evaluate(() => window.mapika2bGpuHarness!.stage());
    offlinePage.off('request', recordBootstrap);
    expect(evidence.bootstrapRequests.filter(({resourceType}) => ['fetch', 'xhr'].includes(resourceType))).toEqual([]);
    await context.setOffline(true);
    offlinePage.on('request', (request) => evidence.requestsDuringOffline.push(request.url()));
    log('Network disabled. Loading from CacheStorage and scoring the same text and image requests.');
    expect(await offlinePage.evaluate(() => navigator.onLine)).toBe(false);
    const offline = await offlinePage.evaluate(
      (installed) => window.mapika2bGpuHarness!.initialize(installed, 'offline'),
      installedArchive
    );
    expect(offline.wasmSha256).toBe(wasmSha256);
    await scoreCases(offlinePage, cases, evidence.offline, save, log);
    expect(evidence.offline).toEqual(evidence.online);
    expect(evidence.requestsDuringOffline).toEqual([]);
  } finally {
    const active = offlinePage ?? page;
    try {
      if (!active.isClosed())
        await active.evaluate(async (cacheName) => {
          await window.mapika2bGpuHarness?.dispose();
          await caches.delete(cacheName);
        }, MAPIKA2B_TEST_CACHE);
    } finally {
      await save();
      await testInfo.attach('mapika2b-offline-webgpu.json', {path: output, contentType: 'application/json'});
    }
  }
});

test('curated Mapika adapter scores a cached image without network access', async ({page, context}, testInfo) => {
  test.setTimeout(30 * 60 * 1000);
  const archive = await verifyInstalledMapikaCandidate();
  if (!archive) throw new Error('The curated offline check requires an installed archive.');
  const snapshot = process.env.WEB_DECISION_MAPIKA2B_SNAPSHOT;
  if (snapshot) await routeMapikaSnapshot(page, snapshot, () => {});
  const [, image] = await readCases();
  if (!image) throw new Error('The Mapika image fixture is missing.');
  expect((await readHardwareReport(page)).status).toBe('Hardware accelerated');
  await startPage(page);
  expect((await readAdapter(page))?.isFallbackAdapter).toBe(false);
  const online = await page.evaluate((item) => window.mapika2bGpuHarness!.curated(item, 'fill'), image);
  await page.close();
  const offlinePage = await context.newPage();
  try {
    await startPage(offlinePage);
    await offlinePage.evaluate(() => window.mapika2bGpuHarness!.stage());
    const requests: string[] = [];
    offlinePage.on('request', (request) => requests.push(request.url()));
    await context.setOffline(true);
    const offline = await offlinePage.evaluate((item) => window.mapika2bGpuHarness!.curated(item, 'offline'), image);
    expect(offline.result).toEqual(online.result);
    expect(requests).toEqual([]);
    const evidence = {archiveSha256: archive.archiveSha256, online: online.result, offline: offline.result, requests};
    if (process.env.WEB_DECISION_MAPIKA2B_CURATED_OFFLINE_REPORT)
      await writeFile(
        process.env.WEB_DECISION_MAPIKA2B_CURATED_OFFLINE_REPORT,
        JSON.stringify(evidence, null, 2) + '\n'
      );
    await testInfo.attach('curated-offline-image.json', {
      body: JSON.stringify(evidence),
      contentType: 'application/json'
    });
  } finally {
    await offlinePage.evaluate((name) => caches.delete(name), CACHE_NAME);
  }
});
