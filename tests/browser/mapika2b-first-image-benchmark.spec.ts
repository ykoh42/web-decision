import {readFile, realpath, writeFile} from 'node:fs/promises';
import {expect, test, type Page, type Request} from '@playwright/test';
import inventory from '../../packages/web-decision/assets/mapika2b/source-inventory.json' with {type: 'json'};
import {MAPIKA2B_REPOSITORY, MAPIKA2B_REVISION} from '../../packages/web-decision/src/mapika2b/contract.mts';
import {archiveIdentity} from '../../scripts/release/curated-gpu-evidence.mts';
import {verifyArchiveInstall} from '../../scripts/release/verify-release-evidence.ts';
import {
  mapikaReferenceMismatches,
  type AcceptedMapikaCase,
  type MapikaAnswerReference,
  type MapikaReference
} from './mapika2b-evidence.mts';
import {readAdapter, readHardwareReport} from './mapika2b-hardware.mts';
import {routeMapikaSnapshot} from './mapika2b-snapshot.mts';

const modelId = 'mapika-decider-2b-v12';
const caseId = 'image-single';
const upstreamVision = new URL(
  `https://huggingface.co/${MAPIKA2B_REPOSITORY}/resolve/${MAPIKA2B_REVISION}/${inventory.weights.vision.file}`
);
const redirects = new Set([301, 302, 303, 307, 308]);

function originalRequest(request: Request): Request {
  let original = request;
  for (let redirect = original.redirectedFrom(); redirect; redirect = original.redirectedFrom()) original = redirect;
  return original;
}

function visionRangeBytes(rangeHeader: string | undefined): number {
  const range = /^bytes=(\d+)-(\d+)$/u.exec(rangeHeader ?? '');
  if (!range) return 0;
  const start = Number(range[1]);
  const end = Number(range[2]);
  const length = end - start + 1;
  return start <= end && end < inventory.weights.vision.bytes && length <= inventory.maxRangeBytes ? length : 0;
}

function observeVisionRequests(page: Page) {
  const active = new Set<Request>();
  const statuses = new Map<Request, number>();
  const limits = {ranges: 0, requestedBytes: 0, completedRanges: 0, maxInFlight: 0};
  const unbounded: string[] = [];
  const failed: string[] = [];
  page.on('request', (request) => {
    const url = new URL(request.url());
    if (request.method() !== 'GET' || url.origin !== upstreamVision.origin || url.pathname !== upstreamVision.pathname)
      return;
    const range = request.headers().range;
    const bytes = visionRangeBytes(range);
    if (!bytes) unbounded.push(range ?? '<missing>');
    else limits.ranges++;
    limits.requestedBytes += bytes;
    active.add(request);
    limits.maxInFlight = Math.max(limits.maxInFlight, active.size);
  });
  page.on('response', (response) => {
    const request = response.request();
    if (active.has(originalRequest(request))) statuses.set(request, response.status());
  });
  page.on('requestfinished', (request) => {
    const original = originalRequest(request);
    if (!active.has(original)) return;
    const status = statuses.get(request);
    statuses.delete(request);
    if (status !== undefined && redirects.has(status)) return;
    active.delete(original);
    if (status === 206) limits.completedRanges++;
    else failed.push(String(status));
  });
  page.on('requestfailed', (request) => {
    statuses.delete(request);
    if (active.delete(originalRequest(request))) failed.push(request.failure()?.errorText ?? 'Request failed');
  });
  return () => ({...limits, inFlight: active.size, unbounded: [...unbounded], failed: [...failed]});
}

function benchmarkSettings() {
  const snapshot = process.env.WEB_DECISION_MAPIKA2B_SNAPSHOT;
  const archive = process.env.WEB_DECISION_RELEASE_ARCHIVE;
  const packagePath = process.env.WEB_DECISION_RELEASE_PACKAGE_DIR;
  const bundle = process.env.WEB_DECISION_INSTALLED_BUNDLE;
  if (!snapshot || !archive || !packagePath || !bundle)
    throw new Error('Set the local text snapshot and exact archive, installed package and bundle paths.');
  const visionDelayMs = Number(process.env.WEB_DECISION_MAPIKA2B_BENCH_VISION_DELAY_MS ?? 0);
  const liveVision = process.env.WEB_DECISION_MAPIKA2B_BENCH_LIVE_VISION === '1';
  if (
    !Number.isSafeInteger(visionDelayMs) ||
    visionDelayMs < 0 ||
    visionDelayMs > 10_000 ||
    (liveVision && visionDelayMs)
  )
    throw new Error('Set a vision delay from 0 to 10000 ms for local vision, or enable live vision without a delay.');
  return {snapshot, archive, packagePath, bundle, visionDelayMs, liveVision};
}

async function loadBenchmarkCase() {
  const root = new URL('../../scripts/mapika/', import.meta.url);
  const prepared: MapikaReference<AcceptedMapikaCase> = JSON.parse(
    await readFile(new URL('reference-fp32-prepared.json', root), 'utf8')
  );
  const answers: MapikaReference<MapikaAnswerReference> = JSON.parse(
    await readFile(new URL('reference-fp32-answers.json', root), 'utf8')
  );
  const imageCase = prepared.cases.find(({id}) => id === caseId);
  const expected = answers.cases.find(({id}) => id === caseId)?.result;
  if (!imageCase || !expected) throw new Error('The Mapika image reference case is missing.');
  return {request: imageCase.request, expected};
}

test('benchmarks the exact installed Mapika package before and after its first image', async ({page}, testInfo) => {
  test.skip(process.env.WEB_DECISION_MAPIKA2B_BENCH_GPU !== '1', 'Enable the installed Mapika image benchmark.');
  test.skip(testInfo.project.name !== 'chromium', 'This benchmark requires Chromium WebGPU.');
  test.setTimeout(30 * 60 * 1000);
  const settings = benchmarkSettings();
  const archive = await verifyArchiveInstall(settings.archive, settings.packagePath);
  expect(archive.bundlePath).toBe(await realpath(settings.bundle));
  const {request, expected} = await loadBenchmarkCase();
  const routeErrors: string[] = [];
  await routeMapikaSnapshot(page, settings.snapshot, (error) => routeErrors.push(error), settings);
  const vision = observeVisionRequests(page);
  const hardware = await readHardwareReport(page);
  expect(hardware.status).toBe('Hardware accelerated');
  await page.goto('/');
  await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');
  const adapter = await readAdapter(page);
  expect(adapter?.isFallbackAdapter).toBe(false);
  const stages = new Map<string, ReturnType<typeof vision>>();
  await page.exposeFunction('recordMapikaImageBenchmark', (name: string) => {
    stages.set(name, vision());
  });
  const observed = await page.evaluate(
    async ({id, request}) => {
      const {recordMapikaImageBenchmark} = window as Window & {
        recordMapikaImageBenchmark?: (name: string) => Promise<void>;
      };
      const start = performance.now();
      const model = await window.webDecision.loadDecisionModel({modelId: id, cache: false});
      const loadMs = performance.now() - start;
      async function imageDecision() {
        const imageStart = performance.now();
        const result = await model.systemOne(request as Parameters<typeof model.systemOne>[0]);
        return {ms: performance.now() - imageStart, result: JSON.parse(JSON.stringify(result))};
      }
      try {
        await recordMapikaImageBenchmark?.('afterLoad');
        const first = await imageDecision();
        await recordMapikaImageBenchmark?.('afterFirstImage');
        const repeat = await imageDecision();
        return {loadMs, first, repeat};
      } finally {
        await model.dispose();
      }
    },
    {id: modelId, request} as const
  );
  await expect.poll(() => vision().inFlight).toBe(0);
  const afterLoad = stages.get('afterLoad');
  const afterFirstImage = stages.get('afterFirstImage');
  const afterRepeat = vision();
  expect(routeErrors).toEqual([]);
  expect(afterLoad?.ranges).toBe(0);
  expect(afterFirstImage?.ranges).toBeGreaterThan(0);
  expect(afterRepeat.ranges).toBe(afterFirstImage?.ranges);
  expect(afterRepeat.requestedBytes).toBe(afterFirstImage?.requestedBytes);
  expect(afterRepeat.completedRanges + afterRepeat.failed.length).toBe(afterRepeat.ranges);
  expect(afterRepeat.unbounded).toEqual([]);
  if (!settings.liveVision) expect(afterRepeat.failed).toEqual([]);
  expect(mapikaReferenceMismatches(observed.first.result, expected)).toEqual([]);
  expect(observed.repeat.result).toEqual(observed.first.result);
  const report = JSON.stringify({
    archive: archiveIdentity(archive),
    modelId,
    caseId,
    source: settings.liveVision ? 'local-text/live-pinned-vision' : 'local-text/local-vision',
    sourceRevision: MAPIKA2B_REVISION,
    visionDelayMs: settings.visionDelayMs,
    cache: false,
    hardware,
    adapter,
    browser: {userAgent: await page.evaluate(() => navigator.userAgent)},
    timings: {loadMs: observed.loadMs, firstImageMs: observed.first.ms, repeatImageMs: observed.repeat.ms},
    vision: {afterLoad, afterFirstImage, afterRepeat},
    result: observed.first.result
  });
  console.log(`MAPIKA_FIRST_IMAGE_BENCHMARK:${report}`);
  const reportPath = process.env.WEB_DECISION_MAPIKA2B_BENCH_REPORT;
  if (reportPath) await writeFile(reportPath, `${report}\n`);
  await testInfo.attach('mapika-first-image-benchmark.json', {body: report, contentType: 'application/json'});
});
