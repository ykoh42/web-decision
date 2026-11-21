import {readFile, writeFile} from 'node:fs/promises';
import {expect, test} from '@playwright/test';
import type {AcceptedMapikaCase, MapikaReference} from './mapika2b-evidence.mts';
import {readAdapter, readHardwareReport} from './mapika2b-hardware.mts';
import {verifyInstalledMapikaCandidate} from './mapika2b-installed.mts';
import {routeMapikaSnapshot} from './mapika2b-snapshot.mts';

test.skip(
  process.env.WEB_DECISION_REAL_GPU !== '1' || process.env.WEB_DECISION_MAPIKA2B_GPU !== '1',
  'Set WEB_DECISION_REAL_GPU and WEB_DECISION_MAPIKA2B_GPU to verify the installed candidate.'
);
test.skip(({browserName}) => browserName !== 'chromium', 'This test checks hardware acceleration in Chromium.');

test('installed curated adapter loads pinned artifacts and answers a real request', async ({page}) => {
  test.setTimeout(60 * 60 * 1000);
  const installed = await verifyInstalledMapikaCandidate();
  if (!installed) throw new Error('The curated adapter test requires an installed candidate archive.');
  const snapshot = process.env.WEB_DECISION_MAPIKA2B_SNAPSHOT;
  if (snapshot) await routeMapikaSnapshot(page, snapshot, () => {});
  const report = JSON.parse(
    await readFile(new URL('../../scripts/mapika/reference-prepared.json', import.meta.url), 'utf8')
  ) as MapikaReference<AcceptedMapikaCase>;
  const item = report.cases.find(({id}) => id === 'typed-packed');
  if (!item) throw new Error('The pinned Mapika request is missing.');
  expect((await readHardwareReport(page)).status).toBe('Hardware accelerated');
  await page.goto('/mapika2b-harness.html');
  await page.waitForFunction(() => !!window.mapika2bGpuHarness);
  expect((await readAdapter(page))?.isFallbackAdapter).toBe(false);
  const {result, progress} = await page.evaluate((request) => window.mapika2bGpuHarness!.curated(request), item);
  expect(progress.length).toBe(320);
  expect(progress.at(-1)?.fraction).toBe(1);
  expect(progress.at(-1)?.loadedBytes).toBe(progress.at(-1)?.totalBytes);
  expect(result.model).toBe('decider-2b-v11');
  expect(result.usage).toEqual({input_tokens: 260, output_tokens: 0});
  expect(result.answers['10']).toMatchObject({type: 'choice', choice: '10'});
  expect(result.answers['2']).toMatchObject({type: 'noul', noul: 1});
  expect(result.answers.rating).toMatchObject({type: 'score', score: 0.83});
});

test('installed curated adapter rejects modified packaged graph bytes', async ({page}) => {
  const installed = await verifyInstalledMapikaCandidate();
  if (!installed) throw new Error('The curated adapter test requires an installed candidate archive.');
  await page.route('**/text.wasm', (route) => route.fulfill({status: 200, body: 'modified'}));
  const report = JSON.parse(
    await readFile(new URL('../../scripts/mapika/reference-prepared.json', import.meta.url), 'utf8')
  ) as MapikaReference<AcceptedMapikaCase>;
  const item = report.cases.find(({id}) => id === 'typed-packed');
  if (!item) throw new Error('The pinned Mapika request is missing.');
  await page.goto('/mapika2b-harness.html');
  await page.waitForFunction(() => !!window.mapika2bGpuHarness);
  const failure = await page.evaluate(async (fixture) => {
    try {
      await window.mapika2bGpuHarness!.curated(fixture);
      return null;
    } catch (cause) {
      return {code: (cause as {code?: unknown}).code, message: String(cause)};
    }
  }, item);
  expect(failure).toMatchObject({code: 'INVALID_ASSET'});
  expect(failure?.message).toMatch(/integrity check failed/iu);
});

test('installed curated adapter scores a long state', async ({page}, testInfo) => {
  test.skip(
    process.env.WEB_DECISION_MAPIKA2B_LONG_GPU !== '1',
    'Set WEB_DECISION_MAPIKA2B_LONG_GPU for the long-state test.'
  );
  test.setTimeout(90 * 60 * 1000);
  const installed = await verifyInstalledMapikaCandidate();
  if (!installed) throw new Error('The curated adapter test requires an installed candidate archive.');
  const snapshot = process.env.WEB_DECISION_MAPIKA2B_SNAPSHOT;
  if (snapshot) await routeMapikaSnapshot(page, snapshot, () => {});
  const report = JSON.parse(
    await readFile(new URL('../../scripts/mapika/reference-prepared.json', import.meta.url), 'utf8')
  ) as MapikaReference<AcceptedMapikaCase>;
  const item = report.cases.find(({id}) => id === 'typed-packed');
  if (!item) throw new Error('The pinned Mapika request is missing.');
  expect((await readHardwareReport(page)).status).toBe('Hardware accelerated');
  await page.goto('/mapika2b-harness.html');
  await page.waitForFunction(() => !!window.mapika2bGpuHarness);
  expect((await readAdapter(page))?.isFallbackAdapter).toBe(false);
  const repetitions = Number(process.env.WEB_DECISION_MAPIKA2B_LONG_STATE_TOKENS ?? 8192);
  if (!Number.isSafeInteger(repetitions) || repetitions < 1 || repetitions > 32765)
    throw new Error('The Mapika long-state fixture must contain between 1 and 32765 repetitions.');
  const request = {...item, request: {...item.request, state: ' a'.repeat(repetitions)}};
  const start = Date.now();
  const {result} = await page.evaluate((fixture) => window.mapika2bGpuHarness!.curated(fixture), request);
  const evidence = JSON.stringify({archiveSha256: installed.archiveSha256, elapsedMs: Date.now() - start, result});
  if (process.env.WEB_DECISION_MAPIKA2B_LONG_REPORT)
    await writeFile(process.env.WEB_DECISION_MAPIKA2B_LONG_REPORT, evidence + '\n');
  await testInfo.attach('long-state-result.json', {
    body: evidence,
    contentType: 'application/json'
  });
  expect(result.usage.input_tokens).toBeGreaterThanOrEqual(repetitions);
  expect(result.answers['10']?.type).toBe('choice');
  expect(result.answers['2']?.type).toBe('noul');
  expect(result.answers.rating?.type).toBe('score');
});
