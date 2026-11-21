import {createHash} from 'node:crypto';
import {readFile, realpath} from 'node:fs/promises';
import {expect, test} from '@playwright/test';
import {verifyArchiveInstall} from '../../scripts/release/verify-release-evidence.ts';
import type {Mapika2bResult} from '../../packages/web-decision/src/mapika2b/readout.mts';
import {
  MAPIKA2B_DECIDER_REVISION,
  MAPIKA2B_REPOSITORY,
  MAPIKA2B_REVISION
} from '../../packages/web-decision/src/mapika2b/contract.mts';
import {
  mapikaReferenceMismatches,
  requestFromMapikaFixture,
  type AcceptedMapikaCase,
  type MapikaAnswerReference,
  type MapikaReference
} from './mapika2b-evidence.mts';
import {readAdapter, readHardwareReport} from './mapika2b-hardware.mts';
import {routeMapikaSnapshot} from './mapika2b-snapshot.mts';

const modelId = 'mapika-decider-2b-v12';

test('the installed public package loads Mapika by model ID and matches the source', async ({
  page,
  browser
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
  const cases = ['typed-packed', 'image-single'].map((id) => {
    const input = prepared.cases.find((item) => item.id === id);
    const expected = answers.cases.find((item) => item.id === id);
    if (!input || !expected) throw new Error('Missing public Mapika reference case: ' + id);
    return {id, request: requestFromMapikaFixture(input), expected: expected.result};
  });
  const snapshot = process.env.WEB_DECISION_MAPIKA2B_SNAPSHOT;
  if (snapshot) await routeMapikaSnapshot(page, snapshot, () => {});
  const hardware = await readHardwareReport(page);
  expect(hardware.status).toBe('Hardware accelerated');
  await page.goto('/');
  await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');
  const adapter = await readAdapter(page);
  expect(adapter?.isFallbackAdapter).toBe(false);
  const observed = await page.evaluate(
    async ({id, inputs}) => {
      const api = window.webDecision as unknown as {
        listModels(): readonly {modelId: string}[];
        describeModel(id: string): {modelId: string};
        loadDecisionModel(options: {modelId: string; cache: boolean}): Promise<{
          modelId: string;
          systemOne(request: unknown): Promise<Mapika2bResult>;
          dispose(): Promise<void>;
        }>;
      };
      if (!api.listModels().some((item) => item.modelId === id) || api.describeModel(id).modelId !== id)
        throw new Error('The public catalog does not expose Mapika.');
      const model = await api.loadDecisionModel({modelId: id, cache: false});
      try {
        const results = [];
        for (const input of inputs)
          results.push(JSON.parse(JSON.stringify(await model.systemOne(input))) as Mapika2bResult);
        return {modelId: model.modelId, results};
      } finally {
        await model.dispose();
      }
    },
    {id: modelId, inputs: cases.map(({request}) => request)}
  );
  expect(observed.modelId).toBe(modelId);
  for (const [index, item] of cases.entries())
    expect(mapikaReferenceMismatches(observed.results[index], item.expected), item.id).toEqual([]);
  await testInfo.attach('mapika-public-evidence.json', {
    body: JSON.stringify({
      archive: {
        sha256: archive.archiveSha256,
        packageTreeSha256: archive.packageTreeSha256,
        bundleSha256: archive.bundleSha256,
        packageName: archive.packageName,
        packageVersion: archive.packageVersion
      },
      references: {
        preparedSha256: createHash('sha256').update(preparedBytes).digest('hex'),
        answersSha256: createHash('sha256').update(answerBytes).digest('hex')
      },
      browser: {name: testInfo.project.name, version: browser.version()},
      hardware,
      adapter,
      modelId,
      results: observed.results
    }),
    contentType: 'application/json'
  });
});
