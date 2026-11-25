import assert from 'node:assert/strict';
import {corpusTitle, gemmaSplitCandidate} from './strands-gemma4-v2-split-candidate.mjs';

export const persistentTitle =
  'the exact bfa9922b archive records the official subset and five cases in a persistent Chromium profile';
export const persistentConfig = `import {defineConfig} from '@playwright/test';
import base from './checkout/playwright.config.ts';
const {globalTeardown: _teardown, webServer: _server, ...browserConfig} = base;
export default defineConfig({
  ...browserConfig,
  testDir: '.',
  testMatch: 'persistent-public-archive.spec.ts',
  workers: 1,
  retries: 0,
  projects: [{name: 'chromium'}],
  use: {...base.use, baseURL: 'http://127.0.0.1:42713', serviceWorkers: 'block'}
});
`;
export const warmTitle =
  'the exact bfa9922b archive records the official subset and five cases from a rehashed offline seed';
export const warmConfig = persistentConfig.replace(
  'persistent-public-archive.spec.ts',
  'persistent-warm-archive.spec.ts'
);

export function derivePersistentSpec(standard: string) {
  let output = standard;
  const once = (source: string, replacement: string) => {
    assert.equal(output.split(source).length, 2, 'Missing unique producer marker: ' + source);
    output = output.replace(source, replacement);
  };
  once(
    "import {readFile, realpath} from 'node:fs/promises';",
    "import {mkdtemp, readFile, realpath} from 'node:fs/promises';"
  );
  once(
    "import {expect, test, type Page, type Response} from '@playwright/test';",
    "import {expect, test as baseTest, type Page, type Response} from '@playwright/test';"
  );
  once(
    'import lock from',
    "import {checkedInventory, checkpointUrl, observedCheckpointRange, requiredOfflineWeightBytes, routeVerifiedLocalCheckpoint} from './local-checkpoint.mjs';\nimport lock from"
  );
  once(
    "test.use({serviceWorkers: 'block'});",
    `const test = baseTest.extend({
  context: async ({playwright, browserName}, use, info) => {
    expect(browserName).toBe('chromium');
    const {launchOptions, channel, headless, baseURL} = info.project.use;
    const profileRoot = await realpath(requiredEnv('WEB_DECISION_PERSISTENT_PROFILE_ROOT'));
    const profile = await mkdtemp(join(profileRoot, 'chromium-'));
    expect(await realpath(profile)).toBe(profile);
    const context = await playwright.chromium.launchPersistentContext(profile, {
      ...launchOptions,
      ...(channel === undefined ? {} : {channel}),
      ...(headless === undefined ? {} : {headless}),
      ...(baseURL === undefined ? {} : {baseURL}),
      serviceWorkers: 'block'
    });
    try {
      await use(context);
    } finally {
      await context.close();
    }
  }
});`
  );
  once(
    'test(' + JSON.stringify(corpusTitle) + ', async ({page, browser}, info) => {',
    'test(' + JSON.stringify(persistentTitle) + ', async ({page}, info) => {'
  );
  once(
    '  const sourceCommit = cleanSourceCommit(root);',
    `  const sourceCommit = cleanSourceCommit(root);
  expect(sourceCommit).toBe('${gemmaSplitCandidate.sourceCommit}');
  const browser = page.context().browser();
  if (!browser) throw new Error('Persistent Chromium browser identity unavailable.');
  const checkpointMode = requiredEnv('WEB_DECISION_PERSISTENT_CHECKPOINT_MODE');
  expect(['online', 'local']).toContain(checkpointMode);
  let localCheckpoint: Awaited<ReturnType<typeof routeVerifiedLocalCheckpoint>> | undefined;`
  );
  once(
    "  await page.goto(exactPage?.url() ?? '/');",
    `  expect(exactPage).toBeDefined();
  const checkpointInventory = await checkedInventory(verified.archive.installedRoot);
  try {
    await page.goto(exactPage!.url());
    const storageBefore = await page.evaluate(() => navigator.storage.estimate());
    expect(storageBefore.quota ?? 0).toBeGreaterThanOrEqual(requiredOfflineWeightBytes);
    if (checkpointMode === 'local')
      localCheckpoint = await routeVerifiedLocalCheckpoint(page.context(), requiredEnv('WEB_DECISION_LOCAL_CHECKPOINT'), checkpointInventory);`
  );
  once(
    '  );\n  const upstreamResponse = page.waitForResponse(isPinnedRangeResponse, {timeout: 600_000});',
    `  ).then(async (response) => {
    expect(sha256(await response.body())).toBe(graphPin.sha256);
  });
  const upstreamResponse = page.waitForResponse(isPinnedRangeResponse, {timeout: 600_000})
    .then((response) => observedCheckpointRange(response, checkpointInventory));`
  );
  once(
    `  try {
    const [receivedGraph, receivedUpstream, {cases, results}] = await Promise.all([
      graphResponse,
      upstreamResponse,
      load.then(() => collectCases(page, references))
    ]);
    expect(sha256(await receivedGraph.body())).toBe(graphPin.sha256);
    expect(receivedUpstream.request().headers().range).toMatch(/^bytes=\\d+-\\d+$/u);`,
    `    const [, checkpointRange] = await Promise.all([graphResponse, upstreamResponse, load]);
    expect(checkpointRange.url).toBe(checkpointUrl);
    const {cases, results} = await collectCases(page, references);`
  );
  once(
    "    const single = references['image-single']!;",
    `    expect(rawEvidence).toBeDefined();
    expect(metrics.questions).toBe(1066);
    expect(metrics.meetsUpstreamMetricThresholds).toBe(true);
    const single = references['image-single']!;`
  );
  once(
    '    const offline = await checkOffline(page, single, singleResult);',
    `    const offline = await checkOffline(page, single, singleResult);
    const storageAfterOffline = await page.evaluate(() => navigator.storage.estimate());
    const checkpointTransport = {
      mode: checkpointMode === 'local' ? 'verified-local-checkpoint-http-range' : 'pinned-public-upstream-http-range',
      wholeCheckpointOnlineDownloadVerified: false,
      sourceUrl: checkpointUrl,
      observation: 'model-load-range',
      firstObservedRange: checkpointRange,
      ...(localCheckpoint ? {localRoute: localCheckpoint.evidence()} : {})
    };`
  );
  once(
    "      caseMode: 'five-cases-external-32k-required',",
    `      caseMode: 'five-cases-external-32k-required',
      storage: {
        mode: 'persistent-chromium-profile',
        requiredOfflineWeightBytes,
        graphObservation: 'model-load-response',
        before: storageBefore,
        afterOffline: storageAfterOffline
      },
      checkpointTransport,`
  );
  once(
    '    await closeSession(page);\n  }\n});',
    `    try {
      await closeSession(page);
    } finally {
      await localCheckpoint?.close();
    }
  }
});`
  );
  return output;
}

export function deriveWarmSpec(standard: string) {
  let output = derivePersistentSpec(standard);
  const once = (source: string, replacement: string) => {
    assert.equal(output.split(source).length, 2, 'Missing unique warm producer marker: ' + source);
    output = output.replace(source, replacement);
  };
  once('{mkdtemp, readFile, realpath}', '{readFile, realpath}');
  once(
    "import {checkedInventory, checkpointUrl, observedCheckpointRange, requiredOfflineWeightBytes, routeVerifiedLocalCheckpoint} from './local-checkpoint.mjs';",
    `import {checkedInventory, checkpointUrl, requiredOfflineWeightBytes} from './local-checkpoint.mjs';
import {assertClosedPrivateProfile, blockWarmRemote, probePinnedPublicRange, warmOrigin, warmProfileName, warmRehash, type WarmClone} from './verifier/scripts/release/strands-gemma4-v2-warm-seed.mjs';`
  );
  once(
    "const profile = await mkdtemp(join(profileRoot, 'chromium-'));\n    expect(await realpath(profile)).toBe(profile);",
    'const profile = join(profileRoot, warmProfileName);\n    expect(baseURL).toBe(warmOrigin);\n    await assertClosedPrivateProfile(profile);'
  );
  once(JSON.stringify(persistentTitle), JSON.stringify(warmTitle));
  once(
    `  const checkpointMode = requiredEnv('WEB_DECISION_PERSISTENT_CHECKPOINT_MODE');
  expect(['online', 'local']).toContain(checkpointMode);
  let localCheckpoint: Awaited<ReturnType<typeof routeVerifiedLocalCheckpoint>> | undefined;`,
    `  expect(requiredEnv('WEB_DECISION_PERSISTENT_CHECKPOINT_MODE')).toBe('warm');
  const admission = JSON.parse(await readFile(requiredEnv('WEB_DECISION_WARM_STARTED'), 'utf8')) as {warmSeed: WarmClone};
  const {copiedProfilePathSha256, ...warmSeed} = admission.warmSeed;
  const profile = join(await realpath(requiredEnv('WEB_DECISION_PERSISTENT_PROFILE_ROOT')), warmProfileName);
  expect(copiedProfilePathSha256).toBe(sha256(Buffer.from(profile)));`
  );
  const probeStart = output.indexOf('function isPinnedRangeResponse');
  const probeEnd = output.indexOf('type CorpusRequests', probeStart);
  assert.ok(probeStart > 0 && probeEnd > probeStart);
  output = output.slice(0, probeStart) + output.slice(probeEnd);
  once('type Page, type Response', 'type Page');
  once(
    '  const exactPage = rawMode',
    `  const browserRemoteAttempts = await blockWarmRemote(page.context(), origin);
  await page.context().setOffline(true);
  const exactPage = rawMode`
  );
  once(
    `    if (checkpointMode === 'local')
      localCheckpoint = await routeVerifiedLocalCheckpoint(page.context(), requiredEnv('WEB_DECISION_LOCAL_CHECKPOINT'), checkpointInventory);`,
    `    const checkpointRange = await probePinnedPublicRange(checkpointInventory);
    expect(checkpointRange.url).toBe(checkpointUrl);`
  );
  once(
    `  const upstreamResponse = page.waitForResponse(isPinnedRangeResponse, {timeout: 600_000})
    .then((response) => observedCheckpointRange(response, checkpointInventory));
  const load = page.evaluate(async (id) => {
    const api = window.webDecision as unknown as Api;
    window.gemmaV2ReleaseSession = await api.loadDecisionModel({modelId: id, cache: true});
  }, modelId);
    const [, checkpointRange] = await Promise.all([graphResponse, upstreamResponse, load]);
    expect(checkpointRange.url).toBe(checkpointUrl);`,
    `    const packagedResources = ['graph.wasm', 'local.bin', 'global.bin', 'scoring.bin'];
    await Promise.all([
      graphResponse,
      page.evaluate(async (urls) => {
        for (const url of urls) {
          const response = await fetch(url, {cache: 'no-store'});
          if (!response.ok) throw new Error('Unable to read exact release asset.');
          await response.arrayBuffer();
        }
      }, packagedResources.map((resource) => exactPage!.url('dist/gemma4-v2/' + resource)))
    ]);
    const rehash = await warmRehash(page, modelId, checkpointInventory);`
  );
  once(
    "mode: checkpointMode === 'local' ? 'verified-local-checkpoint-http-range' : 'pinned-public-upstream-http-range',",
    "mode: 'pinned-public-upstream-http-range',"
  );
  once(
    "      observation: 'model-load-range',\n      firstObservedRange: checkpointRange,\n      ...(localCheckpoint ? {localRoute: localCheckpoint.evidence()} : {})",
    "      observation: 'independent-online-range-probe',\n      firstObservedRange: checkpointRange"
  );
  once("        graphObservation: 'model-load-response',", "        graphObservation: 'independent-archive-response',");
  once(
    '    const report = {',
    `    expect(browserRemoteAttempts).toEqual([]);
    const report = {
      warmSeed: {
        ...warmSeed, rehash, browserRemoteAttempts,
        copy: {mode: 'closed-profile-copy', copiedProfilePathSha256}
      },`
  );
  once(
    '    assertions: {checkpointRangeObserved: true, installedGraphResponseSha256: graphPin.sha256},',
    '    assertions: {independentPublicRangeObserved: true, installedGraphResponseSha256: graphPin.sha256},'
  );
  once(
    '    try {\n      await closeSession(page);\n    } finally {\n      await localCheckpoint?.close();\n    }',
    '    await closeSession(page);'
  );
  return output;
}
