import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createReadStream, readFileSync} from 'node:fs';
import {readFile, stat, writeFile} from 'node:fs/promises';
import {createServer} from 'node:https';
import {extname, resolve, sep} from 'node:path';
import {expect, test} from '@playwright/test';
import type {RepresentativeDecisionCase} from './kev06-expanded-inference.js';

const enabled = process.env.WEB_DECISION_KEV06_DIRECT_GPU === '1';
const DIRECT_ARTIFACT_ROOT = 'https://direct-kev-package-test.invalid/';
const EXPANDED_FIXTURE_SHA256 = '05e7f81d2ec5e8a306c0a8034253e3daa3983c088b8ccd9c29b5130b2eecbd41';
const LOGIT_TOLERANCE = 0.15;
const PROBABILITY_TOLERANCE = 0.001;
const PROGRESS_PREFIX = '__DIRECT_KEV_PROGRESS__';
const DIAGNOSTIC_STORAGE_KEY = 'direct-kev-r6-webgpu-diagnostics';
const fixtureBytes = readFileSync(new URL('../fixtures/kev06-expanded-inference.json', import.meta.url));
// Test-only self-signed localhost certificate; never use it outside this fixture.
const testTlsKey = readFileSync(new URL('../fixtures/direct-kev-test-key.pem.fixture', import.meta.url));
const testTlsCert = readFileSync(new URL('../fixtures/direct-kev-test-cert.pem.fixture', import.meta.url));
const fixtureSha256 = createHash('sha256').update(fixtureBytes).digest('hex');
const fixture = JSON.parse(fixtureBytes.toString('utf8')) as {
  readonly decisionCases: readonly RepresentativeDecisionCase[];
};

test.use({ignoreHTTPSErrors: true});

interface GateDecision {
  id: string;
  tokenCount: number;
  selected: string;
  probabilities: readonly Readonly<{option: string; logit: number; probability: number}>[];
}

interface GateOutput {
  adapter: Readonly<{
    info: Readonly<{vendor: string; architecture: string; device: string; description: string}>;
    isFallbackAdapter: boolean | null;
    limits: Readonly<{maxBufferSize: number; maxStorageBufferBindingSize: number}>;
  }>;
  nativeInputLimits: Readonly<{
    sourceRepository: string;
    sourceRevision: string;
    sourceFile: string;
    maxStateTokens: number;
    maxBranchTokens: number;
    maxPositionEmbeddings: number;
    aggregateLimit: null;
  }>;
  publicOptionsRejected: boolean;
  onlineFetchPaths: readonly string[];
  offlineFetchPaths: readonly string[];
  qualityIssues: readonly string[];
  online: readonly GateDecision[];
  cacheDiagnostics: Readonly<{
    cacheNames: readonly string[];
    namespace: string;
    cacheKeyCount: number;
    cacheKeyUrls: readonly string[];
    cachedAssetNames: readonly string[];
    missingAssetNames: readonly string[];
    cachePutFailures: readonly string[];
    storageEstimate: Readonly<{usage: number | null; quota: number | null}>;
  }>;
  boundaryProbe?: Readonly<{
    requestedTokenCount: number;
    acceptedTokenCount?: number;
    outputFinite?: boolean;
    probabilityTotal?: number;
    atLimitError?: string;
    overLimitErrorCode?: string;
    overLimitMessage?: string;
  }>;
  boundaryProbeError?: string;
  offlineBaseline: GateDecision;
}

interface DirectCacheDiagnostics {
  cacheNames: readonly string[];
  namespace: string;
  cacheKeyCount: number;
  cacheKeyUrls: readonly string[];
  cachedAssetNames: readonly string[];
  missingAssetNames: readonly string[];
  cachePutFailures: readonly string[];
  storageEstimate: Readonly<{usage: number | null; quota: number | null}>;
}

interface DefaultSourceSmokeOutput {
  onlineBaseline: GateDecision;
  offlineBaseline: GateDecision;
  cacheDiagnostics: DirectCacheDiagnostics;
}

interface BuildReport {
  status: string;
  kevCheckpointRevision: string;
  baseRevision: string;
  tvmLogits: readonly number[];
  selectedOptionIndexMatchesUpstreamQ4: boolean;
  tvmWebToolchain: {
    sourceRevision: string;
    sourceTag: string;
    librarySha256: Readonly<Record<string, string>>;
    emscriptenRuntimeDependency: Readonly<{
      package: string;
      packageVersion: string;
      soname: string;
      librarySha256: string;
    }>;
  };
  qualityGates: {
    cpuBaseline: {status: string};
    webgpuExpandedMatrix: {
      status: string;
      fixtureSha256: string;
      decisionCases: readonly Readonly<{id: string; tokenCount: number}>[];
      maxAbsoluteQ4LogitDifference: number;
      q4ProbabilityComparison?: string;
      outputsMustBeFinite?: boolean;
      probabilitiesMustSumToOne?: boolean;
      maxProbabilitySumError?: number;
      allSelectionsMustMatchPinnedQ4: boolean;
      baselineCpuGpuParityRequired: boolean;
    };
  };
  browserPackage: {manifestSha256: string; parameterCount: number; license: string};
}

interface PackageManifest {
  files: Readonly<Record<string, Readonly<{bytes: number; sha256: string}>>>;
  [key: string]: unknown;
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    return Object.fromEntries(
      Object.keys(record)
        .sort()
        .map((key) => [key, canonical(record[key])])
    );
  }
  return value;
}

function sha256(value: Uint8Array): string {
  return createHash('sha256').update(value).digest('hex');
}

function softmax(logits: readonly number[]): number[] {
  const maximum = Math.max(...logits);
  const values = logits.map((logit) => Math.exp(logit - maximum));
  const total = values.reduce((sum, value) => sum + value, 0);
  return values.map((value) => value / total);
}

function maxDifference(left: readonly number[], right: readonly number[]): number {
  assert.equal(left.length, right.length, 'Observed and reference vector lengths differ.');
  return Math.max(...left.map((value, index) => Math.abs(value - right[index]!)));
}

async function withPackageFileServer<T>(root: string, run: (origin: string) => Promise<T>): Promise<T> {
  const server = createServer({key: testTlsKey, cert: testTlsCert}, async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url ?? '/', 'http://localhost').pathname).replace(/^\/+/, '');
      const filePath = resolve(root, pathname);
      if (!filePath.startsWith(`${root}${sep}`)) {
        response.writeHead(400).end('Invalid package path.');
        return;
      }
      const info = await stat(filePath);
      if (!info.isFile()) {
        response.writeHead(404).end('Package file not found.');
        return;
      }
      response.writeHead(200, {
        'access-control-allow-origin': '*',
        'cache-control': 'no-store',
        'content-length': info.size,
        'content-type': contentType(filePath)
      });
      createReadStream(filePath).pipe(response);
    } catch {
      response.writeHead(404).end('Package file not found.');
    }
  });
  await new Promise<void>((resolveListen, rejectListen) => {
    server.once('error', rejectListen);
    server.listen(0, '127.0.0.1', resolveListen);
  });
  const address = server.address();
  if (!address || typeof address === 'string')
    throw new Error('The direct package file server did not bind a TCP port.');
  try {
    return await run(`https://127.0.0.1:${address.port}`);
  } finally {
    await new Promise<void>((resolveClose, rejectClose) => {
      server.close((error) => (error ? rejectClose(error) : resolveClose()));
    });
  }
}

function assertQ4Match(
  actual: GateDecision,
  expected: RepresentativeDecisionCase,
  label: string
): Readonly<{maxAbsLogitDelta: number; maxAbsProbabilityDelta: number}> {
  assert.equal(actual.id, expected.id, `${label} case order mismatch.`);
  assert.equal(actual.tokenCount, expected.tokenCount, `${label} token count mismatch for ${expected.id}.`);
  assert.deepEqual(
    actual.probabilities.map(({option}) => option),
    expected.reference.options,
    `${label} option order mismatch for ${expected.id}.`
  );
  assert.equal(
    actual.selected,
    expected.reference.selected,
    `${label} selected a different option for ${expected.id}.`
  );
  const maxAbsLogitDelta = maxDifference(
    actual.probabilities.map(({logit}) => logit),
    expected.reference.logits
  );
  const maxAbsProbabilityDelta = maxDifference(
    actual.probabilities.map(({probability}) => probability),
    expected.reference.probabilities
  );
  const logits = actual.probabilities.map(({logit}) => logit);
  const probabilities = actual.probabilities.map(({probability}) => probability);
  assert.ok([...logits, ...probabilities].every(Number.isFinite), `${label} output is non-finite for ${expected.id}.`);
  const probabilityTotal = probabilities.reduce((total, value) => total + value, 0);
  assert.ok(
    Math.abs(probabilityTotal - 1) <= 0.000001,
    `${label} probabilities are not normalized for ${expected.id}: ${probabilityTotal}.`
  );
  assert.ok(
    maxAbsLogitDelta <= LOGIT_TOLERANCE,
    `${label} logits exceed Q4 tolerance for ${expected.id}: ${maxAbsLogitDelta}.`
  );
  return {maxAbsLogitDelta, maxAbsProbabilityDelta};
}

let activeBuildRoot: string | undefined;
let activeProgressEvents: unknown[] = [];
let activeNetworkRequests: string[] = [];
let activeDiagnosticDetails: Record<string, unknown> = {};
let activeProgressWrite: Promise<void> = Promise.resolve();

function persistProgress(status: string, details: Record<string, unknown> = {}): Promise<void> {
  if (!activeBuildRoot) return Promise.resolve();
  const path = resolve(activeBuildRoot, 'direct-webgpu-diagnostics.json');
  const snapshot = {
    status,
    fixtureSha256,
    ...activeDiagnosticDetails,
    stageEvents: [...activeProgressEvents],
    networkRequests: [...activeNetworkRequests],
    latestStage: activeProgressEvents.at(-1) ?? null,
    ...details
  };
  activeProgressWrite = activeProgressWrite
    .catch(() => undefined)
    .then(() => writeFile(path, `${JSON.stringify(snapshot, null, 2)}\n`));
  return activeProgressWrite;
}

test.afterEach(async ({page}, testInfo) => {
  if (!activeBuildRoot) return;
  let pageProgress: unknown = null;
  try {
    const saved = await page.evaluate((key) => localStorage.getItem(key), DIAGNOSTIC_STORAGE_KEY);
    pageProgress = saved ? JSON.parse(saved) : null;
  } catch {
    // The page may have closed or timed out while WebGPU was evaluating.
  }
  await activeProgressWrite.catch(() => undefined);
  if (testInfo.status !== 'passed') {
    await persistProgress('failed-after-test', {testStatus: testInfo.status, pageProgress});
  }
  activeBuildRoot = undefined;
});

test('direct Kev browser package passes online, offline, and pinned Q4 GPU gates', async ({page, browserName}) => {
  test.skip(
    !enabled,
    'Set WEB_DECISION_KEV06_DIRECT_GPU=1 and the direct package/report paths for real Chrome GPU validation.'
  );
  test.setTimeout(30 * 60 * 1000);
  assert.equal(browserName, 'chromium');
  assert.equal(fixtureSha256, EXPANDED_FIXTURE_SHA256, 'Expanded Kev fixture digest differs from its pin.');

  const buildRoot = requiredEnvironment('WEB_DECISION_KEV06_DIRECT_BUILD');
  activeBuildRoot = buildRoot;
  activeProgressEvents = [];
  activeNetworkRequests = [];
  activeDiagnosticDetails = {};
  activeProgressWrite = Promise.resolve();
  const packageRoot = resolve(buildRoot, 'browser-package');
  const report = JSON.parse(await readFile(resolve(buildRoot, 'direct-tvm-report.json'), 'utf8')) as BuildReport;
  const manifestBytes = await readFile(resolve(packageRoot, 'manifest.json'));
  const manifest = JSON.parse(manifestBytes.toString('utf8')) as PackageManifest;
  const modelConfig = JSON.parse(await readFile(resolve(packageRoot, 'model_config.json'), 'utf8')) as {
    max_position_embeddings?: unknown;
  };
  const manifestSha256 = createHash('sha256')
    .update(`${JSON.stringify(canonical(manifest))}\n`)
    .digest('hex');

  expect(report.status).toBe('passed');
  expect(report.qualityGates.cpuBaseline.status).toBe('passed');
  expect(report.qualityGates.webgpuExpandedMatrix.status).toBe('pending-real-chrome-webgpu');
  expect(report.qualityGates.webgpuExpandedMatrix.fixtureSha256).toBe(EXPANDED_FIXTURE_SHA256);
  expect(report.qualityGates.webgpuExpandedMatrix.decisionCases).toEqual(
    fixture.decisionCases.map(({id, tokenCount}) => ({id, tokenCount}))
  );
  expect(report.qualityGates.webgpuExpandedMatrix.maxAbsoluteQ4LogitDifference).toBe(LOGIT_TOLERANCE);
  if (report.qualityGates.webgpuExpandedMatrix.q4ProbabilityComparison !== undefined) {
    expect(report.qualityGates.webgpuExpandedMatrix.q4ProbabilityComparison).toBe('descriptive-only');
    expect(report.qualityGates.webgpuExpandedMatrix.outputsMustBeFinite).toBe(true);
    expect(report.qualityGates.webgpuExpandedMatrix.probabilitiesMustSumToOne).toBe(true);
    expect(report.qualityGates.webgpuExpandedMatrix.maxProbabilitySumError).toBe(0.000001);
  }
  expect(report.qualityGates.webgpuExpandedMatrix.allSelectionsMustMatchPinnedQ4).toBe(true);
  expect(report.qualityGates.webgpuExpandedMatrix.baselineCpuGpuParityRequired).toBe(true);
  expect(report.browserPackage.manifestSha256).toBe(manifestSha256);
  expect(report.browserPackage.parameterCount).toBe(511);
  expect(report.browserPackage.license).toBe('Apache-2.0');
  expect(report.kevCheckpointRevision).toBe('abe7110dedfd68a185839550079bcf7c26f561ac');
  expect(report.baseRevision).toBe('da87bfb608c14b7cf20ba1ce41287e8de496c0cd');
  expect(report.tvmWebToolchain).toEqual({
    sourceRevision: '7b5acb32bd271342a24130f4ef439ff6c2e52c6c',
    sourceTag: 'v0.27.0.post1',
    librarySha256: {
      'web/dist/wasm/wasm_runtime.bc': '1974cd7846e3735735fe48828856b7113257a426c24a49e44f45c79c412b972f',
      'web/dist/wasm/tvmjs_support.bc': '24088d9f55002852d513af36b64770dacb1825e5dffebb95d23b43f104b19fc9',
      'web/dist/wasm/webgpu_runtime.bc': 'c4baf79ed300de425db4d81743ab89d36cee323e40e18904c4ef7f196f2fe0aa'
    },
    emscriptenRuntimeDependency: {
      package: 'libatomic1',
      packageVersion: '14.2.0-4ubuntu2~24.04.1',
      soname: 'libatomic.so.1',
      librarySha256: '9d8a34468e614a71a6427dbabb0ac24eb8cb1e324dc6aa3bf95eb6f59201f000'
    }
  });
  expect(manifest).toMatchObject({
    formatVersion: 1,
    model: 'kev-0.6b-q4f16',
    backend: 'tvm-webgpu',
    parameterCount: 511,
    maxStateTokens: 8192,
    maxBranchTokens: 8192,
    checkpoint: {
      repository: 'jaredpalmer/kev-0.6b',
      revision: 'abe7110dedfd68a185839550079bcf7c26f561ac'
    },
    base: {
      repository: 'Qwen/Qwen3-0.6B-Base',
      revision: 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
    },
    licenses: expect.arrayContaining([
      expect.objectContaining({file: 'notices/kev-LICENSE', spdxId: 'Apache-2.0'}),
      expect.objectContaining({file: 'notices/qwen3-0.6b-base-LICENSE', spdxId: 'Apache-2.0'})
    ])
  });
  expect(Object.hasOwn(manifest, 'maxInputTokens')).toBe(false);
  expect(modelConfig.max_position_embeddings).toBe(32768);
  const nativeKevInputLimits = {
    sourceRepository: 'jaredpalmer/kev',
    sourceRevision: '37cb2f6155afec1054de2cc78dfd7c7dd24a956d',
    sourceFile: 'kev/serve.py',
    maxStateTokens: 8192,
    maxBranchTokens: 8192,
    maxPositionEmbeddings: modelConfig.max_position_embeddings,
    aggregateLimit: null
  } as const;

  for (const name of ['notices/kev-LICENSE', 'notices/qwen3-0.6b-base-LICENSE']) {
    const bytes = await readFile(resolve(packageRoot, name));
    expect(manifest.files[name]).toEqual({bytes: bytes.byteLength, sha256: sha256(bytes)});
    expect(bytes.toString('utf8')).toContain('Apache License');
  }

  const resolvedPackageRoot = resolve(packageRoot);
  await withPackageFileServer(resolvedPackageRoot, async (packageOrigin) => {
    page.on('console', (message) => {
      if (message.type() === 'error') console.error(`Browser console error: ${message.text()}`);
    });
    page.on('pageerror', (error) => console.error(`Browser page error: ${error.stack ?? error.message}`));
    page.on('crash', () => console.error('Browser page crashed.'));
    page.on('request', (request) => {
      if (request.url().startsWith(DIRECT_ARTIFACT_ROOT)) console.log(`Browser model request: ${request.url()}`);
    });
    page.on('requestfailed', (request) =>
      console.error(`Browser request failed: ${request.url()} (${request.failure()?.errorText ?? 'unknown'})`)
    );
    await page.route(`${DIRECT_ARTIFACT_ROOT}**`, async (route) => {
      const pathname = new URL(route.request().url()).pathname;
      await route.continue({url: `${packageOrigin}${pathname}`});
    });
    await page.goto('/');
    await page.addScriptTag({type: 'module', url: '/direct-kev-package-harness.ts?direct-r6=10'});
    await page.waitForFunction(() => typeof window.runDirectKevPackageGate === 'function');
    const cpuLogits = report.tvmLogits;
    const baseline = fixture.decisionCases[0];
    assert.ok(baseline, 'The pinned baseline case is missing.');
    assert.equal(report.selectedOptionIndexMatchesUpstreamQ4, true);
    const cpuProbabilities = softmax(cpuLogits);
    const cpuSelected = baseline.reference.options[cpuLogits.indexOf(Math.max(...cpuLogits))];
    assert.equal(cpuSelected, baseline.reference.selected, 'Retained CPU baseline changed the pinned selection.');
    const cpuBaseline = {logits: cpuLogits, probabilities: cpuProbabilities, selected: cpuSelected};
    await page.evaluate(() => localStorage.removeItem('direct-kev-r6-webgpu-diagnostics'));
    let defaultSourceRequests: string[] = [];
    let defaultOfflineRequestStart: number | undefined;
    page.on('console', (message) => {
      const text = message.text();
      if (!text.startsWith(PROGRESS_PREFIX)) return;
      try {
        const event = JSON.parse(text.slice(PROGRESS_PREFIX.length)) as {stage?: unknown};
        activeProgressEvents.push(event);
        if (event.stage === 'default-source-offline-load-started')
          defaultOfflineRequestStart = defaultSourceRequests.length;
        void persistProgress('in-progress');
      } catch {
        // Malformed progress messages should not interrupt the browser gate.
      }
    });
    let observed: GateOutput;
    try {
      observed = (await page.evaluate(
        ({manifestSha256, cases, cpuBaseline}) => window.runDirectKevPackageGate(manifestSha256, cases, cpuBaseline),
        {manifestSha256, cases: fixture.decisionCases, cpuBaseline}
      )) as GateOutput;
    } catch (cause) {
      const saved = await page
        .evaluate(() => localStorage.getItem('direct-kev-r6-webgpu-diagnostics'))
        .catch(() => null);
      await persistProgress('failed-page-evaluate', {
        cpuBaseline,
        error: String(cause),
        pageProgress: saved ? JSON.parse(saved) : null
      });
      throw cause;
    }

    const matrixDiagnostics = observed.online.map((actual, index) => {
      const expected = fixture.decisionCases[index];
      return {
        id: actual.id,
        tokenCount: actual.tokenCount,
        selectedMatches: actual.selected === expected?.reference.selected,
        maxAbsLogitDelta: expected
          ? maxDifference(
              actual.probabilities.map(({logit}) => logit),
              expected.reference.logits
            )
          : null,
        maxAbsProbabilityDelta: expected
          ? maxDifference(
              actual.probabilities.map(({probability}) => probability),
              expected.reference.probabilities
            )
          : null
      };
    });
    expect(observed.adapter.isFallbackAdapter).toBe(false);
    activeDiagnosticDetails = {
      adapter: observed.adapter,
      nativeKevInputLimits,
      cpuBaseline,
      qualityIssues: observed.qualityIssues,
      matrixDiagnostics,
      online: observed.online,
      cacheDiagnostics: observed.cacheDiagnostics,
      boundaryProbe: observed.boundaryProbe ?? null,
      boundaryProbeError: observed.boundaryProbeError ?? null,
      offlineBaseline: observed.offlineBaseline
    };
    const directGateStatus =
      observed.qualityIssues.length === 0 &&
      !observed.boundaryProbeError &&
      observed.boundaryProbe?.acceptedTokenCount === 8192 &&
      !observed.boundaryProbe.atLimitError &&
      observed.boundaryProbe.outputFinite === true &&
      observed.boundaryProbe.overLimitErrorCode === 'INVALID_INPUT'
        ? 'direct-gate-passed'
        : 'direct-gate-failed';
    await persistProgress(directGateStatus);

    expect(observed.qualityIssues).toEqual([]);
    expect(observed.nativeInputLimits).toEqual(nativeKevInputLimits);
    expect(observed.publicOptionsRejected).toBe(true);
    expect(observed.online).toHaveLength(4);
    expect(observed.online.map(({id, tokenCount}) => ({id, tokenCount}))).toEqual(
      fixture.decisionCases.map(({id, tokenCount}) => ({id, tokenCount}))
    );
    const onlineComparisons = observed.online.map((actual, index) => {
      const expected = fixture.decisionCases[index];
      assert.ok(expected, `Missing fixture case ${index}.`);
      return {id: actual.id, ...assertQ4Match(actual, expected, 'Online WebGPU')};
    });

    const onlineBaseline = observed.online[0];
    assert.ok(onlineBaseline, 'The pinned baseline GPU result is missing.');
    const baselineCpuGpuParity = {
      maxAbsLogitDelta: maxDifference(
        onlineBaseline.probabilities.map(({logit}) => logit),
        cpuLogits
      ),
      maxAbsProbabilityDelta: maxDifference(
        onlineBaseline.probabilities.map(({probability}) => probability),
        cpuProbabilities
      ),
      selectionMatches: onlineBaseline.selected === cpuSelected
    };
    assert.ok(baselineCpuGpuParity.maxAbsLogitDelta <= LOGIT_TOLERANCE, 'Baseline CPU/GPU logits differ.');
    assert.ok(
      baselineCpuGpuParity.maxAbsProbabilityDelta <= PROBABILITY_TOLERANCE,
      'Baseline CPU/GPU probabilities differ.'
    );
    assert.equal(baselineCpuGpuParity.selectionMatches, true, 'Baseline CPU/GPU selected options differ.');
    expect(observed.boundaryProbeError).toBeUndefined();
    expect(observed.boundaryProbe?.requestedTokenCount).toBe(8192);
    expect(observed.boundaryProbe?.atLimitError).toBeUndefined();
    expect(observed.boundaryProbe?.acceptedTokenCount).toBe(8192);
    expect(observed.boundaryProbe?.outputFinite).toBe(true);
    expect(observed.boundaryProbe?.probabilityTotal).toBeCloseTo(1, 6);
    expect(observed.boundaryProbe?.overLimitErrorCode).toBe('INVALID_INPUT');
    expect(observed.boundaryProbe?.overLimitMessage).toContain('8192 tokens');
    expect(observed.boundaryProbe?.overLimitMessage).toContain('no truncation was applied');

    const offlineBaseline = observed.offlineBaseline;
    const offlineParity = {
      maxAbsLogitDelta: maxDifference(
        offlineBaseline.probabilities.map(({logit}) => logit),
        onlineBaseline.probabilities.map(({logit}) => logit)
      ),
      maxAbsProbabilityDelta: maxDifference(
        offlineBaseline.probabilities.map(({probability}) => probability),
        onlineBaseline.probabilities.map(({probability}) => probability)
      ),
      selectionMatches: offlineBaseline.selected === onlineBaseline.selected
    };
    assert.ok(offlineParity.maxAbsLogitDelta <= LOGIT_TOLERANCE, 'Offline baseline logits differ from online.');
    assert.ok(
      offlineParity.maxAbsProbabilityDelta <= PROBABILITY_TOLERANCE,
      'Offline baseline probabilities differ from online.'
    );
    assert.equal(offlineParity.selectionMatches, true, 'Offline baseline selection differs from online.');
    expect(observed.offlineFetchPaths).toEqual([]);

    const fetchedRuntimeFiles = [...new Set(observed.onlineFetchPaths)]
      .filter((name) => name !== 'manifest.json')
      .sort();
    const manifestRuntimeFiles = Object.keys(manifest.files)
      .filter((name) => !name.startsWith('notices/'))
      .sort();
    expect(fetchedRuntimeFiles).toEqual(manifestRuntimeFiles);

    const defaultSourceRequestHandler = (request: import('@playwright/test').Request) => {
      const url = request.url();
      if (!url.startsWith(DIRECT_ARTIFACT_ROOT) && !url.includes('kev06-stable')) return;
      defaultSourceRequests.push(url);
      activeNetworkRequests.push(url);
      void persistProgress('in-progress');
    };
    page.on('request', defaultSourceRequestHandler);
    let defaultSourceSmoke: DefaultSourceSmokeOutput;
    try {
      defaultSourceSmoke = (await page.evaluate(
        ({manifestSha256, cases, manifestFiles}) =>
          window.runDirectKevDefaultSourceSmoke(manifestSha256, cases, manifestFiles),
        {manifestSha256, cases: fixture.decisionCases, manifestFiles: manifest.files}
      )) as DefaultSourceSmokeOutput;
    } finally {
      page.off('request', defaultSourceRequestHandler);
    }
    assert.notEqual(defaultOfflineRequestStart, undefined, 'The default-source offline checkpoint is missing.');
    const defaultSourceOfflineRequests = defaultSourceRequests.slice(defaultOfflineRequestStart);
    expect(defaultSourceRequests.some((url) => url.includes('kev06-stable') && url.includes('manifest.json'))).toBe(
      true
    );
    expect(defaultSourceRequests.some((url) => url.startsWith(`${DIRECT_ARTIFACT_ROOT}weights/params_shard_`))).toBe(
      true
    );
    expect(defaultSourceOfflineRequests).toEqual([]);
    expect(defaultSourceSmoke.cacheDiagnostics.cachePutFailures).toEqual([]);
    expect(
      defaultSourceSmoke.cacheDiagnostics.missingAssetNames.filter((name) => !name.startsWith('notices/'))
    ).toEqual([]);
    const defaultSourceParity = {
      maxAbsLogitDelta: maxDifference(
        defaultSourceSmoke.onlineBaseline.probabilities.map(({logit}) => logit),
        onlineBaseline.probabilities.map(({logit}) => logit)
      ),
      maxAbsProbabilityDelta: maxDifference(
        defaultSourceSmoke.onlineBaseline.probabilities.map(({probability}) => probability),
        onlineBaseline.probabilities.map(({probability}) => probability)
      ),
      selectionMatches: defaultSourceSmoke.onlineBaseline.selected === onlineBaseline.selected,
      offlineMaxAbsLogitDelta: maxDifference(
        defaultSourceSmoke.offlineBaseline.probabilities.map(({logit}) => logit),
        defaultSourceSmoke.onlineBaseline.probabilities.map(({logit}) => logit)
      ),
      offlineMaxAbsProbabilityDelta: maxDifference(
        defaultSourceSmoke.offlineBaseline.probabilities.map(({probability}) => probability),
        defaultSourceSmoke.onlineBaseline.probabilities.map(({probability}) => probability)
      ),
      offlineSelectionMatches:
        defaultSourceSmoke.offlineBaseline.selected === defaultSourceSmoke.onlineBaseline.selected
    };
    assert.ok(
      defaultSourceParity.maxAbsLogitDelta <= LOGIT_TOLERANCE,
      'Default source logits differ from injected source.'
    );
    assert.ok(
      defaultSourceParity.maxAbsProbabilityDelta <= PROBABILITY_TOLERANCE,
      'Default source probabilities differ from injected source.'
    );
    assert.equal(defaultSourceParity.selectionMatches, true, 'Default source selected a different option.');
    assert.ok(defaultSourceParity.offlineMaxAbsLogitDelta <= LOGIT_TOLERANCE, 'Default offline logits differ.');
    assert.ok(
      defaultSourceParity.offlineMaxAbsProbabilityDelta <= PROBABILITY_TOLERANCE,
      'Default offline probabilities differ.'
    );
    assert.equal(defaultSourceParity.offlineSelectionMatches, true, 'Default offline selection differs.');
    activeDiagnosticDetails = {
      ...activeDiagnosticDetails,
      defaultSourceSmoke,
      defaultSourceRequests,
      defaultSourceParity
    };
    await persistProgress('default-source-gate-complete');

    const evidence = {
      status: 'passed',
      modelId: 'kev-0.6b-q4f16',
      backend: 'tvm-webgpu',
      manifestSha256,
      parameterCount: report.browserPackage.parameterCount,
      fixtureSha256,
      nativeKevInputLimits: observed.nativeInputLimits,
      cpuBaseline: {logits: cpuLogits, probabilities: cpuProbabilities, selected: cpuSelected},
      baselineCpuGpuParity,
      onlineComparisons,
      boundaryProbe: observed.boundaryProbe,
      offlineParity,
      defaultSourceSmoke,
      defaultSourceRequests,
      defaultSourceParity,
      onlineFetchPaths: observed.onlineFetchPaths,
      offlineFetchPaths: observed.offlineFetchPaths
    };
    await writeFile(resolve(buildRoot, 'direct-webgpu-validation.json'), `${JSON.stringify(evidence, null, 2)}\n`);
    await persistProgress('passed');
    expect(evidence.status).toBe('passed');
  });
});

function requiredEnvironment(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`Set ${name} to the finished direct Kev build directory.`);
  return resolve(value);
}

function contentType(path: string): string {
  const extension = extname(path);
  if (extension === '.json') return 'application/json';
  if (extension === '.wasm') return 'application/wasm';
  return 'application/octet-stream';
}
