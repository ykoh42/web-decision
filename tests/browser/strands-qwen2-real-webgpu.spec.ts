import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {appendFile, mkdir, mkdtemp, readFile, realpath, rm, writeFile} from 'node:fs/promises';
import os from 'node:os';
import {basename, dirname, extname, relative, resolve} from 'node:path';
import {chromium, expect, test as base, type BrowserContext, type Page} from '@playwright/test';
import {
  assertQwen35EvidenceRecord,
  assertQwen35OfflineResults,
  assertQwen35VisionImageCountCoverage,
  compareQwen35B3S4096Result,
  assertQwen35TvmMainCallSites,
  buildQwen35ResultParityDiagnostic,
  compareQwen35Result,
  loadQwen35CandidateInputs,
  QWEN35_RELEASE_MODEL,
  verifyQwen35ExactPackage,
  writeQwen35OnlineDiagnostic
} from '../../scripts/release/strands-qwen35-release-evidence.ts';
import {
  recordQwen35GateFailure,
  writeQwen35ReportThenThrow,
  type Qwen35GateFailure
} from '../../scripts/release/qwen35-deferred-gates.mts';
import {summarizeCacheStorageUrls} from './qwen35-cache-diagnostics.mts';
import type {CachePutPrototype} from './qwen35-cache-diagnostics.mts';

const enabled = process.env.WEB_DECISION_QWEN35_RELEASE_GPU === '1';
const candidateListPreflightEnabled = process.env.WEB_DECISION_QWEN35_CANDIDATE_LIST_PREFLIGHT === '1';

type PersistentBrowserOptions = NonNullable<Parameters<typeof chromium.launchPersistentContext>[1]>;
type ProjectBrowserUse = Readonly<{
  baseURL?: string;
  channel?: PersistentBrowserOptions['channel'];
  headless?: PersistentBrowserOptions['headless'];
  launchOptions?: Pick<PersistentBrowserOptions, 'args' | 'env' | 'ignoreDefaultArgs'>;
}>;

const test = base.extend({
  context: async ({browserName}, use, testInfo) => {
    const profileDirectory = await mkdtemp(resolve(os.tmpdir(), `web-decision-qwen35-release-${browserName}-`));
    const projectUse = testInfo.project.use as ProjectBrowserUse;
    const launchOptions: PersistentBrowserOptions = {
      ...(projectUse.baseURL ? {baseURL: projectUse.baseURL} : {}),
      ...(projectUse.channel ? {channel: projectUse.channel} : {}),
      ...(typeof projectUse.headless === 'boolean' ? {headless: projectUse.headless} : {}),
      ...projectUse.launchOptions
    };
    let context: BrowserContext | undefined;
    try {
      context = await chromium.launchPersistentContext(profileDirectory, launchOptions);
      await use(context);
    } finally {
      if (context) await context.close();
      await rm(profileDirectory, {recursive: true, force: true});
    }
  }
});

type RecordValue = Record<string, unknown>;
type BrowserImage = Readonly<{base64: string; mimeType: string}>;
type BrowserCase = Readonly<{
  name: string;
  request: RecordValue;
  images: readonly BrowserImage[];
}>;
type BrowserRejectedCase = BrowserCase & Readonly<{expectedErrorCode: string}>;
type CacheMarkerEvidence = Readonly<{format: unknown; markerId: unknown}>;
type Qwen35CacheDiagnostics = {
  cacheNames: string[];
  cachedAssetUrls: string[];
  cacheEntries?: ReturnType<typeof summarizeCacheStorageUrls>;
  markers?: Readonly<{source: CacheMarkerEvidence | null; tokenizer: CacheMarkerEvidence | null}>;
  storageBeforeOnline?: unknown;
  storageAfterOnline?: unknown;
  rangeCachePuts?: unknown;
};

function requiredPath(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required when the Qwen M4 release browser gate is enabled.`);
  return resolve(value);
}

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function publicRequestUrl(value: string): string {
  const url = new URL(value);
  url.username = '';
  url.password = '';
  url.search = '';
  url.hash = '';
  return url.href;
}

async function prepareCases(
  cases: readonly Readonly<{
    name: string;
    request: RecordValue;
    imageFiles?: readonly Readonly<{path: string; sha256: string; bytes: number; mimeType: string}>[];
  }>[],
  artifactRoot: string
): Promise<readonly BrowserCase[]> {
  const prepared: BrowserCase[] = [];
  for (const item of cases) {
    const images: BrowserImage[] = [];
    for (const file of item.imageFiles ?? []) {
      const bytes = await readFile(resolve(artifactRoot, file.path));
      expect(bytes.byteLength, `${item.name} image ${file.path} byte length`).toBe(file.bytes);
      expect(sha256(bytes), `${item.name} image ${file.path} SHA-256`).toBe(file.sha256);
      images.push({base64: bytes.toString('base64'), mimeType: file.mimeType});
    }
    prepared.push({name: item.name, request: item.request, images});
  }
  return prepared;
}

function readM4Host(): Readonly<{machineName: string; machineModel: string; chipType: string}> {
  const raw = execFileSync('system_profiler', ['-json', 'SPHardwareDataType'], {
    encoding: 'utf8',
    maxBuffer: 1024 * 1024
  });
  const report = JSON.parse(raw) as {SPHardwareDataType?: Array<Record<string, unknown>>};
  const hardware = report.SPHardwareDataType?.[0];
  if (!hardware) throw new Error('system_profiler returned no SPHardwareDataType record.');
  return {
    machineName: String(hardware.machine_name ?? ''),
    machineModel: String(hardware.machine_model ?? ''),
    chipType: String(hardware.chip_type ?? '')
  };
}

async function readChromeGpuReport(page: Page): Promise<Readonly<{status: string | null; renderer: string | null}>> {
  await page.goto('chrome://gpu');
  const report = await page.evaluate(() => {
    const parts: string[] = [];
    function visit(root: Document | ShadowRoot) {
      if (root?.textContent) parts.push(root.textContent);
      for (const element of root?.querySelectorAll?.('*') ?? []) if (element.shadowRoot) visit(element.shadowRoot);
    }
    visit(document);
    return parts.join('\n');
  });
  return {
    status: report.match(/WebGPU:\s*(Hardware accelerated|Software only|Disabled|Blocklisted)/iu)?.[1] ?? null,
    renderer: report.match(/ANGLE Metal Renderer:\s*([^\n]+)/iu)?.[1]?.trim() ?? null
  };
}

function browserRequest(item: BrowserCase): BrowserCase {
  return {
    name: item.name,
    request: item.request,
    images: item.images.map(({base64, mimeType}) => ({base64, mimeType}))
  };
}

test('the installed public Qwen candidate bundle lists and describes the curated catalog ID', async ({page}) => {
  test.skip(
    !candidateListPreflightEnabled,
    'Set WEB_DECISION_QWEN35_CANDIDATE_LIST_PREFLIGHT=1 with the exact candidate archive to run this no-inference preflight.'
  );
  const packageEvidence = await verifyQwen35ExactPackage(
    requiredPath('WEB_DECISION_RELEASE_ARCHIVE'),
    requiredPath('WEB_DECISION_INSTALLED_PACKAGE')
  );
  const installedBundle = await realpath(requiredPath('WEB_DECISION_INSTALLED_BUNDLE'));
  const packageMetadata = JSON.parse(
    await readFile(resolve(packageEvidence.installedRoot, 'package.json'), 'utf8')
  ) as {exports?: Record<string, {import?: unknown}>};
  const publicEntry = packageMetadata.exports?.['.']?.import;
  if (publicEntry !== './dist/index.js')
    throw new Error('The installed Qwen candidate archive must expose its public dist/index.js entry.');
  const publicBundle = await realpath(resolve(packageEvidence.installedRoot, publicEntry));
  expect(packageEvidence.archive.archiveFiles).toContain(publicEntry.slice(2));
  expect(installedBundle).toBe(publicBundle);

  await page.goto('/');
  await page.waitForFunction(() => typeof window.webDecision?.listModels === 'function');
  const result = await page.evaluate((catalogModelId) => {
    const api = window.webDecision as unknown as {
      listModels(): readonly {modelId: string}[];
      describeModel(modelId: string): {
        modelId: string;
        checkpoint: {repository: string; revision: string};
      };
    };
    return {
      listedModelIds: api.listModels().map(({modelId}) => modelId),
      description: api.describeModel(catalogModelId)
    };
  }, QWEN35_RELEASE_MODEL.catalogModelId);

  expect(result.listedModelIds).toContain(QWEN35_RELEASE_MODEL.catalogModelId);
  expect(result.description.modelId).toBe(QWEN35_RELEASE_MODEL.catalogModelId);
  expect(result.description.checkpoint.repository).toBe(QWEN35_RELEASE_MODEL.repository);
  expect(result.description.checkpoint.revision).toBe(QWEN35_RELEASE_MODEL.revision);
});

async function inspectExactQwenPackage() {
  const packageEvidence = await verifyQwen35ExactPackage(
    requiredPath('WEB_DECISION_RELEASE_ARCHIVE'),
    requiredPath('WEB_DECISION_INSTALLED_PACKAGE')
  );
  const installedBundle = await realpath(requiredPath('WEB_DECISION_INSTALLED_BUNDLE'));
  const packageMetadata = JSON.parse(
    await readFile(resolve(packageEvidence.installedRoot, 'package.json'), 'utf8')
  ) as {exports?: Record<string, {import?: unknown}>};
  const publicEntry = packageMetadata.exports?.['.']?.import;
  if (publicEntry !== './dist/index.js')
    throw new Error('The installed Qwen candidate archive must expose its public dist/index.js entry.');
  const candidateBundle = await realpath(resolve(packageEvidence.installedRoot, publicEntry));
  expect(installedBundle).toBe(candidateBundle);
  const candidateBundleRelative = relative(packageEvidence.installedRoot, candidateBundle).replaceAll('\\', '/');
  expect(packageEvidence.archive.archiveFiles).toContain(candidateBundleRelative);
  const archivedBundle = execFileSync(
    'tar',
    ['-xOzf', packageEvidence.archive.archivePath, `package/${candidateBundleRelative}`],
    {maxBuffer: 64 * 1024 * 1024}
  );
  const installedBundleSha256 = sha256(await readFile(installedBundle));
  expect(installedBundleSha256).toBe(sha256(archivedBundle));
  const adapterChunkPath = packageEvidence.archive.archiveFiles.find((path) =>
    /^dist\/strands-qwen35-[^/]+\.js$/u.test(path)
  );
  const visionArtifactsChunkPath = packageEvidence.archive.archiveFiles.find((path) =>
    /^dist\/qwen35-vision-artifacts-[^/]+\.js$/u.test(path)
  );
  const tvmRuntimeChunkPath = packageEvidence.archive.archiveFiles.find((path) =>
    /^dist\/tvm\.generated-[^/]+\.js$/u.test(path)
  );
  if (!adapterChunkPath) throw new Error('The exact package archive has no Qwen adapter chunk.');
  if (!visionArtifactsChunkPath) throw new Error('The exact package archive has no Qwen vision artifacts chunk.');
  if (!tvmRuntimeChunkPath) throw new Error('The exact package archive has no pinned TVM.js runtime chunk.');
  const adapterChunk = execFileSync(
    'tar',
    ['-xOzf', packageEvidence.archive.archivePath, `package/${adapterChunkPath}`],
    {maxBuffer: 16 * 1024 * 1024}
  );
  const tvmRuntimeChunk = execFileSync(
    'tar',
    ['-xOzf', packageEvidence.archive.archivePath, `package/${tvmRuntimeChunkPath}`],
    {maxBuffer: 16 * 1024 * 1024}
  );
  const visionArtifactsChunk = execFileSync(
    'tar',
    ['-xOzf', packageEvidence.archive.archivePath, `package/${visionArtifactsChunkPath}`],
    {maxBuffer: 16 * 1024 * 1024}
  );
  const adapterSource = adapterChunk.toString('utf8');
  const visionArtifactsSource = visionArtifactsChunk.toString('utf8');
  expect(adapterSource).toContain(basename(visionArtifactsChunkPath));
  expect(visionArtifactsSource).toContain('openPinnedQwen35VisionArtifacts');
  const mainCallSites = assertQwen35TvmMainCallSites(adapterSource, adapterSource, tvmRuntimeChunk.toString('utf8'));
  return {packageEvidence, installedBundle, installedBundleSha256, mainCallSites};
}

function compareFullContextBatch(
  results: readonly {name: string; result: unknown}[],
  expected: Parameters<typeof compareQwen35B3S4096Result>[1]
) {
  try {
    if (results.length !== 1 || results[0]?.name !== expected.name)
      throw new Error('The public Qwen B3×S4096 case did not return exactly one matching result.');
    compareQwen35B3S4096Result(results[0]?.result, expected, 'Qwen public batch 3 × context 4096');
    return {passed: true, error: null};
  } catch (error) {
    return {passed: false, error: error instanceof Error ? error.message : String(error)};
  }
}

function summarizeParityStatus(statuses: readonly string[]): 'passed' | 'failed' | 'incomplete' {
  if (statuses.includes('failed')) return 'failed';
  return statuses.every((status) => status === 'passed') ? 'passed' : 'incomplete';
}

test('the current Qwen3.5 v1-2610 package matches pinned source behavior on M4 WebGPU', async ({
  page,
  browser,
  browserName
}) => {
  test.skip(
    !enabled,
    'Set WEB_DECISION_QWEN35_RELEASE_GPU=1 with the exact npm package and generated build artifacts to run this M4 test.'
  );
  test.setTimeout(180 * 60 * 1000);
  expect(process.env.WEB_DECISION_REAL_GPU).toBe('1');
  expect(process.env.WEB_DECISION_BROWSER_NAME).toBe('chromium');
  expect(process.env.WEB_DECISION_BROWSER_CHANNEL ?? 'chromium').toBe('chromium');
  expect(process.platform).toBe('darwin');
  expect(os.arch()).toBe('arm64');
  expect(browserName).toBe('chromium');

  const artifactRoot = await realpath(requiredPath('WEB_DECISION_QWEN35_BUILD_ARTIFACT_ROOT'));
  const publicAssetRoot = await realpath(requiredPath('WEB_DECISION_PUBLIC_ASSET_ROOT'));
  expect(publicAssetRoot).toBe(artifactRoot);
  const candidate = await loadQwen35CandidateInputs(artifactRoot);
  const rangeInventory = JSON.parse(
    await readFile(resolve(artifactRoot, 'assets/qwen35/source-range-inventory.json'), 'utf8')
  ) as {
    readonly assets: Readonly<
      Record<string, {readonly ranges: readonly {readonly sha256: string; readonly bytes: number}[]}>
    >;
  };
  const expectedRangeBytesByDigest = [
    ...new Map(
      Object.values(rangeInventory.assets).flatMap(({ranges}) =>
        ranges.map(({sha256: digest, bytes}) => [digest, bytes] as const)
      )
    )
  ];
  assertQwen35VisionImageCountCoverage(candidate.visionCases);
  const {packageEvidence, installedBundle, installedBundleSha256, mainCallSites} = await inspectExactQwenPackage();
  const host = readM4Host();
  expect(host.chipType).toBe('Apple M4 Pro');

  const textCases = await prepareCases(candidate.sourceCases, artifactRoot);
  const visionCases = await prepareCases(candidate.visionCases, artifactRoot);
  const [b3S4096Case] = await prepareCases([candidate.b3S4096Case], artifactRoot);
  if (!b3S4096Case) throw new Error('The pinned Qwen B3×S4096 public case is missing.');
  const boundaryValues = Object.values(candidate.boundaryCases);
  const acceptedBoundaries = boundaryValues.filter((value) => 'expected' in value) as Array<{
    name: string;
    request: RecordValue;
    imageFiles?: readonly Readonly<{path: string; sha256: string; bytes: number; mimeType: string}>[];
    expected: RecordValue;
    candidateUsageExpectation?: unknown;
  }>;
  const rejectedBoundaries = boundaryValues.filter((value) => !('expected' in value)) as Array<RecordValue>;
  const boundaryCases = await prepareCases(acceptedBoundaries, artifactRoot);
  const rejectedCases = await prepareCases(
    rejectedBoundaries.map((value) => ({
      name: String(value.name),
      request: value.request as RecordValue,
      ...(Array.isArray(value.imageFiles)
        ? {
            imageFiles: value.imageFiles as readonly Readonly<{
              path: string;
              sha256: string;
              bytes: number;
              mimeType: string;
            }>[]
          }
        : {})
    })),
    artifactRoot
  );
  const rejectedInputs = rejectedCases.map((item) => ({
    ...item,
    expectedErrorCode: String(rejectedBoundaries.find((value) => value.name === item.name)?.expectedErrorCode)
  }));

  const outputPath = resolve(process.env.WEB_DECISION_QWEN35_REPORT ?? 'reports/qwen35-exact-package-m4-webgpu.json');
  const progressPath = resolve(dirname(outputPath), `${basename(outputPath, extname(outputPath))}.progress.jsonl`);
  await mkdir(dirname(progressPath), {recursive: true});
  await writeFile(progressPath, '');
  const appendProgress = async (event: RecordValue) => {
    await appendFile(
      progressPath,
      `${JSON.stringify({schemaVersion: 1, ...event, recordedAt: new Date().toISOString()})}\n`
    );
  };
  await appendProgress({
    schemaVersion: 1,
    type: 'run-started',
    modelId: QWEN35_RELEASE_MODEL.catalogModelId,
    packageArchiveSha256: packageEvidence.archive.archiveSha256,
    candidateBundleSha256: installedBundleSha256,
    generatedManifestSha256: candidate.manifestSha256,
    assetManifestSha256: candidate.assetManifestSha256
  });
  await page.exposeFunction('__webDecisionQwen35Progress', async (event: RecordValue) => appendProgress(event));

  await page.goto('/');
  await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');
  await page.evaluate(async () => {
    const api = window.webDecision as unknown as {clearModelCache(): Promise<boolean>};
    await api.clearModelCache();
  });
  const storageBeforeOnline = await page.evaluate(async () => {
    try {
      return navigator.storage?.estimate ? await navigator.storage.estimate() : null;
    } catch (cause) {
      const error = cause as {name?: unknown; message?: unknown};
      return {
        errorName: typeof error?.name === 'string' ? error.name : 'UnknownError',
        errorMessage: typeof error?.message === 'string' ? error.message : String(cause)
      };
    }
  });
  await page.evaluate(async (rangeSizes) => {
    const diagnosticsUrl = new URL('/qwen35-cache-diagnostics.mts', location.href);
    const {createCachePutTracker, observeRangeCachePuts} = await import(diagnosticsUrl.href);
    const tracker = createCachePutTracker(new Map(rangeSizes));
    const restore = observeRangeCachePuts(Cache.prototype as unknown as CachePutPrototype, tracker, location.href);
    Object.defineProperty(window, '__webDecisionQwen35CachePutTelemetry', {
      value: {snapshot: () => tracker.snapshot(), restore},
      configurable: true
    });
  }, expectedRangeBytesByDigest);
  await page.evaluate(async () => {
    const moduleUrl = new URL('/qwen35-tvm-dispatch-capture.mts', location.origin);
    const {installQwen35TvmDispatchCapture} = await import(moduleUrl.href);
    const capture = installQwen35TvmDispatchCapture(GPUComputePassEncoder.prototype);
    Object.defineProperty(window, '__webDecisionQwen35DispatchCapture', {value: capture, configurable: false});
  });
  const onlineNetworkRequests: Array<{url: string; method: string; rangeHeader: string | null}> = [];
  const onRequest = (request: {url(): string; method(): string; headers(): Record<string, string>}) => {
    const headers = request.headers();
    onlineNetworkRequests.push({
      url: publicRequestUrl(request.url()),
      method: request.method(),
      rangeHeader: headers.range ?? headers.Range ?? null
    });
  };
  page.on('request', onRequest);

  const online = await page.evaluate(
    async ({
      modelId,
      textCases: textInput,
      visionCases: visionInput,
      boundaryCases: boundaryInput,
      b3S4096Case,
      rejectedInputs: rejected
    }: {
      modelId: string;
      textCases: readonly BrowserCase[];
      visionCases: readonly BrowserCase[];
      boundaryCases: readonly BrowserCase[];
      b3S4096Case: BrowserCase;
      rejectedInputs: readonly BrowserRejectedCase[];
    }) => {
      type ModelDescription = {
        modelId: string;
        backend: {name: string; requiredFeatures: string[]; requiredLimits: Record<string, number>};
        checkpoint: {repository: string; revision: string};
        assets: {manifestSha256: string};
        features: {
          typedTextQuestionTypes?: string[];
          imageChoice?: {status: string};
        };
      };
      type DecisionModel = {
        modelId: string;
        description: ModelDescription;
        systemOne(request: unknown, options?: unknown): Promise<unknown>;
        dispose(): Promise<void>;
      };
      type LoadProgress = {
        loadedBytes: number;
        totalBytes: number;
        fraction: number;
        phase: 'model-assets' | 'vision-weights' | 'upstream-source';
      };
      type Api = {
        listModels(): readonly ModelDescription[];
        describeModel(id: string): ModelDescription;
        loadDecisionModel(options: {
          modelId: string;
          cache: boolean;
          offline: boolean;
          onProgress?: (progress: LoadProgress) => void;
        }): Promise<DecisionModel>;
      };
      const api = window.webDecision as unknown as Api;
      const reportProgress = async (event: Record<string, unknown>) => {
        const callback = (
          window as unknown as {
            __webDecisionQwen35Progress?: (value: Record<string, unknown>) => Promise<void>;
          }
        ).__webDecisionQwen35Progress;
        if (!callback) throw new Error('The Qwen progress writer is unavailable.');
        await callback(event);
      };
      let progressWrites = Promise.resolve();
      const enqueueProgress = (event: Record<string, unknown>) => {
        progressWrites = progressWrites.then(() => reportProgress(event));
      };
      const loadModel = async (phase: 'online' | 'offline') => {
        const offline = phase === 'offline';
        const lastProgressByPhase = new Map<LoadProgress['phase'], number>();
        await reportProgress({type: 'model-load-started', phase, modelId, cache: true, offline});
        let model: DecisionModel;
        try {
          model = await api.loadDecisionModel({
            modelId,
            cache: true,
            offline,
            onProgress(progress) {
              const previousBytes = lastProgressByPhase.get(progress.phase);
              const phaseChanged = previousBytes === undefined;
              const intervalElapsed =
                previousBytes !== undefined && progress.loadedBytes - previousBytes >= 256 * 1024 * 1024;
              if (!phaseChanged && !intervalElapsed && progress.fraction < 1) return;
              lastProgressByPhase.set(progress.phase, progress.loadedBytes);
              enqueueProgress({
                type: 'model-load-progress',
                phase,
                modelPhase: progress.phase,
                loadedBytes: progress.loadedBytes,
                totalBytes: progress.totalBytes,
                fraction: progress.fraction
              });
            }
          });
          await progressWrites;
        } catch (error) {
          await progressWrites.catch(() => {});
          await reportProgress({
            type: 'model-load-failed',
            phase,
            modelId,
            error: error instanceof Error ? error.message : String(error)
          });
          throw error;
        }
        await reportProgress({
          type: 'model-loaded',
          phase,
          modelId: model.modelId,
          checkpoint: model.description.checkpoint,
          manifestSha256: model.description.assets.manifestSha256
        });
        return model;
      };
      function readPinnedDescription() {
        const listed = api.listModels().map((model) => model.modelId);
        if (!listed.includes(modelId)) throw new Error(`Current curated package does not list ${modelId}.`);
        const description = api.describeModel(modelId);
        if (description.modelId !== modelId) throw new Error('describeModel returned a different model ID.');
        if (description.backend.name !== 'tvm-webgpu')
          throw new Error('Qwen public descriptor is not direct TVM WebGPU.');
        if (!description.assets.manifestSha256) throw new Error('Qwen public descriptor has no pinned asset manifest.');
        for (const type of ['choice', 'score', 'noul']) {
          if (!description.features.typedTextQuestionTypes?.includes(type))
            throw new Error(`Qwen descriptor does not claim ${type} support.`);
        }
        if (description.features.imageChoice?.status !== 'implemented')
          throw new Error('Qwen descriptor does not claim official vision support.');
        return {listed, description};
      }
      async function readAdapterEvidence(description: ModelDescription) {
        const adapter = await navigator.gpu?.requestAdapter({powerPreference: 'high-performance'});
        if (!adapter) throw new Error('WebGPU is unavailable.');
        const adapterWithFallback = adapter as GPUAdapter & {readonly isFallbackAdapter?: boolean};
        const info = adapter.info;
        const infoWithFallback = info as GPUAdapterInfo & {readonly isFallbackAdapter?: boolean};
        const isFallbackAdapter = adapterWithFallback.isFallbackAdapter ?? infoWithFallback.isFallbackAdapter ?? null;
        if (isFallbackAdapter !== false)
          throw new Error(`A physical WebGPU adapter is required; got ${String(isFallbackAdapter)}.`);
        if (
          !String(info.vendor ?? '')
            .toLowerCase()
            .includes('apple')
        )
          throw new Error(`Expected Apple WebGPU hardware; got ${String(info.vendor)}.`);
        for (const feature of description.backend.requiredFeatures)
          if (!adapter.features.has(feature as GPUFeatureName))
            throw new Error(`WebGPU adapter is missing required feature ${feature}.`);
        for (const [name, minimum] of Object.entries(description.backend.requiredLimits)) {
          const value = (adapter.limits as unknown as Record<string, number>)[name];
          if (typeof value !== 'number' || value < minimum)
            throw new Error(`WebGPU adapter limit ${name} is below the model requirement.`);
        }
        return {
          info: {
            vendor: info.vendor,
            architecture: info.architecture,
            device: info.device,
            description: info.description
          },
          isFallbackAdapter,
          features: [...adapter.features].sort(),
          limits: {
            maxBufferSize: adapter.limits.maxBufferSize,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize
          }
        };
      }
      const {listed, description} = readPinnedDescription();
      const adapterEvidence = await readAdapterEvidence(description);
      const materializeRequest = (item: {
        request: Record<string, unknown>;
        images: readonly {base64: string; mimeType: string}[];
      }) => ({
        ...item.request,
        questions: Array.isArray(item.request.questions)
          ? new Map(item.request.questions as Array<[string, unknown]>)
          : item.request.questions,
        ...(item.images.length > 0
          ? {images: item.images.map(({base64, mimeType}) => `data:${mimeType};base64,${base64}`)}
          : {})
      });
      const run = async (
        model: DecisionModel,
        cases: readonly {
          name: string;
          request: Record<string, unknown>;
          images: readonly {base64: string; mimeType: string}[];
        }[],
        group: string
      ) => {
        const capture = (
          window as unknown as {
            __webDecisionQwen35DispatchCapture?: {
              activeGroup: string | null;
              dispatchCounts: Record<string, number>;
              tvmRuntimeDispatchCounts: Record<string, number>;
              tvmRuntimeCallers: Record<string, string>;
            };
          }
        ).__webDecisionQwen35DispatchCapture;
        if (!capture) throw new Error('The Qwen TVM WebGPU dispatch capture is unavailable.');
        if (capture.activeGroup !== null) throw new Error('A Qwen TVM dispatch capture group is already active.');
        const values: Array<{name: string; result: unknown}> = [];
        capture.activeGroup = group;
        try {
          for (const item of cases) {
            const result = await model.systemOne(materializeRequest(item), {});
            await reportProgress({type: 'case-completed', phase: 'online', group, name: item.name, result});
            values.push({name: item.name, result});
          }
        } finally {
          capture.activeGroup = null;
        }
        return values;
      };
      const failures: Array<{name: string; code: string; message: string}> = [];
      let onlineModel = await loadModel('online');
      try {
        if (onlineModel.modelId !== modelId || onlineModel.description.modelId !== modelId)
          throw new Error('Loaded model is not the curated Qwen ID.');
        if (onlineModel.description.checkpoint.repository !== 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610')
          throw new Error('Loaded Qwen checkpoint repository mismatch.');
        if (onlineModel.description.checkpoint.revision !== '216d1ccb46a9e9b036eb9187b248db98480f77d4')
          throw new Error('Loaded Qwen checkpoint revision mismatch.');
        const textResults = await run(onlineModel, textInput, 'text');
        const visionResults = await run(onlineModel, visionInput, 'vision');
        const boundaryResults = await run(onlineModel, boundaryInput, 'accepted-boundary');
        const b3S4096Results = await run(onlineModel, [b3S4096Case], 'batch-3-context-4096');
        const graphDispatchCapture = (
          window as unknown as {
            __webDecisionQwen35DispatchCapture: {
              activeGroup: string | null;
              dispatchCounts: Record<string, number>;
              tvmRuntimeDispatchCounts: Record<string, number>;
              tvmRuntimeCallers: Record<string, string>;
            };
          }
        ).__webDecisionQwen35DispatchCapture;
        if (graphDispatchCapture.activeGroup !== null)
          throw new Error('Qwen TVM graph dispatch capture was not closed after inference.');
        for (const item of rejected) {
          let failure: {name: string; code: string; message: string};
          try {
            await onlineModel.systemOne(materializeRequest(item), {});
            failure = {
              name: item.name,
              code: 'UNEXPECTED_SUCCESS',
              message: 'Request was accepted past the pinned source limit.'
            };
          } catch (error) {
            const rejected = error as {code?: string; name?: string; message?: string};
            failure = {
              name: item.name,
              code: rejected.code ?? rejected.name ?? 'ERROR',
              message: rejected.message ?? ''
            };
          }
          failures.push(failure);
          await reportProgress({type: 'case-completed', phase: 'online', group: 'rejected-boundary', ...failure});
        }
        return {
          listed,
          description,
          userAgent: navigator.userAgent,
          adapter: adapterEvidence,
          textResults,
          visionResults,
          boundaryResults,
          b3S4096Results,
          graphDispatchCounts: {...graphDispatchCapture.dispatchCounts},
          tvmRuntimeDispatchCounts: {...graphDispatchCapture.tvmRuntimeDispatchCounts},
          tvmRuntimeCallers: {...graphDispatchCapture.tvmRuntimeCallers},
          failures,
          manifestSha256: onlineModel.description.assets.manifestSha256
        };
      } finally {
        await onlineModel.dispose();
        onlineModel = null as unknown as DecisionModel;
      }
    },
    {
      modelId: QWEN35_RELEASE_MODEL.catalogModelId,
      textCases: textCases.map(browserRequest),
      visionCases: visionCases.map(browserRequest),
      boundaryCases: boundaryCases.map(browserRequest),
      b3S4096Case: browserRequest(b3S4096Case),
      rejectedInputs
    }
  );
  page.off('request', onRequest);

  const diagnosticPath = outputPath.replace(/\.json$/u, '.online-diagnostic.json');
  const {passed: b3S4096Passed, error: b3S4096ParityError} = compareFullContextBatch(
    online.b3S4096Results,
    candidate.b3S4096Case
  );
  const sourceParity = {
    text: textCases.map((item, index) => {
      const sourceCase = candidate.sourceCases[index]!;
      return {
        name: item.name,
        inputs: {
          request: sourceCase.request,
          requestMetadata: sourceCase.requestMetadata ?? null,
          imageFiles: sourceCase.imageFiles ?? [],
          imageGeometry: sourceCase.imageGeometry ?? []
        },
        comparison: buildQwen35ResultParityDiagnostic(
          online.textResults[index]?.result,
          sourceCase.expected,
          `text/${item.name}`,
          undefined,
          sourceCase.candidateUsageExpectation
        )
      };
    }),
    vision: visionCases.map((item, index) => {
      const sourceCase = candidate.visionCases[index]!;
      return {
        name: item.name,
        inputs: {
          request: sourceCase.request,
          requestMetadata: sourceCase.requestMetadata ?? null,
          imageFiles: sourceCase.imageFiles ?? [],
          imageGeometry: sourceCase.imageGeometry ?? []
        },
        comparison: buildQwen35ResultParityDiagnostic(
          online.visionResults[index]?.result,
          sourceCase.expected,
          `vision/${item.name}`,
          sourceCase.precisionReference
        )
      };
    }),
    acceptedBoundaries: acceptedBoundaries.map((item, index) => {
      const observed = online.boundaryResults[index];
      return {
        name: item.name,
        inputs: {
          request: item.request,
          imageFiles: item.imageFiles ?? []
        },
        comparison: buildQwen35ResultParityDiagnostic(
          observed?.result,
          item.expected,
          `accepted-boundary/${item.name}`,
          undefined,
          item.candidateUsageExpectation
        )
      };
    }),
    b3S4096: {
      name: candidate.b3S4096Case.name,
      sourceReference: candidate.b3S4096Case.sourceReference,
      batchSize: candidate.b3S4096Case.batchSize,
      sequenceLength: candidate.b3S4096Case.sequenceLength,
      expectedUsage: candidate.b3S4096Case.expectedUsage,
      expectedProbabilities: candidate.b3S4096Case.expectedProbabilities,
      actualResult: online.b3S4096Results[0]?.result ?? null,
      status: b3S4096Passed ? 'passed' : 'failed',
      error: b3S4096ParityError
    },
    rejectedBoundaries: rejectedInputs.map((item, index) => {
      const observed = online.failures[index];
      const sourceCase = rejectedBoundaries[index];
      const matches = observed ? observed.name === item.name && observed.code === item.expectedErrorCode : null;
      return {
        name: item.name,
        sourceInput: {
          request: sourceCase?.request ?? item.request,
          expectedErrorCode: item.expectedErrorCode
        },
        candidateFailure: observed ?? null,
        status: matches === true ? 'passed' : matches === false ? 'failed' : 'incomplete'
      };
    })
  };
  const parityStatuses = [
    ...sourceParity.text.map(({comparison}) => comparison.status),
    ...sourceParity.vision.map(({comparison}) => comparison.status),
    ...sourceParity.acceptedBoundaries.map(({comparison}) => comparison.status),
    sourceParity.b3S4096.status,
    ...sourceParity.rejectedBoundaries.map(({status}) => status)
  ];
  const parityStatus = summarizeParityStatus(parityStatuses);
  await writeQwen35OnlineDiagnostic(diagnosticPath, {
    schemaVersion: 2,
    status: 'online-inference-returned-unverified',
    modelId: QWEN35_RELEASE_MODEL.catalogModelId,
    package: {
      archiveSha256: packageEvidence.archive.archiveSha256,
      candidateBundleSha256: installedBundleSha256
    },
    artifacts: {
      generatedManifestSha256: candidate.manifestSha256,
      assetManifestSha256: candidate.assetManifestSha256
    },
    browser: {name: browserName, channel: process.env.WEB_DECISION_BROWSER_CHANNEL, version: browser.version()},
    hardware: {platform: os.platform(), arch: os.arch(), chipType: host.chipType},
    gpu: {adapter: online.adapter.info, isFallbackAdapter: online.adapter.isFallbackAdapter},
    tvmExecutionEvidence: {
      mainCallSites,
      graphDispatchCounts: online.graphDispatchCounts,
      tvmRuntimeDispatchCounts: online.tvmRuntimeDispatchCounts,
      graphDispatchCallers: online.tvmRuntimeCallers
    },
    onlineNetworkRequests,
    parity: {status: parityStatus, sourceParity},
    results: {
      text: online.textResults,
      vision: online.visionResults,
      boundaries: online.boundaryResults,
      b3S4096: online.b3S4096Results,
      rejectedBoundaries: online.failures
    }
  });

  const gateFailures: Qwen35GateFailure[] = [];
  await recordQwen35GateFailure(gateFailures, 'online', 'checkpoint repository', () =>
    expect(online.description.checkpoint.repository).toBe(QWEN35_RELEASE_MODEL.repository)
  );
  await recordQwen35GateFailure(gateFailures, 'online', 'checkpoint revision', () =>
    expect(online.description.checkpoint.revision).toBe(QWEN35_RELEASE_MODEL.revision)
  );
  await recordQwen35GateFailure(gateFailures, 'online', 'asset manifest', () => {
    expect(online.description.assets.manifestSha256).toBe(candidate.assetManifestSha256);
    expect(online.manifestSha256).toBe(candidate.assetManifestSha256);
  });
  const candidateExecution = {
    backend: online.description.backend.name,
    runtime: 'TVM.js',
    graphMainCallSites: mainCallSites,
    graphDispatchCounts: online.graphDispatchCounts,
    tvmRuntimeDispatchCounts: online.tvmRuntimeDispatchCounts,
    graphDispatchCallers: online.tvmRuntimeCallers,
    graphsExecuted: {
      r19:
        mainCallSites.r19 &&
        online.textResults.length > 0 &&
        (online.graphDispatchCounts.text ?? 0) > 0 &&
        (online.tvmRuntimeDispatchCounts.text ?? 0) > 0 &&
        (online.graphDispatchCounts['batch-3-context-4096'] ?? 0) > 0 &&
        (online.tvmRuntimeDispatchCounts['batch-3-context-4096'] ?? 0) > 0,
      vision:
        mainCallSites.vision &&
        online.visionResults.length > 0 &&
        (online.graphDispatchCounts.vision ?? 0) > 0 &&
        (online.tvmRuntimeDispatchCounts.vision ?? 0) > 0
    },
    tvmUsed:
      online.description.backend.name === 'tvm-webgpu' &&
      mainCallSites.tvmRuntimeDispatch &&
      mainCallSites.r19 &&
      mainCallSites.vision &&
      online.textResults.length > 0 &&
      online.visionResults.length > 0 &&
      (online.graphDispatchCounts.text ?? 0) > 0 &&
      (online.tvmRuntimeDispatchCounts.text ?? 0) > 0 &&
      (online.graphDispatchCounts.vision ?? 0) > 0 &&
      (online.tvmRuntimeDispatchCounts.vision ?? 0) > 0 &&
      (online.graphDispatchCounts['batch-3-context-4096'] ?? 0) > 0
  };
  const tvmUsed = candidateExecution.tvmUsed;
  await recordQwen35GateFailure(gateFailures, 'online', 'direct TVM WebGPU execution', () => {
    expect(candidateExecution.backend).toBe('tvm-webgpu');
    expect(candidateExecution.runtime).toBe('TVM.js');
    expect(candidateExecution.graphMainCallSites).toEqual({r19: true, vision: true, tvmRuntimeDispatch: true});
    expect(candidateExecution.graphDispatchCounts.text).toBeGreaterThan(0);
    expect(candidateExecution.graphDispatchCounts.vision).toBeGreaterThan(0);
    expect(candidateExecution.graphDispatchCounts['batch-3-context-4096']).toBeGreaterThan(0);
    expect(candidateExecution.tvmRuntimeDispatchCounts.text).toBeGreaterThan(0);
    expect(candidateExecution.tvmRuntimeDispatchCounts.vision).toBeGreaterThan(0);
    expect(candidateExecution.tvmRuntimeDispatchCounts['batch-3-context-4096']).toBeGreaterThan(0);
    expect(candidateExecution.graphDispatchCallers.text).toContain('tvm.generated');
    expect(candidateExecution.graphDispatchCallers.vision).toContain('tvm.generated');
    expect(candidateExecution.graphDispatchCallers['batch-3-context-4096']).toContain('tvm.generated');
    expect(candidateExecution.graphsExecuted).toEqual({r19: true, vision: true});
    expect(tvmUsed).toBe(true);
  });
  for (const [index, item] of textCases.entries())
    await recordQwen35GateFailure(gateFailures, 'online', `source parity ${item.name}`, () =>
      compareQwen35Result(
        online.textResults[index]?.result,
        candidate.sourceCases[index]!.expected,
        `Qwen source parity ${item.name}`,
        undefined,
        candidate.sourceCases[index]!.candidateUsageExpectation
      )
    );
  for (const [index, item] of visionCases.entries())
    await recordQwen35GateFailure(gateFailures, 'online', `official vision ${item.name}`, () =>
      compareQwen35Result(
        online.visionResults[index]?.result,
        candidate.visionCases[index]!.expected,
        `Qwen official vision ${item.name}`,
        candidate.visionCases[index]!.precisionReference
      )
    );
  for (const [index, item] of acceptedBoundaries.entries())
    await recordQwen35GateFailure(gateFailures, 'online', `boundary ${item.name}`, () =>
      compareQwen35Result(
        online.boundaryResults[index]?.result,
        item.expected,
        `Qwen boundary ${item.name}`,
        undefined,
        item.candidateUsageExpectation
      )
    );
  await recordQwen35GateFailure(gateFailures, 'online', 'rejected boundaries', () =>
    expect(online.failures.map(({name, code}) => ({name, code}))).toEqual(
      rejectedInputs.map(({name, expectedErrorCode}) => ({name, code: expectedErrorCode}))
    )
  );
  await recordQwen35GateFailure(gateFailures, 'online', 'batch 3 × context 4096 source parity', () => {
    expect(b3S4096Passed, b3S4096ParityError ?? undefined).toBe(true);
  });

  let cacheDiagnostics: Qwen35CacheDiagnostics = {
    cacheNames: [],
    cachedAssetUrls: []
  };
  await recordQwen35GateFailure(gateFailures, 'online', 'asset cache diagnostics', async () => {
    const rawCacheDiagnostics = await page.evaluate(async () => {
      const cacheNames = await caches.keys();
      const urls: string[] = [];
      const sourceMarkerUrl = 'https://huggingface.co/qwen35-source-offline-complete-v1.json';
      const tokenizerMarkerUrl = 'https://huggingface.co/web-decision-qwen35-tokenizer-offline-complete-v1.json';
      const markers: {source: CacheMarkerEvidence | null; tokenizer: CacheMarkerEvidence | null} = {
        source: null,
        tokenizer: null
      };
      for (const name of cacheNames) {
        const cache = await caches.open(name);
        for (const request of await cache.keys()) urls.push(request.url);
        for (const [key, url] of [
          ['source', sourceMarkerUrl],
          ['tokenizer', tokenizerMarkerUrl]
        ] as const) {
          const response = await cache.match(url);
          if (!response) continue;
          try {
            const marker = (await response.json()) as {format?: unknown; markerId?: unknown};
            markers[key] = {format: marker.format ?? null, markerId: marker.markerId ?? null};
          } catch {
            markers[key] = {format: null, markerId: null};
          }
        }
      }
      const telemetry = (
        window as unknown as {
          __webDecisionQwen35CachePutTelemetry?: {snapshot(): unknown; restore(): void};
        }
      ).__webDecisionQwen35CachePutTelemetry;
      const storageEstimate = await (async () => {
        try {
          return navigator.storage?.estimate ? await navigator.storage.estimate() : null;
        } catch (cause) {
          const error = cause as {name?: unknown; message?: unknown};
          return {
            errorName: typeof error?.name === 'string' ? error.name : 'UnknownError',
            errorMessage: typeof error?.message === 'string' ? error.message : String(cause)
          };
        }
      })();
      const cachePuts = telemetry?.snapshot() ?? null;
      telemetry?.restore();
      return {
        cacheNames: cacheNames.sort(),
        cachedAssetUrls: urls.sort(),
        markers,
        storageEstimate,
        cachePuts
      };
    });
    cacheDiagnostics = {
      cacheNames: rawCacheDiagnostics.cacheNames,
      cachedAssetUrls: rawCacheDiagnostics.cachedAssetUrls.map(publicRequestUrl),
      cacheEntries: summarizeCacheStorageUrls(
        rawCacheDiagnostics.cachedAssetUrls,
        'https://huggingface.co/qwen35-source-offline-complete-v1.json',
        'https://huggingface.co/web-decision-qwen35-tokenizer-offline-complete-v1.json'
      ),
      markers: rawCacheDiagnostics.markers,
      storageBeforeOnline,
      storageAfterOnline: rawCacheDiagnostics.storageEstimate,
      rangeCachePuts: rawCacheDiagnostics.cachePuts
    };
    expect(rawCacheDiagnostics.cacheNames).toContain('web-decision-assets-v1');
    expect(rawCacheDiagnostics.cachedAssetUrls.length).toBeGreaterThan(0);
  });
  await recordQwen35GateFailure(gateFailures, 'online', 'complete verified cache markers', () => {
    expect(cacheDiagnostics.markers?.source, JSON.stringify(cacheDiagnostics)).not.toBeNull();
    expect(cacheDiagnostics.markers?.tokenizer, JSON.stringify(cacheDiagnostics)).not.toBeNull();
  });
  const rangedHuggingFaceRequests = onlineNetworkRequests.filter((request) => {
    const hostname = new URL(request.url).hostname.toLowerCase();
    return (
      (hostname === 'hf.co' ||
        hostname.endsWith('.hf.co') ||
        hostname === 'huggingface.co' ||
        hostname.endsWith('.huggingface.co')) &&
      request.method === 'GET' &&
      request.rangeHeader !== null
    );
  });
  await recordQwen35GateFailure(gateFailures, 'online', 'ranged Hugging Face requests', () =>
    expect(rangedHuggingFaceRequests.length).toBeGreaterThan(0)
  );

  await recordQwen35GateFailure(gateFailures, 'online', 'diagnostic update', async () => {
    await writeQwen35OnlineDiagnostic(diagnosticPath, {
      schemaVersion: 2,
      status: 'online-inference-returned-unverified',
      modelId: QWEN35_RELEASE_MODEL.catalogModelId,
      package: {
        archiveSha256: packageEvidence.archive.archiveSha256,
        candidateBundleSha256: installedBundleSha256
      },
      artifacts: {
        generatedManifestSha256: candidate.manifestSha256,
        assetManifestSha256: candidate.assetManifestSha256
      },
      browser: {name: browserName, channel: process.env.WEB_DECISION_BROWSER_CHANNEL, version: browser.version()},
      hardware: {platform: os.platform(), arch: os.arch(), chipType: host.chipType},
      gpu: {adapter: online.adapter.info, isFallbackAdapter: online.adapter.isFallbackAdapter},
      onlineNetworkRequests,
      cacheDiagnostics,
      parity: {status: parityStatus, sourceParity, gateFailures},
      results: {
        text: online.textResults,
        vision: online.visionResults,
        boundaries: online.boundaryResults,
        rejectedBoundaries: online.failures
      }
    });
  });

  const offlineNetworkAttempts: string[] = [];
  let offlineRouteInstalled = false;
  await recordQwen35GateFailure(gateFailures, 'offline', 'network blocker setup', async () => {
    await page.route('**/*', async (route) => {
      offlineNetworkAttempts.push(publicRequestUrl(route.request().url()));
      await route.abort('internetdisconnected');
    });
    offlineRouteInstalled = true;
  });
  const offlineVisionIndices = visionCases.map((_, index) => index);
  const offlineCases = [...textCases, b3S4096Case, ...offlineVisionIndices.map((index) => visionCases[index]!)];
  let offline: Array<{name: string; result: unknown}> = [];
  let offlineError: string | null = null;
  try {
    if (offlineRouteInstalled)
      offline = await page.evaluate(
        async ({modelId, cases}: {modelId: string; cases: readonly BrowserCase[]}) => {
          type Api = {
            loadDecisionModel(options: {modelId: string; cache: boolean; offline: boolean}): Promise<{
              modelId: string;
              description: {checkpoint: {repository: string; revision: string}; assets: {manifestSha256: string}};
              systemOne(request: unknown, options?: unknown): Promise<unknown>;
              dispose(): Promise<void>;
            }>;
          };
          const api = window.webDecision as unknown as Api;
          const reportProgress = async (event: Record<string, unknown>) => {
            const callback = (
              window as unknown as {
                __webDecisionQwen35Progress?: (value: Record<string, unknown>) => Promise<void>;
              }
            ).__webDecisionQwen35Progress;
            if (!callback) throw new Error('The Qwen progress writer is unavailable.');
            await callback(event);
          };
          await reportProgress({
            type: 'model-load-started',
            phase: 'offline',
            modelId,
            cache: true,
            offline: true
          });
          let model: Awaited<ReturnType<Api['loadDecisionModel']>>;
          try {
            model = await api.loadDecisionModel({modelId, cache: true, offline: true});
          } catch (error) {
            await reportProgress({
              type: 'model-load-failed',
              phase: 'offline',
              modelId,
              error: error instanceof Error ? error.message : String(error)
            });
            throw error;
          }
          await reportProgress({
            type: 'model-loaded',
            phase: 'offline',
            modelId: model.modelId,
            checkpoint: model.description.checkpoint,
            manifestSha256: model.description.assets.manifestSha256
          });
          try {
            const results: Array<{name: string; result: unknown}> = [];
            for (const item of cases) {
              const request = {
                ...item.request,
                questions: Array.isArray(item.request.questions)
                  ? new Map(item.request.questions as Array<[string, unknown]>)
                  : item.request.questions,
                ...(item.images.length > 0
                  ? {images: item.images.map(({base64, mimeType}) => `data:${mimeType};base64,${base64}`)}
                  : {})
              };
              const result = await model.systemOne(request, {});
              await reportProgress({
                type: 'case-completed',
                phase: 'offline',
                group: 'cache-reuse',
                name: item.name,
                result
              });
              results.push({name: item.name, result});
            }
            return results;
          } finally {
            await model.dispose();
          }
        },
        {modelId: QWEN35_RELEASE_MODEL.catalogModelId, cases: offlineCases.map(browserRequest)}
      );
  } catch (error) {
    offlineError = error instanceof Error ? error.message : String(error);
    await recordQwen35GateFailure(gateFailures, 'offline', 'cache inference', () => {
      throw error;
    });
  } finally {
    if (offlineRouteInstalled)
      await recordQwen35GateFailure(gateFailures, 'offline', 'network blocker cleanup', async () => {
        await page.unrouteAll();
      });
  }
  const offlineDiagnosticPath = outputPath.replace(/\.json$/u, '.offline-diagnostic.json');
  await recordQwen35GateFailure(gateFailures, 'offline', 'offline diagnostic write', async () => {
    await writeFile(
      offlineDiagnosticPath,
      `${JSON.stringify(
        {
          schemaVersion: 1,
          status: 'offline-inference-returned-unverified',
          modelId: QWEN35_RELEASE_MODEL.catalogModelId,
          onlineResults: [
            ...online.textResults,
            ...online.b3S4096Results,
            ...offlineVisionIndices.map((index) => online.visionResults[index]!)
          ],
          offlineResults: offline,
          offlineError,
          offlineNetworkAttempts
        },
        null,
        2
      )}\n`
    );
  });
  await recordQwen35GateFailure(gateFailures, 'offline', 'network attempts', () =>
    expect(offlineNetworkAttempts).toEqual([])
  );
  const onlineOfflineResults = [
    ...online.textResults,
    ...online.b3S4096Results,
    ...offlineVisionIndices.map((index) => online.visionResults[index]!)
  ];
  await recordQwen35GateFailure(gateFailures, 'offline', 'online/offline parity', () =>
    assertQwen35OfflineResults(onlineOfflineResults, offline)
  );

  let chromeGpu: {status: string | null; renderer: string | null} = {status: null, renderer: null};
  await recordQwen35GateFailure(gateFailures, 'adapter', 'Chrome GPU report', async () => {
    const gpuPage = await page.context().newPage();
    try {
      chromeGpu = await readChromeGpuReport(gpuPage);
    } finally {
      await gpuPage.close();
    }
  });
  await recordQwen35GateFailure(gateFailures, 'adapter', 'hardware acceleration', () =>
    expect(chromeGpu.status).toMatch(/hardware accelerated/iu)
  );
  await recordQwen35GateFailure(gateFailures, 'adapter', 'M4 Pro renderer', () =>
    expect(chromeGpu.renderer).toMatch(/Apple M4 Pro/iu)
  );
  await recordQwen35GateFailure(gateFailures, 'adapter', 'Apple adapter vendor', () =>
    expect(String(online.adapter.info.vendor).toLowerCase()).toContain('apple')
  );
  await recordQwen35GateFailure(gateFailures, 'adapter', 'physical adapter', () =>
    expect(online.adapter.isFallbackAdapter).toBe(false)
  );
  await recordQwen35GateFailure(gateFailures, 'online', 'source parity status', () =>
    expect(parityStatus).toBe('passed')
  );

  const report = {
    schemaVersion: 1,
    status: 'candidate-browser-evidence-passed',
    generatedAt: new Date().toISOString(),
    modelId: QWEN35_RELEASE_MODEL.catalogModelId,
    source: {
      baseRepository: QWEN35_RELEASE_MODEL.baseRepository,
      baseRevision: QWEN35_RELEASE_MODEL.baseRevision,
      checkpointRepository: QWEN35_RELEASE_MODEL.repository,
      checkpointRevision: QWEN35_RELEASE_MODEL.revision,
      strandsRepository: QWEN35_RELEASE_MODEL.strandsRepository,
      strandsRevision: QWEN35_RELEASE_MODEL.strandsRevision,
      runner: 'StrandsDeciderModel.load/SystemOneEngine.evaluate',
      weightsLoaded: true,
      tvmUsed: false,
      b3S4096SourceReference: candidate.b3S4096Case.sourceReference
    },
    candidateExecution,
    package: {
      name: packageEvidence.archive.packageName,
      version: packageEvidence.archive.packageVersion,
      archiveSha256: packageEvidence.archive.archiveSha256,
      packageTreeSha256: packageEvidence.archive.packageTreeSha256,
      bundlePath: packageEvidence.archive.bundlePath,
      bundleSha256: packageEvidence.archive.bundleSha256,
      candidateBundlePath: installedBundle,
      candidateBundleSha256: installedBundleSha256,
      exactInstallMatch: true
    },
    artifacts: {
      root: artifactRoot,
      manifestPath: candidate.manifestPath,
      manifestSha256: candidate.manifestSha256,
      manifestRawSha256: candidate.manifestRawSha256,
      assetManifestSha256: candidate.assetManifestSha256,
      manifestFiles: candidate.manifestFiles,
      filesSha256: candidate.artifactSha256,
      evidenceInputSha256: candidate.evidenceInputSha256,
      fixtureSha256: candidate.fixtureSha256
    },
    browser: {
      name: browserName,
      channel: process.env.WEB_DECISION_BROWSER_CHANNEL,
      version: browser.version(),
      userAgent: online.userAgent
    },
    hardware: {platform: os.platform(), arch: os.arch(), chipType: host.chipType, machineModel: host.machineModel},
    gpu: {
      api: 'webgpu',
      adapter: online.adapter.info,
      isFallbackAdapter: online.adapter.isFallbackAdapter,
      features: online.adapter.features,
      limits: online.adapter.limits,
      renderer: chromeGpu.renderer,
      chromeGpuStatus: chromeGpu.status,
      chromeGpu
    },
    checks: {
      curatedPublicApi: online.listed.includes(QWEN35_RELEASE_MODEL.catalogModelId),
      sourceParity: true,
      typedChoiceScoreNoul: true,
      officialVision: true,
      maximumContextBoundary: true,
      maximumChoiceOptionsBoundary: true,
      maximumScoreLevelsBoundary: true,
      maximumImageCountBoundary: true,
      batchBoundaries: true,
      fullContextBatchBoundary: b3S4096Passed,
      onlineInference: true,
      offlineCacheReuse: true
    },
    onlineNetworkRequests,
    rangedHuggingFaceRequests,
    offlineNetworkAttempts,
    cacheDiagnostics,
    results: {
      text: online.textResults,
      vision: online.visionResults,
      boundaries: online.boundaryResults,
      b3S4096: online.b3S4096Results,
      rejectedBoundaries: online.failures,
      offline
    },
    fullContextBatch: {
      sourceReference: candidate.b3S4096Case.sourceReference,
      batchSize: candidate.b3S4096Case.batchSize,
      sequenceLength: candidate.b3S4096Case.sequenceLength,
      expectedUsage: candidate.b3S4096Case.expectedUsage,
      expectedProbabilities: candidate.b3S4096Case.expectedProbabilities,
      result: online.b3S4096Results[0]?.result ?? null,
      parityPassed: b3S4096Passed
    },
    evidenceScope:
      'Exact installed npm archive browser candidate evidence; this report does not mark or release the model.'
  };
  await recordQwen35GateFailure(gateFailures, 'report', 'evidence schema', () => {
    assertQwen35EvidenceRecord(report);
  });
  await writeQwen35ReportThenThrow(gateFailures, async (failures) => {
    await mkdir(dirname(outputPath), {recursive: true});
    if (failures.length > 0) {
      const failedReport = {
        schemaVersion: 1,
        status: 'candidate-browser-evidence-failed',
        generatedAt: new Date().toISOString(),
        modelId: QWEN35_RELEASE_MODEL.catalogModelId,
        package: report.package,
        artifacts: report.artifacts,
        browser: report.browser,
        hardware: report.hardware,
        gpu: report.gpu,
        onlineDiagnosticPath: diagnosticPath,
        offlineDiagnosticPath,
        gateFailures: failures,
        candidateExecution,
        sourceParity,
        onlineNetworkRequests,
        rangedHuggingFaceRequests,
        offlineNetworkAttempts,
        cacheDiagnostics,
        results: report.results
      };
      await writeFile(outputPath, `${JSON.stringify(failedReport, null, 2)}\n`);
      return;
    }
    await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`);
  });
  console.info(
    `QWEN35_CANDIDATE_EVIDENCE:${JSON.stringify({path: outputPath, archiveSha256: packageEvidence.archive.archiveSha256})}`
  );
});
