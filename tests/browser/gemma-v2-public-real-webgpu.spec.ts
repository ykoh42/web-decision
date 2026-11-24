import {readFile, realpath} from 'node:fs/promises';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {expect, test, type Page, type Response} from '@playwright/test';
import lock from '../../packages/web-decision/src/catalog/artifacts.lock.json' with {type: 'json'};
import {archiveIdentity, cleanSourceCommit, sha256} from '../../scripts/release/curated-gpu-evidence.mjs';
import {collectReleaseEvidenceInputs} from '../../scripts/release/verify-release-evidence.ts';
import {
  GEMMA_V2_OFFICIAL_GOLDEN_SHA256,
  GEMMA_V2_OFFICIAL_SUBSET_PATH,
  GEMMA_V2_OFFICIAL_SUBSET_SHA256,
  inspectGemma4V2Conformance
} from '../../scripts/release/strands-gemma4-v2-conformance.mjs';
import {
  type CpuCase,
  GEMMA_V2_CPU_REFERENCE_PATH,
  GEMMA_V2_CPU_REFERENCE_SHA256,
  validateGemmaV2CpuReferences
} from '../../scripts/release/strands-gemma4-v2-source-reference.mjs';
import {routeGemmaExactArchive, verifyGemmaV2ScorerResources} from './gemma-v2-exact-archive-page.mjs';
import {findGemmaV2ScorerPin, gemmaV2RawProbabilityEncoding, openGemmaV2RawCapture} from './gemma-v2-raw-capture.mjs';
import {collectHardwareEvidence} from './kev06-release-evidence.ts';
import {readAdapter, readHardwareReport} from './mapika2b-hardware.mts';
import {
  gemmaV2PublicCodeRow,
  gemmaV2PublicProbabilityEncoding,
  gemmaV2PublicRunAttachment,
  gemmaV2PublicTitle,
  gemmaV2PublicVectorsAttachment,
  readGemmaV2PublicGoldenRequests,
  type PublicGemmaV2Result
} from './gemma-v2-public-evidence.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const modelId = 'strands-decider-e2b-gemma4-v2-2611';
const graphPin = lock.byModelId[modelId].manifest.files['graph.wasm'];
const upstream =
  'https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v2-2611/resolve/a5f32ea1bffe2e91c25c0c641d444e8115e28212/';
type Session = {
  systemOne(request: unknown): Promise<PublicGemmaV2Result>;
  prepareOffline(): Promise<void>;
  isOfflineReady(): Promise<boolean>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
};
type Api = {loadDecisionModel(options: {modelId: string; cache: boolean; offline?: true}): Promise<Session>};

declare global {
  interface Window {
    gemmaV2ReleaseSession?: Session;
  }
}

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error('Set ' + name + ' for the exact public archive check.');
  return value;
}

async function verifyInstalled(archive: string, bundle: string) {
  const root = resolve(dirname(bundle), '..');
  const evidence = await collectReleaseEvidenceInputs({archivePath: archive, installedPackagePath: root});
  const metadata = JSON.parse(await readFile(join(root, 'package.json'), 'utf8')) as {private?: boolean};
  expect(metadata.private).not.toBe(true);
  expect(await realpath(evidence.archive.bundlePath)).toBe(bundle);
  expect(evidence.archive.archiveSha256).toBe(requiredEnv('WEB_DECISION_RELEASE_ARCHIVE_SHA256'));
  expect(evidence.models.some((model) => model.modelId === modelId)).toBe(true);
  const graph = await readFile(join(root, 'dist/gemma4-v2/graph.wasm'));
  expect({bytes: graph.byteLength, sha256: sha256(graph)}).toEqual(graphPin);
  return evidence;
}

async function callSystemOne(page: Page, request: unknown) {
  return page.evaluate((input) => window.gemmaV2ReleaseSession!.systemOne(input), request);
}

async function collectCases(page: Page, references: Record<string, CpuCase>) {
  const results = new Map<string, PublicGemmaV2Result>();
  const cases = {};
  for (const [name, reference] of Object.entries(references)) {
    const input = JSON.parse(reference.requestJson) as unknown;
    if (!results.has(reference.requestSha256)) {
      const started = performance.now();
      results.set(reference.requestSha256, await callSystemOne(page, input));
      console.log(
        'GEMMA_PUBLIC_CASE',
        JSON.stringify({case: name, elapsedMs: Math.round(performance.now() - started)})
      );
    }
    const observed = results.get(reference.requestSha256)!;
    expect(observed.model).toBe(modelId);
    expect(observed.usage).toEqual({input_tokens: reference.inputTokens, output_tokens: reference.outputTokens});
    const result = observed.answers[reference.questionName];
    expect(result).toBeDefined();
    Object.assign(cases, {
      [name]: {
        requestSha256: reference.requestSha256,
        inputTokens: observed.usage.input_tokens,
        outputTokens: observed.usage.output_tokens,
        imageCount: reference.imageCount,
        result
      }
    });
  }
  return {cases, results};
}

async function assertImageRecovery(page: Page, image: CpuCase, expected: PublicGemmaV2Result) {
  const request = JSON.parse(image.requestJson) as {images: string[]};
  await expect(
    callSystemOne(page, {...request, images: [Buffer.from('not an image').toString('base64')]})
  ).rejects.toThrow();
  expect(await callSystemOne(page, request)).toEqual(expected);
}

async function closeSession(page: Page) {
  await page.evaluate(async () => {
    const session = window.gemmaV2ReleaseSession;
    delete window.gemmaV2ReleaseSession;
    await session?.dispose();
  });
}

async function checkOffline(page: Page, reference: CpuCase, expected: PublicGemmaV2Result) {
  await page.evaluate(async () => {
    const model = window.gemmaV2ReleaseSession!;
    await model.releaseCachedMemory();
    await model.prepareOffline();
    if (!(await model.isOfflineReady())) throw new Error('The installed Gemma cache is incomplete.');
  });
  await closeSession(page);
  const context = page.context();
  const blockedRequests: string[] = [];
  await context.route('**/*', (route) => {
    blockedRequests.push(route.request().url());
    return route.abort();
  });
  await context.setOffline(true);
  const request = JSON.parse(reference.requestJson) as unknown;
  const result = await page.evaluate(
    async ({id, input}) => {
      const api = window.webDecision as unknown as Api;
      const session = await api.loadDecisionModel({modelId: id, cache: true, offline: true});
      try {
        if (!(await session.isOfflineReady())) throw new Error('The installed offline Gemma cache is incomplete.');
        return await session.systemOne(input);
      } finally {
        await session.dispose();
      }
    },
    {id: modelId, input: request}
  );
  expect(result).toEqual(expected);
  expect(blockedRequests).toEqual([]);
  return {cacheStatus: 'complete', blockedRequests, result: result.answers[reference.questionName]};
}

function requiredConfiguredBaseURL(baseURL: string | undefined): string {
  if (!baseURL) throw new Error('The Gemma public evidence test needs a configured base URL.');
  return baseURL;
}

function isPinnedRangeResponse(response: Response): boolean {
  if (response.status() !== 206) return false;
  let request = response.request();
  while (true) {
    if (request.url() === upstream + 'model.safetensors') return true;
    const source = request.redirectedFrom();
    if (!source) return false;
    request = source;
  }
}

type CorpusRequests = ReturnType<typeof readGemmaV2PublicGoldenRequests>;
type RawCapture = Awaited<ReturnType<typeof openGemmaV2RawCapture>>;

async function collectCorpus(page: Page, requests: CorpusRequests, raw?: RawCapture) {
  const vectors: string[] = [];
  try {
    for (const {line, golden} of requests) {
      const started = performance.now();
      const invoke = () => callSystemOne(page, golden.request);
      const observed = raw
        ? await raw.capture(Object.keys(golden.questions), invoke)
        : {result: await invoke(), frames: undefined};
      expect(observed.result.model).toBe(modelId);
      vectors.push(JSON.stringify(gemmaV2PublicCodeRow(line, golden, observed.result, observed.frames)));
      console.log(
        'GEMMA_PUBLIC_ROW',
        JSON.stringify({
          line,
          completed: vectors.length,
          total: requests.length,
          elapsedMs: Math.round(performance.now() - started)
        })
      );
    }
    return {bytes: Buffer.from(vectors.join('\n') + '\n'), rawEvidence: raw?.evidence()};
  } finally {
    await raw?.close();
  }
}

test.use({serviceWorkers: 'block'});

test(gemmaV2PublicTitle, async ({page, browser}, info) => {
  test.skip(process.env.WEB_DECISION_GEMMA_V2_PUBLIC_EVIDENCE !== '1', 'Enable exact public Gemma v2 evidence.');
  test.setTimeout(86_400_000);
  const sourceCommit = cleanSourceCommit(root);
  const archive = requiredEnv('WEB_DECISION_RELEASE_ARCHIVE');
  const bundle = await realpath(requiredEnv('WEB_DECISION_INSTALLED_BUNDLE'));
  const verified = await verifyInstalled(archive, bundle);
  const goldenPath = requiredEnv('WEB_DECISION_GEMMA_V2_OFFICIAL_GOLDEN');
  const subsetPath = join(root, GEMMA_V2_OFFICIAL_SUBSET_PATH);
  const cpuPath = join(root, GEMMA_V2_CPU_REFERENCE_PATH);
  const [golden, selection, cpu] = await Promise.all([readFile(goldenPath), readFile(subsetPath), readFile(cpuPath)]);
  const requests = readGemmaV2PublicGoldenRequests(golden, selection);
  const references = validateGemmaV2CpuReferences(cpu);
  expect(browser.browserType().name()).toBe('chromium');
  expect(info.project.name).toBe('chromium');
  const hardware = await readHardwareReport(page);
  expect(hardware.status).toBe('Hardware accelerated');
  expect(hardware.renderer).toBeTruthy();
  const host = collectHardwareEvidence();
  const rawMode = process.env.WEB_DECISION_GEMMA_V2_RAW_CAPTURE === '1';
  const origin = new URL(requiredConfiguredBaseURL(info.project.use.baseURL)).origin;
  const exactPage = rawMode
    ? await routeGemmaExactArchive(
        page.context(),
        origin,
        verified.archive.installedRoot,
        verified.archive.archiveFiles
      )
    : undefined;
  await page.goto(exactPage?.url() ?? '/');
  await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');
  const adapter = await readAdapter(page);
  expect(adapter?.isFallbackAdapter).toBe(false);
  const graphResponse = page.waitForResponse(
    (response) => new URL(response.url()).pathname.endsWith('/gemma4-v2/graph.wasm'),
    {timeout: 120_000}
  );
  const upstreamResponse = page.waitForResponse(isPinnedRangeResponse, {timeout: 600_000});
  const load = page.evaluate(async (id) => {
    const api = window.webDecision as unknown as Api;
    window.gemmaV2ReleaseSession = await api.loadDecisionModel({modelId: id, cache: true});
  }, modelId);
  try {
    const [receivedGraph, receivedUpstream, {cases, results}] = await Promise.all([
      graphResponse,
      upstreamResponse,
      load.then(() => collectCases(page, references))
    ]);
    expect(sha256(await receivedGraph.body())).toBe(graphPin.sha256);
    expect(receivedUpstream.request().headers().range).toMatch(/^bytes=\d+-\d+$/u);

    const scorer = exactPage
      ? await findGemmaV2ScorerPin(verified.archive.installedRoot, verified.archive.archiveFiles)
      : undefined;
    const raw = scorer && exactPage ? await openGemmaV2RawCapture(page, exactPage.url(scorer.path), scorer) : undefined;
    const {bytes: vectorBytes, rawEvidence} = await collectCorpus(page, requests, raw);
    const metrics = inspectGemma4V2Conformance(
      golden,
      vectorBytes,
      {bytes: selection, sha256: GEMMA_V2_OFFICIAL_SUBSET_SHA256},
      rawEvidence ? 'raw-fp32' : undefined
    );
    const single = references['image-single']!;
    const singleResult = results.get(single.requestSha256)!;
    await assertImageRecovery(page, single, singleResult);
    const offlineStarted = performance.now();
    console.log('GEMMA_PUBLIC_OFFLINE', 'start');
    const offline = await checkOffline(page, single, singleResult);
    console.log(
      'GEMMA_PUBLIC_OFFLINE',
      JSON.stringify({phase: 'complete', elapsedMs: Math.round(performance.now() - offlineStarted)})
    );
    const after = await verifyInstalled(archive, bundle);
    expect(archiveIdentity(after.archive)).toEqual(archiveIdentity(verified.archive));
    expect(cleanSourceCommit(root)).toBe(sourceCommit);
    for (const [path, hash] of [
      [goldenPath, GEMMA_V2_OFFICIAL_GOLDEN_SHA256],
      [subsetPath, GEMMA_V2_OFFICIAL_SUBSET_SHA256],
      [cpuPath, GEMMA_V2_CPU_REFERENCE_SHA256]
    ] as const)
      expect(sha256(await readFile(path))).toBe(hash);
    const transport = exactPage?.evidence();
    if (rawEvidence && transport) {
      expect(transport.tokenizerServed).toBe(true);
      await verifyGemmaV2ScorerResources(
        after.archive.installedRoot,
        after.archive.archiveFiles,
        transport.served,
        rawEvidence.path
      );
    }
    const report = {
      schemaVersion: 1,
      modelId,
      source: 'pinned-upstream',
      sourceCommit,
      archive: archiveIdentity(verified.archive),
      catalogSha256: verified.catalogSha256,
      graphSha256: graphPin.sha256,
      sourceRevision: 'c3c845782095b31d438963ac2afcdfe41b49f757',
      checkpointRevision: 'a5f32ea1bffe2e91c25c0c641d444e8115e28212',
      bundledManifestSha256: lock.byModelId[modelId].manifest.rawSha256,
      cpuFixtureSha256: GEMMA_V2_CPU_REFERENCE_SHA256,
      browser: {name: browser.browserType().name(), version: browser.version()},
      hardware,
      host,
      adapter,
      assertions: {checkpointRangeObserved: true, installedGraphResponseSha256: graphPin.sha256},
      cases,
      offline,
      recovery: 'passed',
      cleanup: 'passed',
      conformance: {
        probabilityEncoding: rawEvidence ? gemmaV2RawProbabilityEncoding : gemmaV2PublicProbabilityEncoding,
        ...metrics
      },
      ...(rawEvidence && transport ? {rawCapture: {...rawEvidence, ...transport}} : {})
    };
    await info.attach(gemmaV2PublicVectorsAttachment, {body: vectorBytes, contentType: 'application/x-ndjson'});
    await info.attach(gemmaV2PublicRunAttachment, {body: JSON.stringify(report), contentType: 'application/json'});
  } finally {
    await closeSession(page);
  }
});
