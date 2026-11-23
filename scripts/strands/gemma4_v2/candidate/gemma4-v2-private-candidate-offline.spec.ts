import {createHash} from 'node:crypto';
import {open, readFile, mkdtemp, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {chromium, expect, test, type BrowserContext} from '@playwright/test';
import {readHardwareReport} from '../../../../tests/browser/mapika2b-hardware.mts';
import {gemmaV2EvidenceFingerprint, verifyGemmaV2Candidate} from './gemma4-v2-candidate-evidence.mts';
import {verifyGemmaV2Cache} from './gemma4-v2-cache-evidence.mts';
import {fileURLToPath} from 'node:url';

const modelId = 'strands-decider-e2b-gemma4-v2-2611';
const upstream =
  'https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v2-2611/resolve/a5f32ea1bffe2e91c25c0c641d444e8115e28212/**';
const expectedWeightBytes = 9_638_447_838;
const sha256 = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');

type ImageResult = {
  answers: {color: {choice: string; probabilities: Record<string, number>}};
  usage: {input_tokens: number; output_tokens: number};
};

async function readOfflineReference(root: string) {
  const requestPath = process.env.WEB_DECISION_REFERENCE_REQUEST;
  const resultPath = process.env.WEB_DECISION_REFERENCE_RESULT;
  if (requestPath === undefined && resultPath === undefined) {
    const expected = JSON.parse(await readFile(resolve(root, 'e2b-v2-vision-reference/result.json'), 'utf8'));
    return {expected: expected as ImageResult, exact: null};
  }
  if (!requestPath || !resultPath)
    throw new Error('Set WEB_DECISION_REFERENCE_REQUEST and WEB_DECISION_REFERENCE_RESULT together.');
  const [requestBytes, resultBytes] = await Promise.all([readFile(requestPath), readFile(resultPath)]);
  const requestJson = requestBytes.toString('utf8');
  JSON.parse(requestJson);
  const expected = JSON.parse(resultBytes.toString('utf8')) as ImageResult;
  return {expected, exact: {requestPath, resultPath, requestBytes, resultBytes, requestJson}};
}

async function pageFor(context: BrowserContext) {
  const page = await context.newPage();
  await page.goto(`http://127.0.0.1:${process.env.WEB_DECISION_BROWSER_PORT || 41773}/`);
  await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');
  return page;
}

test('the exact private archive prepares and runs its model offline after a browser restart', async () => {
  test.skip(
    process.env.WEB_DECISION_GEMMA_V2_CANDIDATE_OFFLINE !== '1',
    'Enable the private candidate disk and GPU test'
  );
  test.setTimeout(3_600_000);
  const report = process.env.WEB_DECISION_GEMMA_V2_CANDIDATE_REPORT;
  const bundle = process.env.WEB_DECISION_INSTALLED_BUNDLE;
  if (!report || !bundle) throw new Error('The exact candidate report and installed bundle are required.');
  const evidence = await verifyGemmaV2Candidate(report, bundle);
  const fingerprint = await gemmaV2EvidenceFingerprint(report, fileURLToPath(import.meta.url));
  console.log(
    `PRIVATE_CANDIDATE_EVIDENCE ${JSON.stringify({archiveSha256: evidence.archiveSha256, sourceSnapshotSha256: evidence.sourceSnapshotSha256, fingerprint})}`
  );
  const root = resolve(process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT!);
  const checkpointRoot = resolve(root, 'strands-e2b-v2');
  const {expected, exact} = await readOfflineReference(root);
  const checkpoint = await open(resolve(checkpointRoot, 'model.safetensors'), 'r');
  const profile = await mkdtemp(join(tmpdir(), 'web-decision-gemma-v2-private-offline-'));
  const names = new Set([
    'config.json',
    'decision_config.json',
    'processor_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'readout.safetensors'
  ]);
  let requests = 0;
  const options = {
    headless: process.env.WEB_DECISION_REAL_GPU_HEADLESS === '1',
    args: ['--enable-unsafe-webgpu', '--disable-background-networking']
  };
  let context: BrowserContext | undefined;
  try {
    context = await chromium.launchPersistentContext(profile, options);
    const hardware = await context.newPage();
    expect((await readHardwareReport(hardware)).status).toBe('Hardware accelerated');
    await hardware.close();
    await context.route(upstream, async (route) => {
      requests++;
      const name = new URL(route.request().url()).pathname.split('/').at(-1)!;
      if (name === 'model.safetensors') {
        const match = /^bytes=(\d+)-(\d+)$/u.exec(route.request().headers().range ?? '');
        if (!match) throw new Error('Missing source byte range');
        const start = Number(match[1]);
        const end = Number(match[2]);
        const bytes = Buffer.alloc(end - start + 1);
        const result = await checkpoint.read(bytes, 0, bytes.length, start);
        if (result.bytesRead !== bytes.length) throw new Error('Short local checkpoint read');
        await route.fulfill({
          status: 206,
          headers: {
            'content-range': `bytes ${start}-${end}/10246621918`,
            'access-control-allow-origin': '*',
            'access-control-expose-headers': '*'
          },
          body: bytes
        });
        return;
      }
      if (!names.has(name)) throw new Error(`Unexpected upstream asset ${name}`);
      await route.fulfill({
        status: 200,
        headers: {'access-control-allow-origin': '*'},
        body: await readFile(resolve(checkpointRoot, name))
      });
    });
    const online = await pageFor(context);
    const prepared = await online.evaluate(async (id) => {
      const estimate = await navigator.storage.estimate();
      if (
        estimate.quota !== undefined &&
        estimate.usage !== undefined &&
        estimate.quota - estimate.usage < 9_750_000_000
      )
        throw new Error('Insufficient browser storage quota for the full model cache.');
      const persisted = await navigator.storage.persist();
      const api = window.webDecision as unknown as {
        loadDecisionModel(options: unknown): Promise<{
          prepareOffline(options: {onProgress(value: {loadedBytes: number}): void}): Promise<void>;
          isOfflineReady(): Promise<boolean>;
          dispose(): Promise<void>;
        }>;
      };
      const model = await api.loadDecisionModel({modelId: id});
      let progressBytes = 0;
      try {
        await model.prepareOffline({
          onProgress: ({loadedBytes}) => {
            progressBytes = loadedBytes;
          }
        });
        return {ready: await model.isOfflineReady(), persisted, progressBytes};
      } finally {
        await model.dispose();
      }
    }, modelId);
    expect(prepared.ready).toBe(true);
    expect(prepared.progressBytes).toBe(expectedWeightBytes);
    console.log(`PRIVATE_CANDIDATE_STORAGE ${JSON.stringify(prepared)}`);
    const onlineRequests = requests;
    await context.close();
    context = await chromium.launchPersistentContext(profile, options);
    const offline = await pageFor(context);
    await context.setOffline(true);
    const browserRequests: string[] = [];
    offline.on('request', (request) => browserRequests.push(request.url()));
    const cacheVerification = await offline.evaluate(verifyGemmaV2Cache);
    const actual = await offline.evaluate(
      async ({id, requestJson}) => {
        function defaultRequest() {
          const canvas = document.createElement('canvas');
          canvas.width = 64;
          canvas.height = 64;
          const ctx = canvas.getContext('2d');
          if (!ctx) throw new Error('Canvas unavailable');
          ctx.fillStyle = 'rgb(220,20,30)';
          ctx.fillRect(0, 0, 32, 64);
          ctx.fillStyle = 'rgb(15,40,210)';
          ctx.fillRect(32, 0, 32, 64);
          return {
            state: 'The image contains two color regions.',
            images: [canvas.toDataURL('image/png').split(',')[1]!],
            questions: {
              color: {type: 'choice', instructions: 'Which color is on the left?', criteria: {red: 'red', blue: 'blue'}}
            }
          };
        }

        const attempts: string[] = [];
        const originalFetch = globalThis.fetch.bind(globalThis);
        globalThis.fetch = (input, init) => {
          attempts.push(new Request(input, init).url);
          return originalFetch(input, init);
        };
        const api = window.webDecision as unknown as {
          loadDecisionModel(options: unknown): Promise<{
            isOfflineReady(): Promise<boolean>;
            systemOne(request: unknown): Promise<ImageResult>;
            dispose(): Promise<void>;
          }>;
        };
        const request = requestJson === null ? defaultRequest() : JSON.parse(requestJson);
        const model = await api.loadDecisionModel({modelId: id, offline: true});
        try {
          const ready = await model.isOfflineReady();
          const answer = await model.systemOne(request);
          return {ready, answer, attempts};
        } finally {
          await model.dispose();
        }
      },
      {id: modelId, requestJson: exact?.requestJson ?? null}
    );
    expect(actual.ready).toBe(true);
    expect(cacheVerification.verifiedBytes).toBeGreaterThanOrEqual(expectedWeightBytes);
    expect(cacheVerification.verifiedEntries).toBeGreaterThan(0);
    expect(actual.attempts).toEqual([]);
    expect(browserRequests).toEqual([]);
    expect(requests).toBe(onlineRequests);
    const {choice, probabilities} = expected.answers.color;
    expect(actual.answer.answers.color.choice).toBe(choice);
    const delta = Math.abs(actual.answer.answers.color.probabilities[choice]! - probabilities[choice]!);
    expect(delta).toBeLessThan(0.02);
    console.log(
      `PRIVATE_CANDIDATE_OFFLINE ${JSON.stringify({delta, persisted: prepared.persisted, requests: actual.attempts.length, verifiedBytes: cacheVerification.verifiedBytes, verifiedEntries: cacheVerification.verifiedEntries})}`
    );
    await verifyGemmaV2Candidate(report, bundle);
    expect(await gemmaV2EvidenceFingerprint(report, fileURLToPath(import.meta.url))).toBe(fingerprint);
    if (exact) {
      const [requestAfter, resultAfter] = await Promise.all([readFile(exact.requestPath), readFile(exact.resultPath)]);
      expect(requestAfter).toEqual(exact.requestBytes);
      expect(resultAfter).toEqual(exact.resultBytes);
      expect(actual.answer.usage).toEqual(expected.usage);
      console.log(
        `PRIVATE_CANDIDATE_OFFLINE_REFERENCE ${JSON.stringify({
          archiveSha256: evidence.archiveSha256,
          sourceSnapshotSha256: evidence.sourceSnapshotSha256,
          fingerprint,
          requestSha256: sha256(exact.requestBytes),
          referenceSha256: sha256(exact.resultBytes),
          reference: expected,
          actual: actual.answer,
          delta,
          browserRequests,
          fetchAttempts: actual.attempts,
          ...cacheVerification
        })}`
      );
    }
  } finally {
    await context?.close();
    await checkpoint.close();
    if (process.env.WEB_DECISION_KEEP_OFFLINE_PROFILE === '1')
      console.log(`Preserved temporary test profile: ${profile}`);
    else await rm(profile, {recursive: true, force: true});
  }
});
