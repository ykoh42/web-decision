import {open, readFile, realpath, type FileHandle} from 'node:fs/promises';
import {dirname, resolve} from 'node:path';
import {expect, test, type Route} from '@playwright/test';
import {readHardwareReport} from '../../../../tests/browser/mapika2b-hardware.mts';
import {gemmaV2EvidenceFingerprint, verifyGemmaV2Candidate} from './gemma4-v2-candidate-evidence.mts';
import {verifyGemmaV2Cache} from './gemma4-v2-cache-evidence.mts';
import {fileURLToPath} from 'node:url';

const checkpointUrl =
  'https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v2-2611/resolve/a5f32ea1bffe2e91c25c0c641d444e8115e28212/model.safetensors';
async function serveRange(route: Route, checkpoint: FileHandle) {
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
}

test('browser can read the Content-Range exposed by the checkpoint fixture', async ({page}) => {
  test.skip(process.env.WEB_DECISION_GEMMA_V2_CANDIDATE !== '1', 'Enable the private candidate test');
  const root = resolve(process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT!);
  const checkpoint = await open(resolve(root, 'strands-e2b-v2/model.safetensors'), 'r');
  page.once('close', () => {
    void checkpoint.close();
  });
  await page.route(checkpointUrl, (route) => serveRange(route, checkpoint));
  await page.goto('/');
  const observed = await page.evaluate(async (url) => {
    const response = await fetch(url, {headers: {Range: 'bytes=0-0'}});
    return {
      status: response.status,
      range: response.headers.get('content-range'),
      bytes: (await response.arrayBuffer()).byteLength
    };
  }, checkpointUrl);
  expect(observed).toEqual({status: 206, range: 'bytes 0-0/10246621918', bytes: 1});
});

test('installed private candidate exposes the experimental model ID', async ({page}) => {
  test.skip(process.env.WEB_DECISION_GEMMA_V2_CANDIDATE !== '1', 'Enable the private candidate test');
  await page.goto('/');
  await page.waitForFunction(() => typeof window.webDecision?.listModels === 'function');
  const models = await page.evaluate(() => window.webDecision.listModels());
  expect(models).toEqual([{modelId: 'strands-decider-e2b-gemma4-v2-2611', status: 'private-test-candidate'}]);
});

test('browser rehashes candidate CacheStorage bytes', async ({page}) => {
  test.skip(process.env.WEB_DECISION_GEMMA_V2_CANDIDATE !== '1', 'Enable the private candidate test');
  await page.goto('/');
  await page.evaluate(async () => {
    const bytes = new Uint8Array([1, 2, 3]);
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (value) =>
      value.toString(16).padStart(2, '0')
    ).join('');
    const cache = await caches.open('web-decision-assets-v1');
    await cache.put(`/.web-decision-cache/sha256/?digest=${digest}`, new Response(bytes));
  });
  expect(await page.evaluate(verifyGemmaV2Cache)).toEqual({verifiedBytes: 3, verifiedEntries: 1});
});

test('private installed candidate loads Gemma v2 by model ID and matches the reference', async ({page}) => {
  test.skip(process.env.WEB_DECISION_GEMMA_V2_CANDIDATE !== '1', 'Enable the private candidate GPU test');
  test.setTimeout(600000);
  const reportPath = process.env.WEB_DECISION_GEMMA_V2_CANDIDATE_REPORT;
  if (!reportPath) throw new Error('The exact private candidate report is required.');
  const bundle = await realpath(process.env.WEB_DECISION_INSTALLED_BUNDLE!);
  const evidence = await verifyGemmaV2Candidate(reportPath, bundle);
  const fingerprint = await gemmaV2EvidenceFingerprint(reportPath, fileURLToPath(import.meta.url));
  console.log(
    `PRIVATE_CANDIDATE_EVIDENCE ${JSON.stringify({archiveSha256: evidence.archiveSha256, sourceSnapshotSha256: evidence.sourceSnapshotSha256, fingerprint})}`
  );
  const installed = resolve(dirname(bundle), '..');
  const metadata = JSON.parse(await readFile(resolve(installed, 'package.json'), 'utf8'));
  expect(metadata.private).toBe(true);
  expect(metadata.version).toBe('0.0.0-gemma4-v2-private-test');
  const root = resolve(process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT!);
  const expected = JSON.parse(await readFile(resolve(root, 'e2b-v2-vision-reference/result.json'), 'utf8'));
  const checkpointRoot = resolve(root, 'strands-e2b-v2');
  const checkpoint = await open(resolve(checkpointRoot, 'model.safetensors'), 'r');
  page.once('close', () => {
    void checkpoint.close();
  });
  const names = new Set([
    'config.json',
    'decision_config.json',
    'processor_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'readout.safetensors'
  ]);
  await page.route(
    'https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v2-2611/resolve/a5f32ea1bffe2e91c25c0c641d444e8115e28212/**',
    async (route) => {
      const name = new URL(route.request().url()).pathname.split('/').at(-1)!;
      if (name === 'model.safetensors') {
        await serveRange(route, checkpoint);
        return;
      }
      if (!names.has(name)) throw new Error(`Unexpected upstream asset ${name}`);
      await route.fulfill({
        status: 200,
        headers: {'access-control-allow-origin': '*'},
        body: await readFile(resolve(checkpointRoot, name))
      });
    }
  );
  expect((await readHardwareReport(page)).status).toBe('Hardware accelerated');
  await page.goto('/');
  await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');
  const actual = await page.evaluate(async () => {
    const api = window.webDecision as unknown as {
      listModels(): Array<{modelId: string; status: string}>;
      loadDecisionModel(options: {modelId: string; cache: boolean}): Promise<{
        systemOne(
          request: unknown
        ): Promise<{answers: {color: {choice: string; probabilities: Record<string, number>}}}>;
        dispose(): Promise<void>;
      }>;
    };
    if (api.listModels()[0]?.status !== 'private-test-candidate') throw new Error('Unexpected package entry');
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Canvas unavailable');
    context.fillStyle = 'rgb(220,20,30)';
    context.fillRect(0, 0, 32, 64);
    context.fillStyle = 'rgb(15,40,210)';
    context.fillRect(32, 0, 32, 64);
    const image = canvas.toDataURL('image/png').split(',')[1]!;
    const model = await api.loadDecisionModel({modelId: 'strands-decider-e2b-gemma4-v2-2611', cache: false});
    try {
      return await model.systemOne({
        state: 'The image contains two color regions.',
        images: [image],
        questions: {
          color: {type: 'choice', instructions: 'Which color is on the left?', criteria: {red: 'red', blue: 'blue'}}
        }
      });
    } finally {
      await model.dispose();
    }
  });
  expect(actual.answers.color.choice).toBe(expected.answers.color.choice);
  const delta = Math.abs(actual.answers.color.probabilities.red! - expected.answers.color.probabilities.red);
  console.log(
    `private candidate: upstream red=${expected.answers.color.probabilities.red}; installed=${actual.answers.color.probabilities.red}; delta=${delta}`
  );
  expect(delta).toBeLessThan(0.02);
  await verifyGemmaV2Candidate(reportPath, bundle);
  expect(await gemmaV2EvidenceFingerprint(reportPath, fileURLToPath(import.meta.url))).toBe(fingerprint);
});
