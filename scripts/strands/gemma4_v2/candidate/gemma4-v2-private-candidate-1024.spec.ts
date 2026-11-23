import {open, readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {expect, test} from '@playwright/test';
import {readHardwareReport} from '../../../../tests/browser/mapika2b-hardware.mts';
import {gemmaV2EvidenceFingerprint, verifyGemmaV2Candidate} from './gemma4-v2-candidate-evidence.mts';

test('installed candidate matches the official 1024-token text request', async ({page}) => {
  test.skip(process.env.WEB_DECISION_GEMMA_V2_CANDIDATE !== '1', 'Enable the private candidate GPU test');
  test.setTimeout(600000);
  const report = process.env.WEB_DECISION_GEMMA_V2_CANDIDATE_REPORT;
  const bundle = process.env.WEB_DECISION_INSTALLED_BUNDLE;
  if (!report || !bundle) throw new Error('The exact candidate report and installed bundle are required.');
  const evidence = await verifyGemmaV2Candidate(report, bundle);
  const fingerprint = await gemmaV2EvidenceFingerprint(report, fileURLToPath(import.meta.url));
  console.log(
    `PRIVATE_CANDIDATE_EVIDENCE ${JSON.stringify({archiveSha256: evidence.archiveSha256, sourceSnapshotSha256: evidence.sourceSnapshotSha256, fingerprint})}`
  );
  const root = resolve(process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT!);
  const reference = resolve(root, 'e2b-v2-text-1024-reference');
  const request = JSON.parse(await readFile(resolve(reference, 'request.json'), 'utf8'));
  const tokens = JSON.parse(await readFile(resolve(reference, 'tokens.json'), 'utf8')) as {
    state: number[];
    questions: number[][];
  };
  const expected = JSON.parse(await readFile(resolve(reference, 'result.json'), 'utf8'));
  expect(tokens.state.length + tokens.questions[0]!.length).toBe(1024);
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
    }
  );
  expect((await readHardwareReport(page)).status).toBe('Hardware accelerated');
  await page.goto('/');
  await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');
  const actual = await page.evaluate(async (input) => {
    const api = window.webDecision as unknown as {
      loadDecisionModel(options: unknown): Promise<{
        systemOne(request: unknown): Promise<{
          answers: {q: {choice: string; probabilities: Record<string, number>}};
          usage: {input_tokens: number};
        }>;
        dispose(): Promise<void>;
      }>;
    };
    const model = await api.loadDecisionModel({modelId: 'strands-decider-e2b-gemma4-v2-2611', cache: false});
    try {
      return await model.systemOne(input);
    } finally {
      await model.dispose();
    }
  }, request);
  expect(actual.usage.input_tokens).toBe(1024);
  expect(actual.answers.q.choice).toBe(expected.answers.q.choice);
  for (const name of ['yes', 'no']) {
    const delta = Math.abs(actual.answers.q.probabilities[name]! - expected.answers.q.probabilities[name]);
    console.log(
      `1024 tokens ${name}: upstream=${expected.answers.q.probabilities[name]}, installed=${actual.answers.q.probabilities[name]}, delta=${delta}`
    );
    expect(delta).toBeLessThan(0.02);
  }
  await verifyGemmaV2Candidate(report, bundle);
  expect(await gemmaV2EvidenceFingerprint(report, fileURLToPath(import.meta.url))).toBe(fingerprint);
});
