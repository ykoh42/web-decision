import {createHash} from 'node:crypto';
import {open, readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';
import {readHardwareReport} from './mapika2b-hardware.mts';

type Snapshot = {count: number; bytes: number};
type Answer = {answers: {color: {choice: string; probabilities: Record<string, number>}}};

test('multi-image session releases, reloads and disposes GPU weights after cancellation', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the experimental probe');
  test.setTimeout(1_800_000);
  const root = resolve(process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT!);
  const wasmPath = process.env.WEB_DECISION_STRANDS_V2_IMAGE_WASM;
  const expectedHash = process.env.WEB_DECISION_STRANDS_V2_IMAGE_WASM_SHA256;
  const split = process.env.WEB_DECISION_STRANDS_V2_IMAGE_SPLIT === '1';
  if (!wasmPath || !expectedHash || !/^[0-9a-f]{64}$/u.test(expectedHash))
    throw new Error('Explicit WASM path and SHA-256 are required.');
  const wasm = await readFile(resolve(root, wasmPath));
  expect(createHash('sha256').update(wasm).digest('hex')).toBe(expectedHash);
  const request = JSON.parse(await readFile(resolve(root, 'e2b-v2-multi-reference/request.json'), 'utf8'));
  const official = JSON.parse(await readFile(resolve(root, 'e2b-v2-multi-reference/result.json'), 'utf8'));
  const checkpointDirectory = resolve(root, 'strands-e2b-v2');
  const checkpoint = await open(resolve(checkpointDirectory, 'model.safetensors'), 'r');
  const graphPaths: Record<string, string | Buffer> = {
    'linked.wasm': wasm,
    'local.bin': resolve(root, 'e2b-v2-positions/local.bin'),
    'global.bin': resolve(root, 'e2b-v2-positions/global.bin')
  };
  const allowed = new Set([
    'config.json',
    'decision_config.json',
    'processor_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'readout.safetensors'
  ]);
  try {
    await page.route(
      'https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v2-2611/resolve/a5f32ea1bffe2e91c25c0c641d444e8115e28212/**',
      async (route) => {
        const name = new URL(route.request().url()).pathname.split('/').at(-1)!;
        if (!allowed.has(name)) throw new Error(`Unexpected static asset ${name}`);
        await route.fulfill({
          status: 200,
          headers: {'access-control-allow-origin': '*'},
          body: await readFile(resolve(checkpointDirectory, name))
        });
      }
    );
    await page.route('**/strands-v2-lifecycle-source/model.safetensors', async (route) => {
      const match = /^bytes=(\d+)-(\d+)$/u.exec(route.request().headers().range ?? '');
      if (!match) throw new Error('Missing source byte range');
      const start = Number(match[1]),
        end = Number(match[2]);
      const bytes = Buffer.alloc(end - start + 1);
      const result = await checkpoint.read(bytes, 0, bytes.length, start);
      if (result.bytesRead !== bytes.length) throw new Error('Short local checkpoint read');
      await route.fulfill({status: 206, headers: {'content-range': `bytes ${start}-${end}/10246621918`}, body: bytes});
    });
    await page.route('**/strands-v2-lifecycle-assets/**', async (route) => {
      const name = new URL(route.request().url()).pathname.split('/').at(-1)!;
      const path = graphPaths[name];
      if (!path) throw new Error(`Unknown lifecycle graph ${name}`);
      await route.fulfill({status: 200, body: typeof path === 'string' ? await readFile(path) : path});
    });
    expect((await readHardwareReport(page)).status).toBe('Hardware accelerated');
    await page.goto('/strands-v2-multi-image-lifecycle-probe.html');
    await page.waitForFunction(() => typeof window.runGemma4ImageLifecycle === 'function');
    const result = (await page.evaluate(({request, split}) => window.runGemma4ImageLifecycle(request, split), {
      request,
      split
    })) as {
      first: Answer;
      retry: Answer;
      cancelled: boolean;
      imageSyncs: number;
      firstLayers: number;
      cancelLayers: number;
      retryLayers: number;
      initial: Snapshot;
      afterFirst: Snapshot;
      afterRelease: Snapshot;
      afterCancel: Snapshot;
      afterCancelRelease: Snapshot;
      afterRetry: Snapshot;
      afterFinalRelease: Snapshot;
      afterDispose: Snapshot;
      firstLoads: number;
      cancelLoads: number;
      retryLoads: number;
      releasesAfterFirst: number;
      totalReleases: number;
    };
    console.log(`MULTI_IMAGE_LIFECYCLE ${JSON.stringify(result)}`);
    expect(result.cancelled).toBe(true);
    expect(result.imageSyncs).toBe(2);
    if (split) {
      expect(result.firstLayers).toBe(35);
      expect(result.cancelLayers).toBeGreaterThan(result.firstLayers);
      expect(result.cancelLayers).toBeLessThan(result.firstLayers + 35);
      expect(result.retryLayers - result.cancelLayers).toBe(35);
    }
    expect(result.first.answers.color.choice).toBe(official.answers.color.choice);
    expect(
      Math.abs(result.first.answers.color.probabilities.green! - official.answers.color.probabilities.green)
    ).toBeLessThan(0.02);
    expect(result.retry.answers.color.choice).toBe(result.first.answers.color.choice);
    for (const key of ['green', 'yellow'])
      expect(
        Math.abs(result.first.answers.color.probabilities[key]! - result.retry.answers.color.probabilities[key]!)
      ).toBeLessThan(0.000001);
    expect(result.firstLoads).toBe(1);
    expect(result.cancelLoads).toBe(2);
    expect(result.retryLoads).toBe(3);
    expect(result.releasesAfterFirst).toBe(1);
    expect(result.totalReleases).toBe(3);
    expect(result.afterRelease.count).toBeLessThan(result.afterFirst.count);
    expect(result.afterRelease.bytes).toBeLessThan(result.afterFirst.bytes);
    if (split) {
      expect(result.afterCancel.bytes - result.afterCancelRelease.bytes).toBeGreaterThanOrEqual(674_178_816);
      expect(result.afterRetry.bytes - result.afterFinalRelease.bytes).toBeGreaterThanOrEqual(674_178_816);
    } else {
      expect(result.afterCancelRelease).toEqual(result.afterRelease);
      expect(result.afterFinalRelease).toEqual(result.afterRelease);
      expect(result.afterRetry).toEqual(result.afterFirst);
    }
    expect(result.afterDispose).toEqual({count: 0, bytes: 0});
  } finally {
    await checkpoint.close();
  }
});
