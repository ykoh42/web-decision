import {createHash} from 'node:crypto';
import {open, readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';

test('actual image through the TVM vision and text runner produces the official choice', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the experimental probe');
  test.setTimeout(600000);
  const root = resolve(process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT!);
  const input = JSON.parse(await readFile(resolve(root, 'e2b-v2-vision-reference/tokens.json'), 'utf8'));
  const official = JSON.parse(await readFile(resolve(root, 'e2b-v2-vision-reference/result.json'), 'utf8'));
  const checkpoint = await open(resolve(root, 'strands-e2b-v2/model.safetensors'), 'r');
  page.once('close', () => {
    void checkpoint.close();
  });
  await page.route('**/strands-v2-image-e2e-source/model.safetensors', async (route) => {
    const match = /^bytes=(\d+)-(\d+)$/u.exec(route.request().headers().range ?? '');
    if (!match) throw new Error('Missing source byte range');
    const start = Number(match[1]);
    const end = Number(match[2]);
    const bytes = Buffer.alloc(end - start + 1);
    const result = await checkpoint.read(bytes, 0, bytes.length, start);
    if (result.bytesRead !== bytes.length) throw new Error('Short local checkpoint read');
    await route.fulfill({status: 206, headers: {'content-range': `bytes ${start}-${end}/10246621918`}, body: bytes});
  });
  const paths: Record<string, string> = {
    'linked.wasm': process.env.WEB_DECISION_STRANDS_V2_IMAGE_WASM ?? 'e2b-v2-linked-complete/layer.wasm',
    'readout.bin': 'e2b-v2-dynamic-readout/input-2.bin',
    'local.bin': 'e2b-v2-positions/local.bin',
    'global.bin': 'e2b-v2-positions/global.bin'
  };
  const expectedWasmHash = process.env.WEB_DECISION_STRANDS_V2_IMAGE_WASM_SHA256;
  const split = process.env.WEB_DECISION_STRANDS_V2_IMAGE_SPLIT === '1';
  const hostReadout = process.env.WEB_DECISION_STRANDS_V2_HOST_READOUT === '1';
  if ((split || hostReadout) && !expectedWasmHash)
    throw new Error('Split image probe requires the exact graph SHA-256.');
  if (expectedWasmHash !== undefined) {
    if (!/^[0-9a-f]{64}$/u.test(expectedWasmHash)) throw new Error('Invalid graph SHA-256.');
    const wasm = await readFile(resolve(root, paths['linked.wasm']!));
    expect(createHash('sha256').update(wasm).digest('hex')).toBe(expectedWasmHash);
  }
  await page.route('**/strands-v2-image-e2e-assets/**', async (route) => {
    const name = new URL(route.request().url()).pathname.split('/').at(-1)!;
    const path = paths[name];
    if (!path) throw new Error(`Unknown image runner fixture ${name}`);
    await route.fulfill({status: 200, body: await readFile(resolve(root, path))});
  });
  await page.goto('/strands-v2-image-e2e-probe.html');
  const capability = await page.evaluate(async () => {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) return {available: false, fallback: null};
    const fallback =
      adapter.info?.isFallbackAdapter ??
      (adapter as GPUAdapter & {isFallbackAdapter?: boolean}).isFallbackAdapter ??
      null;
    return {available: true, fallback};
  });
  console.log(`WEBGPU_ADAPTER ${JSON.stringify(capability)}`);
  expect(capability.available).toBe(true);
  expect(capability.fallback).toBe(false);
  await page.waitForFunction(() => typeof window.runGemma4V2ImageE2E === 'function');
  const actual = (await page.evaluate(
    ({value, split, hostReadout}) => window.runGemma4V2ImageE2E(value, split, hostReadout),
    {
      value: input,
      split,
      hostReadout
    }
  )) as {
    tokens: number;
    layers: number;
    logits: number[];
    answer: {choice: string; probabilities: Record<string, number>};
  };
  expect(actual.tokens).toBe(256);
  if (split) expect(actual.layers).toBe(35);
  expect(actual.logits).toHaveLength(255);
  expect(actual.logits.every(Number.isFinite)).toBe(true);
  expect(actual.answer.choice).toBe(official.answers.color.choice);
  const difference = Math.abs(actual.answer.probabilities.red! - official.answers.color.probabilities.red);
  console.log(
    `Official BF16 red probability=${official.answers.color.probabilities.red}; TVM=${actual.answer.probabilities.red}; difference=${difference}`
  );
  expect(difference).toBeLessThan(0.02);
});

test('FP64 WASM readout exactly matches frozen Strands logits', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_HOST_READOUT !== '1', 'Enable the host readout probe');
  const root = resolve(process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT!);
  const wasm = await readFile(resolve(root, process.env.WEB_DECISION_STRANDS_V2_IMAGE_WASM!));
  expect(createHash('sha256').update(wasm).digest('hex')).toBe(process.env.WEB_DECISION_STRANDS_V2_IMAGE_WASM_SHA256);
  const assets: Record<string, Buffer> = {
    'linked.wasm': wasm,
    'readout.bin': await readFile(resolve(root, 'e2b-v2-dynamic-readout/input-2.bin')),
    'fp64-hidden.f32le': await readFile(resolve(root, 'e2b-v2-release-diagnosis/fp64-hidden.f32le')),
    'fp64-logits.f32le': await readFile(resolve(root, 'e2b-v2-release-diagnosis/fp64-logits.f32le'))
  };
  await page.route('**/strands-v2-image-e2e-assets/**', async (route) => {
    const name = new URL(route.request().url()).pathname.split('/').at(-1)!;
    const body = assets[name];
    if (!body) throw new Error(`Unknown FP64 readout fixture ${name}`);
    await route.fulfill({status: 200, body});
  });
  await page.goto('/strands-v2-image-e2e-probe.html');
  await page.waitForFunction(() => typeof window.checkGemma4V2HostReadout === 'function');
  expect(await page.evaluate(() => window.checkGemma4V2HostReadout())).toBe(1530);
});
