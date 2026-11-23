import {createHash} from 'node:crypto';
import {open, readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';
import {readHardwareReport} from './mapika2b-hardware.mts';

test('two actual images through TVM produce the official choice', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the experimental probe');
  test.setTimeout(600000);
  const root = resolve(process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT!);
  const input = JSON.parse(await readFile(resolve(root, 'e2b-v2-two-image-reference/tokens.json'), 'utf8'));
  const official = JSON.parse(await readFile(resolve(root, 'e2b-v2-two-image-reference/result.json'), 'utf8'));
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
  const split = process.env.WEB_DECISION_STRANDS_V2_IMAGE_SPLIT === '1';
  const expectedHash = process.env.WEB_DECISION_STRANDS_V2_IMAGE_WASM_SHA256;
  if (split && (!expectedHash || !/^[0-9a-f]{64}$/u.test(expectedHash)))
    throw new Error('Split image probe requires the exact graph SHA-256.');
  if (expectedHash) {
    const wasm = await readFile(resolve(root, paths['linked.wasm']!));
    expect(createHash('sha256').update(wasm).digest('hex')).toBe(expectedHash);
  }
  await page.route('**/strands-v2-image-e2e-assets/**', async (route) => {
    const name = new URL(route.request().url()).pathname.split('/').at(-1)!;
    const path =
      paths[name] ??
      (/^debug-layer(?:0|4|34)-(?:hidden|output)\.bin$/u.test(name)
        ? `e2b-v2-two-image-reference/${name}`
        : /^debug-ple(?:0|4|34)\.bin$/u.test(name)
          ? `e2b-v2-two-image-reference/text-per-layer-${name.slice(9, -4)}.bin`
          : undefined);
    if (!path) throw new Error(`Unknown image runner fixture ${name}`);
    await route.fulfill({status: 200, body: await readFile(resolve(root, path))});
  });
  expect((await readHardwareReport(page)).status).toBe('Hardware accelerated');
  await page.goto('/strands-v2-two-image-e2e-probe.html');
  await page.waitForFunction(() => typeof window.runGemma4V2TwoImageE2E === 'function');
  const actual = (await page.evaluate(({value, split}) => window.runGemma4V2TwoImageE2E(value, split), {
    value: input,
    split
  })) as {
    differences: Record<string, {max: number; mean: number; rms: number}>;
    layers: number;
    tokens: number[];
    results: Array<{logits: number[]; answer: {choice: string; probabilities: Record<string, number>}}>;
  };
  console.log('Second question stage differences:', JSON.stringify(actual.differences));
  expect(actual.tokens).toEqual([256, 256]);
  if (split) expect(actual.layers).toBe(70);
  expect(actual.results).toHaveLength(2);
  for (const [index, name] of ['color', 'color_second'].entries()) {
    const result = actual.results[index]!;
    const reference = official.answers[name];
    expect(result.logits).toHaveLength(255);
    expect(result.logits.every(Number.isFinite)).toBe(true);
    expect(result.answer.choice).toBe(reference.choice);
    const difference = Math.abs(result.answer.probabilities.red! - reference.probabilities.red);
    console.log(
      `${name}: official red=${reference.probabilities.red}; TVM=${result.answer.probabilities.red}; difference=${difference}`
    );
    expect(difference).toBeLessThan(0.02);
  }
});
