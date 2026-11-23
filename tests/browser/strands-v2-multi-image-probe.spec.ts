import {open, readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';
import {readHardwareReport} from './mapika2b-hardware.mts';

test('two real images with different aspect ratios match the official decision', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the experimental probe');
  test.setTimeout(900000);
  const root = resolve(process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT!);
  const input = JSON.parse(await readFile(resolve(root, 'e2b-v2-multi-reference/tokens.json'), 'utf8'));
  const request = JSON.parse(await readFile(resolve(root, 'e2b-v2-multi-reference/request.json'), 'utf8'));
  const official = JSON.parse(await readFile(resolve(root, 'e2b-v2-multi-reference/result.json'), 'utf8'));
  const checkpoint = await open(resolve(root, 'strands-e2b-v2/model.safetensors'), 'r');
  page.once('close', () => {
    void checkpoint.close();
  });
  await page.route('**/strands-v2-multi-image-source/model.safetensors', async (route) => {
    const match = /^bytes=(\d+)-(\d+)$/u.exec(route.request().headers().range ?? '');
    if (!match) throw new Error('Missing source byte range');
    const start = Number(match[1]),
      end = Number(match[2]);
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
  await page.route('**/strands-v2-multi-image-assets/**', async (route) => {
    const name = new URL(route.request().url()).pathname.split('/').at(-1)!;
    const path = paths[name];
    if (!path) throw new Error(`Unknown multi-image fixture ${name}`);
    await route.fulfill({status: 200, body: await readFile(resolve(root, path))});
  });
  expect((await readHardwareReport(page)).status).toBe('Hardware accelerated');
  await page.goto('/strands-v2-multi-image-probe.html');
  await page.waitForFunction(() => typeof window.runGemma4V2MultipleImages === 'function');
  const actual = (await page.evaluate((value) => window.runGemma4V2MultipleImages(value), {
    ...input,
    encoded: request.images
  })) as {
    tokens: number[];
    logits: number[];
    answer: {choice: string; probabilities: Record<string, number>};
  };
  expect(actual.tokens).toEqual(input.images.map((span: {start: number; end: number}) => span.end - span.start));
  expect(actual.logits).toHaveLength(255);
  expect(actual.logits.every(Number.isFinite)).toBe(true);
  expect(actual.answer.choice).toBe(official.answers.color.choice);
  for (const choice of ['green', 'yellow']) {
    const difference = Math.abs(actual.answer.probabilities[choice]! - official.answers.color.probabilities[choice]);
    console.log(
      `${choice}: official=${official.answers.color.probabilities[choice]}, TVM=${actual.answer.probabilities[choice]}, difference=${difference}`
    );
    expect(difference).toBeLessThan(0.02);
  }
});
