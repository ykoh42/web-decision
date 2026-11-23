import {createHash} from 'node:crypto';
import {open, readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';

test('35 actual layers preserve logits when FFN is chunked', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable private GPU probe.');
  test.setTimeout(900000);
  const root = resolve(process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT!);
  const checkpoint = await open(resolve(root, 'strands-e2b-v2/model.safetensors'), 'r');
  page.once('close', () => {
    void checkpoint.close();
  });
  await page.route('**/strands-v2-optimized-full-source/model.safetensors', async (route) => {
    const match = /^bytes=(\d+)-(\d+)$/u.exec(route.request().headers().range ?? '');
    if (!match) throw new Error('Missing range');
    const start = Number(match[1]),
      end = Number(match[2]);
    const bytes = Buffer.alloc(end - start + 1);
    const result = await checkpoint.read(bytes, 0, bytes.length, start);
    if (result.bytesRead !== bytes.length) throw new Error('Short read');
    await route.fulfill({status: 206, headers: {'content-range': `bytes ${start}-${end}/10246621918`}, body: bytes});
  });
  const paths: Record<string, string> = {
    'linked.wasm': 'e2b-v2-split-transposed-full/layer.wasm',
    'readout.bin': 'e2b-v2-dynamic-readout/input-2.bin',
    'local.bin': 'e2b-v2-positions/local.bin',
    'global.bin': 'e2b-v2-positions/global.bin'
  };
  const wasm = await readFile(resolve(root, paths['linked.wasm']!));
  expect(createHash('sha256').update(wasm).digest('hex')).toBe(
    '88d91503237cdaa5f1c7b63cf02f2a0b82a44f44699d3d13838623738a5c7b47'
  );
  await page.route('**/strands-v2-optimized-full-assets/**', async (route) => {
    const name = new URL(route.request().url()).pathname.split('/').at(-1)!;
    const path = paths[name];
    if (!path) throw new Error(`Unexpected ${name}`);
    await route.fulfill({status: 200, body: await readFile(resolve(root, path))});
  });
  await page.goto('/strands-v2-optimized-full.html');
  await page.waitForFunction(() => typeof window.runOptimizedFull === 'function');
  const result = (await page.evaluate(() => window.runOptimizedFull())) as {
    layers: number;
    logits: number;
    maxError: number;
  };
  console.log(`OPTIMIZED_FULL_PROBE ${JSON.stringify(result)}`);
  expect(result.layers).toBe(35);
  expect(result.logits).toBe(255);
  expect(result.maxError).toBeLessThan(0.01);
});
