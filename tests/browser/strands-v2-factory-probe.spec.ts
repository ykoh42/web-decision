import {createHash} from 'node:crypto';
import {open, readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';
import {readHardwareReport} from './mapika2b-hardware.mts';

test('verified graph factory loads and runs 35 Gemma decoder layers on WebGPU', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the experimental probe');
  test.setTimeout(300000);
  const root = resolve(process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT!);
  const split = Number(process.env.WEB_DECISION_STRANDS_V2_DYNAMIC_SPLIT ?? '24');
  const unique = JSON.parse(await readFile(resolve(root, 'e2b-v2-compact-text-rows/unique.json'), 'utf8')) as number[];
  const indexBytes = await readFile(resolve(root, 'e2b-v2-compact-text-rows/indices.bin'));
  const indices = new Int32Array(indexBytes.buffer, indexBytes.byteOffset, indexBytes.byteLength / 4);
  const ids = Array.from(indices, (index) => {
    const id = unique[index];
    if (id === undefined) throw new Error('Invalid compact token index');
    return id;
  });
  const checkpoint = await open(resolve(root, 'strands-e2b-v2/model.safetensors'), 'r');
  page.once('close', () => {
    void checkpoint.close();
  });
  await page.route('**/strands-v2-local-source/model.safetensors', async (route) => {
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
    'linked.wasm': process.env.WEB_DECISION_STRANDS_V2_RUNNER_WASM ?? 'e2b-v2-linked-readout/layer.wasm',
    'readout.bin': 'e2b-v2-dynamic-readout/input-2.bin',
    'local.bin': 'e2b-v2-positions/local.bin',
    'global.bin': 'e2b-v2-positions/global.bin'
  };
  await page.route('**/strands-v2-runner-assets/**', async (route) => {
    const name = new URL(route.request().url()).pathname.split('/').at(-1)!;
    const path = paths[name];
    if (!path) throw new Error(`Unknown runner fixture ${name}`);
    await route.fulfill({status: 200, body: await readFile(resolve(root, path))});
  });
  const pins: Record<string, {bytes: number; sha256: string}> = {};
  for (const name of ['linked.wasm', 'local.bin', 'global.bin']) {
    const bytes = await readFile(resolve(root, paths[name]!));
    pins[name] = {bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex')};
  }
  expect((await readHardwareReport(page)).status).toBe('Hardware accelerated');
  await page.goto('/strands-v2-factory-probe.html');
  await page.waitForFunction(() => typeof window.runGemma4V2FactoryProbe === 'function');
  const actual = await page.evaluate(({ids, split, pins}) => window.runGemma4V2FactoryProbe(ids, split, pins), {
    ids,
    split,
    pins
  });
  const bytes = await readFile(resolve(root, 'e2b-v2-dynamic-readout/expected.bin'));
  const expected = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  expect(actual.length).toBe(expected.length);
  let maximum = 0;
  for (let i = 0; i < expected.length; i++) {
    if (!Number.isFinite(actual[i])) throw new Error(`Non-finite value at ${i}`);
    maximum = Math.max(maximum, Math.abs(actual[i]! - expected[i]!));
  }
  console.log(`length=${ids.length} split=${split} max error=${maximum}`);
  expect(maximum).toBeLessThan(0.001);
});
