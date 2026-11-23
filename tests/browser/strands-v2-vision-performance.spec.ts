import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';
import {startLocalRangeServer} from './strands-v2-local-range-server.mts';

async function sha256(path: string) {
  return createHash('sha256')
    .update(await readFile(path))
    .digest('hex');
}

test('compares two and eight layer synchronization on one device and image', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the private GPU probe.');
  test.setTimeout(600000);
  const root = process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT;
  const wasm = process.env.WEB_DECISION_STRANDS_V2_IMAGE_WASM;
  const expectedHash = process.env.WEB_DECISION_STRANDS_V2_IMAGE_WASM_SHA256;
  if (!root || !wasm || !expectedHash || !/^[a-f0-9]{64}$/u.test(expectedHash))
    throw new Error('Set the local root, graph and graph SHA-256.');
  const graph = resolve(root, wasm);
  const pixels = resolve(root, 'e2b-v2-vision-reference/input-pixels.bin');
  const positions = resolve(root, 'e2b-v2-vision-reference/input-positions.bin');
  const hashes = {graph: await sha256(graph), pixels: await sha256(pixels), positions: await sha256(positions)};
  expect(hashes.graph).toBe(expectedHash);
  const range = await startLocalRangeServer(
    new Map([
      ['/graph.wasm', graph],
      ['/input-pixels.bin', pixels],
      ['/input-positions.bin', positions],
      ['/model.safetensors', resolve(root, 'strands-e2b-v2/model.safetensors')]
    ])
  );
  try {
    await page.goto('/strands-v2-vision-performance.html');
    await page.waitForFunction(() => typeof window.runVisionPerformance === 'function');
    const result = await page.evaluate((base) => window.runVisionPerformance(base, 3), range.base);
    console.log(`VISION_SYNC_PERFORMANCE ${JSON.stringify({hashes, result, range: range.stats()})}`);
    expect(result.failure).toBeNull();
    expect(result.trials.filter((trial) => trial.warmup)).toHaveLength(2);
    for (const interval of [2, 8]) {
      const measured = result.trials.filter((trial) => !trial.warmup && trial.interval === interval);
      expect(measured).toHaveLength(3);
      expect(measured.every((trial) => trial.elapsedMs > 0 && trial.maxAbsoluteError < 0.00001)).toBe(true);
    }
  } finally {
    await page.close();
    await range.close();
  }
});
