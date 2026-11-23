import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';

test('private rotary graph covers the 131072 boundary when the adapter permits it', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the private GPU probe.');
  test.setTimeout(300000);
  const wasm = process.env.WEB_DECISION_STRANDS_V2_ROTARY_BOUNDARY_WASM;
  if (!wasm) throw new Error('Set WEB_DECISION_STRANDS_V2_ROTARY_BOUNDARY_WASM to the test-only graph.');
  await page.route('**/strands-v2-rotary-boundary/graph.wasm', async (route) => {
    await route.fulfill({status: 200, contentType: 'application/wasm', body: await readFile(resolve(wasm))});
  });
  await page.goto('/strands-v2-rotary-boundary.html');
  await page.waitForFunction(() => typeof window.runGemma4RotaryBoundary === 'function');
  const result = await page.evaluate(() => window.runGemma4RotaryBoundary());
  console.log(`ROTARY_BOUNDARY_PROBE ${JSON.stringify(result)}`);
  expect(result.cases.map((item) => item.length).sort((a, b) => a - b)).toEqual([131071, 131072, 131073]);
  for (const item of result.cases) {
    if (item.status === 'passed') expect(item.maxError).toBeLessThanOrEqual(0.02);
    else
      expect(item.requiredBytes).toBeGreaterThan(
        Math.min(result.adapterLimits.maxBufferSize, result.adapterLimits.maxStorageBufferBindingSize)
      );
  }
  test.skip(
    result.cases.every((item) => item.status === 'unsupported'),
    'Adapter storage limits do not permit the probe.'
  );
});
