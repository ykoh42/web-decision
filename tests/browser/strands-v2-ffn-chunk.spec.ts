import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';

test('private FFN chunks match the checkpoint tail reference', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the private GPU probe.');
  test.setTimeout(300000);
  const directory = process.env.WEB_DECISION_STRANDS_V2_FFN_FIXTURES;
  if (!directory) throw new Error('Set WEB_DECISION_STRANDS_V2_FFN_FIXTURES to the private FFN fixture directory.');
  await page.route('**/strands-v2-ffn-chunk/**', async (route) => {
    const name = new URL(route.request().url()).pathname.split('/').at(-1)!;
    if (!/^(?:tail\.wasm|inputs\.json|expected\.bin|input-(?:[0-9]|10)\.bin)$/u.test(name))
      throw new Error(`Unexpected FFN fixture ${name}`);
    await route.fulfill({status: 200, body: await readFile(resolve(directory, name))});
  });
  await page.goto('/strands-v2-ffn-chunk.html');
  await page.waitForFunction(() => typeof window.runGemma4FfnChunks === 'function');
  const result = await page.evaluate(() => window.runGemma4FfnChunks());
  console.log(`FFN_CHUNK_PROBE ${JSON.stringify(result)}`);
  expect(result.results.map((item) => item.chunk)).toEqual([1, 2, 3, 6, 7]);
  for (const item of result.results) {
    expect(item.tokens).toBe(7);
    expect(item.chunks).toBe(Math.ceil(7 / item.chunk));
    expect(item.maxReferenceError).toBeLessThanOrEqual(0.01);
  }
});
