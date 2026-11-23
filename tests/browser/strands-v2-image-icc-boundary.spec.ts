import {readFile} from 'node:fs/promises';
import {expect, test} from '@playwright/test';

test('browser ICC decompression matches Pillow 1 MiB boundary', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the private browser probe.');
  const dir = new URL(
    '../../packages/web-decision/test/fixtures/strands-gemma4-v2-image-icc-boundary/',
    import.meta.url
  );
  const allowed = (await readFile(new URL('icc-1048576.png', dir))).toString('base64');
  const rejected = (await readFile(new URL('icc-1048577.png', dir))).toString('base64');
  await page.goto('/strands-v2-image-compatibility-probe.html');
  await page.waitForFunction(() => typeof window.imageCompatibility === 'function');
  const result = await page.evaluate((image) => window.imageCompatibility(image), allowed);
  expect([result.width, result.height, result.tokens]).toEqual([1, 1, 256]);
  await expect(page.evaluate((image) => window.imageCompatibility(image), rejected)).rejects.toThrow(
    /decompression limit/i
  );
});
