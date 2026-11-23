import {readFile} from 'node:fs/promises';
import {expect, test} from '@playwright/test';

test('browser image decoding matches the pinned Python base64 padding boundaries', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the private browser probe.');
  const file = new URL(
    '../../packages/web-decision/test/fixtures/qwen35-image-input/png-orientation-1.png',
    import.meta.url
  );
  const png = await readFile(file);
  expect(png.length % 3).toBe(0);
  await page.goto('/strands-v2-image-compatibility-probe.html');
  await page.waitForFunction(() => typeof window.imageCompatibility === 'function');
  for (let trailingBytes = 0; trailingBytes < 3; trailingBytes++) {
    const payload = Buffer.concat([png, Buffer.alloc(trailingBytes)])
      .toString('base64')
      .replace(/=+$/u, '');
    for (let padding = 0; padding <= 5; padding++) {
      const encoded = payload + '='.repeat(padding);
      const expected = trailingBytes === 0 || padding === 3 - trailingBytes;
      if (expected) {
        const actual = await page.evaluate((value) => window.imageCompatibility(value), encoded);
        expect([actual.width, actual.height], `${trailingBytes}:${padding}`).toEqual([64, 32]);
      } else {
        await expect(page.evaluate((value) => window.imageCompatibility(value), encoded)).rejects.toThrow(/base64/i);
      }
    }
  }
});
