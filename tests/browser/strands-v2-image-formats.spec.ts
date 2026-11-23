import {readFile} from 'node:fs/promises';
import {expect, test} from '@playwright/test';

type Row = {name: string; width: number; height: number; rgbSha256: string};

test('browser color modes and bit depths match the original Pillow RGB output', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the private browser probe.');
  const directory = new URL(
    '../../packages/web-decision/test/fixtures/strands-gemma4-v2-image-formats/',
    import.meta.url
  );
  const rows = JSON.parse(await readFile(new URL('reference.json', directory), 'utf8')) as Row[];
  await page.goto('/strands-v2-image-compatibility-probe.html');
  await page.waitForFunction(() => typeof window.imageCompatibility === 'function');
  for (const row of rows) {
    const encoded = (await readFile(new URL(row.name, directory))).toString('base64');
    const actual = await page.evaluate((image) => window.imageCompatibility(image), encoded);
    expect({width: actual.width, height: actual.height, rgbSha256: actual.rgbSha256}, row.name).toEqual({
      width: row.width,
      height: row.height,
      rgbSha256: row.rgbSha256
    });
  }
  console.log(`Matched ${rows.length} Pillow image format cases in the browser`);
});
