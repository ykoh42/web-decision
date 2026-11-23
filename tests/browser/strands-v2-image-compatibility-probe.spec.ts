import {readFile} from 'node:fs/promises';
import {expect, test} from '@playwright/test';

type Row = {
  name: string;
  width: number;
  height: number;
  tokens: number;
  rgbSha256: string;
  pixelsSha256: string;
  positionsSha256: string;
};

test('browser Qwen and Gemma image paths match pinned Pillow compatibility cases', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the experimental probe');
  test.setTimeout(180000);
  const directory = new URL(
    '../../packages/web-decision/test/fixtures/strands-gemma4-v2-image-compatibility/',
    import.meta.url
  );
  const rows = JSON.parse(await readFile(new URL('reference.json', directory), 'utf8')) as Row[];
  await page.goto('/strands-v2-image-compatibility-probe.html');
  await page.waitForFunction(() => typeof window.imageCompatibility === 'function');
  for (const row of rows) {
    const encoded = (await readFile(new URL(row.name, directory))).toString('base64');
    const {name, ...expected} = row;
    const actual = await page.evaluate((image) => window.imageCompatibility(image), encoded);
    expect(actual, name).toEqual(expected);
  }
  console.log(`Matched ${rows.length} Pillow image fixtures in the browser`);
});
