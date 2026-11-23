import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {expect, test} from '@playwright/test';

type Row = {
  name: string;
  file: string;
  status: string;
  encodedSha256: string;
  decodedSize: number[];
  fittedSize: number[];
  processedSize: number[];
  imageGridThw: number[];
  pixelShape: number[];
  pixelValuesSha256: string;
};

test('Qwen browser preprocessing preserves every pinned image reference', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the private browser probe.');
  test.setTimeout(180000);
  const directory = new URL('../../packages/web-decision/test/fixtures/qwen35-image-input/', import.meta.url);
  const goldens = JSON.parse(await readFile(new URL('goldens.json', directory), 'utf8')) as {cases: Row[]};
  await page.goto('/strands-qwen35-image-goldens.html');
  await page.waitForFunction(() => typeof window.preprocessQwen35Golden === 'function');
  for (const row of goldens.cases) {
    const bytes = await readFile(new URL(row.file, directory));
    expect(createHash('sha256').update(bytes).digest('hex'), row.name).toBe(row.encodedSha256);
    const encoded = bytes.toString('base64');
    if (row.status !== 'generated') {
      await expect(page.evaluate((image) => window.preprocessQwen35Golden(image), encoded)).rejects.toThrow(
        /aspect ratio/i
      );
      continue;
    }
    const actual = await page.evaluate((image) => window.preprocessQwen35Golden(image), encoded);
    expect(actual, row.name).toEqual({
      decodedSize: row.decodedSize,
      fittedSize: row.fittedSize,
      processedSize: row.processedSize,
      imageGridThw: row.imageGridThw,
      pixelShape: row.pixelShape,
      pixelValuesSha256: row.pixelValuesSha256
    });
  }
  console.log(`Matched ${goldens.cases.length} pinned Qwen image cases in the browser`);
});
