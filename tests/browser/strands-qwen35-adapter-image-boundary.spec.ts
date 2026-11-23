import {readFile} from 'node:fs/promises';
import {expect, test} from '@playwright/test';

type Row = {name: string; file: string; pixelShape: number[]; pixelValuesSha256: string; imageGridThw: number[]};
test('Qwen adapter passes ordered real image preprocessing to its graph boundary', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the private browser probe.');
  const dir = new URL('../../packages/web-decision/test/fixtures/qwen35-image-input/', import.meta.url);
  const goldens = JSON.parse(await readFile(new URL('goldens.json', dir), 'utf8')) as {cases: Row[]};
  const rows = ['png-orientation-6', 'jpeg-orientation-6', 'gif-first-transparent'].map((name) => {
    const row = goldens.cases.find((item) => item.name === name);
    if (!row) throw new Error(`Missing pinned image ${name}`);
    return row;
  });
  const images = await Promise.all(
    rows.map(async (row) => (await readFile(new URL(row.file, dir))).toString('base64'))
  );
  await page.goto('/strands-qwen35-adapter-image-boundary.html');
  await page.waitForFunction(() => typeof window.qwenImageAdapterBoundary === 'function');
  const result = (await page.evaluate(({images, rows}) => window.qwenImageAdapterBoundary(images, rows), {
    images,
    rows: rows.map((row) => row.pixelShape[0]!)
  })) as {
    invalidRejected: boolean;
    observations: Array<{imageCount: number; hashes: string[]; grid: number[]}>;
    answer: {answers: {q: {noul: number}}};
    releases: number;
    disposals: number;
  };
  expect(result.invalidRejected).toBe(true);
  expect(result.observations).toEqual(
    Array.from({length: 2}, () => ({
      imageCount: 3,
      hashes: rows.map((row) => row.pixelValuesSha256),
      grid: rows.flatMap((row) => row.imageGridThw)
    }))
  );
  expect(result.answer.answers.q.noul).toBe(0.8);
  expect(result.releases).toBe(1);
  expect(result.disposals).toBe(1);
});
