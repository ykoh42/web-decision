import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';
import {readHardwareReport} from './mapika2b-hardware.mts';

type Row = {name: string; width: number; height: number; tokens: number; pixelsSha256: string; positionsSha256: string};
type Insertion = {name: string; n: number; spans: number[][]; sha256: string};
async function fixtures() {
  const directory = new URL('../../packages/web-decision/test/fixtures/strands-gemma4-v2-matrix/', import.meta.url);
  const reference = JSON.parse(await readFile(new URL('reference.json', directory), 'utf8')) as {
    images: Row[];
    invalid: string[];
    multi: string[][];
    insertions: Insertion[];
  };
  const input = new Map(
    await Promise.all(
      reference.images.map(
        async (row) => [row.name, (await readFile(new URL(row.name, directory))).toString('base64')] as const
      )
    )
  );
  return {reference, input};
}
const expected = (row: Row) => ({
  width: row.width,
  height: row.height,
  tokens: row.tokens,
  pixelsSha256: row.pixelsSha256,
  positionsSha256: row.positionsSha256
});

test('image preprocessing matches Python across sizes, boundaries and multiple images', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the experimental probe');
  test.setTimeout(180000);
  const {reference, input} = await fixtures();
  await page.goto('/strands-v2-image-matrix-probe.html');
  await page.waitForFunction(() => typeof window.gemma4ImageMatrix === 'function');
  for (const row of reference.images) {
    const result = await page.evaluate((data) => window.gemma4ImageMatrix([data]), input.get(row.name)!);
    expect(result).toEqual([expected(row)]);
  }
  for (const names of reference.multi) {
    const values = names.map((name) => input.get(name)!);
    const results = names.map((name) => expected(reference.images.find((row) => row.name === name)!));
    expect(await page.evaluate((data) => window.gemma4ImageMatrix(data), values)).toEqual(results);
  }
  expect(await page.evaluate(() => window.gemma4ImageMatrix([]))).toEqual([]);
  expect(await page.evaluate((data) => window.gemma4InvalidImages(data), reference.invalid)).toEqual(
    reference.invalid.map(() => true)
  );
  const dataUri = 'data:image/png;base64,' + input.get('64x64.png');
  expect(await page.evaluate((data) => window.gemma4ImageMatrix([data]), dataUri)).toEqual([
    expected(reference.images.find((row) => row.name === '64x64.png')!)
  ]);
});

test('actual WebGPU image insertion matches Python BF16 at boundaries and with multiple images', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the experimental probe');
  test.setTimeout(180000);
  const root = resolve(process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT!);
  const wasm = process.env.WEB_DECISION_STRANDS_V2_IMAGE_WASM ?? 'e2b-v2-linked-complete/layer.wasm';
  await page.route('**/strands-v2-image-matrix/linked.wasm', async (route) => {
    await route.fulfill({status: 200, body: await readFile(resolve(root, wasm))});
  });
  const {reference} = await fixtures();
  expect((await readHardwareReport(page)).status).toBe('Hardware accelerated');
  await page.goto('/strands-v2-image-matrix-probe.html');
  await page.waitForFunction(() => typeof window.gemma4ImageInsertions === 'function');
  expect(await page.evaluate((data) => window.gemma4ImageInsertions(data), reference.insertions)).toEqual(
    reference.insertions.map((item) => item.sha256)
  );
});
