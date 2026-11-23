import {createHash} from 'node:crypto';
import {open} from 'node:fs/promises';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';

test('pinned upstream supports verified browser byte ranges', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the experimental probe');
  test.setTimeout(60000);
  const offset = 1422042672;
  const length = 4096;
  const handle = await open(resolve(process.env.WEB_DECISION_STRANDS_V2_CHECKPOINT!, 'model.safetensors'), 'r');
  const expected = Buffer.alloc(length);
  try {
    const result = await handle.read(expected, 0, length, offset);
    expect(result.bytesRead).toBe(length);
  } finally {
    await handle.close();
  }
  await page.goto('/strands-v2-layer-probe.html');
  const actual = await page.evaluate(
    async ({start, count}) => {
      const url =
        'https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v2-2611/resolve/a5f32ea1bffe2e91c25c0c641d444e8115e28212/model.safetensors';
      const response = await fetch(url, {headers: {Range: `bytes=${start}-${start + count - 1}`}});
      return {
        status: response.status,
        contentRange: response.headers.get('content-range'),
        bytes: Array.from(new Uint8Array(await response.arrayBuffer()))
      };
    },
    {start: offset, count: length}
  );
  expect(actual.status).toBe(206);
  expect(actual.contentRange).toBe(`bytes ${offset}-${offset + length - 1}/10246621918`);
  expect(createHash('sha256').update(Uint8Array.from(actual.bytes)).digest('hex')).toBe(
    createHash('sha256').update(expected).digest('hex')
  );
});
