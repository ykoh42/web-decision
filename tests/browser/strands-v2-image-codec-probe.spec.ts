import {readFile} from 'node:fs/promises';
import {expect, test} from '@playwright/test';

test('verified JPEG and WebP decoders work online and from offline cache', async ({page, context}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the experimental probe');
  test.setTimeout(120000);
  const fixture = new URL('../../packages/web-decision/test/fixtures/qwen35-image-input/', import.meta.url);
  const rows = JSON.parse(
    await readFile(
      new URL('../../packages/web-decision/test/fixtures/strands-gemma4-v2-encoded-image.json', import.meta.url),
      'utf8'
    )
  ) as Array<{name: string; tokens: number; pixelsSha256: string; positionsSha256: string}>;
  const names = ['jpeg-orientation-6.jpg', 'webp-alpha.webp'];
  const images = await Promise.all(
    names.map(async (name) => (await readFile(new URL(name, fixture))).toString('base64'))
  );
  const expected = names.map((name) => {
    const item = rows.find((row) => row.name === name)!;
    return {tokens: item.tokens, pixelHash: item.pixelsSha256, positionHash: item.positionsSha256};
  });
  const cacheName = 'gemma-v2-image-codecs-' + Date.now();
  await page.goto('/strands-v2-image-codec-probe.html');
  await page.waitForFunction(() => typeof window.initGemmaImageCache === 'function');
  try {
    expect(await page.evaluate((cache) => window.gemmaImageOfflineReady(cache), cacheName)).toBe(false);
    const online = await page.evaluate(({cache, inputs}) => window.initGemmaImageCache(cache, inputs), {
      cache: cacheName,
      inputs: images
    });
    expect(online).toEqual(expected);
    await context.setOffline(true);
    const offline = await page.evaluate(({cache, inputs}) => window.testGemmaImageOffline(cache, inputs), {
      cache: cacheName,
      inputs: images
    });
    expect(offline).toEqual(expected);
    expect(await page.evaluate((cache) => window.gemmaImageOfflineReady(cache), cacheName)).toBe(true);
    await page.evaluate((cache) => window.corruptGemmaImageCache(cache), cacheName);
    expect(await page.evaluate((cache) => window.gemmaImageOfflineReady(cache), cacheName)).toBe(false);
    await expect(
      page.evaluate(({cache, inputs}) => window.testGemmaImageOffline(cache, inputs), {
        cache: cacheName,
        inputs: images
      })
    ).rejects.toThrow();
  } finally {
    await context.setOffline(false);
    await page.evaluate((cache) => window.clearGemmaImageCache(cache), cacheName);
  }
});
