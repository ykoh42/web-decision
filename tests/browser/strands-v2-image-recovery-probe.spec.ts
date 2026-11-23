import {readFile} from 'node:fs/promises';
import {expect, test} from '@playwright/test';

test('browser JPEG initialization failure and cancellation allow session reuse', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the experimental probe');
  test.setTimeout(120000);
  let attempts = 0;
  let entered!: () => void;
  let release!: () => void;
  const requestStarted = new Promise<void>((resolve) => {
    entered = resolve;
  });
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route('**/mozjpeg_dec.wasm**', async (route) => {
    if (route.request().resourceType() !== 'fetch') {
      await route.continue();
      return;
    }
    attempts++;
    if (attempts === 1) {
      await route.fulfill({status: 503, body: 'temporary failure'});
      return;
    }
    if (attempts === 2) {
      entered();
      await gate;
    }
    await route.continue();
  });
  const fixtures = new URL('../../packages/web-decision/test/fixtures/qwen35-image-input/', import.meta.url);
  const jpeg = (await readFile(new URL('jpeg-orientation-6.jpg', fixtures))).toString('base64');
  const png = (await readFile(new URL('single-pixel.png', fixtures))).toString('base64');
  await page.goto('/strands-v2-image-recovery-probe.html');
  await page.waitForFunction(() => typeof window.initializeImageRecovery === 'function');
  await page.evaluate(({j, p}) => window.initializeImageRecovery(j, p), {j: jpeg, p: png});
  expect((await page.evaluate(() => window.callRecovery('invalid'))).ok).toBe(false);
  const failed = await page.evaluate(() => window.callRecovery('jpeg'));
  expect(failed.ok).toBe(false);
  expect(failed.message).toMatch(/fetch|503/i);
  expect(attempts).toBe(1);
  expect((await page.evaluate(() => window.callRecovery('png'))).ok).toBe(true);
  await page.evaluate(() => window.startCancelledImage());
  try {
    await requestStarted;
    await page.evaluate(() => window.abortImage());
    const cancelled = await page.evaluate(() => window.cancelledImageResult());
    expect(cancelled.ok).toBe(false);
    expect(cancelled.name).toBe('AbortError');
    await page.evaluate(() => window.startRetryImage());
  } finally {
    release();
  }
  expect((await page.evaluate(() => window.retryImageResult())).ok).toBe(true);
  expect((await page.evaluate(() => window.callRecovery('jpeg'))).ok).toBe(true);
  expect(attempts).toBe(2);
  const result = await page.evaluate(() => window.finishImageRecovery());
  expect(result.calls).toBe(3);
  expect(result.disposed).toBe(true);
  expect(result.afterDispose.ok).toBe(false);
});
