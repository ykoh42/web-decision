import {expect, test} from '@playwright/test';

test('offline harness bootstraps without downloading weights or initializing WebGPU', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the experimental probe');
  const failures: string[] = [];
  const requests: string[] = [];
  const binaries: string[] = [];
  page.on('pageerror', (error) => failures.push(error.message));
  page.on('request', (request) => requests.push(request.url()));
  page.on('response', (response) => {
    if ((response.headers()['content-type'] ?? '').includes('application/wasm')) binaries.push(response.url());
  });
  await page.goto('/strands-v2-offline-full-probe.html');
  await page.waitForFunction(() => typeof window.gemmaOfflineBootstrap === 'function', undefined, {timeout: 15000});
  const cache = `web-decision-offline-bootstrap-${Date.now()}`;
  try {
    const result = await page.evaluate(
      ({name}) =>
        window.gemmaOfflineBootstrap({
          base: 'http://127.0.0.1:1/',
          cache: name,
          pins: {},
          requiredBytes: 0,
          checkCapacity: false
        }),
      {name: cache}
    );
    expect(typeof result.persisted).toBe('boolean');
    expect(failures).toEqual([]);
    expect(requests.filter((url) => url.includes('model.safetensors') || url.includes('huggingface.co'))).toEqual([]);
    expect(binaries).toEqual([]);
  } finally {
    await page.evaluate((name) => caches.delete(name), cache);
  }
});
