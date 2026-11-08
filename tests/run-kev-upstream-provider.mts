import assert from 'node:assert/strict';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {chromium, type Browser} from '@playwright/test';

const root = dirname(fileURLToPath(import.meta.url));
const vite = await createServer({configFile: join(root, 'browser/kev-upstream-provider.vite.config.mts')});
let browser: Browser | undefined;
try {
  await vite.listen();
  browser = await chromium.launch({
    ...(process.env.CHROME_PATH ? {executablePath: process.env.CHROME_PATH} : {}),
    headless: true
  });
  const page = await browser.newPage();
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const localUrl = vite.resolvedUrls?.local[0];
  assert.ok(localUrl, 'Vite must expose a local URL');
  await page.goto(new URL('/kev-upstream-provider.html', localUrl).href, {waitUntil: 'load'});
  await page.waitForFunction(() => document.querySelector('#result')?.textContent !== 'running', undefined, {
    timeout: 30000
  });
  const resultText = await page.locator('#result').textContent();
  assert.ok(resultText, 'The browser test must produce a result');
  const result = JSON.parse(resultText);
  assert.equal(result.ok, true, result.error ?? JSON.stringify(result));
  assert.deepEqual(errors, []);
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser?.close();
  await vite.close();
}
