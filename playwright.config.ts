import {defineConfig} from '@playwright/test';
import {mkdirSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

const port = Number(process.env.WEB_DECISION_BROWSER_PORT || 41773);
const baseURL = `http://127.0.0.1:${port}`;
const realGPU =
  process.env.WEB_DECISION_REAL_GPU === '1' ||
  process.env.WEB_DECISION_KANANA_CURATED_REAL_GPU === '1' ||
  process.env.WEB_DECISION_KANANA_LOCAL_GPU === '1';
const requestedBrowser = process.env.WEB_DECISION_BROWSER_NAME || 'chromium';
if (!['chromium', 'firefox', 'webkit'].includes(requestedBrowser)) {
  throw new Error(`Unsupported browser: ${requestedBrowser}`);
}
const browserName = requestedBrowser as 'chromium' | 'firefox' | 'webkit';
const channel =
  process.env.WEB_DECISION_BROWSER_CHANNEL || (realGPU && browserName === 'chromium' ? 'chromium' : undefined);
const chromiumArgs = [
  ...(realGPU ? ['--enable-unsafe-webgpu'] : []),
  ...(process.env.WEB_DECISION_GPU_BROWSER_MARKER ? [process.env.WEB_DECISION_GPU_BROWSER_MARKER] : [])
];
// macOS 27 can block Firefox's shared app-data directory from Playwright.
const firefoxAppDataHome =
  process.platform === 'darwin' && browserName === 'firefox'
    ? join(tmpdir(), `web-decision-firefox-home-${port}`)
    : null;

if (firefoxAppDataHome) {
  mkdirSync(firefoxAppDataHome, {recursive: true});
  process.env.WEB_DECISION_PLAYWRIGHT_FIREFOX_HOME = firefoxAppDataHome;
}

export default defineConfig({
  testDir: './tests/browser',
  testMatch: '**/*.spec.{mjs,ts}',
  timeout: 30000,
  workers: 1,
  reporter: 'list',
  projects: [{name: browserName}],
  ...(firefoxAppDataHome ? {globalTeardown: './tests/browser/firefox-app-data-teardown.ts'} : {}),
  use: {
    baseURL,
    browserName,
    ...(realGPU ? {headless: false} : {}),
    ...(channel ? {channel} : {}),
    launchOptions: {
      args: realGPU && browserName === 'chromium' ? chromiumArgs : [],
      ...(realGPU && browserName === 'chromium' ? {ignoreDefaultArgs: ['--enable-unsafe-swiftshader']} : {}),
      ...(firefoxAppDataHome ? {env: {...process.env, CFFIXED_USER_HOME: firefoxAppDataHome}} : {})
    }
  },
  webServer: {
    command: 'pnpm exec vite --config tests/browser/vite.config.ts',
    url: baseURL,
    reuseExistingServer: false,
    timeout: 30000
  }
});
