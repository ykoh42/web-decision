import {defineConfig} from '@playwright/test';
import {fileURLToPath} from 'node:url';
import base from '../../../../playwright.config.ts';

const root = fileURLToPath(new URL('../../../../', import.meta.url));
const port = Number(process.env.WEB_DECISION_BROWSER_PORT || 41773);
export default defineConfig({
  ...base,
  testDir: fileURLToPath(new URL('.', import.meta.url)),
  testMatch: '**/*.spec.ts',
  ...(base.globalTeardown
    ? {
        globalTeardown: fileURLToPath(
          new URL('../../../../tests/browser/firefox-app-data-teardown.ts', import.meta.url)
        )
      }
    : {}),
  webServer: {
    command: 'pnpm exec vite --config tests/browser/vite.config.ts',
    cwd: root,
    url: `http://127.0.0.1:${port}`,
    reuseExistingServer: false,
    timeout: 30000
  }
});
