import {defineConfig} from '@playwright/test';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import workspaceConfig from '../../../playwright.config.mjs';

const testDirectory = dirname(fileURLToPath(import.meta.url));
const workspaceRoot = resolve(testDirectory, '../../..');

export default defineConfig({
  ...workspaceConfig,
  testDir: testDirectory,
  testMatch: '**/tamev-micro-wasm.spec.mjs',
  ...(workspaceConfig.globalTeardown
    ? {globalTeardown: resolve(workspaceRoot, workspaceConfig.globalTeardown)}
    : {}),
  outputDir: resolve(testDirectory, '../build/playwright-results'),
  webServer: {...workspaceConfig.webServer, cwd: workspaceRoot}
});
