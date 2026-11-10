import baseConfig from '../../playwright.config.ts';

const baseURL = 'http://127.0.0.1:8767';

export default {
  ...baseConfig,
  testDir: '../../tests/browser',
  testMatch: 'direct-kev-package.spec.ts',
  use: {
    ...baseConfig.use,
    baseURL,
    browserName: 'chromium',
    channel: 'chrome',
    headless: false,
    ignoreHTTPSErrors: true,
    launchOptions: {
      ...baseConfig.use?.launchOptions,
      args: ['--enable-unsafe-webgpu'],
      ignoreDefaultArgs: ['--enable-unsafe-swiftshader']
    }
  },
  webServer: {
    command: 'node_modules/.bin/vite --config tests/browser/vite.config.ts',
    url: baseURL,
    reuseExistingServer: true,
    timeout: 30_000
  }
};
