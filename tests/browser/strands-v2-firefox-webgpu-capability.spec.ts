import {expect, test} from '@playwright/test';

test('reports Firefox WebGPU adapter and limits without running a model', async ({page}, testInfo) => {
  test.skip(process.env.WEB_DECISION_BROWSER_NAME !== 'firefox', 'Run with the isolated Firefox project');
  await page.goto('/strands-v2-webgpu-capability.html');
  const report = await page.evaluate(async () => {
    type Adapter = {
      info?: {
        vendor?: string;
        architecture?: string;
        device?: string;
        description?: string;
        isFallbackAdapter?: boolean;
      };
      isFallbackAdapter?: boolean;
      features?: Iterable<string>;
      limits: Record<string, number>;
    };
    const api = (navigator as unknown as {gpu?: {requestAdapter(): Promise<Adapter | null>}}).gpu;
    const common = {userAgent: navigator.userAgent, secureContext: isSecureContext, navigatorGpu: !!api};
    if (!api) return {...common, adapter: null, error: null};
    try {
      const adapter = await api.requestAdapter();
      if (!adapter) return {...common, adapter: null, error: null};
      const names = [
        'maxBufferSize',
        'maxStorageBufferBindingSize',
        'maxStorageBuffersPerShaderStage',
        'maxComputeWorkgroupStorageSize',
        'maxComputeInvocationsPerWorkgroup',
        'maxComputeWorkgroupsPerDimension',
        'maxComputeWorkgroupSizeX',
        'maxComputeWorkgroupSizeY',
        'maxComputeWorkgroupSizeZ'
      ];
      return {
        ...common,
        error: null,
        adapter: {
          info: adapter.info
            ? {
                vendor: adapter.info.vendor ?? null,
                architecture: adapter.info.architecture ?? null,
                device: adapter.info.device ?? null,
                description: adapter.info.description ?? null
              }
            : null,
          fallback: adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null,
          features: adapter.features ? [...adapter.features].sort() : [],
          limits: Object.fromEntries(names.map((name) => [name, adapter.limits[name] ?? null]))
        }
      };
    } catch (error) {
      return {...common, adapter: null, error: {name: (error as Error).name, message: (error as Error).message}};
    }
  });
  console.log(`FIREFOX_WEBGPU_CAPABILITY ${JSON.stringify(report)}`);
  await testInfo.attach('firefox-webgpu-capability.json', {
    body: JSON.stringify(report, null, 2),
    contentType: 'application/json'
  });
  expect(report.secureContext).toBe(true);
});
