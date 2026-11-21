import type {Page} from '@playwright/test';

export async function readHardwareReport(page: Page) {
  await page.goto('chrome://gpu');
  return page.evaluate(() => {
    const parts: string[] = [];
    function visit(root: Document | ShadowRoot) {
      if (root.textContent) parts.push(root.textContent);
      for (const element of root.querySelectorAll('*')) if (element.shadowRoot) visit(element.shadowRoot);
    }
    visit(document);
    const text = parts.join('\n');
    return {
      status: text.match(/WebGPU:\s*(Hardware accelerated|Software only|Disabled|Blocklisted)/iu)?.[1] ?? null,
      renderer:
        text.match(/ANGLE Metal Renderer:\s*([^\n]+?)(?=\)\)|\s*GL_VERSION|\n|$)/iu)?.[1]?.trim() ??
        text.match(/GL_RENDERER\s*:\s*([^\n]+)/iu)?.[1]?.trim() ??
        null
    };
  });
}

type CandidateAdapter = {
  readonly info: {
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
    readonly isFallbackAdapter: boolean;
  };
  readonly limits: {
    readonly maxBufferSize: number;
    readonly maxStorageBufferBindingSize: number;
  };
};

export async function readAdapter(page: Page) {
  return page.evaluate(async () => {
    const {gpu} = navigator as Navigator & {
      gpu?: {requestAdapter(options: {powerPreference: 'high-performance'}): Promise<CandidateAdapter | null>};
    };
    const adapter = await gpu?.requestAdapter({powerPreference: 'high-performance'});
    if (!adapter) return null;
    const {vendor, architecture, device, description, isFallbackAdapter} = adapter.info;
    const {maxBufferSize, maxStorageBufferBindingSize} = adapter.limits;
    return {vendor, architecture, device, description, isFallbackAdapter, maxBufferSize, maxStorageBufferBindingSize};
  });
}
