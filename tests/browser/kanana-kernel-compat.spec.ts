import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';

const enabled = process.env.WEB_DECISION_KANANA_REAL_GPU === '1';
const shaderMarker = Buffer.from('@group(0) @binding(0)');
const graphFiles = [
  'packages/web-decision/assets/kanana/kanana-text-model.wasm',
  'packages/web-decision/assets/kanana/kanana-vision-model.wasm'
];

function isWgslByte(value: number): boolean {
  return (value >= 32 && value <= 126) || value === 9 || value === 10 || value === 13;
}

function findShaderMarker(wasm: Buffer, offset: number): number | null {
  const marker = wasm.indexOf(shaderMarker, offset);
  return marker < 0 ? null : marker;
}

function findComputeMarker(wasm: Buffer, marker: number): number | null {
  const compute = wasm.indexOf(Buffer.from('@compute '), marker);
  return compute < 0 ? null : compute;
}

function findShaderStart(wasm: Buffer, marker: number): number {
  let start = marker;
  while (start > 0 && isWgslByte(wasm[start - 1]!)) start--;
  return start;
}

function findShaderEnd(wasm: Buffer, compute: number): number {
  let depth = 0;
  for (let end = compute; end < wasm.length; end++) {
    if (wasm[end] === 123) depth++;
    if (wasm[end] !== 125) continue;
    depth--;
    if (depth === 0) return end + 1;
  }
  return wasm.length;
}

function extractComputeShaders(wasm: Buffer): string[] {
  const shaders: string[] = [];
  let offset = 0;
  for (;;) {
    const marker = findShaderMarker(wasm, offset);
    if (marker === null) return shaders;
    const compute = findComputeMarker(wasm, marker);
    if (compute === null) return shaders;
    const start = findShaderStart(wasm, marker);
    const end = findShaderEnd(wasm, compute);
    shaders.push(wasm.subarray(start, end).toString('utf8'));
    offset = end + 1;
  }
}

test('the pinned Kanana TVM kernels fit and compile on the local WebGPU device', async ({page, browser}) => {
  test.skip(!enabled, 'Set WEB_DECISION_KANANA_REAL_GPU=1 to compile Kanana kernels on the local GPU.');
  const graphs = await Promise.all(
    graphFiles.map(async (path) => extractComputeShaders(await readFile(resolve(path))))
  );
  const shaders = graphs.flat();
  const maxStorageBuffers = Math.max(...shaders.map((source) => (source.match(/var<storage,/gu) ?? []).length));
  expect(graphs.map((graph) => graph.length)).toEqual([82, 51]);
  expect(maxStorageBuffers).toBe(5);

  await page.goto('/');
  await page.addScriptTag({path: resolve('.toolchains/apache-tvm-src/web/lib/index.js')});
  const report = await page.evaluate(async (sources) => {
    const runtime = (
      window as unknown as {
        tvmjs: {
          detectGPUDevice(): Promise<{adapter: GPUAdapter; device: GPUDevice}>;
        };
      }
    ).tvmjs;
    const {adapter, device} = await runtime.detectGPUDevice();
    const failures: {index: number; entryPoint?: string; message: string}[] = [];

    for (const [index, source] of sources.entries()) {
      const entryPoint = source.match(/@compute\b[\s\S]{0,200}?\bfn\s+([A-Za-z_][A-Za-z0-9_]*)/u)?.[1];
      if (!entryPoint) {
        failures.push({index, message: 'Could not read the WGSL compute entry point.'});
        continue;
      }
      try {
        const module = device.createShaderModule({code: source});
        const messages = (await module.getCompilationInfo()).messages.filter(({type}) => type === 'error');
        if (messages.length) {
          failures.push({index, entryPoint, message: messages.map(({message}) => message).join('\n')});
          continue;
        }
        await device.createComputePipelineAsync({
          layout: 'auto',
          compute: {module, entryPoint}
        });
      } catch (error) {
        failures.push({index, entryPoint, message: error instanceof Error ? error.message : String(error)});
      }
    }

    const limits = {
      adapter: adapter.limits.maxStorageBuffersPerShaderStage,
      device: device.limits.maxStorageBuffersPerShaderStage
    };
    device.destroy();
    return {limits, compiled: sources.length - failures.length, failures};
  }, shaders);

  console.log(
    `KANANA_KERNEL_COMPATIBILITY ${JSON.stringify({browser: browser.version(), ...report, total: shaders.length})}`
  );
  expect(report.limits.device).toBe(report.limits.adapter);
  expect(report.compiled).toBe(shaders.length);
  expect(report.failures).toEqual([]);
});
