import {createReadStream} from 'node:fs';
import {stat, writeFile} from 'node:fs/promises';
import {createServer, type IncomingMessage, type ServerResponse} from 'node:http';
import {extname, resolve, sep} from 'node:path';
import {expect, test} from '@playwright/test';

const enabled = process.env.WEB_DECISION_KANANA_REAL_GPU === '1';
const imageEnabled = process.env.WEB_DECISION_KANANA_REAL_IMAGE === '1';
const assetsDirectory = resolve(process.env.WEB_DECISION_KANANA_VALIDATION_ASSETS ?? '.');
const visionAssetsDirectory = resolve(process.env.WEB_DECISION_KANANA_VISION_ASSETS ?? '.');
const checkpointPath = resolve(process.env.WEB_DECISION_KANANA_CHECKPOINT ?? '.');
const checkpointDirectory = resolve(checkpointPath, '..');

function replyCors(response: ServerResponse) {
  response.setHeader('Access-Control-Allow-Origin', '*');
  response.setHeader('Access-Control-Allow-Headers', 'Range');
  response.setHeader('Access-Control-Expose-Headers', 'Content-Length, Content-Range');
}

function resolveAssetPath(pathname: string): string | undefined {
  let relativePath = decodeURIComponent(pathname.slice('/assets/'.length));
  const visionAsset = relativePath.startsWith('vision/');
  if (visionAsset) relativePath = relativePath.slice('vision/'.length);
  const modelAsset = ['config.json', 'tokenizer.json', 'tokenizer_config.json'].includes(relativePath);
  const root = visionAsset ? visionAssetsDirectory : modelAsset ? checkpointDirectory : assetsDirectory;
  const filePath = relativePath === 'model.safetensors' ? checkpointPath : resolve(root, relativePath);
  if (filePath !== checkpointPath && !filePath.startsWith(`${root}${sep}`)) return undefined;
  return filePath;
}

type AssetRange = {status: 200 | 206; start: number; end: number};

function parseAssetRange(range: string | undefined, size: number, response: ServerResponse): AssetRange | undefined {
  let start = 0;
  let end = size - 1;
  let status: 200 | 206 = 200;
  if (range) {
    const match = /^bytes=(\d+)-(\d+)$/.exec(range);
    if (!match) {
      response.writeHead(416, {'Content-Range': `bytes */${size}`}).end();
      return undefined;
    }
    start = Number(match[1]);
    end = Number(match[2]);
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || end >= size) {
      response.writeHead(416, {'Content-Range': `bytes */${size}`}).end();
      return undefined;
    }
    status = 206;
  }
  return {status, start, end};
}

function assetHeaders(filePath: string, size: number, range: AssetRange): Record<string, string | number> {
  const headers: Record<string, string | number> = {
    'Accept-Ranges': 'bytes',
    'Content-Length': range.end - range.start + 1,
    'Content-Type': extname(filePath) === '.json' ? 'application/json' : 'application/octet-stream'
  };
  if (range.status === 206) headers['Content-Range'] = `bytes ${range.start}-${range.end}/${size}`;
  return headers;
}

async function serveRequestedAsset(request: IncomingMessage, response: ServerResponse): Promise<void> {
  const filePath = resolveAssetPath(new URL(request.url ?? '/', 'http://127.0.0.1').pathname);
  if (!filePath) {
    response.writeHead(404).end();
    return;
  }
  const info = await stat(filePath);
  if (!info.isFile()) {
    response.writeHead(404).end();
    return;
  }
  const range = parseAssetRange(request.headers.range, info.size, response);
  if (!range) return;
  response.writeHead(range.status, assetHeaders(filePath, info.size, range));
  if (request.method === 'HEAD') {
    response.end();
    return;
  }
  createReadStream(filePath, {start: range.start, end: range.end})
    .on('error', () => response.destroy())
    .pipe(response);
}

function serveAsset(request: IncomingMessage, response: ServerResponse) {
  replyCors(response);
  if (request.method === 'OPTIONS') {
    response.writeHead(204);
    response.end();
    return;
  }
  void serveRequestedAsset(request, response).catch((error: unknown) => {
    if (!response.headersSent) response.writeHead(500);
    response.end(error instanceof Error ? error.message : String(error));
  });
}

test('reports the real WebGPU features available to Kanana', async ({page, browserName, browser}) => {
  test.skip(!enabled, 'Set WEB_DECISION_KANANA_REAL_GPU=1 to inspect the local WebGPU adapter.');
  await page.goto('/');
  const gpu = await page.evaluate(async () => {
    const adapter = await navigator.gpu?.requestAdapter({powerPreference: 'high-performance'});
    if (!adapter) return null;
    const requiredFeatures = ['shader-f16', 'subgroups', 'chromium-experimental-subgroup-matrix'].filter((feature) =>
      adapter.features.has(feature as GPUFeatureName)
    ) as GPUFeatureName[];
    const device = await adapter.requestDevice({requiredFeatures});
    const shaderCases = {
      int8Array: [
        'enable chromium_experimental_subgroup_matrix;',
        '@group(0) @binding(0) var<storage, read> values: array<i8>;',
        '@compute @workgroup_size(1) fn main() { let value = values[0]; }'
      ],
      int32BoolConstructor: [
        '@group(0) @binding(0) var<storage, read> values: array<i32>;',
        '@compute @workgroup_size(1) fn main() { let value: bool = bool(values[0]); if (value) {} }'
      ],
      int32BoolComparison: [
        '@group(0) @binding(0) var<storage, read> values: array<i32>;',
        '@compute @workgroup_size(1) fn main() { let value: bool = values[0] != 0; if (value) {} }'
      ],
      int32BoolStoreCast: [
        '@group(0) @binding(0) var<storage, read_write> values: array<i32>;',
        '@compute @workgroup_size(1) fn main() { values[0] = i32(true); }'
      ],
      uint32BoolComparison: [
        '@group(0) @binding(0) var<storage, read> values: array<u32>;',
        '@compute @workgroup_size(1) fn main() { let value: bool = values[0] != 0u; if (value) {} }'
      ],
      negativeInfinityF32: ['@compute @workgroup_size(1) fn main() { let value: f32 = -inff; }'],
      negativeInfinityF32Bitcast: [
        '@compute @workgroup_size(1) fn main() { let value: f32 = bitcast<f32>(0xff800000u); }'
      ],
      negativeInfinityF32RuntimeBitcast: [
        '@compute @workgroup_size(1) fn main(@builtin(local_invocation_index) index: u32) { let value: f32 = bitcast<f32>(0xff800000u ^ (index & 0u)); }'
      ],
      negativeInfinityF16: ['enable f16;', '@compute @workgroup_size(1) fn main() { let value: f16 = -infh; }'],
      negativeInfinityF16FromF32: [
        'enable f16;',
        '@compute @workgroup_size(1) fn main() { let value: f16 = f16(bitcast<f32>(0xff800000u)); }'
      ],
      negativeInfinityF16RuntimeBitcast: [
        'enable f16;',
        '@compute @workgroup_size(1) fn main(@builtin(local_invocation_index) index: u32) { let value: f16 = f16(bitcast<f32>(0xff800000u ^ (index & 0u))); }'
      ]
    };
    const shaderMessages = await Promise.all(
      Object.entries(shaderCases).map(async ([name, lines]) => {
        const shader = device.createShaderModule({code: lines.join('\n')});
        const messages = (await shader.getCompilationInfo()).messages.map(({message, type}) => ({message, type}));
        return [name, messages] as const;
      })
    );
    device.destroy();
    return {
      adapterFeatures: Array.from(adapter.features),
      requestedFeatures: requiredFeatures,
      shaderMessages: Object.fromEntries(shaderMessages),
      limits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage
      }
    };
  });
  expect(gpu).not.toBeNull();
  console.log(`KANANA_GPU_ADAPTER ${JSON.stringify({browserName, version: browser.version(), gpu})}`);
  const invalidShaderCases = new Set([
    'int8Array',
    'negativeInfinityF32',
    'negativeInfinityF32Bitcast',
    'negativeInfinityF16',
    'negativeInfinityF16FromF32'
  ]);
  for (const [name, messages] of Object.entries(gpu?.shaderMessages ?? {}))
    expect(
      messages.some(({type}) => type === 'error'),
      name
    ).toBe(invalidShaderCases.has(name));
});

test('Kanana runs the pinned text decision on the local TVM WebGPU runtime', async ({page, browserName, browser}) => {
  test.skip(
    !enabled || !process.env.WEB_DECISION_KANANA_VALIDATION_ASSETS || !process.env.WEB_DECISION_KANANA_CHECKPOINT,
    'Set the Kanana real GPU environment variables to run the full local checkpoint.'
  );
  test.setTimeout(45 * 60 * 1000);

  const server = createServer(serveAsset);
  await new Promise<void>((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolveListen);
  });
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Kanana asset server did not start.');
  const assetsBase = `http://127.0.0.1:${address.port}/assets/`;
  const progress: Record<string, unknown>[] = [];
  const browserErrors: {text: string; arguments: unknown[]}[] = [];
  const browserErrorDetails: Promise<void>[] = [];

  try {
    page.on('console', (message) => {
      if (message.type() !== 'error') return;
      const error = {text: message.text(), arguments: [] as unknown[]};
      browserErrors.push(error);
      browserErrorDetails.push(
        Promise.all(
          message.args().map((argument) =>
            argument
              .evaluate((value) => {
                if (value === null || typeof value !== 'object') return value;
                const candidate = value as {name?: unknown; message?: unknown; stack?: unknown};
                return {
                  name: candidate.name,
                  message: candidate.message,
                  stack: candidate.stack
                };
              })
              .catch((error: unknown) => ({inspectionError: String(error)}))
          )
        ).then((details) => {
          error.arguments = details;
        })
      );
    });
    await page.exposeFunction('__kananaProgress', (event: Record<string, unknown>) => {
      progress.push(event);
      console.log(`KANANA_PROGRESS ${JSON.stringify(event)}`);
    });
    await page.goto('/');
    await page.addScriptTag({
      path: resolve('.toolchains/apache-tvm-src/web/lib/index.js')
    });
    const result = await page.evaluate(
      async ({assetsBase, referenceUrl}) => {
        const runnerUrl = new URL('/kanana-real-webgpu.ts', location.href).href;
        const runner = await import(/* @vite-ignore */ runnerUrl);
        return runner.runKananaRealWebGpu(assetsBase, new URL('pytorch-reference.json', assetsBase).href, referenceUrl);
      },
      {
        assetsBase,
        referenceUrl: new URL('decision-reference.json', assetsBase).href
      }
    );

    await Promise.all(browserErrorDetails);
    const report = {browserName, version: browser.version(), ...result, progress, browserErrors};
    console.log(`KANANA_RESULT ${JSON.stringify(report)}`);
    const reportPath = process.env.WEB_DECISION_KANANA_REPORT;
    if (reportPath) {
      await writeFile(resolve(reportPath), `${JSON.stringify(report, null, 2)}\n`);
    }
    expect(result.webgpuErrors).toEqual([]);
    expect(result.cleanupErrors).toEqual([]);
    expect(result.kind).not.toBe('execution-failure');
    if (result.kind === 'hidden-state-probe') {
      expect(result.hiddenState.maxAbsDelta).toBeLessThan(0.05);
      expect(result.hiddenState.rmse).toBeLessThan(0.005);
    } else {
      const probabilityTolerance = 0.001;
      expect(result.answer.choice).toBe(result.expected.choice);
      expect(result.answer.type).toBe(result.expected.type);
      expect(Object.keys(result.answer.probabilities ?? {})).toEqual(Object.keys(result.expected.probabilities ?? {}));
      const expectedProbabilities = (result.expected.probabilities ?? {}) as Record<string, number>;
      for (const probability of Object.values(result.answer.probabilities ?? {}))
        expect(Number.isFinite(probability)).toBe(true);
      expect(Number.isFinite(result.answer.confidence)).toBe(true);
      expect(Math.abs(result.answer.confidence - result.expected.confidence)).toBeLessThan(probabilityTolerance);
      for (const [choice, expectedProbability] of Object.entries(expectedProbabilities)) {
        const actualProbability = result.answer.probabilities?.[choice];
        expect(actualProbability).toBeDefined();
        expect(Math.abs((actualProbability ?? Number.NaN) - expectedProbability)).toBeLessThan(probabilityTolerance);
      }
      expect(
        Math.abs(
          (Object.values(result.answer.probabilities ?? {}) as number[]).reduce<number>(
            (sum, value) => sum + value,
            0
          ) - 1
        )
      ).toBeLessThan(1e-5);
    }
  } finally {
    await new Promise<void>((resolveClose, reject) =>
      server.close((error) => (error ? reject(error) : resolveClose()))
    );
  }
});

test('Kanana runs an encoded image through the real vision and text graphs', async ({page, browserName, browser}) => {
  test.skip(
    !enabled ||
      !imageEnabled ||
      !process.env.WEB_DECISION_KANANA_VALIDATION_ASSETS ||
      !process.env.WEB_DECISION_KANANA_VISION_ASSETS ||
      !process.env.WEB_DECISION_KANANA_CHECKPOINT,
    'Set the Kanana text, vision, and checkpoint paths with WEB_DECISION_KANANA_REAL_IMAGE=1.'
  );
  test.setTimeout(45 * 60 * 1000);

  const server = createServer(serveAsset);
  await new Promise<void>((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolveListen);
  });
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Kanana image asset server did not start.');
  const assetsBase = `http://127.0.0.1:${address.port}/assets/`;

  try {
    await page.goto('/');
    await page.addScriptTag({path: resolve('.toolchains/apache-tvm-src/web/lib/index.js')});
    const result = await page.evaluate(
      async ({assetsBase}) => {
        const runnerUrl = new URL('/kanana-real-image-webgpu.ts', location.href).href;
        const runner = await import(/* @vite-ignore */ runnerUrl);
        return runner.runKananaRealImageDecisionWebGpu(assetsBase);
      },
      {assetsBase}
    );

    console.log(`KANANA_REAL_IMAGE_WEBGPU ${JSON.stringify({browserName, version: browser.version(), result})}`);
    expect(result.model).toBe('kakaocorp/kanana-2-v-decide-3b-2611');
    expect(result.textWeights).toBe(354);
    expect(result.visionWeights).toBe(333);
    expect(result.imageTokenThw).toEqual([1, 8, 8]);
    expect(result.answer.type).toBe('choice');
    expect(result.probabilitySum).toBeCloseTo(1, 6);
    expect(result.webgpuErrors).toEqual([]);
  } finally {
    await new Promise<void>((resolveClose, reject) =>
      server.close((error) => (error ? reject(error) : resolveClose()))
    );
  }
});
