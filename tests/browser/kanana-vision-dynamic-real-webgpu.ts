import {openSafetensorsWeightSource} from '../../packages/web-decision/src/kanana/weight-source.mjs';
import {setupKananaTvmRuntime} from '../../packages/web-decision/src/kanana/tvm-runtime.js';
import {
  createKananaTvmVisionInferenceSession,
  type KananaTvmVisionInferenceSessionOptions
} from '../../packages/web-decision/src/kanana/tvm-session.js';

type TvmJs = {
  instantiate(
    bytes: ArrayBuffer,
    imports: Record<string, unknown>,
    logger: (message: string) => void
  ): Promise<unknown>;
  createPolyfillWASI(): Record<string, unknown>;
  detectGPUDevice(): Promise<{adapter: GPUAdapter; device: GPUDevice} | undefined>;
};

type GraphWeight = {
  name: string;
  parameterName: string;
  shape: number[];
  dtype: 'float32';
  length: number;
};

type TestCase = {
  name: string;
  grid: [number, number, number];
  pixelShape: number[];
  expectedShape: number[];
  files: Record<'pixels' | 'expected', string>;
};

type ValidationWeightSource = {
  records: readonly {name: string; parameterName?: string; shape: readonly number[]; length: number}[];
  readTensor(name: string): Promise<{bytes: Uint8Array}>;
};

type ValidationWeightOpener = (options: {
  base: string;
  manifest: Record<string, unknown>;
  requiredWeights: readonly Record<string, unknown>[];
  cache: false;
}) => Promise<ValidationWeightSource>;

const openWeights = openSafetensorsWeightSource as unknown as ValidationWeightOpener;

declare global {
  interface Window {
    tvmjs?: TvmJs;
    __kananaVisionProgress?(event: Record<string, unknown>): void;
  }
}

async function readJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not read ${url}: HTTP ${response.status}.`);
  return (await response.json()) as T;
}

async function readBytes(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not read ${url}: HTTP ${response.status}.`);
  return new Uint8Array(await response.arrayBuffer());
}

function float16ToFloat32(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength % 2 !== 0) throw new Error('The FP16 weight byte length must be even.');
  const source = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = new Float32Array(bytes.byteLength / 2);
  for (let index = 0; index < values.length; index++) {
    const half = source.getUint16(index * 2, true);
    const sign = half & 0x8000 ? -1 : 1;
    const exponent = (half >>> 10) & 0x1f;
    const fraction = half & 0x03ff;
    if (exponent === 0x1f) throw new Error('The Kanana checkpoint contains a non-finite vision weight.');
    values[index] =
      exponent === 0 ? sign * 2 ** -14 * (fraction / 1024) : sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
  }
  return values;
}

function compare(actual: Float32Array, expected: Float32Array) {
  if (actual.length !== expected.length)
    throw new Error('Kanana vision output shape differs from the PyTorch reference.');
  let maxAbsDelta = 0;
  let sumSquaredDelta = 0;
  for (let index = 0; index < actual.length; index++) {
    const delta = actual[index]! - expected[index]!;
    maxAbsDelta = Math.max(maxAbsDelta, Math.abs(delta));
    sumSquaredDelta += delta * delta;
  }
  return {maxAbsDelta, rmse: Math.sqrt(sumSquaredDelta / actual.length)};
}

export async function runKananaDynamicVisionRealWebGpu(assetsBase: string) {
  const tvmjs = window.tvmjs;
  if (!tvmjs) throw new Error('Apache TVM JS runtime bundle is not loaded.');
  const base = new URL(assetsBase);
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  const [rangeManifest, graphManifest, testCases, buildReport] = await Promise.all([
    readJson<Record<string, unknown>>(new URL('range-manifest.json', base).href),
    readJson<{weights: GraphWeight[]}>(new URL('external-weight-manifest.json', base).href),
    readJson<TestCase[]>(new URL('cases.json', base).href),
    readJson<{maxPatchCount: number}>(new URL('build-report.json', base).href)
  ]);
  const adapterSetup = await tvmjs.detectGPUDevice();
  if (!adapterSetup) throw new Error('This browser does not expose WebGPU.');
  const {adapter, device} = adapterSetup;
  const webgpuErrors: string[] = [];
  device.addEventListener('uncapturederror', (event) => {
    const error = (event as Event & {error?: {message?: string}}).error;
    webgpuErrors.push(error?.message ?? 'Unspecified WebGPU error.');
  });
  const requiredWeights = graphManifest.weights.map((weight) => ({
    ...weight,
    dtype: 'float16',
    length: weight.shape.reduce((size, dimension) => size * dimension, 1) * 2
  }));
  const source = await openWeights({base: base.href, manifest: rangeManifest, requiredWeights, cache: false});
  const wasmResponse = await fetch(new URL('kanana-vision-model.wasm', base));
  if (!wasmResponse.ok) throw new Error(`Could not read the TVM vision WASM: HTTP ${wasmResponse.status}.`);
  const setup = await setupKananaTvmRuntime({
    instantiate: tvmjs.instantiate as never,
    createPolyfillWASI: tvmjs.createPolyfillWASI,
    device,
    bytes: await wasmResponse.arrayBuffer()
  });
  let vision: Awaited<ReturnType<typeof createKananaTvmVisionInferenceSession>> | undefined;
  let loadedWeights = 0;
  const progress = (stage: string, completed: number, total: number) => {
    window.__kananaVisionProgress?.({stage, completed, total});
  };

  try {
    const manifestWeights = graphManifest.weights.map((weight) => ({
      name: weight.name,
      parameterName: weight.parameterName,
      shape: weight.shape,
      dtype: 'float32',
      length: weight.shape.reduce((size, dimension) => size * dimension, 1) * Float32Array.BYTES_PER_ELEMENT
    }));
    vision = await createKananaTvmVisionInferenceSession({
      setup: setup as unknown as KananaTvmVisionInferenceSessionOptions['setup'],
      weightSource: {
        manifest: {weights: manifestWeights},
        async readWeight(record) {
          const tensor = await source.readTensor(record.name);
          const values = float16ToFloat32(tensor.bytes);
          loadedWeights++;
          if (loadedWeights % 16 === 0 || loadedWeights === source.records.length)
            progress('vision weight loading', loadedWeights, source.records.length);
          return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
        }
      },
      patchValues: 3 * 2 * 16 * 16,
      numPositionEmbeddings: 2_304,
      spatialMergeSize: 2,
      maxPatchCount: buildReport.maxPatchCount
    });

    const results = [];
    for (const testCase of testCases) {
      const caseBase = new URL('cases/', base);
      const [pixelBytes, expectedBytes] = await Promise.all([
        readBytes(new URL(testCase.files.pixels, caseBase).href),
        readBytes(new URL(testCase.files.expected, caseBase).href)
      ]);
      const expected = new Float32Array(
        expectedBytes.buffer,
        expectedBytes.byteOffset,
        expectedBytes.byteLength / Float32Array.BYTES_PER_ELEMENT
      );
      if (pixelBytes.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0)
        throw new Error(`${testCase.name} pixels have invalid byte length.`);
      const pixels = new Float32Array(pixelBytes.buffer, pixelBytes.byteOffset, pixelBytes.byteLength / 4);
      const result = await vision.infer(pixels, testCase.grid);
      const comparison = compare(result.values, expected);
      if (!sameShape(result.shape, testCase.expectedShape))
        throw new Error(`${testCase.name} produced [${result.shape}] instead of [${testCase.expectedShape}].`);
      results.push({name: testCase.name, shape: result.shape, ...comparison});
    }
    if (webgpuErrors.length) throw new Error(`WebGPU reported errors: ${webgpuErrors.join('; ')}`);
    return {
      model: 'kakaocorp/kanana-2-v-decide-3b-2611',
      revision: '12820d654ed9789a0f7f48daeee1703c3483c464',
      gpu: {vendor: adapter.info?.vendor ?? '', architecture: adapter.info?.architecture ?? ''},
      weights: loadedWeights,
      loadedBytes: source.records.reduce((total, record) => total + record.length, 0),
      cases: results,
      webgpuErrors
    };
  } finally {
    if (vision) await vision.dispose();
    else setup.dispose();
  }
}

function sameShape(actual: readonly number[], expected: readonly number[]): boolean {
  return actual.length === expected.length && actual.every((dimension, index) => dimension === expected[index]);
}
