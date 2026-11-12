import {openSafetensorsWeightSource} from '../../packages/web-decision/src/kanana/weight-source.mjs';
import {setupKananaTvmRuntime} from '../../packages/web-decision/src/kanana/tvm-runtime.js';

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
    __kananaVisionProgress?(event: Record<string, unknown>): void;
  }
}

async function readJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not read ${url}: HTTP ${response.status}.`);
  return (await response.json()) as T;
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

function compare(actual: Float32Array, expected: number[]) {
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

export async function runKananaVisionRealWebGpu(assetsBase: string) {
  const tvmjs = (window as Window & {tvmjs?: TvmJs}).tvmjs;
  if (!tvmjs) throw new Error('Apache TVM JS runtime bundle is not loaded.');
  const base = new URL(assetsBase);
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  const [rangeManifest, graphManifest, reference, pixelsBuffer] = await Promise.all([
    readJson<Record<string, unknown>>(new URL('range-manifest.json', base).href),
    readJson<{weights: GraphWeight[]}>(new URL('external-weight-manifest.json', base).href),
    readJson<{shape: number[]; values: number[]; pixelShape: number[]; grid: number[][]}>(
      new URL('pytorch-reference.json', base).href
    ),
    fetch(new URL('pixel-values-f32.bin', base)).then(async (response) => {
      if (!response.ok) throw new Error(`Could not read the vision input: HTTP ${response.status}.`);
      return response.arrayBuffer();
    })
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
  const source = await openWeights({
    base: base.href,
    manifest: rangeManifest,
    requiredWeights,
    cache: false
  });
  const wasmResponse = await fetch(new URL('kanana-vision-model.wasm', base));
  if (!wasmResponse.ok) throw new Error(`Could not read the TVM vision WASM: HTTP ${wasmResponse.status}.`);
  const setup = await setupKananaTvmRuntime({
    instantiate: tvmjs.instantiate as never,
    createPolyfillWASI: tvmjs.createPolyfillWASI,
    device,
    bytes: await wasmResponse.arrayBuffer()
  });
  const runtime = setup.runtime as typeof setup.runtime & {
    empty(
      shape: number[],
      dtype: string,
      device: object
    ): {
      readonly shape: readonly number[];
      readonly dtype: string;
      copyFrom(source: unknown): unknown;
      copyFromRawBytes(bytes: Uint8Array): unknown;
      toRawBytes(): Uint8Array;
    };
    cpu(): object;
    webgpu(): object;
    scalar(value: number, dtype: string): unknown;
  };
  const vm = setup.vm as typeof setup.vm & {getFunction(name: string): (...args: unknown[]) => unknown};
  const main = vm.getFunction('main');
  const arity = vm.getFunction('get_function_arity')('main');
  const getParameterName = vm.getFunction('get_function_param_name');
  const parameterNames = Array.from({length: arity as number}, (_, index) =>
    getParameterName('main', runtime.scalar(index, 'int'))
  ) as string[];
  const weights = new Map<string, ReturnType<typeof runtime.empty>>();
  const progress = (stage: string, completed: number, total: number) => {
    window.__kananaVisionProgress?.({stage, completed, total});
  };

  try {
    for (let index = 0; index < source.records.length; index++) {
      const record = source.records[index]!;
      const tensor = await source.readTensor(record.name);
      const values = float16ToFloat32(tensor.bytes);
      const target = runtime.empty([...record.shape], 'float32', runtime.webgpu());
      target.copyFrom(values);
      weights.set(record.parameterName!, target);
      if (index % 16 === 15 || index === source.records.length - 1)
        progress('vision weight loading', index + 1, source.records.length);
    }
    await device.queue.onSubmittedWorkDone();

    const pixelValues = new Float32Array(pixelsBuffer);
    const input = runtime.empty(reference.pixelShape, 'float32', runtime.webgpu());
    input.copyFrom(pixelValues);
    const grid = runtime.empty([1, 3], 'int64', runtime.webgpu());
    const gridBytes = new Uint8Array(BigInt64Array.from(reference.grid.flat(), (value) => BigInt(value)).buffer);
    grid.copyFromRawBytes(gridBytes);
    const args = parameterNames.map((name) => {
      if (name === 'pixel_values') return input;
      if (name === 'image_grid_thw') return grid;
      const weight = weights.get(name);
      if (!weight) throw new Error(`The TVM graph requires an unprovided weight: ${name}.`);
      return weight;
    });
    if (weights.size !== graphManifest.weights.length)
      throw new Error(`Loaded ${weights.size} vision weights; expected ${graphManifest.weights.length}.`);
    runtime.beginScope();
    let output: ReturnType<typeof runtime.empty>;
    try {
      output = (await main(...args)) as ReturnType<typeof runtime.empty>;
      await device.queue.onSubmittedWorkDone();
      const host = runtime.empty([...output.shape], 'float32', runtime.cpu());
      host.copyFrom(output);
      await device.queue.onSubmittedWorkDone();
      const bytes = host.toRawBytes();
      const actual = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
      const comparison = compare(actual, reference.values);
      if (webgpuErrors.length) throw new Error(`WebGPU reported errors: ${webgpuErrors.join('; ')}`);
      return {
        model: 'kakaocorp/kanana-2-v-decide-3b-2611',
        revision: '12820d654ed9789a0f7f48daeee1703c3483c464',
        shape: output.shape,
        ...comparison,
        gpu: {
          vendor: adapter.info?.vendor ?? '',
          architecture: adapter.info?.architecture ?? '',
          maxBufferSize: device.limits.maxBufferSize
        },
        weights: weights.size,
        loadedBytes: source.records.reduce((total, record) => total + record.length, 0),
        webgpuErrors
      };
    } finally {
      runtime.endScope();
    }
  } finally {
    setup.dispose();
  }
}
