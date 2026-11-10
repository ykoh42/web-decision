import {createKananaDecisionSession} from '../../packages/web-decision/src/kanana/decision-session.js';
import {createKananaTvmImageEmbeddingEncoder} from '../../packages/web-decision/src/kanana/image-encoder.ts';
import {preprocessKananaImage} from '../../packages/web-decision/src/kanana/image-processing.js';
import {decodeKananaImageSource} from '../../packages/web-decision/src/kanana/browser-image.js';
import {createKananaTokenizer} from '../../packages/web-decision/src/kanana/tokenizer-asset.mjs';
import {openKananaSource} from '../../packages/web-decision/src/kanana/source.mjs';
import {
  openKananaWeightSource as openPinnedKananaWeightSource,
  openSafetensorsWeightSource as openSafetensorsSource
} from '../../packages/web-decision/src/kanana/weight-source.mjs';
import {
  createKananaTvmInferenceSession,
  createKananaTvmVisionInferenceSession,
  type KananaTvmInferenceSessionOptions,
  type KananaTvmVisionInferenceSessionOptions
} from '../../packages/web-decision/src/kanana/tvm-session.js';
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
  dtype: string;
  length: number;
};

type WeightSource = {
  records: readonly {name: string; parameterName?: string; shape: readonly number[]; dtype: string; length: number}[];
  readTensor(name: string): Promise<{bytes: Uint8Array}>;
};

type WeightOpener = (options: {
  base: string;
  manifest: Record<string, unknown>;
  requiredWeights: readonly Record<string, unknown>[];
  cache: false;
}) => Promise<WeightSource>;

const openTextWeights = openPinnedKananaWeightSource as unknown as WeightOpener;
const openVisionWeights = openSafetensorsSource as unknown as WeightOpener;

declare global {
  interface Window {
    tvmjs?: TvmJs;
  }
}

async function readJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not read ${url}: HTTP ${response.status}.`);
  return (await response.json()) as T;
}

async function readWasm(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not read TVM WASM ${url}: HTTP ${response.status}.`);
  return response.arrayBuffer();
}

function float16ToFloat32(bytes: Uint8Array): Uint8Array {
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
  return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
}

async function syntheticImage(): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('A 2D canvas is required to create the local image fixture.');
  context.fillStyle = '#172b4d';
  context.fillRect(0, 0, 256, 256);
  context.fillStyle = '#e9b44c';
  context.fillRect(32, 32, 96, 128);
  context.fillStyle = '#4b9aaa';
  context.beginPath();
  context.arc(184, 144, 48, 0, Math.PI * 2);
  context.fill();
  const blob = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob(
      (value) => (value ? resolve(value) : reject(new Error('The PNG image fixture could not be encoded.'))),
      'image/png'
    )
  );
  canvas.width = 0;
  canvas.height = 0;
  return blob;
}

function finiteProbabilityOutput(value: Record<string, unknown>) {
  const answers = value.answers as Record<string, Record<string, unknown>> | undefined;
  const answer = answers?.action;
  const probabilities = answer?.probabilities as Record<string, number> | undefined;
  if (
    answer?.type !== 'choice' ||
    typeof answer.choice !== 'string' ||
    !probabilities ||
    Object.keys(probabilities).length !== 2 ||
    !Object.values(probabilities).every(Number.isFinite)
  )
    throw new Error('The full image decision returned an invalid choice result.');
  const probabilitySum = Object.values(probabilities).reduce((total, probability) => total + probability, 0);
  if (Math.abs(probabilitySum - 1) > 1e-6) throw new Error('The image decision probabilities do not sum to one.');
  return {answer, probabilities, probabilitySum};
}

/** Execute the pinned image and text graphs on one local image through the package decision session. */
export async function runKananaRealImageDecisionWebGpu(textAssetsBase: string) {
  const tvmjs = window.tvmjs;
  if (!tvmjs) throw new Error('Apache TVM JS runtime bundle is not loaded.');
  const textBase = new URL(textAssetsBase);
  if (!textBase.pathname.endsWith('/')) textBase.pathname += '/';
  const visionBase = new URL('vision/', textBase);
  const tokenizerAssets = await openKananaSource({base: textBase.href, cacheEnabled: false});
  const [configBytes, tokenizerConfigBytes, tokenizerBytes, textRanges, textManifest, visionRanges, visionManifest] =
    await Promise.all([
      tokenizerAssets.read('config.json'),
      tokenizerAssets.read('tokenizer_config.json'),
      tokenizerAssets.read('tokenizer.json'),
      readJson<Record<string, unknown>>(new URL('range-manifest.json', textBase).href),
      readJson<{weights: readonly Record<string, unknown>[]}>(new URL('external-weight-manifest.json', textBase).href),
      readJson<Record<string, unknown>>(new URL('range-manifest.json', visionBase).href),
      readJson<{weights: GraphWeight[]}>(new URL('external-weight-manifest.json', visionBase).href)
    ]);
  const tokenizer = createKananaTokenizer(
    JSON.parse(new TextDecoder().decode(tokenizerBytes)),
    JSON.parse(new TextDecoder().decode(tokenizerConfigBytes)),
    JSON.parse(new TextDecoder().decode(configBytes))
  );

  const textSource = await openTextWeights({
    base: textBase.href,
    manifest: textRanges,
    requiredWeights: textManifest.weights,
    cache: false
  });
  const visionRequiredWeights = visionManifest.weights.map((weight) => ({
    ...weight,
    dtype: 'float16',
    length: weight.shape.reduce((size, dimension) => size * dimension, 1) * 2
  }));
  const visionSource = await openVisionWeights({
    base: visionBase.href,
    manifest: visionRanges,
    requiredWeights: visionRequiredWeights,
    cache: false
  });
  const [textWasm, visionWasm] = await Promise.all([
    readWasm(new URL('kanana-text-model.wasm', textBase).href),
    readWasm(new URL('kanana-vision-model.wasm', visionBase).href)
  ]);
  const adapterSetup = await tvmjs.detectGPUDevice();
  if (!adapterSetup) throw new Error('This browser does not expose WebGPU.');
  const {adapter, device} = adapterSetup;
  const webgpuErrors: string[] = [];
  device.addEventListener('uncapturederror', (event) => {
    const error = (event as Event & {error?: {message?: string}}).error;
    webgpuErrors.push(error?.message ?? 'Unspecified WebGPU error.');
  });

  const textSetup = (await setupKananaTvmRuntime({
    instantiate: tvmjs.instantiate as never,
    createPolyfillWASI: tvmjs.createPolyfillWASI,
    device,
    bytes: textWasm
  })) as unknown as KananaTvmInferenceSessionOptions['setup'];
  const visionSetup = (await setupKananaTvmRuntime({
    instantiate: tvmjs.instantiate as never,
    createPolyfillWASI: tvmjs.createPolyfillWASI,
    device,
    bytes: visionWasm
  })) as unknown as KananaTvmVisionInferenceSessionOptions['setup'];

  let textInference: Awaited<ReturnType<typeof createKananaTvmInferenceSession>> | undefined;
  let visionInference: Awaited<ReturnType<typeof createKananaTvmVisionInferenceSession>> | undefined;
  let decision: ReturnType<typeof createKananaDecisionSession> | undefined;
  try {
    visionInference = await createKananaTvmVisionInferenceSession({
      setup: visionSetup,
      weightSource: {
        manifest: {
          weights: visionManifest.weights.map((weight) => ({
            ...weight,
            dtype: 'float32',
            length: weight.shape.reduce((size, dimension) => size * dimension, 1) * 4
          }))
        },
        async readWeight(record) {
          const tensor = await visionSource.readTensor(record.name);
          return float16ToFloat32(tensor.bytes);
        }
      },
      patchValues: 1_536,
      numPositionEmbeddings: 2_304,
      spatialMergeSize: 2,
      maxPatchCount: 65_536
    });
    textInference = await createKananaTvmInferenceSession({
      setup: textSetup,
      weightSource: {
        manifest: {weights: textSource.records},
        async readWeight(record) {
          return (await textSource.readTensor(record.name)).bytes;
        }
      }
    });

    const decodedBySource = new WeakMap<object, Awaited<ReturnType<typeof decodeKananaImageSource>>>();
    const preparedByPixels = new WeakMap<Uint8Array, ReturnType<typeof preprocessKananaImage>>();
    async function decodedImage(source: unknown, signal?: AbortSignal) {
      if (source === null || typeof source !== 'object') throw new TypeError('Kanana image must be an object.');
      const key = source as object;
      const cached = decodedBySource.get(key);
      if (cached) return cached;
      const decoded = await decodeKananaImageSource(source, signal ? {signal} : {});
      const prepared = preprocessKananaImage(decoded.rgb, decoded.width, decoded.height);
      decodedBySource.set(key, decoded);
      preparedByPixels.set(decoded.rgb, prepared);
      return decoded;
    }

    const imageEncoder = createKananaTvmImageEmbeddingEncoder({
      vision: visionInference,
      decode: decodedImage,
      preprocess(rgb, width, height) {
        const prepared = preparedByPixels.get(rgb);
        if (!prepared || prepared.width !== width || prepared.height !== height)
          throw new Error('Kanana image preprocessing cache does not match its decoded source.');
        return prepared;
      }
    });
    decision = createKananaDecisionSession({
      tokenizer,
      imageTokenThw: async (source) => {
        const decoded = await decodedImage(source);
        return preparedByPixels.get(decoded.rgb)!.imageTokenThw;
      },
      prepareImageEmbeddings: imageEncoder,
      infer: ({inputIds, attentionMask, imageEmbeddings, imagePositions, positionIds}) =>
        textInference!.infer(inputIds, attentionMask, {
          imageEmbeddings,
          imagePositions,
          positionIds
        }),
      dispose: () => textInference!.dispose()
    });

    const image = await syntheticImage();
    const result = await decision.systemOne({
      state: 'Inspect this synthetic illustration and decide which description fits best.',
      images: [image],
      questions: {
        action: {
          type: 'choice',
          instructions: 'Which object is most prominent?',
          criteria: {
            rectangle: 'A gold rectangle on the left is prominent.',
            circle: 'A teal circle on the right is prominent.'
          }
        }
      }
    });
    await device.queue.onSubmittedWorkDone();
    const output = finiteProbabilityOutput(result);
    if (webgpuErrors.length) throw new Error(`WebGPU reported errors: ${webgpuErrors.join('; ')}`);
    const usage = result.usage as {input_tokens?: number} | undefined;
    if (!usage || !Number.isSafeInteger(usage.input_tokens) || (usage.input_tokens ?? 0) < 1)
      throw new Error('The image decision did not report valid prompt usage.');

    return {
      model: 'kakaocorp/kanana-2-v-decide-3b-2611',
      gpu: {vendor: adapter.info?.vendor ?? '', architecture: adapter.info?.architecture ?? ''},
      textWeights: textManifest.weights.length,
      visionWeights: visionManifest.weights.length,
      imageTokenThw: preparedByPixels.get((await decodedImage(image)).rgb)?.imageTokenThw,
      ...output,
      usage,
      webgpuErrors
    };
  } finally {
    if (decision) await decision.dispose();
    else if (textInference) await textInference.dispose();
    else textSetup.dispose();
    if (visionInference) await visionInference.dispose();
    else visionSetup.dispose();
    device.destroy();
  }
}
