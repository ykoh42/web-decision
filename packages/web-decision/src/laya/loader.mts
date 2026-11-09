import {Tokenizer} from '@huggingface/tokenizers';
import {parseJson, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {createLayaRunner} from './runner.mjs';
import {createLayaSession} from './session.mjs';
import {openLayaSource} from './source.mjs';
import {LAYA_MODEL, LAYA_MODEL_CONFIG, LAYA_SPEC} from './contract.mjs';
import {createLayaTokenizer} from './tokenizer.mjs';
import type {ManifestSourceSpec} from '../asset-source-types.js';
import type {ModelProgress} from '../curated/types.js';
import type {LayaTemperatures} from './protocol.mjs';
import type {LayaSession} from './session.mjs';

interface LayaSignal {
  throwIfAborted(): void;
  addEventListener: AbortSignal['addEventListener'];
  removeEventListener: AbortSignal['removeEventListener'];
  readonly aborted?: boolean;
  readonly reason?: unknown;
}

interface LayaModelConfig {
  readonly modelType: string;
  readonly vocabSize: number;
  readonly specialTokenIds: {readonly sepId: number; readonly padId: number; readonly maskId: number};
}

interface LayaSpec extends ManifestSourceSpec {
  readonly repository: string;
  readonly revision: string;
  readonly modelConfig?: LayaModelConfig;
}

interface LayaCheckedLoadOptions {
  readonly base: URL;
  readonly signal: LayaSignal | null | undefined;
  readonly cacheEnabled: boolean;
  readonly offline: boolean;
  readonly onProgress: ((progress: ModelProgress) => void) | undefined;
}

interface LayaConfigValues {
  readonly model_type?: unknown;
  readonly vocab_size?: unknown;
  readonly mask_token_id?: unknown;
  readonly sep_token_id?: unknown;
  readonly pad_token_id?: unknown;
  readonly laya?: LayaModelConfigValues | null;
}

interface LayaModelConfigValues {
  readonly max_len?: unknown;
  readonly head_max_len?: unknown;
  readonly max_option_tokens?: unknown;
  readonly qtypes?: unknown;
  readonly temperature?: unknown[];
  readonly temperature_by_options?: unknown;
  readonly temperature_clamp?: unknown[];
}

interface LayaOnnxSession {
  readonly inputNames?: Iterable<string>;
  readonly outputNames?: Iterable<string>;
  run(feeds: Record<string, LayaTensor>, outputNames: string[]): Promise<unknown>;
  release(): void | Promise<unknown>;
}

interface LayaTensor {
  dispose?(): void;
}

interface LayaWebGpuEnvironment {
  powerPreference: string;
  forceFallbackAdapter: boolean;
  adapter: LayaGpuAdapter;
}

interface LayaGpuAdapter {
  readonly info?: {readonly isFallbackAdapter?: boolean};
  readonly isFallbackAdapter?: boolean;
}

interface LayaGpuApi {
  requestAdapter(options: {
    powerPreference: 'high-performance';
    forceFallbackAdapter: false;
  }): Promise<LayaGpuAdapter | null>;
}

interface LayaOrtRuntime {
  readonly InferenceSession?: {
    create(
      graph: Uint8Array,
      options: {
        executionProviders: ['webgpu'];
        graphOptimizationLevel: 'all';
        externalData: Array<{path: string; data: Uint8Array}>;
      }
    ): Promise<LayaOnnxSession>;
  };
  readonly Tensor?: unknown;
  readonly env?: {readonly webgpu?: LayaWebGpuEnvironment};
}

interface LayaOrtWithTensor extends LayaOrtRuntime {
  readonly Tensor: new (type: 'int64' | 'bool', data: BigInt64Array | Uint8Array, dims: number[]) => LayaTensor;
}

interface LayaModelDependencies {
  readonly ort?: unknown;
  readonly Tokenizer?: typeof Tokenizer;
  readonly openSource?: typeof openLayaSource;
  readonly gpuApi?: unknown;
  readonly model?: string;
  readonly modelName?: string;
  readonly spec?: LayaSpec;
}

interface LayaLoadOptionsInput {
  readonly model?: unknown;
  readonly assetsUrl?: unknown;
  readonly cache?: unknown;
  readonly offline?: unknown;
  readonly signal?: unknown;
  readonly onProgress?: unknown;
  readonly maxPrefetchBytes?: unknown;
}

const LOAD_KEYS = new Set(['model', 'assetsUrl', 'cache', 'offline', 'signal', 'onProgress', 'maxPrefetchBytes']);
const fail = (
  code: ConstructorParameters<typeof WebDecisionError>[0],
  message: string,
  cause?: unknown
): WebDecisionError => new WebDecisionError(code, message, cause ? {cause} : undefined);
const INPUT_NAMES = ['input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype'];
const OUTPUT_NAMES = ['logits', 'act_logits'];

function validSignal(signal: unknown): signal is LayaSignal | null | undefined {
  return (
    signal == null ||
    (typeof (signal as {throwIfAborted?: unknown}).throwIfAborted === 'function' &&
      typeof (signal as {addEventListener?: unknown}).addEventListener === 'function' &&
      typeof (signal as {removeEventListener?: unknown}).removeEventListener === 'function')
  );
}

export function validateLayaLoadOptions(
  options: unknown,
  model: string = LAYA_MODEL,
  spec: LayaSpec = LAYA_SPEC
): LayaCheckedLoadOptions {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'Laya load options are required.');
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported Laya load option.');
  }
  if (
    (options as LayaLoadOptionsInput).model !== model ||
    ((options as LayaLoadOptionsInput).assetsUrl !== undefined &&
      (typeof (options as LayaLoadOptionsInput).assetsUrl !== 'string' ||
        !(options as LayaLoadOptionsInput).assetsUrl)) ||
    ((options as LayaLoadOptionsInput).cache !== undefined &&
      typeof (options as LayaLoadOptionsInput).cache !== 'boolean') ||
    ((options as LayaLoadOptionsInput).offline !== undefined &&
      typeof (options as LayaLoadOptionsInput).offline !== 'boolean') ||
    ((options as LayaLoadOptionsInput).offline && (options as LayaLoadOptionsInput).cache === false) ||
    ((options as LayaLoadOptionsInput).maxPrefetchBytes !== undefined &&
      (options as LayaLoadOptionsInput).maxPrefetchBytes !== 0) ||
    ((options as LayaLoadOptionsInput).onProgress !== undefined &&
      typeof (options as LayaLoadOptionsInput).onProgress !== 'function') ||
    !validSignal((options as LayaLoadOptionsInput).signal)
  )
    throw fail('INVALID_INPUT', 'Invalid Laya load option values.');

  let base;
  try {
    const sourceUrl =
      (options as LayaLoadOptionsInput).assetsUrl ??
      `https://huggingface.co/${spec.repository}/resolve/${spec.revision}/`;
    base = new URL(sourceUrl as string, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', 'Invalid Laya assetsUrl.', cause);
  }
  if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'Laya assetsUrl must be an HTTP or HTTPS directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {
    base,
    signal: (options as LayaLoadOptionsInput).signal as LayaSignal | null | undefined,
    cacheEnabled: (options as LayaLoadOptionsInput).cache !== false,
    offline: (options as LayaLoadOptionsInput).offline === true,
    onProgress: (options as LayaLoadOptionsInput).onProgress as LayaCheckedLoadOptions['onProgress']
  };
}

function validateLayaModelConfig(config: unknown, modelConfig: LayaModelConfig = LAYA_MODEL_CONFIG): LayaTemperatures {
  const laya = (config as LayaConfigValues | null | undefined)?.laya;
  const specialTokenIds = modelConfig.specialTokenIds;
  const expectedQtypes = {choice: 0, score: 1, noul: 2};
  if (
    (config as LayaConfigValues | null | undefined)?.model_type !== modelConfig.modelType ||
    (config as LayaConfigValues | null | undefined)?.vocab_size !== modelConfig.vocabSize ||
    ((config as LayaConfigValues | null | undefined)?.mask_token_id !== undefined &&
      (config as LayaConfigValues).mask_token_id !== specialTokenIds.maskId) ||
    (config as LayaConfigValues | null | undefined)?.sep_token_id !== specialTokenIds.sepId ||
    (config as LayaConfigValues | null | undefined)?.pad_token_id !== specialTokenIds.padId ||
    laya?.max_len !== 1024 ||
    laya?.head_max_len !== 256 ||
    laya?.max_option_tokens !== 48 ||
    JSON.stringify(laya?.qtypes) !== JSON.stringify(expectedQtypes) ||
    !Array.isArray(laya?.temperature) ||
    laya.temperature.length !== 3 ||
    laya.temperature.some((value: unknown) => !Number.isFinite(value) || (value as number) <= 0) ||
    !laya.temperature_by_options ||
    typeof laya.temperature_by_options !== 'object' ||
    Array.isArray(laya.temperature_by_options) ||
    Object.entries(laya.temperature_by_options as Record<string, unknown>).some(
      ([key, value]) =>
        !/^(choice|score|noul):(2|3-5|6-10|11\+)$/.test(key) || !Number.isFinite(value) || (value as number) <= 0
    ) ||
    !Array.isArray(laya.temperature_clamp) ||
    laya.temperature_clamp.length !== 2 ||
    laya.temperature_clamp[0] !== 0.5 ||
    laya.temperature_clamp[1] !== 5
  )
    throw fail('INVALID_ASSET', 'Laya config does not match the pinned checkpoint.');
  return {
    byType: [...laya.temperature] as number[],
    byOptions: {...(laya.temperature_by_options as Record<string, number>)}
  };
}

function validateSession(session: unknown): LayaOnnxSession {
  const inputs = new Set((session as {inputNames?: Iterable<string>} | null | undefined)?.inputNames ?? []);
  const outputs = new Set((session as {outputNames?: Iterable<string>} | null | undefined)?.outputNames ?? []);
  if (!INPUT_NAMES.every((name) => inputs.has(name)) || !OUTPUT_NAMES.every((name) => outputs.has(name)))
    throw fail('INVALID_ASSET', 'Laya ONNX graph has an unexpected interface.');
  return session as LayaOnnxSession;
}

async function createSession(
  ort: LayaOrtRuntime,
  adapter: LayaGpuAdapter,
  graphBytes: ArrayBuffer,
  dataBytes: ArrayBuffer,
  signal: LayaSignal | null | undefined
): Promise<LayaOnnxSession> {
  if (typeof ort?.InferenceSession?.create !== 'function' || typeof ort?.Tensor !== 'function')
    throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WebGPU is unavailable.');
  const webgpu = ort.env?.webgpu;
  if (!webgpu) throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WebGPU execution provider is unavailable.');
  webgpu.powerPreference = 'high-performance';
  webgpu.forceFallbackAdapter = false;
  webgpu.adapter = adapter;
  return waitForAbort(
    () =>
      ort.InferenceSession!.create(new Uint8Array(graphBytes), {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
        externalData: [{path: 'model_fp16.onnx_data', data: new Uint8Array(dataBytes)}]
      }),
    signal as AbortSignal | null | undefined,
    (lateSession) => lateSession?.release()
  );
}

async function getHardwareAdapter(gpu: unknown, signal: LayaSignal | null | undefined): Promise<LayaGpuAdapter> {
  if (typeof (gpu as LayaGpuApi | null | undefined)?.requestAdapter !== 'function')
    throw fail('UNSUPPORTED_ENVIRONMENT', 'Laya requires WebGPU.');
  let adapter: LayaGpuAdapter | null;
  try {
    adapter = await waitForAbort(
      () => (gpu as LayaGpuApi).requestAdapter({powerPreference: 'high-performance', forceFallbackAdapter: false}),
      signal as AbortSignal | null | undefined
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw fail('MODEL_LOAD_FAILED', 'Could not initialize a hardware WebGPU adapter for Laya.', cause);
  }
  if (!adapter) throw fail('UNSUPPORTED_ENVIRONMENT', 'No hardware WebGPU adapter is available for Laya.');
  if (adapter.info?.isFallbackAdapter === true || adapter.isFallbackAdapter === true)
    throw fail('UNSUPPORTED_ENVIRONMENT', 'Laya does not run on a fallback WebGPU adapter.');
  return adapter;
}

export async function loadLayaModel(
  options: unknown,
  {
    ort,
    Tokenizer: TokenizerClass = Tokenizer,
    openSource = openLayaSource,
    gpuApi = (globalThis.navigator as (Navigator & {readonly gpu?: LayaGpuApi}) | undefined)?.gpu,
    model = LAYA_MODEL,
    modelName = 'Laya',
    spec = LAYA_SPEC
  }: LayaModelDependencies = {}
): Promise<LayaSession> {
  const checked = validateLayaLoadOptions(options, model, spec);
  const {signal} = checked;
  let session: LayaOnnxSession | null | undefined;
  try {
    const runtime = ort ?? (await import('onnxruntime-web/webgpu'));
    signal?.throwIfAborted();
    const source = await openSource({...checked, spec} as Parameters<typeof openLayaSource>[0]);
    const [configBytes, tokenizerConfigBytes, tokenizerBytes] = await Promise.all([
      source.read('config.json'),
      source.read('tokenizer_config.json'),
      source.read('tokenizer.json')
    ]);
    const config = parseJson(configBytes, 'config.json');
    const modelConfig = spec.modelConfig ?? LAYA_MODEL_CONFIG;
    const temperatures = validateLayaModelConfig(config, modelConfig);
    const tokenizer = createLayaTokenizer(
      TokenizerClass,
      parseJson(tokenizerBytes, 'tokenizer.json'),
      parseJson(tokenizerConfigBytes, 'tokenizer_config.json'),
      config as Parameters<typeof createLayaTokenizer>[3],
      modelConfig.specialTokenIds
    );
    signal?.throwIfAborted();

    const [graphBytes, dataBytes] = await Promise.all([
      source.read('onnx/model_fp16.onnx'),
      source.read('onnx/model_fp16.onnx_data')
    ]);
    signal?.throwIfAborted();
    const adapter = await getHardwareAdapter(gpuApi, signal);
    session = validateSession(
      await createSession((ort ?? runtime) as LayaOrtRuntime, adapter, graphBytes, dataBytes, signal)
    );
    signal?.throwIfAborted();
    const runner = createLayaRunner((ort ?? runtime) as LayaOrtWithTensor, session, tokenizer);
    const loaded = createLayaSession({runner, tokenizer, temperatures, model});
    session = null;
    return loaded;
  } catch (cause) {
    await Promise.allSettled([session?.release()].filter(Boolean));
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || (cause as {name?: unknown} | null | undefined)?.name === 'AbortError')
      throw cause;
    throw fail('MODEL_LOAD_FAILED', `Could not load ${modelName} on ONNX Runtime WebGPU.`, cause);
  }
}
