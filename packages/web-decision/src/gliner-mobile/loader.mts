import {Tokenizer} from '@huggingface/tokenizers';
import {parseJson, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {createGlinerMobileRunner} from './runner.mjs';
import {createGlinerMobileSession} from './session.mjs';
import {GLINER_MOBILE_MODEL, GLINER_MOBILE_SPEC} from './contract.mjs';
import {openGlinerMobileSource} from './source.mjs';
import {createGlinerMobileTokenizer} from './tokenizer.mjs';
import type {ModelProgress} from '../curated/types.js';
import type {GlinerMobileSession} from './session.mjs';

interface GlinerLoadOptionsInput {
  readonly model?: unknown;
  readonly assetsUrl?: unknown;
  readonly cache?: unknown;
  readonly offline?: unknown;
  readonly signal?: unknown;
  readonly onProgress?: unknown;
  readonly maxPrefetchBytes?: unknown;
}

interface GlinerCheckedLoadOptions {
  readonly base: URL;
  readonly signal: AbortSignal | null | undefined;
  readonly cacheEnabled: boolean;
  readonly offline: boolean;
  readonly onProgress: ((progress: ModelProgress) => void) | undefined;
}

interface GlinerModelConfig {
  readonly model_type?: unknown;
  readonly vocab_size?: unknown;
  readonly max_position_embeddings?: unknown;
  readonly gliner2?: {
    readonly max_len?: unknown;
    readonly temperature?: unknown;
    readonly source?: unknown;
    readonly inputs?: unknown;
    readonly output?: unknown;
  } | null;
  readonly 'transformers.js_config'?: {
    readonly dtype?: unknown;
    readonly use_external_data_format?: unknown;
  } | null;
}

interface GlinerOnnxSession {
  readonly inputNames?: Iterable<string>;
  readonly outputNames?: Iterable<string>;
  run(feeds: Record<string, unknown>, outputNames: string[]): Promise<unknown>;
  release(): void | Promise<unknown>;
}

interface GlinerWebGpuEnvironment {
  powerPreference: string;
  forceFallbackAdapter: boolean;
  adapter: GlinerGpuAdapter;
}

interface GlinerGpuAdapter {
  readonly info?: {readonly isFallbackAdapter?: boolean};
  readonly isFallbackAdapter?: boolean;
}

interface GlinerGpuApi {
  requestAdapter(options: {
    powerPreference: 'high-performance';
    forceFallbackAdapter: false;
  }): Promise<GlinerGpuAdapter | null>;
}

interface GlinerOrtRuntime {
  readonly InferenceSession?: {
    create(
      graph: Uint8Array,
      options: {
        executionProviders: ['webgpu'];
        graphOptimizationLevel: 'all';
        externalData: Array<{path: string; data: Uint8Array}>;
      }
    ): Promise<GlinerOnnxSession>;
  };
  readonly Tensor?: unknown;
  readonly env?: {readonly webgpu?: GlinerWebGpuEnvironment};
}

interface GlinerSourceReader {
  read(name: string): Promise<ArrayBuffer>;
}

interface GlinerSourceOptions extends Omit<GlinerCheckedLoadOptions, 'signal'> {
  readonly signal: AbortSignal | null | undefined;
}

type GlinerSourceOpener = (options: GlinerSourceOptions) => Promise<GlinerSourceReader>;

interface GlinerLoaderDependencies {
  readonly ort?: unknown;
  readonly Tokenizer?: typeof Tokenizer;
  readonly openSource?: GlinerSourceOpener;
  readonly gpuApi?: unknown;
}

const LOAD_KEYS = new Set(['model', 'assetsUrl', 'cache', 'offline', 'signal', 'onProgress', 'maxPrefetchBytes']);
const fail = (
  code: ConstructorParameters<typeof WebDecisionError>[0],
  message: string,
  cause?: unknown
): WebDecisionError => new WebDecisionError(code, message, cause ? {cause} : undefined);
const INPUT_NAMES = ['input_ids', 'attention_mask', 'marker_positions'];
const OUTPUT_NAMES = ['logits'];

function validSignal(signal: unknown): signal is AbortSignal | null | undefined {
  return (
    signal == null ||
    (typeof (signal as {throwIfAborted?: unknown}).throwIfAborted === 'function' &&
      typeof (signal as {addEventListener?: unknown}).addEventListener === 'function' &&
      typeof (signal as {removeEventListener?: unknown}).removeEventListener === 'function')
  );
}

export function validateGlinerMobileLoadOptions(options: unknown): GlinerCheckedLoadOptions {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'GLiNER load options are required.');
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported GLiNER load option.');
  }
  if (
    (options as GlinerLoadOptionsInput).model !== GLINER_MOBILE_MODEL ||
    ((options as GlinerLoadOptionsInput).assetsUrl !== undefined &&
      (typeof (options as GlinerLoadOptionsInput).assetsUrl !== 'string' ||
        !(options as GlinerLoadOptionsInput).assetsUrl)) ||
    ((options as GlinerLoadOptionsInput).cache !== undefined &&
      typeof (options as GlinerLoadOptionsInput).cache !== 'boolean') ||
    ((options as GlinerLoadOptionsInput).offline !== undefined &&
      typeof (options as GlinerLoadOptionsInput).offline !== 'boolean') ||
    ((options as GlinerLoadOptionsInput).offline && (options as GlinerLoadOptionsInput).cache === false) ||
    ((options as GlinerLoadOptionsInput).maxPrefetchBytes !== undefined &&
      (options as GlinerLoadOptionsInput).maxPrefetchBytes !== 0) ||
    ((options as GlinerLoadOptionsInput).onProgress !== undefined &&
      typeof (options as GlinerLoadOptionsInput).onProgress !== 'function') ||
    !validSignal((options as GlinerLoadOptionsInput).signal)
  )
    throw fail('INVALID_INPUT', 'Invalid GLiNER load option values.');

  let base;
  try {
    const sourceUrl =
      (options as GlinerLoadOptionsInput).assetsUrl ??
      `https://huggingface.co/${GLINER_MOBILE_SPEC.repository}/resolve/${GLINER_MOBILE_SPEC.revision}/`;
    base = new URL(sourceUrl as string, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', 'Invalid GLiNER assetsUrl.', cause);
  }
  if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'GLiNER assetsUrl must be an HTTP or HTTPS directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {
    base,
    signal: (options as GlinerLoadOptionsInput).signal as AbortSignal | null | undefined,
    cacheEnabled: (options as GlinerLoadOptionsInput).cache !== false,
    offline: (options as GlinerLoadOptionsInput).offline === true,
    onProgress: (options as GlinerLoadOptionsInput).onProgress as GlinerCheckedLoadOptions['onProgress']
  };
}

function validateModelConfig(config: unknown): void {
  const gliner = (config as GlinerModelConfig | null | undefined)?.gliner2;
  if (
    (config as GlinerModelConfig | null | undefined)?.model_type !== 'deberta-v2' ||
    (config as GlinerModelConfig | null | undefined)?.vocab_size !== 128011 ||
    (config as GlinerModelConfig | null | undefined)?.max_position_embeddings !== 512 ||
    gliner?.max_len !== 512 ||
    gliner?.temperature !== 1 ||
    gliner?.source !== 'fastino/GLiNER2.5-Decide' ||
    gliner?.inputs !== 'input_ids, attention_mask, marker_positions (index of each [L] token)' ||
    gliner?.output !== "logits (batch, markers); softmax within each question's markers" ||
    (config as GlinerModelConfig | null | undefined)?.['transformers.js_config']?.dtype !== 'q4f16' ||
    JSON.stringify((config as GlinerModelConfig)['transformers.js_config']!.use_external_data_format) !==
      JSON.stringify({'model_q4f16.onnx': 1})
  )
    throw fail('INVALID_ASSET', 'GLiNER config does not match the pinned mobile checkpoint.');
}

function validateSession(session: GlinerOnnxSession | null | undefined): GlinerOnnxSession {
  const inputs = new Set(session?.inputNames ?? []);
  const outputs = new Set(session?.outputNames ?? []);
  if (!INPUT_NAMES.every((name) => inputs.has(name)) || !OUTPUT_NAMES.every((name) => outputs.has(name)))
    throw fail('INVALID_ASSET', 'GLiNER ONNX graph has an unexpected interface.');
  return session as GlinerOnnxSession;
}

async function createSession(
  ort: unknown,
  adapter: GlinerGpuAdapter,
  graphBytes: ArrayBuffer,
  dataBytes: ArrayBuffer,
  signal: AbortSignal | null | undefined
): Promise<GlinerOnnxSession> {
  if (
    typeof (ort as GlinerOrtRuntime | null | undefined)?.InferenceSession?.create !== 'function' ||
    typeof (ort as GlinerOrtRuntime | null | undefined)?.Tensor !== 'function'
  )
    throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WebGPU is unavailable.');
  const webgpu = (ort as GlinerOrtRuntime).env?.webgpu;
  if (!webgpu) throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WebGPU provider is unavailable.');
  webgpu.powerPreference = 'high-performance';
  webgpu.forceFallbackAdapter = false;
  webgpu.adapter = adapter;
  return waitForAbort(
    () =>
      (ort as GlinerOrtRuntime & Required<Pick<GlinerOrtRuntime, 'InferenceSession'>>).InferenceSession.create(
        new Uint8Array(graphBytes),
        {
          executionProviders: ['webgpu'],
          graphOptimizationLevel: 'all',
          externalData: [{path: 'model_q4f16.onnx_data', data: new Uint8Array(dataBytes)}]
        }
      ),
    signal,
    (lateSession) => lateSession?.release()
  );
}

async function getHardwareAdapter(gpu: unknown, signal: AbortSignal | null | undefined): Promise<GlinerGpuAdapter> {
  if (typeof (gpu as GlinerGpuApi | null | undefined)?.requestAdapter !== 'function')
    throw fail('UNSUPPORTED_ENVIRONMENT', 'GLiNER requires WebGPU.');
  let adapter: GlinerGpuAdapter | null | undefined;
  try {
    adapter = await waitForAbort(
      () => (gpu as GlinerGpuApi).requestAdapter({powerPreference: 'high-performance', forceFallbackAdapter: false}),
      signal
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw fail('MODEL_LOAD_FAILED', 'Could not initialize a hardware WebGPU adapter for GLiNER.', cause);
  }
  if (!adapter) throw fail('UNSUPPORTED_ENVIRONMENT', 'No hardware WebGPU adapter is available for GLiNER.');
  if (
    (adapter as GlinerGpuAdapter).info?.isFallbackAdapter === true ||
    (adapter as GlinerGpuAdapter).isFallbackAdapter === true
  )
    throw fail('UNSUPPORTED_ENVIRONMENT', 'GLiNER does not run on a fallback WebGPU adapter.');
  return adapter as GlinerGpuAdapter;
}

export async function loadGlinerMobileModel(
  options: unknown,
  {
    ort,
    Tokenizer: TokenizerClass = Tokenizer,
    openSource = openGlinerMobileSource as GlinerSourceOpener,
    gpuApi = globalThis.navigator?.gpu
  }: GlinerLoaderDependencies = {}
): Promise<GlinerMobileSession> {
  const checked = validateGlinerMobileLoadOptions(options);
  const {signal} = checked;
  let session: GlinerOnnxSession | null | undefined;
  try {
    const runtime: unknown = ort ?? (await import('onnxruntime-web/webgpu'));
    signal?.throwIfAborted();
    const source = await openSource(checked);
    const [configBytes, specialTokenMapBytes, tokenizerConfigBytes, tokenizerBytes] = await Promise.all([
      source.read('config.json'),
      source.read('special_tokens_map.json'),
      source.read('tokenizer_config.json'),
      source.read('tokenizer.json')
    ]);
    const config = parseJson(configBytes, 'config.json');
    validateModelConfig(config);
    const tokenizer = createGlinerMobileTokenizer(
      TokenizerClass,
      parseJson(tokenizerBytes, 'tokenizer.json'),
      parseJson(tokenizerConfigBytes, 'tokenizer_config.json'),
      parseJson(specialTokenMapBytes, 'special_tokens_map.json'),
      config
    );
    signal?.throwIfAborted();

    const [graphBytes, dataBytes] = await Promise.all([
      source.read('onnx/model_q4f16.onnx'),
      source.read('onnx/model_q4f16.onnx_data')
    ]);
    signal?.throwIfAborted();
    const adapter = await getHardwareAdapter(gpuApi, signal);
    session = validateSession(await createSession(ort ?? runtime, adapter, graphBytes, dataBytes, signal));
    signal?.throwIfAborted();
    const runner = createGlinerMobileRunner(ort ?? runtime, session);
    const model = createGlinerMobileSession({runner, tokenizer, model: GLINER_MOBILE_MODEL});
    session = null;
    return model;
  } catch (cause) {
    await Promise.allSettled([session?.release()].filter(Boolean));
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || (cause as {name?: unknown} | null | undefined)?.name === 'AbortError')
      throw cause;
    throw fail('MODEL_LOAD_FAILED', 'Could not load GLiNER on ONNX Runtime WebGPU.', cause);
  }
}
