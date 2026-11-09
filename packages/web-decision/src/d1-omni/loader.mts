import {Tokenizer} from '@huggingface/tokenizers';
import {parseJson, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {createD1OmniSession} from './session.mjs';
import {D1_OMNI_ASSETS_URL, D1_OMNI_MODEL, D1_OMNI_SPEC, D1_OMNI_TEMPERATURES} from './contract.mjs';
import {openD1OmniSource} from './source.mjs';
import {createD1OmniTokenizer} from './tokenizer.mjs';
import {createD1OmniRunner, D1_OMNI_INPUT_NAMES, D1_OMNI_OUTPUT_NAMES} from './runner.mjs';
import type {ModelProgress} from '../curated/types.js';
import type {D1OmniSession} from './session.mjs';

interface D1OmniLoadOptionsInput {
  readonly model?: unknown;
  readonly assetsUrl?: unknown;
  readonly cache?: unknown;
  readonly offline?: unknown;
  readonly signal?: unknown;
  readonly onProgress?: unknown;
  readonly maxPrefetchBytes?: unknown;
}

interface D1OmniCheckedLoadOptions {
  readonly base: URL;
  readonly signal: AbortSignal | null | undefined;
  readonly cacheEnabled: boolean;
  readonly offline: boolean;
  readonly onProgress: ((progress: ModelProgress) => void) | undefined;
}

interface D1OmniJsonConfig {
  readonly model_type?: unknown;
  readonly max_length?: unknown;
  readonly bos_token_id?: unknown;
  readonly pad_token_id?: unknown;
  readonly temperatures?: unknown;
  readonly d1_omni?: {readonly max_length?: unknown} | null;
  readonly 'transformers.js_config'?: {readonly dtype?: unknown} | null;
}

interface D1OmniOnnxSession {
  readonly inputNames?: Iterable<string>;
  readonly outputNames?: Iterable<string>;
  run(feeds: Record<string, unknown>, outputNames: string[]): Promise<unknown>;
  release(): void | Promise<unknown>;
}

interface D1OmniGpuAdapter {
  readonly info?: {readonly isFallbackAdapter?: boolean};
  readonly isFallbackAdapter?: boolean;
}

interface D1OmniGpuApi {
  requestAdapter(options: {
    powerPreference: 'high-performance';
    forceFallbackAdapter: false;
  }): Promise<D1OmniGpuAdapter | null>;
}

interface D1OmniWebGpuEnvironment {
  powerPreference: string;
  forceFallbackAdapter: boolean;
  adapter: D1OmniGpuAdapter;
}

interface D1OmniOrtRuntime {
  readonly InferenceSession?: {
    create(
      graph: Uint8Array,
      options: {
        executionProviders: ['webgpu'];
        graphOptimizationLevel: 'all';
        externalData: Array<{path: string; data: Uint8Array}>;
      }
    ): Promise<D1OmniOnnxSession>;
  };
  readonly Tensor?: unknown;
  readonly env?: {readonly webgpu?: D1OmniWebGpuEnvironment};
}

interface D1OmniSource {
  read(name: string): Promise<ArrayBuffer>;
}

interface D1OmniLoaderDependencies {
  readonly ort?: unknown;
  readonly Tokenizer?: typeof Tokenizer | undefined;
  readonly openSource?: typeof openD1OmniSource | undefined;
  readonly gpuApi?: unknown;
}

const LOAD_KEYS = new Set(['model', 'assetsUrl', 'cache', 'offline', 'signal', 'onProgress', 'maxPrefetchBytes']);
const fail = (
  code: ConstructorParameters<typeof WebDecisionError>[0],
  message: string,
  cause?: unknown
): WebDecisionError => new WebDecisionError(code, message, cause ? {cause} : undefined);

function validSignal(signal: unknown): signal is AbortSignal | null | undefined {
  return (
    signal == null ||
    (typeof (signal as {throwIfAborted?: unknown}).throwIfAborted === 'function' &&
      typeof (signal as {addEventListener?: unknown}).addEventListener === 'function' &&
      typeof (signal as {removeEventListener?: unknown}).removeEventListener === 'function')
  );
}

export function validateD1OmniLoadOptions(options: unknown): D1OmniCheckedLoadOptions {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'd1-omni load options are required.');
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported d1-omni load option.');
  }
  const assetsUrl =
    (options as D1OmniLoadOptionsInput).assetsUrl === undefined
      ? D1_OMNI_ASSETS_URL
      : (options as D1OmniLoadOptionsInput).assetsUrl;
  if (
    (options as D1OmniLoadOptionsInput).model !== D1_OMNI_MODEL ||
    typeof assetsUrl !== 'string' ||
    !assetsUrl ||
    ((options as D1OmniLoadOptionsInput).cache !== undefined &&
      typeof (options as D1OmniLoadOptionsInput).cache !== 'boolean') ||
    ((options as D1OmniLoadOptionsInput).offline !== undefined &&
      typeof (options as D1OmniLoadOptionsInput).offline !== 'boolean') ||
    ((options as D1OmniLoadOptionsInput).offline && (options as D1OmniLoadOptionsInput).cache === false) ||
    ((options as D1OmniLoadOptionsInput).maxPrefetchBytes !== undefined &&
      (options as D1OmniLoadOptionsInput).maxPrefetchBytes !== 0) ||
    ((options as D1OmniLoadOptionsInput).onProgress !== undefined &&
      typeof (options as D1OmniLoadOptionsInput).onProgress !== 'function') ||
    !validSignal((options as D1OmniLoadOptionsInput).signal)
  )
    throw fail('INVALID_INPUT', 'Invalid d1-omni load option values.');

  let base: URL;
  try {
    base = new URL(assetsUrl, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', 'Invalid d1-omni assetsUrl.', cause);
  }
  if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'd1-omni assetsUrl must be an HTTP or HTTPS directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {
    base,
    signal: (options as D1OmniLoadOptionsInput).signal as AbortSignal | null | undefined,
    cacheEnabled: (options as D1OmniLoadOptionsInput).cache !== false,
    offline: (options as D1OmniLoadOptionsInput).offline === true,
    onProgress: (options as D1OmniLoadOptionsInput).onProgress as D1OmniCheckedLoadOptions['onProgress']
  };
}

function validateConfig(config: unknown): Readonly<Record<string, number>> {
  const temperatures = (config as D1OmniJsonConfig | null | undefined)?.temperatures;
  const expected = D1_OMNI_TEMPERATURES;
  if (
    (config as D1OmniJsonConfig | null | undefined)?.model_type !== 'd1_omni' ||
    (config as D1OmniJsonConfig | null | undefined)?.max_length !== D1_OMNI_SPEC.maxTokens ||
    (config as D1OmniJsonConfig | null | undefined)?.bos_token_id !== 1 ||
    (config as D1OmniJsonConfig | null | undefined)?.pad_token_id !== 0 ||
    (config as D1OmniJsonConfig | null | undefined)?.['transformers.js_config']?.dtype !== 'q8' ||
    (config as D1OmniJsonConfig | null | undefined)?.d1_omni?.max_length !== D1_OMNI_SPEC.maxTokens ||
    !temperatures ||
    Object.keys(expected).some((key) => (temperatures as Record<string, unknown>)[key] !== expected[key])
  )
    throw fail('INVALID_ASSET', 'd1-omni config does not match the pinned Q8 checkpoint.');
  return expected;
}

function validateSession(session: D1OmniOnnxSession | null | undefined): D1OmniOnnxSession {
  const inputs = new Set(session?.inputNames ?? []);
  const outputs = new Set(session?.outputNames ?? []);
  if (
    !D1_OMNI_INPUT_NAMES.every((name) => inputs.has(name)) ||
    !D1_OMNI_OUTPUT_NAMES.every((name) => outputs.has(name))
  )
    throw fail('INVALID_ASSET', 'd1-omni ONNX graph has an unexpected interface.');
  return session as D1OmniOnnxSession;
}

async function getHardwareAdapter(gpu: unknown, signal: AbortSignal | null | undefined): Promise<D1OmniGpuAdapter> {
  if (typeof (gpu as D1OmniGpuApi | null | undefined)?.requestAdapter !== 'function')
    throw fail('UNSUPPORTED_ENVIRONMENT', 'd1-omni requires WebGPU.');
  let adapter: D1OmniGpuAdapter | null | undefined;
  try {
    adapter = await waitForAbort(
      () => (gpu as D1OmniGpuApi).requestAdapter({powerPreference: 'high-performance', forceFallbackAdapter: false}),
      signal
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw fail('MODEL_LOAD_FAILED', 'Could not initialize a hardware WebGPU adapter for d1-omni.', cause);
  }
  if (!adapter) throw fail('UNSUPPORTED_ENVIRONMENT', 'No hardware WebGPU adapter is available for d1-omni.');
  if (
    (adapter as D1OmniGpuAdapter).info?.isFallbackAdapter === true ||
    (adapter as D1OmniGpuAdapter).isFallbackAdapter === true
  )
    throw fail('UNSUPPORTED_ENVIRONMENT', 'd1-omni does not run on a fallback WebGPU adapter.');
  return adapter as D1OmniGpuAdapter;
}

async function createSession(
  ort: unknown,
  adapter: D1OmniGpuAdapter,
  graphBytes: ArrayBuffer,
  dataBytes: ArrayBuffer,
  signal: AbortSignal | null | undefined
): Promise<D1OmniOnnxSession> {
  if (
    typeof (ort as D1OmniOrtRuntime | null | undefined)?.InferenceSession?.create !== 'function' ||
    typeof (ort as D1OmniOrtRuntime | null | undefined)?.Tensor !== 'function'
  )
    throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WebGPU is unavailable.');
  if (!(ort as D1OmniOrtRuntime).env?.webgpu)
    throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WebGPU provider is unavailable.');
  (ort as D1OmniOrtRuntime).env!.webgpu!.powerPreference = 'high-performance';
  (ort as D1OmniOrtRuntime).env!.webgpu!.forceFallbackAdapter = false;
  (ort as D1OmniOrtRuntime).env!.webgpu!.adapter = adapter;
  return waitForAbort(
    () =>
      (ort as D1OmniOrtRuntime & Required<Pick<D1OmniOrtRuntime, 'InferenceSession'>>).InferenceSession.create(
        new Uint8Array(graphBytes),
        {
          executionProviders: ['webgpu'],
          graphOptimizationLevel: 'all',
          externalData: [{path: 'model_quantized.onnx_data', data: new Uint8Array(dataBytes)}]
        }
      ),
    signal,
    (lateSession) => lateSession?.release()
  );
}

export async function loadD1OmniModel(
  options: unknown,
  {
    ort,
    Tokenizer: TokenizerClass = Tokenizer,
    openSource = openD1OmniSource,
    gpuApi = (globalThis.navigator as (Navigator & {readonly gpu?: unknown}) | undefined)?.gpu
  }: D1OmniLoaderDependencies = {}
): Promise<D1OmniSession> {
  const checked = validateD1OmniLoadOptions(options);
  const {signal} = checked;
  let session: D1OmniOnnxSession | null | undefined;
  try {
    const runtime: unknown = ort ?? (await import('onnxruntime-web/webgpu'));
    signal?.throwIfAborted();
    const source: D1OmniSource = await openSource({...checked, spec: D1_OMNI_SPEC});
    const [configBytes, tokenizerConfigBytes, tokenizerBytes] = await Promise.all([
      source.read('config.json'),
      source.read('tokenizer_config.json'),
      source.read('tokenizer.json')
    ]);
    const config = parseJson(configBytes, 'config.json');
    const temperatures = validateConfig(config);
    const tokenizer = createD1OmniTokenizer(
      TokenizerClass,
      parseJson(tokenizerBytes, 'tokenizer.json'),
      parseJson(tokenizerConfigBytes, 'tokenizer_config.json')
    );
    signal?.throwIfAborted();

    const [graphBytes, dataBytes] = await Promise.all([
      source.read('onnx/model_quantized.onnx'),
      source.read('onnx/model_quantized.onnx_data')
    ]);
    signal?.throwIfAborted();
    const adapter = await getHardwareAdapter(gpuApi, signal);
    session = validateSession(await createSession(ort ?? runtime, adapter, graphBytes, dataBytes, signal));
    signal?.throwIfAborted();
    const runner = createD1OmniRunner(ort ?? runtime, session, tokenizer.ids['<|pad|>'] as number);
    const model = createD1OmniSession({runner, tokenizer, temperatures, model: D1_OMNI_MODEL});
    session = null;
    return model;
  } catch (cause) {
    await Promise.allSettled([session?.release()].filter(Boolean));
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || (cause as {name?: unknown} | null | undefined)?.name === 'AbortError')
      throw cause;
    throw fail('MODEL_LOAD_FAILED', 'Could not load d1-omni Q8 on ONNX Runtime WebGPU.', cause);
  }
}
