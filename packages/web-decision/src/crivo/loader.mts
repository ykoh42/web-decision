import {Tokenizer} from '@huggingface/tokenizers';
import {parseJson, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {CRIVO_REPOSITORY, CRIVO_REVISION, getCrivoSpec} from './contract.mjs';
import {createCrivoRunner} from './runner.mjs';
import {createCrivoSession} from './session.mjs';
import {openCrivoSource} from './source.mjs';
import {createCrivoTokenizer, validateCrivoConfig} from './tokenizer.mjs';
import type {ModelProgress} from '../curated/types.js';
import type {CrivoAbortSignalLike, CrivoTokenizer} from './protocol.mjs';
import type {CrivoSpec} from './contract.mjs';

interface CrivoLoadOptionsInput {
  readonly model?: unknown;
  readonly assetsUrl?: unknown;
  readonly cache?: unknown;
  readonly offline?: unknown;
  readonly signal?: unknown;
  readonly onProgress?: unknown;
}

interface CrivoCheckedLoadOptions {
  readonly base: URL;
  readonly spec: CrivoSpec;
  readonly signal: AbortSignal | null | undefined;
  readonly cacheEnabled: boolean;
  readonly offline: boolean;
  readonly onProgress: ((progress: ModelProgress) => void) | undefined;
}

interface CrivoGpuAdapter {
  readonly info?: {readonly isFallbackAdapter?: boolean};
  readonly isFallbackAdapter?: boolean;
}

interface CrivoGpuApi {
  requestAdapter(options: {
    powerPreference: 'high-performance';
    forceFallbackAdapter: false;
  }): Promise<CrivoGpuAdapter | null>;
}

interface CrivoOnnxSession {
  readonly inputNames?: string[];
  readonly outputNames?: string[];
  release(): unknown;
}

interface CrivoInferenceSessionOptions {
  executionProviders: string[];
  graphOptimizationLevel: 'all';
  intraOpNumThreads?: number;
  interOpNumThreads?: number;
}

interface CrivoOrtRuntime {
  readonly InferenceSession?: {
    create(data: Uint8Array, options: CrivoInferenceSessionOptions): Promise<CrivoOnnxSession>;
  };
  readonly Tensor?: unknown;
  readonly env?: {
    readonly webgpu?: {
      powerPreference?: string;
      forceFallbackAdapter?: boolean;
      adapter?: CrivoGpuAdapter;
    };
    readonly wasm?: {numThreads?: number};
  };
}

interface CrivoSource {
  read(name: string): Promise<ArrayBuffer>;
}

interface CrivoRunner {
  run(rows: readonly unknown[]): unknown;
  dispose(): unknown;
}

interface CrivoLoadedSessionInput {
  readonly runner: CrivoRunner;
  readonly tokenizer: CrivoTokenizer;
  readonly model: string;
}

interface CrivoLoaderRuntime {
  readonly ort?: unknown;
  readonly Tokenizer?: typeof Tokenizer;
  readonly gpuApi?: unknown;
  readonly openSource?: (options: CrivoCheckedLoadOptions, spec: CrivoSpec) => Promise<CrivoSource>;
  readonly createTokenizer?: (data: unknown, config: unknown) => CrivoTokenizer;
  readonly createRunner?: (ort: unknown, session: CrivoOnnxSession) => CrivoRunner;
  readonly createLoadedSession?: (options: CrivoLoadedSessionInput) => unknown;
}

interface CrivoLoaderRuntimeDefaults extends CrivoLoaderRuntime {
  readonly Tokenizer: typeof Tokenizer;
  readonly openSource: NonNullable<CrivoLoaderRuntime['openSource']>;
  readonly createTokenizer: NonNullable<CrivoLoaderRuntime['createTokenizer']>;
  readonly createRunner: NonNullable<CrivoLoaderRuntime['createRunner']>;
  readonly createLoadedSession: NonNullable<CrivoLoaderRuntime['createLoadedSession']>;
}

const LOAD_KEYS = new Set(['model', 'assetsUrl', 'cache', 'offline', 'signal', 'onProgress']);
const fail = (code: ConstructorParameters<typeof WebDecisionError>[0], message: string, cause?: unknown) =>
  new WebDecisionError(code, message, cause ? {cause} : undefined);

function validSignal(signal: unknown): signal is AbortSignal | null | undefined {
  return (
    signal == null ||
    (typeof (signal as CrivoAbortSignalLike).throwIfAborted === 'function' &&
      typeof (signal as CrivoAbortSignalLike).addEventListener === 'function' &&
      typeof (signal as CrivoAbortSignalLike).removeEventListener === 'function')
  );
}

export function validateCrivoLoadOptions(options: unknown): CrivoCheckedLoadOptions {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'Crivo load options are required.');
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported Crivo load option.');
  }
  const spec = getCrivoSpec((options as CrivoLoadOptionsInput).model);
  if (
    !spec ||
    ((options as CrivoLoadOptionsInput).assetsUrl !== undefined &&
      (typeof (options as CrivoLoadOptionsInput).assetsUrl !== 'string' ||
        !(options as CrivoLoadOptionsInput).assetsUrl)) ||
    ((options as CrivoLoadOptionsInput).cache !== undefined &&
      typeof (options as CrivoLoadOptionsInput).cache !== 'boolean') ||
    ((options as CrivoLoadOptionsInput).offline !== undefined &&
      typeof (options as CrivoLoadOptionsInput).offline !== 'boolean') ||
    ((options as CrivoLoadOptionsInput).offline && (options as CrivoLoadOptionsInput).cache !== true) ||
    ((options as CrivoLoadOptionsInput).onProgress !== undefined &&
      typeof (options as CrivoLoadOptionsInput).onProgress !== 'function') ||
    !validSignal((options as CrivoLoadOptionsInput).signal)
  )
    throw fail('INVALID_INPUT', 'Invalid Crivo load option values.');

  let base: URL;
  try {
    const sourceUrl =
      (options as CrivoLoadOptionsInput).assetsUrl ??
      `https://huggingface.co/${CRIVO_REPOSITORY}/resolve/${CRIVO_REVISION}/models/32m/`;
    base = new URL(sourceUrl as string, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', 'Invalid Crivo assetsUrl.', cause);
  }
  if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'Crivo assetsUrl must be an HTTP or HTTPS directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {
    base,
    spec,
    signal: (options as CrivoLoadOptionsInput).signal as AbortSignal | null | undefined,
    cacheEnabled: (options as CrivoLoadOptionsInput).cache !== false,
    offline: (options as CrivoLoadOptionsInput).offline === true,
    onProgress: (options as CrivoLoadOptionsInput).onProgress as CrivoCheckedLoadOptions['onProgress']
  };
}

function validateSession(session: unknown): CrivoOnnxSession {
  const inputs = (session as CrivoOnnxSession | null | undefined)?.inputNames ?? [];
  const outputs = (session as CrivoOnnxSession | null | undefined)?.outputNames ?? [];
  const expectedInputs = ['input_ids', 'attention_mask', 'opt_positions', 'opt_mask', 'qtype'];
  if (
    inputs.length !== expectedInputs.length ||
    expectedInputs.some((name) => !inputs.includes(name)) ||
    outputs.length !== 1 ||
    outputs[0] !== 'logits'
  )
    throw fail('INVALID_ASSET', 'Crivo ONNX graph has an unexpected interface.');
  return session as CrivoOnnxSession;
}

async function getHardwareAdapter(gpu: unknown, signal: AbortSignal | null | undefined): Promise<CrivoGpuAdapter> {
  if (typeof (gpu as CrivoGpuApi | null | undefined)?.requestAdapter !== 'function')
    throw fail('UNSUPPORTED_ENVIRONMENT', 'Crivo FP32 requires WebGPU.');
  let adapter: CrivoGpuAdapter | null | undefined;
  try {
    adapter = await waitForAbort(
      () => (gpu as CrivoGpuApi).requestAdapter({powerPreference: 'high-performance', forceFallbackAdapter: false}),
      signal
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw fail('MODEL_LOAD_FAILED', 'Could not initialize a hardware WebGPU adapter for Crivo.', cause);
  }
  if (!adapter) throw fail('UNSUPPORTED_ENVIRONMENT', 'No hardware WebGPU adapter is available for Crivo.');
  if (adapter.info?.isFallbackAdapter === true || adapter.isFallbackAdapter === true)
    throw fail('UNSUPPORTED_ENVIRONMENT', 'Crivo FP32 does not run on a fallback WebGPU adapter.');
  return adapter;
}

async function createInferenceSession(
  ort: unknown,
  spec: CrivoSpec,
  bytes: ArrayBuffer,
  gpuApi: unknown,
  signal: AbortSignal | null | undefined
): Promise<CrivoOnnxSession> {
  if (
    typeof (ort as CrivoOrtRuntime | null | undefined)?.InferenceSession?.create !== 'function' ||
    typeof (ort as CrivoOrtRuntime | null | undefined)?.Tensor !== 'function'
  )
    throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime Web is unavailable.');
  const options = {
    executionProviders: [spec.backend === 'onnxruntime-web-webgpu' ? 'webgpu' : 'wasm'],
    graphOptimizationLevel: 'all'
  } as CrivoInferenceSessionOptions;
  if (spec.backend === 'onnxruntime-web-webgpu') {
    const adapter = await getHardwareAdapter(gpuApi, signal);
    const webgpu = (ort as CrivoOrtRuntime).env?.webgpu;
    if (!webgpu) throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WebGPU is unavailable.');
    webgpu.powerPreference = 'high-performance';
    webgpu.forceFallbackAdapter = false;
    webgpu.adapter = adapter;
  } else {
    const wasm = (ort as CrivoOrtRuntime).env?.wasm;
    if (!wasm) throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WASM is unavailable.');
    wasm.numThreads = 1;
    options.intraOpNumThreads = 1;
    options.interOpNumThreads = 1;
  }
  return waitForAbort(
    () =>
      (ort as CrivoOrtRuntime & Required<Pick<CrivoOrtRuntime, 'InferenceSession'>>).InferenceSession.create(
        new Uint8Array(bytes),
        options
      ),
    signal,
    (lateSession) => lateSession?.release()
  );
}

async function load(options: unknown, runtime: CrivoLoaderRuntime, spec: CrivoSpec): Promise<unknown> {
  const checked = validateCrivoLoadOptions(options);
  const {signal} = checked;
  let session;
  try {
    signal?.throwIfAborted();
    const source = await (runtime.openSource as NonNullable<CrivoLoaderRuntime['openSource']>)(
      {...checked, spec},
      spec
    );
    const [configBytes, tokenizerConfigBytes, tokenizerBytes] = await Promise.all([
      source.read('config.json'),
      source.read('tokenizer_config.json'),
      source.read('tokenizer.json')
    ]);
    const config = parseJson(configBytes, 'config.json');
    const tokenizerConfig = parseJson(tokenizerConfigBytes, 'tokenizer_config.json');
    const tokenizerJson = parseJson(tokenizerBytes, 'tokenizer.json');
    validateCrivoConfig(config, tokenizerConfig);
    const tokenizer = (runtime.createTokenizer as NonNullable<CrivoLoaderRuntime['createTokenizer']>)(
      tokenizerJson,
      tokenizerConfig
    );
    signal?.throwIfAborted();
    const graphBytes = await source.read(spec.graphFile);
    session = await createInferenceSession(runtime.ort, spec, graphBytes, runtime.gpuApi, signal);
    validateSession(session);
    signal?.throwIfAborted();
    const runner = (runtime.createRunner as NonNullable<CrivoLoaderRuntime['createRunner']>)(runtime.ort, session);
    const loaded = (runtime.createLoadedSession as NonNullable<CrivoLoaderRuntime['createLoadedSession']>)({
      runner,
      tokenizer,
      model: spec.model
    });
    session = null;
    return loaded;
  } catch (cause) {
    await Promise.allSettled([session?.release()].filter(Boolean));
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || (cause as {name?: unknown} | null | undefined)?.name === 'AbortError')
      throw cause;
    throw fail('MODEL_LOAD_FAILED', 'Could not load ' + spec.model + '.', cause);
  }
}

function runtimeDefaults(runtime: CrivoLoaderRuntime = {}): CrivoLoaderRuntimeDefaults {
  return {
    ...runtime,
    Tokenizer: runtime.Tokenizer ?? Tokenizer,
    gpuApi: runtime.gpuApi ?? globalThis.navigator?.gpu,
    openSource: runtime.openSource ?? openCrivoSource,
    createTokenizer:
      runtime.createTokenizer ?? ((data, config) => createCrivoTokenizer(runtime.Tokenizer ?? Tokenizer, data, config)),
    createRunner: runtime.createRunner ?? ((ort, session) => createCrivoRunner(ort, session)),
    createLoadedSession: runtime.createLoadedSession ?? createCrivoSession
  } as CrivoLoaderRuntimeDefaults;
}

export async function loadCrivoModel(options: unknown, runtime: CrivoLoaderRuntime = {}): Promise<unknown> {
  const checked = validateCrivoLoadOptions(options);
  const defaults = runtimeDefaults(runtime);
  const ort =
    defaults.ort ??
    (await import(
      checked.spec.backend === 'onnxruntime-web-webgpu' ? 'onnxruntime-web/webgpu' : 'onnxruntime-web/wasm'
    ));
  return load(options, {...defaults, ort}, checked.spec);
}
