import {Tokenizer} from '@huggingface/tokenizers';
import type {ModelProgress} from '../curated/types.js';
import {parseJson, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {SAFEDECIDE_MODEL, SAFEDECIDE_MODEL_CONFIG, SAFEDECIDE_SPEC} from './contract.mjs';
import {createSafeDecideRunner} from './runner.mjs';
import {createSafeDecideSession} from './session.mjs';
import {openSafeDecideSource} from './source.mjs';
import {createSafeDecideTokenizer} from './tokenizer.mjs';

type SafeDecideModelConfig = typeof SAFEDECIDE_MODEL_CONFIG;
type SafeDecideSpec = typeof SAFEDECIDE_SPEC;
type SafeDecideTokenizer = ReturnType<typeof createSafeDecideTokenizer>;
type SafeDecideRunner = ReturnType<typeof createSafeDecideRunner>;
type SafeDecideAlpha = 0.05 | 0.1 | 0.2;
type SafeDecideQuestionType = 'choice' | 'noul' | 'score';
type SafeDecideSignal = Pick<AbortSignal, 'throwIfAborted' | 'addEventListener' | 'removeEventListener'>;
type SafeDecideCreateTokenizer = (
  data: object,
  tokenizerConfig: object,
  modelConfig: SafeDecideModelConfig,
  maxSequenceTokens: number
) => SafeDecideTokenizer;
type SafeDecideWaitForAbort = <T>(
  operation: () => Promise<T>,
  signal: SafeDecideSignal | null | undefined,
  cleanup?: ((value: T) => unknown) | undefined
) => Promise<T>;

interface SafeDecideLoadOptionsCandidate {
  readonly model?: unknown;
  readonly assetsUrl?: unknown;
  readonly cache?: unknown;
  readonly offline?: unknown;
  readonly signal?: unknown;
  readonly onProgress?: unknown;
}

interface SafeDecideCheckedOptions {
  readonly base: URL;
  readonly signal: SafeDecideSignal | null | undefined;
  readonly cacheEnabled: boolean;
  readonly offline: boolean;
  readonly onProgress: ((progress: ModelProgress) => void) | undefined;
}

interface SafeDecideConfigCandidate {
  readonly model_type?: unknown;
  readonly vocab_size?: unknown;
  readonly hidden_size?: unknown;
  readonly num_hidden_layers?: unknown;
  readonly max_position_embeddings?: unknown;
  readonly bos_token_id?: unknown;
  readonly eos_token_id?: unknown;
  readonly pad_token_id?: unknown;
  readonly mask_token_id?: unknown;
  readonly num_labels?: unknown;
}

interface SafeDecideRuleCandidate {
  readonly alpha?: unknown;
  readonly threshold?: unknown;
  readonly delta?: unknown;
  readonly statement?: unknown;
}

interface SafeDecideModelDecisionConfigCandidate {
  readonly base_model?: unknown;
  readonly profile?: unknown;
  readonly max_length?: unknown;
  readonly temperatures?: unknown;
  readonly question_bias?: unknown;
  readonly selective_rules?: unknown;
  readonly selective_rule?: unknown;
}

interface SafeDecideValidatedRule {
  readonly alpha: SafeDecideAlpha;
  readonly threshold: number | null;
  readonly delta: number;
  readonly statement: string;
}

interface SafeDecideValidatedConfig {
  readonly base_model: 'jhu-clsp/mmBERT-small';
  readonly profile: 'clean';
  readonly max_length: number;
  readonly temperatures: Record<SafeDecideQuestionType, number>;
  readonly selective_rules: SafeDecideValidatedRule[];
  readonly selective_rule: SafeDecideValidatedRule;
}

interface SafeDecideOnnxSession {
  readonly inputNames?: readonly string[] | null;
  readonly outputNames?: readonly string[] | null;
  run(feeds: Record<string, unknown>, outputNames: string[]): Promise<unknown>;
  release(): void | Promise<unknown>;
}

interface SafeDecideAdapter {
  readonly info?: {readonly isFallbackAdapter?: unknown} | null;
  readonly isFallbackAdapter?: unknown;
}

interface SafeDecideGpuApi {
  requestAdapter?: (options: {
    readonly powerPreference: 'high-performance';
    readonly forceFallbackAdapter: false;
  }) => Promise<SafeDecideAdapter | null | undefined>;
}

interface SafeDecideWebGpuEnvironment {
  powerPreference?: string;
  forceFallbackAdapter?: boolean;
  adapter?: SafeDecideAdapter;
}

interface SafeDecideOrt {
  readonly Tensor?: unknown;
  readonly env?: {readonly webgpu?: SafeDecideWebGpuEnvironment | null} | null;
  readonly InferenceSession?: {
    create?: (
      graph: Uint8Array,
      options: {readonly executionProviders: ['webgpu']; readonly graphOptimizationLevel: 'all'}
    ) => Promise<SafeDecideOnnxSession>;
  } | null;
}

interface SafeDecideRuntimeOverrides {
  readonly fake?: unknown;
  readonly spec?: unknown;
  readonly model?: unknown;
  readonly Tokenizer?: unknown;
  readonly gpuApi?: unknown;
  readonly openSource?: unknown;
  readonly createTokenizer?: unknown;
  readonly createRunner?: unknown;
  readonly createLoadedSession?: unknown;
  readonly ort?: unknown;
  readonly [key: string]: unknown;
}

type SafeDecideOpenSource = (
  options: Omit<Parameters<typeof openSafeDecideSource>[0], 'signal' | 'spec'> & {
    readonly signal?: SafeDecideSignal | null | undefined;
    readonly spec: SafeDecideSpec;
  }
) => ReturnType<typeof openSafeDecideSource>;
type SafeDecideRuntimeCreateRunner = (ort: unknown, session: SafeDecideOnnxSession, padId: number) => SafeDecideRunner;

interface SafeDecideRuntime {
  readonly Tokenizer: typeof Tokenizer;
  readonly gpuApi: SafeDecideGpuApi | null | undefined;
  readonly openSource: SafeDecideOpenSource;
  readonly createTokenizer: SafeDecideCreateTokenizer;
  readonly createRunner: SafeDecideRuntimeCreateRunner;
  readonly createLoadedSession: typeof createSafeDecideSession;
  readonly ort?: SafeDecideOrt | null | undefined;
}

const LOAD_KEYS = new Set(['model', 'assetsUrl', 'cache', 'offline', 'signal', 'onProgress']);
const fail = (
  code: 'INVALID_INPUT' | 'INVALID_ASSET' | 'UNSUPPORTED_ENVIRONMENT' | 'MODEL_LOAD_FAILED',
  message: string,
  cause?: unknown
): WebDecisionError => new WebDecisionError(code, message, cause ? {cause} : undefined);

function validSignal(signal: unknown): boolean {
  return (
    signal == null ||
    (typeof (signal as {throwIfAborted?: unknown}).throwIfAborted === 'function' &&
      typeof (signal as {addEventListener?: unknown}).addEventListener === 'function' &&
      typeof (signal as {removeEventListener?: unknown}).removeEventListener === 'function')
  );
}

export function validateSafeDecideLoadOptions(
  options: unknown,
  model: string = SAFEDECIDE_MODEL
): SafeDecideCheckedOptions {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'SafeDecide load options are required.');
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported SafeDecide load option.');
  }
  if (
    (options as SafeDecideLoadOptionsCandidate).model !== model ||
    ((options as SafeDecideLoadOptionsCandidate).assetsUrl !== undefined &&
      (typeof (options as SafeDecideLoadOptionsCandidate).assetsUrl !== 'string' ||
        !(options as SafeDecideLoadOptionsCandidate).assetsUrl)) ||
    ((options as SafeDecideLoadOptionsCandidate).cache !== undefined &&
      typeof (options as SafeDecideLoadOptionsCandidate).cache !== 'boolean') ||
    ((options as SafeDecideLoadOptionsCandidate).offline !== undefined &&
      typeof (options as SafeDecideLoadOptionsCandidate).offline !== 'boolean') ||
    ((options as SafeDecideLoadOptionsCandidate).offline &&
      (options as SafeDecideLoadOptionsCandidate).cache !== true) ||
    ((options as SafeDecideLoadOptionsCandidate).onProgress !== undefined &&
      typeof (options as SafeDecideLoadOptionsCandidate).onProgress !== 'function') ||
    !validSignal((options as SafeDecideLoadOptionsCandidate).signal)
  )
    throw fail('INVALID_INPUT', 'Invalid SafeDecide load option values.');

  let base: URL;
  try {
    const sourceUrl =
      ((options as SafeDecideLoadOptionsCandidate).assetsUrl as string | undefined) ??
      `https://huggingface.co/${SAFEDECIDE_SPEC.repository}/resolve/${SAFEDECIDE_SPEC.revision}/`;
    base = new URL(sourceUrl, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', 'Invalid SafeDecide assetsUrl.', cause);
  }
  if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'SafeDecide assetsUrl must be an HTTP or HTTPS directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {
    base,
    signal: (options as SafeDecideLoadOptionsCandidate).signal as SafeDecideSignal | null | undefined,
    cacheEnabled: (options as SafeDecideLoadOptionsCandidate).cache === true,
    offline: (options as SafeDecideLoadOptionsCandidate).offline === true,
    onProgress: (options as SafeDecideLoadOptionsCandidate).onProgress as
      | ((progress: ModelProgress) => void)
      | undefined
  };
}

function positiveTemperatures(temperatures: unknown): unknown {
  return (
    temperatures &&
    !Array.isArray(temperatures) &&
    ['choice', 'noul', 'score'].every(
      (type) =>
        Number.isFinite((temperatures as Record<string, unknown>)[type]) &&
        ((temperatures as Record<string, unknown>)[type] as number) > 0
    ) &&
    Object.keys(temperatures as object).every((type) => ['choice', 'noul', 'score'].includes(type))
  );
}

function validateRules(config: SafeDecideModelDecisionConfigCandidate): void {
  const expected = [
    {alpha: 0.05, threshold: null},
    {alpha: 0.1, threshold: 0.7},
    {alpha: 0.2, threshold: 0.5}
  ];
  if (
    !Array.isArray(config.selective_rules) ||
    (config.selective_rules as unknown[]).length !== expected.length ||
    (config.selective_rules as unknown[]).some(
      (rule: unknown, index: number) =>
        (rule as SafeDecideRuleCandidate | null | undefined)?.alpha !== expected[index]!.alpha ||
        (rule as SafeDecideRuleCandidate | null | undefined)?.threshold !== expected[index]!.threshold ||
        (rule as SafeDecideRuleCandidate | null | undefined)?.delta !== 0.1 ||
        typeof (rule as SafeDecideRuleCandidate | null | undefined)?.statement !== 'string' ||
        !(rule as SafeDecideRuleCandidate).statement
    ) ||
    (config.selective_rule as SafeDecideRuleCandidate | null | undefined)?.alpha !== 0.1 ||
    (config.selective_rule as SafeDecideRuleCandidate).threshold !== 0.7 ||
    (config.selective_rule as SafeDecideRuleCandidate).delta !== 0.1 ||
    (config.selective_rule as SafeDecideRuleCandidate).statement !==
      ((config.selective_rules as unknown[])[1] as SafeDecideRuleCandidate).statement
  )
    throw fail('INVALID_ASSET', 'SafeDecide certified action rules do not match the pinned checkpoint.');
}

export function validateSafeDecideModelConfig(
  config: unknown,
  safedecideConfig: unknown,
  tokenizerConfig: unknown,
  modelConfig: SafeDecideModelConfig = SAFEDECIDE_MODEL_CONFIG
): SafeDecideValidatedConfig {
  if (
    (config as SafeDecideConfigCandidate | null | undefined)?.model_type !== modelConfig.modelType ||
    (config as SafeDecideConfigCandidate | null | undefined)?.vocab_size !== modelConfig.vocabSize ||
    (config as SafeDecideConfigCandidate | null | undefined)?.hidden_size !== modelConfig.hiddenSize ||
    (config as SafeDecideConfigCandidate | null | undefined)?.num_hidden_layers !== modelConfig.layers ||
    (config as SafeDecideConfigCandidate | null | undefined)?.max_position_embeddings !== 8192 ||
    (config as SafeDecideConfigCandidate | null | undefined)?.bos_token_id !== modelConfig.specialTokenIds.bosId ||
    (config as SafeDecideConfigCandidate | null | undefined)?.eos_token_id !== modelConfig.specialTokenIds.eosId ||
    (config as SafeDecideConfigCandidate | null | undefined)?.pad_token_id !== modelConfig.specialTokenIds.padId ||
    (config as SafeDecideConfigCandidate | null | undefined)?.mask_token_id !== modelConfig.specialTokenIds.maskId ||
    ((config as SafeDecideConfigCandidate | null | undefined)?.num_labels !== undefined &&
      (config as SafeDecideConfigCandidate).num_labels !== 1) ||
    (safedecideConfig as SafeDecideModelDecisionConfigCandidate | null | undefined)?.base_model !==
      'jhu-clsp/mmBERT-small' ||
    (safedecideConfig as SafeDecideModelDecisionConfigCandidate | null | undefined)?.profile !== 'clean' ||
    (safedecideConfig as SafeDecideModelDecisionConfigCandidate | null | undefined)?.max_length !==
      SAFEDECIDE_SPEC.maxSequenceTokens ||
    !positiveTemperatures(
      (safedecideConfig as SafeDecideModelDecisionConfigCandidate | null | undefined)?.temperatures
    ) ||
    ((safedecideConfig as SafeDecideModelDecisionConfigCandidate | null | undefined)?.question_bias &&
      Object.keys((safedecideConfig as SafeDecideModelDecisionConfigCandidate).question_bias as object).length > 0) ||
    (
      tokenizerConfig as
        | {readonly max_length?: unknown; readonly truncation_side?: unknown; readonly padding_side?: unknown}
        | null
        | undefined
    )?.max_length !== 384 ||
    (
      tokenizerConfig as
        | {readonly max_length?: unknown; readonly truncation_side?: unknown; readonly padding_side?: unknown}
        | null
        | undefined
    )?.truncation_side !== 'right' ||
    (
      tokenizerConfig as
        | {readonly max_length?: unknown; readonly truncation_side?: unknown; readonly padding_side?: unknown}
        | null
        | undefined
    )?.padding_side !== 'right'
  )
    throw fail('INVALID_ASSET', 'SafeDecide configuration does not match the pinned checkpoint.');
  validateRules(safedecideConfig as SafeDecideModelDecisionConfigCandidate);
  return safedecideConfig as SafeDecideValidatedConfig;
}

function validateSession(session: SafeDecideOnnxSession): SafeDecideOnnxSession {
  const inputs: readonly string[] = (session as SafeDecideOnnxSession | null | undefined)?.inputNames ?? [];
  const outputs: readonly string[] = (session as SafeDecideOnnxSession | null | undefined)?.outputNames ?? [];
  if (
    inputs.length !== 2 ||
    !inputs.includes('input_ids') ||
    !inputs.includes('attention_mask') ||
    outputs.length !== 1 ||
    outputs[0] !== 'logits'
  )
    throw fail('INVALID_ASSET', 'SafeDecide ONNX graph has an unexpected interface.');
  return session;
}

async function getHardwareAdapter(
  gpu: SafeDecideGpuApi | null | undefined,
  signal: SafeDecideSignal | null | undefined
): Promise<SafeDecideAdapter> {
  if (typeof gpu?.requestAdapter !== 'function') throw fail('UNSUPPORTED_ENVIRONMENT', 'SafeDecide requires WebGPU.');
  let adapter;
  try {
    adapter = await (waitForAbort as SafeDecideWaitForAbort)(
      () => gpu.requestAdapter!({powerPreference: 'high-performance', forceFallbackAdapter: false}),
      signal
    );
  } catch (cause) {
    signal?.throwIfAborted();
    throw fail('MODEL_LOAD_FAILED', 'Could not initialize a hardware WebGPU adapter for SafeDecide.', cause);
  }
  if (!adapter) throw fail('UNSUPPORTED_ENVIRONMENT', 'No hardware WebGPU adapter is available for SafeDecide.');
  if (adapter.info?.isFallbackAdapter === true || adapter.isFallbackAdapter === true)
    throw fail('UNSUPPORTED_ENVIRONMENT', 'SafeDecide does not run on a fallback WebGPU adapter.');
  return adapter;
}

async function createSession(
  ort: SafeDecideOrt | null | undefined,
  adapter: SafeDecideAdapter,
  graphBytes: ArrayBuffer,
  signal: SafeDecideSignal | null | undefined
): Promise<SafeDecideOnnxSession> {
  if (typeof ort?.InferenceSession?.create !== 'function' || typeof ort?.Tensor !== 'function')
    throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WebGPU is unavailable.');
  const webgpu = ort.env?.webgpu;
  if (!webgpu) throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime WebGPU execution provider is unavailable.');
  webgpu.powerPreference = 'high-performance';
  webgpu.forceFallbackAdapter = false;
  webgpu.adapter = adapter;
  return (waitForAbort as SafeDecideWaitForAbort)(
    () =>
      ort.InferenceSession!.create!(new Uint8Array(graphBytes), {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all'
      }),
    signal,
    (lateSession) => lateSession?.release()
  );
}

async function load(
  options: unknown,
  runtime: SafeDecideRuntime,
  spec: SafeDecideSpec,
  model: string
): Promise<unknown> {
  const checked = validateSafeDecideLoadOptions(options, model);
  const {signal} = checked;
  let session: SafeDecideOnnxSession | null | undefined;
  try {
    signal?.throwIfAborted();
    const source = await runtime.openSource({...checked, spec});
    const [configBytes, safeConfigBytes, tokenizerConfigBytes, tokenizerBytes] = await Promise.all([
      source.read('config.json'),
      source.read('safedecide_config.json'),
      source.read('tokenizer_config.json'),
      source.read('tokenizer.json')
    ]);
    const config = parseJson(configBytes, 'config.json');
    const safedecideConfig = validateSafeDecideModelConfig(
      config,
      parseJson(safeConfigBytes, 'safedecide_config.json'),
      parseJson(tokenizerConfigBytes, 'tokenizer_config.json'),
      spec.modelConfig
    );
    const tokenizer = runtime.createTokenizer(
      parseJson(tokenizerBytes, 'tokenizer.json') as object,
      parseJson(tokenizerConfigBytes, 'tokenizer_config.json') as object,
      spec.modelConfig,
      spec.maxSequenceTokens
    );
    signal?.throwIfAborted();
    const adapter = await getHardwareAdapter(runtime.gpuApi, signal);
    signal?.throwIfAborted();
    const graphBytes = await source.read('onnx/model.onnx');
    signal?.throwIfAborted();
    session = await createSession(runtime.ort, adapter, graphBytes, signal);
    validateSession(session);
    signal?.throwIfAborted();
    const runner = runtime.createRunner(runtime.ort, session, tokenizer.special.padId);
    const loaded = runtime.createLoadedSession({runner, tokenizer, config: safedecideConfig, model});
    session = null;
    return loaded;
  } catch (cause) {
    await Promise.allSettled([session?.release()].filter(Boolean));
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError || (cause as {name?: unknown} | null | undefined)?.name === 'AbortError')
      throw cause;
    throw fail('MODEL_LOAD_FAILED', `Could not load ${model} on ONNX Runtime WebGPU.`, cause);
  }
}

function runtimeDefaults(runtime: SafeDecideRuntimeOverrides = {}): SafeDecideRuntime {
  return {
    ...runtime,
    Tokenizer: (runtime.Tokenizer ?? Tokenizer) as typeof Tokenizer,
    gpuApi: (runtime.gpuApi ?? (globalThis.navigator as (Navigator & {gpu?: unknown}) | undefined)?.gpu) as
      | SafeDecideGpuApi
      | undefined,
    openSource: (runtime.openSource ?? openSafeDecideSource) as unknown as SafeDecideOpenSource,
    createTokenizer:
      (runtime.createTokenizer as SafeDecideCreateTokenizer | undefined) ??
      ((data, tokenizerConfig, modelConfig, maxSequenceTokens) =>
        createSafeDecideTokenizer(
          (runtime.Tokenizer ?? Tokenizer) as typeof Tokenizer,
          data,
          tokenizerConfig,
          modelConfig,
          maxSequenceTokens
        )),
    createRunner:
      (runtime.createRunner as SafeDecideRuntimeCreateRunner | undefined) ??
      ((ort, session, padId) => createSafeDecideRunner(ort, session, padId)),
    createLoadedSession:
      (runtime.createLoadedSession as typeof createSafeDecideSession | undefined) ?? createSafeDecideSession
  } as SafeDecideRuntime;
}

export async function loadSafeDecideModel(
  options: unknown,
  runtime: SafeDecideRuntimeOverrides = {}
): Promise<unknown> {
  validateSafeDecideLoadOptions(options);
  const defaults = runtimeDefaults(runtime);
  const ort = defaults.ort ?? ((await import('onnxruntime-web/webgpu')) as SafeDecideOrt);
  return load(options, {...defaults, ort}, SAFEDECIDE_SPEC, SAFEDECIDE_MODEL);
}

export async function loadFakeSafeDecide(
  options: unknown,
  deps: SafeDecideRuntimeOverrides | null | undefined
): Promise<unknown> {
  if (deps?.fake !== true) throw new TypeError('Fake dependencies are required.');
  return load(
    options,
    runtimeDefaults(deps as SafeDecideRuntimeOverrides),
    ((deps as SafeDecideRuntimeOverrides).spec ?? SAFEDECIDE_SPEC) as SafeDecideSpec,
    ((deps as SafeDecideRuntimeOverrides).model ?? SAFEDECIDE_MODEL) as string
  );
}
