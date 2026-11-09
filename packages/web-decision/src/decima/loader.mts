import {Tokenizer} from '@huggingface/tokenizers';
import * as defaultOrt from 'onnxruntime-web/wasm';
import {parseJson, waitForAbort} from '../assets.js';
import type {AssetSourceFile, ManifestSourceSpec} from '../asset-source-types.js';
import {WebDecisionError} from '../kev.js';
import type {ModelProgress} from '../curated/types.js';
import {DECIMA_MODELS} from './contract.mjs';
import {createDecimaRunner, type DecimaOrtApi} from './runner.mjs';
import {createDecimaSession, type DecimaSession} from './session.mjs';
import {openDecimaSource, type DecimaSource, type DecimaSourceOptions} from './source.mjs';
import {createDecimaTokenizer} from './tokenizer.mjs';
import type {DecimaInferenceKind, DecimaOrdinalConfig, DecimaRuntimeConfig} from './protocol.mjs';
import type {InferenceSession} from 'onnxruntime-web/wasm';

interface DecimaLoadOptions {
  readonly model: string;
  readonly assetsUrl?: string | undefined;
  readonly cache?: boolean | undefined;
  readonly offline?: boolean | undefined;
  readonly signal?: AbortSignal | null | undefined;
  readonly onProgress?: ((progress: ModelProgress) => void) | undefined;
  readonly maxPrefetchBytes?: number | undefined;
}

interface DecimaModelSpec extends ManifestSourceSpec {
  readonly model: string;
  readonly repository: string;
  readonly revision: string;
  readonly files: Readonly<Record<string, AssetSourceFile>>;
  readonly maxStateTokens: number;
  readonly sourceMaxStateTokens: number;
  readonly maxChoiceTokens: number;
  readonly temperature: number;
  readonly stateOverflow: 'truncate' | 'reject';
  readonly choiceFit: boolean;
  readonly typedQuestionTypes: readonly DecimaInferenceKind[];
  readonly cpuWasmVerification: string;
}

interface DecimaJsonConfig extends DecimaRuntimeConfig {
  readonly max_state_tokens: number;
  readonly max_choice_tokens: number;
  readonly question_in_state: boolean;
  readonly question_in_choices: boolean;
  readonly state_prefix: string;
  readonly choice_prefix: string;
  readonly choice_fit?: boolean | undefined;
  readonly ordinal: DecimaOrdinalConfig;
}

interface DecimaModelConfig extends DecimaJsonConfig {
  readonly max_state_tokens: number;
  readonly choice_fit: boolean;
  readonly state_overflow: 'truncate' | 'reject';
  readonly supported_question_types: DecimaInferenceKind[];
}

interface DecimaCheckedOptions {
  readonly model: string;
  readonly spec: DecimaModelSpec;
  readonly base: URL;
  readonly signal: AbortSignal | null | undefined;
  readonly cacheEnabled: boolean;
  readonly offline: boolean;
  readonly onProgress: ((progress: ModelProgress) => void) | undefined;
}

interface DecimaOpenSourceOptions extends Omit<DecimaSourceOptions, 'signal'> {
  readonly signal?: AbortSignal | null | undefined;
}

type DecimaSourceOpener = (options: DecimaOpenSourceOptions) => Promise<DecimaSource>;

interface DecimaLoadDependencies {
  readonly ort?: DecimaOrtApi | undefined;
  readonly Tokenizer?: typeof Tokenizer | undefined;
  readonly openSource?: DecimaSourceOpener | undefined;
}

const LOAD_KEYS = new Set(['model', 'assetsUrl', 'cache', 'offline', 'signal', 'onProgress', 'maxPrefetchBytes']);
const fail = (
  code: ConstructorParameters<typeof WebDecisionError>[0],
  message: string,
  cause?: unknown
): WebDecisionError => new WebDecisionError(code, message, cause ? {cause} : undefined);

export function validateDecimaLoadOptions(options: unknown): DecimaCheckedOptions {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'Decima load options are required.');
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported Decima load option.');
  }
  const spec = (DECIMA_MODELS as Readonly<Record<string, DecimaModelSpec>>)[(options as DecimaLoadOptions).model];
  if (
    !spec ||
    ((options as DecimaLoadOptions).assetsUrl !== undefined &&
      (typeof (options as DecimaLoadOptions).assetsUrl !== 'string' || !(options as DecimaLoadOptions).assetsUrl)) ||
    ((options as DecimaLoadOptions).cache !== undefined && typeof (options as DecimaLoadOptions).cache !== 'boolean') ||
    ((options as DecimaLoadOptions).offline !== undefined &&
      typeof (options as DecimaLoadOptions).offline !== 'boolean') ||
    ((options as DecimaLoadOptions).offline && (options as DecimaLoadOptions).cache === false) ||
    ((options as DecimaLoadOptions).maxPrefetchBytes !== undefined &&
      (options as DecimaLoadOptions).maxPrefetchBytes !== 0) ||
    ((options as DecimaLoadOptions).onProgress !== undefined &&
      typeof (options as DecimaLoadOptions).onProgress !== 'function') ||
    ((options as DecimaLoadOptions).signal != null &&
      typeof ((options as DecimaLoadOptions).signal as AbortSignal).throwIfAborted !== 'function')
  )
    throw fail('INVALID_INPUT', 'Invalid Decima load option values.');
  let base: URL;
  try {
    const sourceUrl =
      (options as DecimaLoadOptions).assetsUrl ??
      `https://huggingface.co/${spec.repository}/resolve/${spec.revision}/onnx/int8/`;
    base = new URL(sourceUrl, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', 'Invalid Decima assetsUrl.', cause);
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'Decima assetsUrl must be an HTTP or HTTPS directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {
    model: (options as DecimaLoadOptions).model,
    spec,
    base,
    signal: (options as DecimaLoadOptions).signal,
    cacheEnabled: (options as DecimaLoadOptions).cache !== false,
    offline: (options as DecimaLoadOptions).offline === true,
    onProgress: (options as DecimaLoadOptions).onProgress
  };
}

function validateConfig(config: unknown, spec: DecimaModelSpec): DecimaModelConfig {
  if (
    (config as DecimaJsonConfig | null | undefined)?.max_state_tokens !== spec.sourceMaxStateTokens ||
    (config as DecimaJsonConfig | null | undefined)?.max_choice_tokens !== spec.maxChoiceTokens ||
    (config as DecimaJsonConfig | null | undefined)?.temperature !== spec.temperature ||
    (config as DecimaJsonConfig | null | undefined)?.question_in_state !== true ||
    (config as DecimaJsonConfig | null | undefined)?.question_in_choices !== true ||
    typeof (config as DecimaJsonConfig | null | undefined)?.state_prefix !== 'string' ||
    typeof (config as DecimaJsonConfig | null | undefined)?.choice_prefix !== 'string' ||
    ((config as DecimaJsonConfig | null | undefined)?.choice_fit ?? false) !== spec.choiceFit ||
    !Array.isArray((config as DecimaJsonConfig).ordinal?.ord_g_w) ||
    !Array.isArray((config as DecimaJsonConfig).ordinal?.ord_gap_w) ||
    (config as DecimaJsonConfig).ordinal.ord_g_w.length !== (config as DecimaJsonConfig).ordinal.ord_gap_w.length ||
    !Number.isFinite((config as DecimaJsonConfig).ordinal.ord_g_b) ||
    !Number.isFinite((config as DecimaJsonConfig).ordinal.ord_gap_b) ||
    [...(config as DecimaJsonConfig).ordinal.ord_g_w, ...(config as DecimaJsonConfig).ordinal.ord_gap_w].some(
      (value) => !Number.isFinite(value)
    )
  )
    throw fail('INVALID_ASSET', 'Decima configuration does not match the pinned model.');
  return {
    ...(config as DecimaJsonConfig),
    max_state_tokens: spec.maxStateTokens,
    choice_fit: spec.choiceFit,
    state_overflow: spec.stateOverflow,
    supported_question_types: spec.typedQuestionTypes as DecimaInferenceKind[]
  };
}

async function createSession(
  ort: DecimaOrtApi,
  modelBytes: ArrayBuffer,
  signal?: AbortSignal | null
): Promise<InferenceSession> {
  if (typeof ort?.InferenceSession?.create !== 'function')
    throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime Web is unavailable.');
  return waitForAbort(
    () =>
      ort.InferenceSession.create(modelBytes, {
        executionProviders: ['wasm'],
        intraOpNumThreads: 1,
        interOpNumThreads: 1,
        graphOptimizationLevel: 'all'
      }),
    signal,
    (lateSession) => lateSession?.release()
  );
}

export async function loadDecimaModel(
  options: unknown,
  {
    ort = defaultOrt,
    Tokenizer: TokenizerClass = Tokenizer,
    openSource = openDecimaSource as unknown as DecimaSourceOpener
  }: DecimaLoadDependencies = {}
): Promise<DecimaSession> {
  const checked = validateDecimaLoadOptions(options);
  const {signal, spec} = checked;
  let encoder: InferenceSession | null | undefined;
  let scorer: InferenceSession | null | undefined;
  try {
    const source = await openSource({...checked, signal});
    const [tokenizerBytes, tokenizerConfigBytes, configBytes] = await Promise.all([
      source.read('tokenizer/tokenizer.json'),
      source.read('tokenizer/tokenizer_config.json'),
      source.read('decima.json')
    ]);
    const tokenizerData = parseJson(tokenizerBytes, 'tokenizer/tokenizer.json');
    const tokenizerConfig = parseJson(tokenizerConfigBytes, 'tokenizer/tokenizer_config.json');
    const config = validateConfig(parseJson(configBytes, 'decima.json'), spec);
    const tokenizer = createDecimaTokenizer(TokenizerClass, tokenizerData, tokenizerConfig);
    signal?.throwIfAborted();

    const encoderBytes = await source.read('encoder.onnx');
    encoder = await createSession(ort, encoderBytes, signal);
    signal?.throwIfAborted();
    const scorerBytes = await source.read('scorer.onnx');
    scorer = await createSession(ort, scorerBytes, signal);
    signal?.throwIfAborted();

    const runner = createDecimaRunner(ort, encoder, scorer, tokenizer);
    const model = createDecimaSession({runner, tokenizer, config, model: spec.model});
    encoder = scorer = null;
    return model;
  } catch (cause) {
    await Promise.allSettled([scorer?.release(), encoder?.release()].filter(Boolean));
    signal?.throwIfAborted();
    if (
      cause instanceof WebDecisionError ||
      (cause as {readonly name?: unknown} | null | undefined)?.name === 'AbortError'
    )
      throw cause;
    throw fail('MODEL_LOAD_FAILED', `Could not load ${spec.model}.`, cause);
  }
}
