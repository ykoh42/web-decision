import {Tokenizer} from '@huggingface/tokenizers';
import * as defaultOrt from 'onnxruntime-web/wasm';
import {parseJson, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import type {ModelProgress} from '../curated/types.js';
import type {InferenceSession} from 'onnxruntime-web/wasm';
import {ETTIN_FILES, ETTIN_MODEL, ETTIN_PROFILE} from './contract.mjs';
import {createEttinRunner} from './runner.mjs';
import type {EttinOrt} from './runner.mjs';
import {createEttinSession} from './session.mjs';
import {openEttinSource} from './source.mjs';
import {createEttinTokenizer} from './tokenizer.mjs';

const LOAD_KEYS = new Set(['model', 'assetsUrl', 'cache', 'offline', 'signal', 'onProgress', 'maxPrefetchBytes']);
type EttinErrorCode = ConstructorParameters<typeof WebDecisionError>[0];

interface CheckedEttinLoadOptions {
  readonly base: URL;
  readonly signal: AbortSignal | null | undefined;
  readonly cacheEnabled: boolean;
  readonly offline: boolean;
  readonly onProgress: ((progress: ModelProgress) => void) | undefined;
}

interface EttinSource {
  read(name: string): Promise<ArrayBuffer>;
}

type EttinSourceOpener = (options: CheckedEttinLoadOptions) => Promise<EttinSource>;

interface EttinLoadDependencies {
  readonly ort?: EttinOrt;
  readonly Tokenizer?: unknown;
  readonly openSource?: EttinSourceOpener;
}

const fail = (code: EttinErrorCode, message: string, cause?: unknown) =>
  new WebDecisionError(code, message, cause ? {cause} : undefined);

export function validateEttinLoadOptions(options: unknown): CheckedEttinLoadOptions {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'Decider Ettin load options are required.');
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported Decider Ettin load option.');
  }
  const values = Object.fromEntries(
    Reflect.ownKeys(options).map((key) => [key, (options as Record<string, unknown>)[key as string]])
  ) as Record<string, unknown>;
  if (
    values.model !== ETTIN_MODEL ||
    (values.assetsUrl !== undefined && (typeof values.assetsUrl !== 'string' || !values.assetsUrl)) ||
    (values.cache !== undefined && typeof values.cache !== 'boolean') ||
    (values.offline !== undefined && typeof values.offline !== 'boolean') ||
    (values.offline && values.cache === false) ||
    (values.maxPrefetchBytes !== undefined && values.maxPrefetchBytes !== 0) ||
    (values.onProgress !== undefined && typeof values.onProgress !== 'function') ||
    (values.signal != null &&
      (typeof (values.signal as AbortSignal).throwIfAborted !== 'function' ||
        typeof (values.signal as AbortSignal).addEventListener !== 'function' ||
        typeof (values.signal as AbortSignal).removeEventListener !== 'function'))
  )
    throw fail('INVALID_INPUT', 'Invalid Decider Ettin load option values.');

  let base;
  try {
    const sourceUrl =
      values.assetsUrl ?? `https://huggingface.co/${ETTIN_PROFILE.repository}/resolve/${ETTIN_PROFILE.revision}/`;
    base = new URL(sourceUrl, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', 'Invalid Decider Ettin assetsUrl.', cause);
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'Decider Ettin assetsUrl must be an HTTP or HTTPS directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';

  return {
    base,
    signal: values.signal as AbortSignal | null | undefined,
    cacheEnabled: values.cache !== false,
    offline: values.offline === true,
    onProgress: values.onProgress as ((progress: ModelProgress) => void) | undefined
  };
}

function validateConfig(config: unknown): unknown {
  if (
    (config as Record<string, unknown> | null | undefined)?.base_model !== 'jhu-clsp/ettin-encoder-68m' ||
    (config as Record<string, unknown> | null | undefined)?.max_len !== ETTIN_PROFILE.maxSequenceTokens ||
    (config as Record<string, unknown> | null | undefined)?.max_options !== ETTIN_PROFILE.maxOptions ||
    (config as Record<string, unknown> | null | undefined)?.max_score_levels !== ETTIN_PROFILE.maxScoreLevels ||
    (config as Record<string, unknown> | null | undefined)?.onnx_opset !== 18 ||
    (config as Record<string, unknown> | null | undefined)?.onnx_batch_size !== 1 ||
    (config as Record<string, unknown> | null | undefined)?.onnx_sequence_dynamic !== true ||
    JSON.stringify((config as Record<string, unknown> | null | undefined)?.onnx_inputs) !==
      JSON.stringify(['input_ids', 'attention_mask', 'decision_pos', 'option_pos', 'option_mask', 'score_n']) ||
    JSON.stringify((config as Record<string, unknown> | null | undefined)?.onnx_outputs) !==
      JSON.stringify(['choice_logits', 'noul_logits', 'score_logits'])
  )
    throw fail('INVALID_ASSET', 'Decider Ettin config does not match the pinned ONNX graph.');
  return config;
}

function validateManifest(manifest: unknown): unknown {
  if (
    (manifest as Record<string, unknown> | null | undefined)?.version !== 'epoch2-int8' ||
    (manifest as Record<string, unknown> | null | undefined)?.modelFile !== 'model_int8.onnx' ||
    (manifest as Record<string, unknown> | null | undefined)?.sizeBytes !== ETTIN_FILES['model_int8.onnx']!.bytes ||
    (manifest as Record<string, unknown> | null | undefined)?.sha256 !== ETTIN_FILES['model_int8.onnx']!.sha256 ||
    (manifest as Record<string, unknown> | null | undefined)?.maxOptions !== ETTIN_PROFILE.maxOptions ||
    (manifest as Record<string, unknown> | null | undefined)?.maxScoreLevels !== ETTIN_PROFILE.maxScoreLevels ||
    (manifest as Record<string, unknown> | null | undefined)?.maxSequenceLength !== ETTIN_PROFILE.maxSequenceTokens
  )
    throw fail('INVALID_ASSET', 'Decider Ettin manifest does not match the pinned model.');
  return manifest;
}

function createSession(
  ort: EttinOrt,
  bytes: ArrayBuffer,
  signal: AbortSignal | null | undefined
): Promise<InferenceSession> {
  if (typeof ort?.InferenceSession?.create !== 'function')
    throw fail('UNSUPPORTED_ENVIRONMENT', 'ONNX Runtime Web is unavailable.');
  return waitForAbort(
    () =>
      ort.InferenceSession.create(new Uint8Array(bytes), {
        executionProviders: ['wasm'],
        intraOpNumThreads: 1,
        interOpNumThreads: 1,
        graphOptimizationLevel: 'all'
      }),
    signal,
    (lateSession: InferenceSession) => lateSession?.release()
  ) as Promise<InferenceSession>;
}

function validateSession(session: unknown): InferenceSession {
  const inputs =
    (
      session as {
        readonly inputNames?: readonly string[];
        readonly outputNames?: readonly string[];
      } | null
    )?.inputNames ?? [];
  const outputs =
    (
      session as {
        readonly inputNames?: readonly string[];
        readonly outputNames?: readonly string[];
      } | null
    )?.outputNames ?? [];
  const expectedInputs = ['input_ids', 'attention_mask', 'decision_pos', 'option_pos', 'option_mask', 'score_n'];
  const expectedOutputs = ['choice_logits', 'noul_logits', 'score_logits'];
  if (
    inputs.length !== expectedInputs.length ||
    expectedInputs.some((name) => !inputs.includes(name)) ||
    outputs.length !== expectedOutputs.length ||
    expectedOutputs.some((name) => !outputs.includes(name))
  )
    throw fail('INVALID_ASSET', 'Decider Ettin ONNX graph has an unexpected interface.');
  return session as Parameters<typeof createEttinRunner>[1];
}

export async function loadEttin(
  options: unknown,
  {
    ort = defaultOrt,
    Tokenizer: TokenizerClass = Tokenizer,
    openSource = openEttinSource as EttinSourceOpener
  }: EttinLoadDependencies = {}
) {
  const checked = validateEttinLoadOptions(options);
  const {signal} = checked;
  let session: InferenceSession | null | undefined;
  try {
    const source = await openSource(checked);
    const [configBytes, manifestBytes, tokenizerBytes, tokenizerConfigBytes] = await Promise.all([
      source.read('config.json'),
      source.read('manifest.json'),
      source.read('tokenizer.json'),
      source.read('tokenizer_config.json')
    ]);
    validateConfig(parseJson(configBytes, 'config.json'));
    validateManifest(parseJson(manifestBytes, 'manifest.json'));
    const tokenizer = createEttinTokenizer(
      TokenizerClass,
      parseJson(tokenizerBytes, 'tokenizer.json'),
      parseJson(tokenizerConfigBytes, 'tokenizer_config.json')
    );
    signal?.throwIfAborted();

    const modelBytes = await source.read('model_int8.onnx');
    session = await createSession(ort, modelBytes, signal);
    validateSession(session);
    signal?.throwIfAborted();
    const runner = createEttinRunner(ort, session);
    const model = createEttinSession({runner, tokenizer});
    session = null;
    return model;
  } catch (cause) {
    await Promise.allSettled([session?.release()].filter(Boolean));
    signal?.throwIfAborted();
    if (
      cause instanceof WebDecisionError ||
      (cause as {readonly name?: unknown} | null | undefined)?.name === 'AbortError'
    )
      throw cause;
    throw fail('MODEL_LOAD_FAILED', `Could not load ${ETTIN_MODEL}.`, cause);
  }
}
