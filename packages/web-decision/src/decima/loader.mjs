import {Tokenizer} from '@huggingface/tokenizers';
import * as defaultOrt from 'onnxruntime-web/wasm';
import {parseJson, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {DECIMA_MODELS} from './contract.mjs';
import {createDecimaRunner} from './runner.mjs';
import {createDecimaSession} from './session.mjs';
import {openDecimaSource} from './source.mjs';
import {createDecimaTokenizer} from './tokenizer.mjs';

const LOAD_KEYS = new Set(['model', 'assetsUrl', 'cache', 'offline', 'signal', 'onProgress', 'maxPrefetchBytes']);
const fail = (code, message, cause) => new WebDecisionError(code, message, cause ? {cause} : undefined);

export function validateDecimaLoadOptions(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options))
    throw fail('INVALID_INPUT', 'Decima load options are required.');
  for (const key of Reflect.ownKeys(options)) {
    const descriptor = Object.getOwnPropertyDescriptor(options, key);
    if (typeof key !== 'string' || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, 'value'))
      throw fail('INVALID_INPUT', 'Unsupported Decima load option.');
  }
  const spec = DECIMA_MODELS[options.model];
  if (
    !spec ||
    (options.assetsUrl !== undefined && (typeof options.assetsUrl !== 'string' || !options.assetsUrl)) ||
    (options.cache !== undefined && typeof options.cache !== 'boolean') ||
    (options.offline !== undefined && typeof options.offline !== 'boolean') ||
    (options.offline && options.cache === false) ||
    (options.maxPrefetchBytes !== undefined && options.maxPrefetchBytes !== 0) ||
    (options.onProgress !== undefined && typeof options.onProgress !== 'function') ||
    (options.signal != null && typeof options.signal.throwIfAborted !== 'function')
  )
    throw fail('INVALID_INPUT', 'Invalid Decima load option values.');
  let base;
  try {
    const sourceUrl =
      options.assetsUrl ?? `https://huggingface.co/${spec.repository}/resolve/${spec.revision}/onnx/int8/`;
    base = new URL(sourceUrl, globalThis.location?.href);
  } catch (cause) {
    throw fail('INVALID_INPUT', 'Invalid Decima assetsUrl.', cause);
  }
  if (!['https:', 'http:'].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw fail('INVALID_INPUT', 'Decima assetsUrl must be an HTTP or HTTPS directory.');
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return {
    model: options.model,
    spec,
    base,
    signal: options.signal,
    cacheEnabled: options.cache !== false,
    offline: options.offline === true,
    onProgress: options.onProgress
  };
}

function validateConfig(config, spec) {
  if (
    config?.max_state_tokens !== spec.sourceMaxStateTokens ||
    config?.max_choice_tokens !== spec.maxChoiceTokens ||
    config?.temperature !== spec.temperature ||
    config?.question_in_state !== true ||
    config?.question_in_choices !== true ||
    typeof config?.state_prefix !== 'string' ||
    typeof config?.choice_prefix !== 'string' ||
    (config?.choice_fit ?? false) !== spec.choiceFit ||
    !Array.isArray(config.ordinal?.ord_g_w) ||
    !Array.isArray(config.ordinal?.ord_gap_w) ||
    config.ordinal.ord_g_w.length !== config.ordinal.ord_gap_w.length ||
    !Number.isFinite(config.ordinal.ord_g_b) ||
    !Number.isFinite(config.ordinal.ord_gap_b) ||
    [...config.ordinal.ord_g_w, ...config.ordinal.ord_gap_w].some((value) => !Number.isFinite(value))
  )
    throw fail('INVALID_ASSET', 'Decima configuration does not match the pinned model.');
  return {
    ...config,
    max_state_tokens: spec.maxStateTokens,
    choice_fit: spec.choiceFit,
    state_overflow: spec.stateOverflow,
    supported_question_types: spec.typedQuestionTypes
  };
}

async function createSession(ort, modelBytes, signal) {
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
  options,
  {ort = defaultOrt, Tokenizer: TokenizerClass = Tokenizer, openSource = openDecimaSource} = {}
) {
  const checked = validateDecimaLoadOptions(options);
  const {signal, spec} = checked;
  let encoder;
  let scorer;
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
    if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
    throw fail('MODEL_LOAD_FAILED', `Could not load ${spec.model}.`, cause);
  }
}
