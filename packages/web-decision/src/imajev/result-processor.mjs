// Test-only integration: no model loader, WebGPU route or asset verification.
import {IMAJEV_MAX_OPTIONS, compileTextChoice, decodeReadout} from './protocol.mjs';
import {safeImajevAnswer} from './safe-outcome.mjs';
import {WebDecisionError} from '../kev.js';
import {validateRequestFields} from '../request-fields.mjs';

const fail = (code, message, cause) => new WebDecisionError(code, message, cause ? {cause} : undefined);
const MAX_VOCAB = 248320;

function snapshotMetadata(manifest, calibration) {
  if (manifest?.version !== 1 || !Array.isArray(manifest.codes) || manifest.codes.length !== 256)
    throw fail('INVALID_ASSET', 'Invalid Imajev readout metadata.');
  const codes = Array.from(manifest.codes, (record) => ({code: record?.code, token_id: record?.token_id}));
  if (
    codes.some(
      ({code, token_id}) =>
        typeof code !== 'string' || !code || !Number.isSafeInteger(token_id) || token_id < 0 || token_id >= MAX_VOCAB
    ) ||
    new Set(codes.map((x) => x.code)).size !== 256 ||
    new Set(codes.map((x) => x.token_id)).size !== 256
  )
    throw fail('INVALID_ASSET', 'Invalid Imajev readout code bindings.');
  if (
    calibration?.schema_version !== '1.0' ||
    typeof calibration.calibration_version !== 'string' ||
    !calibration.calibration_version ||
    calibration.unknown_offsets !== undefined ||
    !calibration.temperatures ||
    typeof calibration.temperatures !== 'object' ||
    Array.isArray(calibration.temperatures)
  )
    throw fail('INVALID_ASSET', 'Invalid Imajev calibration metadata.');
  const temperatures = Object.fromEntries(Object.entries(calibration.temperatures));
  if (Object.values(temperatures).some((value) => !Number.isFinite(value) || value <= 0))
    throw fail('INVALID_ASSET', 'Invalid Imajev calibration temperature.');
  return {
    manifest: {version: 1, codes},
    calibration: {schema_version: '1.0', calibration_version: calibration.calibration_version, temperatures}
  };
}

/**
 * Turn package-shaped text requests into independent Imajev questions, then
 * decode one FP32 256-score row for each prepared question. Only pass assets
 * whose byte hashes and graph compatibility were checked by the future loader.
 * The configured maximum must come from the verified graph, not the model card.
 */
export function createImajevResultProcessor({tokenizer, manifest, calibration, maxInputTokens} = {}) {
  if (!tokenizer || typeof tokenizer.encode !== 'function') throw fail('INVALID_ASSET', 'A tokenizer is required.');
  if (!Number.isSafeInteger(maxInputTokens) || maxInputTokens < 2 || maxInputTokens > 4096)
    throw fail('INVALID_ASSET', 'A verified graph token limit between 2 and 4096 is required.');
  const metadata = snapshotMetadata(manifest, calibration);
  const preparedRequests = new WeakMap();
  const checkedTokenizer = {
    encode(text, options) {
      try {
        const values = tokenizer.encode(text, options)?.ids;
        if (!values || typeof values[Symbol.iterator] !== 'function') throw new Error('Missing token IDs');
        const ids = Array.from(values);
        if (ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= MAX_VOCAB))
          throw new Error('Invalid token ID');
        return {ids};
      } catch (cause) {
        throw fail('INVALID_ASSET', 'Tokenizer returned invalid input IDs.', cause);
      }
    }
  };

  function prepare(request) {
    request = validateRequestFields(request, 8, Infinity, Infinity, IMAJEV_MAX_OPTIONS);
    if (
      !request ||
      typeof request !== 'object' ||
      Array.isArray(request) ||
      typeof request.state !== 'string' ||
      !Array.isArray(request.questions) ||
      request.questions.length < 1 ||
      request.questions.length > 8
    )
      throw fail('INVALID_INPUT', 'Expected a text state and 1 to 8 choice questions.');
    const internal = [];
    for (const question of request.questions) {
      if (!question || typeof question !== 'object' || Array.isArray(question))
        throw fail('INVALID_INPUT', 'Each question must be an object.');
      let packed;
      try {
        packed = compileTextChoice(
          {state: request.state, question: question.instructions, options: question.options},
          checkedTokenizer,
          metadata.manifest,
          maxInputTokens
        );
      } catch (cause) {
        const assetError =
          cause?.code === 'INVALID_ASSET' ||
          /Tokenizer and readout binding|Invalid readout binding/i.test(String(cause?.message));
        throw fail(assetError ? 'INVALID_ASSET' : 'INVALID_INPUT', 'Could not prepare an Imajev question.', cause);
      }
      if (
        packed.inputIds.length < 2 ||
        packed.inputIds.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= MAX_VOCAB)
      )
        throw fail('INVALID_ASSET', 'Tokenizer returned invalid input IDs.');
      internal.push(packed);
    }
    const questions = Object.freeze(
      internal.map((item) =>
        Object.freeze({
          inputIds: item.inputIds,
          labelIds: item.labelIds,
          lastIndex: item.inputIds.length - 1
        })
      )
    );
    const prepared = Object.freeze({tokenCount: Math.max(...questions.map((q) => q.inputIds.length)), questions});
    preparedRequests.set(prepared, internal);
    return prepared;
  }

  function finish(prepared, rows) {
    const internal = preparedRequests.get(prepared);
    if (!internal) throw fail('INVALID_INPUT', 'Request was not prepared by this processor.');
    if (
      !Array.isArray(rows) ||
      rows.length !== internal.length ||
      rows.some(
        (row) =>
          !(row instanceof Float32Array) ||
          row.length !== 256 ||
          (typeof SharedArrayBuffer !== 'undefined' && row.buffer instanceof SharedArrayBuffer)
      )
    )
      throw fail('INFERENCE_ERROR', 'Expected one independent FP32 256-score row per question.');
    let answers;
    try {
      answers = internal.map((packed, i) =>
        safeImajevAnswer(decodeReadout(packed, new Float32Array(rows[i]), metadata.calibration))
      );
    } catch (cause) {
      throw fail('INFERENCE_ERROR', 'Invalid Imajev readout.', cause);
    }
    return Object.freeze({tokenCount: prepared.tokenCount, answers: Object.freeze(answers)});
  }
  return Object.freeze({prepare, finish});
}
