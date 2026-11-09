import {IMAJEV_MAX_OPTIONS, compileTextChoice, decodeReadout} from './protocol.mjs';
import {safeImajevAnswer, type ImajevAnswer, type ImajevAnswerProbability} from './safe-outcome.mjs';
import {WebDecisionError} from '../kev.js';
import {validateRequestFields} from '../request-fields.mjs';

export type {ImajevAnswer, ImajevAnswerProbability};

interface ImajevTokenizerEncoding {
  readonly ids?: unknown;
}

interface ImajevTokenizer {
  encode(text: string, options: {readonly add_special_tokens: false}): ImajevTokenizerEncoding | null | undefined;
}

interface ImajevReadoutCodeCandidate {
  readonly code?: unknown;
  readonly token_id?: unknown;
}

interface ImajevReadoutCode {
  readonly code: string;
  readonly token_id: number;
}

interface ImajevReadoutManifest {
  readonly version: 1;
  readonly codes: readonly ImajevReadoutCode[];
}

interface ImajevCalibrationCandidate {
  readonly schema_version?: unknown;
  readonly calibration_version?: unknown;
  readonly unknown_offsets?: unknown;
  readonly temperatures?: unknown;
}

interface ImajevCalibration {
  readonly schema_version: '1.0';
  readonly calibration_version: string;
  readonly temperatures: Readonly<Record<string, number>>;
}

interface ImajevMetadata {
  readonly manifest: ImajevReadoutManifest;
  readonly calibration: ImajevCalibration;
}

interface ImajevRequestCandidate {
  readonly state?: unknown;
  readonly questions?: unknown;
}

interface ImajevQuestionCandidate {
  readonly instructions?: unknown;
  readonly options?: unknown;
}

interface CompiledTextChoice {
  readonly prompt: string;
  readonly rendered: string;
  readonly inputIds: readonly number[];
  readonly labelIds: readonly number[];
  readonly options: readonly string[];
  readonly unknownIndex: number;
}

export interface PreparedImajevQuestion {
  readonly inputIds: readonly number[];
  readonly labelIds: readonly number[];
  readonly lastIndex: number;
}

export interface PreparedImajevRequest {
  readonly tokenCount: number;
  readonly questions: readonly PreparedImajevQuestion[];
}

export interface ImajevProcessedResult {
  readonly tokenCount: number;
  readonly answers: readonly ImajevAnswer[];
}

export interface ImajevResultProcessor {
  prepare(request: unknown): PreparedImajevRequest;
  finish(prepared: PreparedImajevRequest, rows: unknown): ImajevProcessedResult;
}

interface ImajevResultProcessorOptions {
  readonly tokenizer?: unknown;
  readonly manifest?: unknown;
  readonly calibration?: unknown;
  readonly maxInputTokens?: unknown;
}

const fail = (code: 'INVALID_ASSET' | 'INVALID_INPUT' | 'INFERENCE_ERROR', message: string, cause?: unknown) =>
  new WebDecisionError(code, message, cause ? {cause} : undefined);
const MAX_VOCAB = 248320;

function snapshotMetadata(manifest: unknown, calibration: unknown): ImajevMetadata {
  if (
    (manifest as {readonly version?: unknown} | null | undefined)?.version !== 1 ||
    !Array.isArray((manifest as {readonly codes?: unknown} | null | undefined)?.codes) ||
    (manifest as {readonly codes: unknown[]}).codes.length !== 256
  )
    throw fail('INVALID_ASSET', 'Invalid Imajev readout metadata.');
  const codes = Array.from(
    (manifest as {readonly codes: unknown[]}).codes,
    (rawRecord): {code: unknown; token_id: unknown} => {
      const record = rawRecord as ImajevReadoutCodeCandidate | null | undefined;
      return {code: record?.code, token_id: record?.token_id};
    }
  );
  if (
    codes.some(
      ({code, token_id}) =>
        typeof code !== 'string' ||
        !code ||
        !Number.isSafeInteger(token_id) ||
        (token_id as number) < 0 ||
        (token_id as number) >= MAX_VOCAB
    ) ||
    new Set(codes.map((x) => x.code)).size !== 256 ||
    new Set(codes.map((x) => x.token_id)).size !== 256
  )
    throw fail('INVALID_ASSET', 'Invalid Imajev readout code bindings.');
  const calibrationCandidate = calibration as ImajevCalibrationCandidate | null | undefined;
  if (
    calibrationCandidate?.schema_version !== '1.0' ||
    typeof calibrationCandidate.calibration_version !== 'string' ||
    !calibrationCandidate.calibration_version ||
    calibrationCandidate.unknown_offsets !== undefined ||
    !calibrationCandidate.temperatures ||
    typeof calibrationCandidate.temperatures !== 'object' ||
    Array.isArray(calibrationCandidate.temperatures)
  )
    throw fail('INVALID_ASSET', 'Invalid Imajev calibration metadata.');
  const temperatures = Object.fromEntries(Object.entries(calibrationCandidate.temperatures));
  if (Object.values(temperatures).some((value: unknown) => !Number.isFinite(value) || (value as number) <= 0))
    throw fail('INVALID_ASSET', 'Invalid Imajev calibration temperature.');
  return {
    manifest: {version: 1, codes: codes as ImajevReadoutCode[]},
    calibration: {
      schema_version: '1.0',
      calibration_version: calibrationCandidate.calibration_version,
      temperatures: temperatures as Readonly<Record<string, number>>
    }
  };
}

/**
 * Turn package-shaped text requests into independent Imajev questions, then
 * decode one FP32 256-score row for each prepared question. Only pass assets
 * whose byte hashes and graph compatibility were checked by the future loader.
 * The configured maximum must come from the verified graph, not the model card.
 */
export function createImajevResultProcessor({
  tokenizer,
  manifest,
  calibration,
  maxInputTokens
}: ImajevResultProcessorOptions = {}): ImajevResultProcessor {
  if (!tokenizer || typeof (tokenizer as ImajevTokenizer).encode !== 'function')
    throw fail('INVALID_ASSET', 'A tokenizer is required.');
  if (!Number.isSafeInteger(maxInputTokens) || (maxInputTokens as number) < 2 || (maxInputTokens as number) > 4096)
    throw fail('INVALID_ASSET', 'A verified graph token limit between 2 and 4096 is required.');
  const metadata = snapshotMetadata(manifest, calibration);
  const preparedRequests = new WeakMap<PreparedImajevRequest, CompiledTextChoice[]>();
  const checkedTokenizer: ImajevTokenizer = {
    encode(text, options) {
      try {
        const values = (tokenizer as ImajevTokenizer).encode(text, options)?.ids;
        if (!values || typeof (values as {readonly [Symbol.iterator]?: unknown})[Symbol.iterator] !== 'function')
          throw new Error('Missing token IDs');
        const ids = Array.from(values as Iterable<unknown>);
        if (ids.some((id) => !Number.isSafeInteger(id) || (id as number) < 0 || (id as number) >= MAX_VOCAB))
          throw new Error('Invalid token ID');
        return {ids: ids as number[]};
      } catch (cause) {
        throw fail('INVALID_ASSET', 'Tokenizer returned invalid input IDs.', cause);
      }
    }
  };

  function prepare(request: unknown): PreparedImajevRequest {
    request = validateRequestFields(request, 8, Infinity, Infinity, IMAJEV_MAX_OPTIONS) as ImajevRequestCandidate;
    if (
      !request ||
      typeof request !== 'object' ||
      Array.isArray(request) ||
      typeof (request as ImajevRequestCandidate).state !== 'string' ||
      !Array.isArray((request as ImajevRequestCandidate).questions) ||
      (request as {readonly questions: unknown[]}).questions.length < 1 ||
      (request as {readonly questions: unknown[]}).questions.length > 8
    )
      throw fail('INVALID_INPUT', 'Expected a text state and 1 to 8 choice questions.');
    const internal: CompiledTextChoice[] = [];
    for (const question of (request as ImajevRequestCandidate).questions as unknown[]) {
      if (!question || typeof question !== 'object' || Array.isArray(question))
        throw fail('INVALID_INPUT', 'Each question must be an object.');
      let packed: CompiledTextChoice;
      try {
        packed = compileTextChoice(
          {
            state: (request as ImajevRequestCandidate).state,
            question: (question as ImajevQuestionCandidate).instructions,
            options: (question as ImajevQuestionCandidate).options
          },
          checkedTokenizer,
          metadata.manifest,
          maxInputTokens as number
        ) as CompiledTextChoice;
      } catch (cause: unknown) {
        const assetError =
          (cause as {readonly code?: unknown} | null | undefined)?.code === 'INVALID_ASSET' ||
          /Tokenizer and readout binding|Invalid readout binding/i.test(
            String((cause as {readonly message?: unknown} | null | undefined)?.message)
          );
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

  function finish(prepared: PreparedImajevRequest, rows: unknown): ImajevProcessedResult {
    const internal = preparedRequests.get(prepared);
    if (!internal) throw fail('INVALID_INPUT', 'Request was not prepared by this processor.');
    if (
      !Array.isArray(rows) ||
      rows.length !== internal.length ||
      rows.some(
        (row: unknown) =>
          !(row instanceof Float32Array) ||
          row.length !== 256 ||
          (typeof SharedArrayBuffer !== 'undefined' && row.buffer instanceof SharedArrayBuffer)
      )
    )
      throw fail('INFERENCE_ERROR', 'Expected one independent FP32 256-score row per question.');
    let answers: ImajevAnswer[];
    try {
      answers = internal.map((packed, i) =>
        safeImajevAnswer(
          decodeReadout(packed, new Float32Array((rows as unknown[])[i] as Float32Array), metadata.calibration)
        )
      );
    } catch (cause) {
      throw fail('INFERENCE_ERROR', 'Invalid Imajev readout.', cause);
    }
    return Object.freeze({tokenCount: prepared.tokenCount, answers: Object.freeze(answers)});
  }
  return Object.freeze({prepare, finish});
}
