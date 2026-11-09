import {WebDecisionError} from '../kev.js';
import {ETTIN_MAX_OPTIONS, ETTIN_MAX_SCORE_LEVELS} from './contract.mjs';
import type {InferenceSession, InferenceSessionFactory, Tensor} from 'onnxruntime-web/wasm';
import type {PackedEttinQuestion, EttinRunnerResult} from './protocol.mjs';

export interface EttinOrt {
  readonly Tensor: typeof Tensor;
  readonly InferenceSession: InferenceSessionFactory;
}

interface EttinTensorData {
  readonly type?: unknown;
  readonly dims?: unknown;
  readonly data?: unknown;
}

const inferenceFailure = (cause?: unknown) =>
  new WebDecisionError('INFERENCE_ERROR', 'Decider Ettin ONNX inference failed.', cause ? {cause} : undefined);

function logits(tensor: unknown, count: number, name: string): number[] {
  const candidate = tensor as EttinTensorData | null | undefined;
  if (
    candidate?.type !== 'float32' ||
    !Array.isArray(candidate.dims) ||
    candidate.dims.length !== 2 ||
    candidate.dims[0] !== 1 ||
    candidate.dims[1] !== count ||
    !ArrayBuffer.isView(candidate.data) ||
    (candidate.data as unknown as ArrayLike<number>).length !== count
  )
    throw inferenceFailure(new TypeError(`Decider Ettin returned an invalid ${name} tensor.`));
  const values = Array.from(candidate.data as unknown as ArrayLike<number>);
  if (values.some((value) => !Number.isFinite(value)))
    throw inferenceFailure(new TypeError(`Decider Ettin returned non-finite ${name} logits.`));
  return values;
}

export function createEttinRunner(ort: EttinOrt, session: InferenceSession) {
  if (typeof ort?.Tensor !== 'function' || typeof session?.run !== 'function' || typeof session?.release !== 'function')
    throw new TypeError('Decider Ettin requires an ONNX Runtime Web session.');

  let closed = false;
  let disposing: Promise<void> | undefined;

  return {
    async run(question: PackedEttinQuestion): Promise<EttinRunnerResult> {
      if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
      if (
        !Array.isArray(question?.inputIds) ||
        question.inputIds.length < 3 ||
        !Array.isArray(question.optionPositions) ||
        question.optionPositions.length !== ETTIN_MAX_OPTIONS ||
        !Array.isArray(question.optionMask) ||
        question.optionMask.length !== ETTIN_MAX_OPTIONS ||
        !Number.isSafeInteger(question.scoreCount) ||
        question.scoreCount < 0 ||
        question.scoreCount > ETTIN_MAX_SCORE_LEVELS
      )
        throw inferenceFailure(new TypeError('Decider Ettin received an invalid packed question.'));

      const ids = BigInt64Array.from(question.inputIds, BigInt);
      const mask = new BigInt64Array(ids.length).fill(1n);
      const optionMask = Uint8Array.from(question.optionMask, (value) => (value ? 1 : 0));
      let output: InferenceSession.ReturnType;
      try {
        output = await session.run({
          input_ids: new ort.Tensor('int64', ids, [1, ids.length]),
          attention_mask: new ort.Tensor('int64', mask, [1, ids.length]),
          decision_pos: new ort.Tensor('int64', BigInt64Array.of(0n), [1]),
          option_pos: new ort.Tensor('int64', BigInt64Array.from(question.optionPositions, BigInt), [
            1,
            ETTIN_MAX_OPTIONS
          ]),
          option_mask: new ort.Tensor('bool', optionMask, [1, ETTIN_MAX_OPTIONS]),
          score_n: new ort.Tensor('int64', BigInt64Array.of(BigInt(question.scoreCount)), [1])
        });
      } catch (cause) {
        throw inferenceFailure(cause);
      }

      return {
        choiceLogits: logits(output?.choice_logits, ETTIN_MAX_OPTIONS, 'choice'),
        noulLogits: logits(output?.noul_logits, 2, 'Noul'),
        scoreLogits: logits(output?.score_logits, ETTIN_MAX_SCORE_LEVELS, 'score')
      };
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = Promise.resolve()
        .then(() => session.release())
        .catch((cause) => {
          throw inferenceFailure(cause);
        });
      return disposing;
    }
  };
}
