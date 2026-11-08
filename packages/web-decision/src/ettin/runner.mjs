import {WebDecisionError} from '../kev.js';
import {ETTIN_MAX_OPTIONS, ETTIN_MAX_SCORE_LEVELS} from './contract.mjs';

const inferenceFailure = (cause) =>
  new WebDecisionError('INFERENCE_ERROR', 'Decider Ettin ONNX inference failed.', cause ? {cause} : undefined);

function logits(tensor, count, name) {
  if (
    tensor?.type !== 'float32' ||
    !Array.isArray(tensor.dims) ||
    tensor.dims.length !== 2 ||
    tensor.dims[0] !== 1 ||
    tensor.dims[1] !== count ||
    !ArrayBuffer.isView(tensor.data) ||
    tensor.data.length !== count
  )
    throw inferenceFailure(new TypeError(`Decider Ettin returned an invalid ${name} tensor.`));
  const values = Array.from(tensor.data);
  if (values.some((value) => !Number.isFinite(value)))
    throw inferenceFailure(new TypeError(`Decider Ettin returned non-finite ${name} logits.`));
  return values;
}

export function createEttinRunner(ort, session) {
  if (typeof ort?.Tensor !== 'function' || typeof session?.run !== 'function' || typeof session?.release !== 'function')
    throw new TypeError('Decider Ettin requires an ONNX Runtime Web session.');

  let closed = false;
  let disposing;

  return {
    async run(question) {
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
      let output;
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
