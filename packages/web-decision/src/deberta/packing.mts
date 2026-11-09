/** Experimental text packing for the pinned Open-Jev DeBERTa Q4 model. */
import {validateDebertaTensors} from './validate.mjs';
interface DebertaTokenizer {
  encode(text: string, options: {add_special_tokens: false}): {ids: Iterable<number>};
}
interface DebertaRequest {
  state: string;
  questions: readonly {instructions: string; options: readonly string[]}[];
}
interface DebertaPacked {
  pairs: number;
  realTokens: number;
  optionGroups: readonly (readonly string[])[];
}

const markers = Object.freeze({cls: 1, sep: 2, state: 128001, q: 128002, opt: 128003});
const reserved = new Set<number>(Object.values(markers));
export const DEBERTA_MAX_OPTIONS = 255;
const fail: (message: string) => never = (message) => {
  throw new RangeError(message);
};
function encode(tokenizer: DebertaTokenizer, text: string, label: string) {
  const ids = Array.from(tokenizer.encode(text, {add_special_tokens: false}).ids);
  if (ids.some((id) => reserved.has(id))) fail(`${label} contains a reserved model marker`);
  return ids;
}

/** No silent truncation: reject rather than change the submitted state or options. */
export function packDebertaRequest(tokenizer: DebertaTokenizer, request: DebertaRequest) {
  if (!request || typeof request !== 'object' || Array.isArray(request) || typeof request.state !== 'string')
    fail('A request with a string state is required');
  if (!Array.isArray(request.questions) || request.questions.length === 0) fail('At least one question is required');
  const state = encode(tokenizer, request.state, 'state');
  if (state.length > 256) fail('state exceeds 256 tokens');
  const questions = request.questions.map((q: DebertaRequest['questions'][number], i: number) => {
    if (!q || typeof q !== 'object' || typeof q.instructions !== 'string' || !q.instructions.trim())
      fail(`Question ${i} needs instructions`);
    if (!Array.isArray(q.options) || q.options.length < 2 || q.options.length > DEBERTA_MAX_OPTIONS)
      fail(`Question ${i} needs 2 to ${DEBERTA_MAX_OPTIONS} options`);
    const seen = new Set<string>(),
      options = q.options.map((option: unknown) => {
        if (typeof option !== 'string' || !option.trim() || seen.has(option))
          fail(`Question ${i} has an empty or duplicate option`);
        seen.add(option);
        const ids = encode(tokenizer, option, 'option');
        if (!ids.length || ids.length > 512) fail(`Question ${i} has an option with an invalid token length`);
        return {text: option, ids};
      });
    const instructions = encode(tokenizer, q.instructions, 'instructions');
    if (!instructions.length || instructions.length > 512)
      fail(`Question ${i} has instructions with an invalid token length`);
    return {instructions, options};
  });
  const pairs = questions.reduce((count, q) => count + q.options.length, 0);
  const length =
    3 +
    state.length +
    questions.reduce(
      (n, q) => n + 1 + q.instructions.length + q.options.reduce((sum, o) => sum + 1 + o.ids.length, 0),
      0
    );
  if (length > 512) fail('Packed input exceeds 512 tokens; no truncation was applied');
  const ids = [markers.cls, markers.state, ...state],
    seg = Array(ids.length).fill(-1),
    pairQ: number[] = [],
    pairOpt: number[] = [],
    groups: (readonly string[])[] = [];
  for (const [index, q] of questions.entries()) {
    const questionSlot = pairs + index;
    ids.push(markers.q, ...q.instructions);
    seg.push(-1, ...q.instructions.map(() => questionSlot));
    const group = [];
    for (const option of q.options) {
      const slot = pairOpt.length;
      ids.push(markers.opt, ...option.ids);
      seg.push(-1, ...option.ids.map(() => slot));
      pairQ.push(questionSlot);
      pairOpt.push(slot);
      group.push(option.text);
    }
    groups.push(Object.freeze(group));
  }
  ids.push(markers.sep);
  seg.push(-1);
  if (ids.length !== length) fail('Packed input length mismatch');
  const tensor = validateDebertaTensors({
    groups: groups.map((x) => x.length),
    inputs: {input_ids: ids, attention_mask: Array(ids.length).fill(1), seg, pair_q: pairQ, pair_opt: pairOpt}
  });
  return Object.freeze({...tensor, optionGroups: Object.freeze(groups)});
}

/** Interpret finite pair scores using the checkpoint's temperature; outputs are not calibrated confidence. */
export function unpackDebertaScores(packed: DebertaPacked, logits: unknown) {
  if (!Array.isArray(logits) || logits.length !== packed.pairs || logits.some((v) => !Number.isFinite(v)))
    fail('Invalid model scores');
  let offset = 0;
  return {
    tokenCount: packed.realTokens,
    answers: packed.optionGroups.map((options) => {
      const scores = logits.slice(offset, offset + options.length) as number[];
      offset += options.length;
      const max = Math.max(...scores),
        exps = scores.map((value) => Math.exp((value - max) / 1.05)),
        sum = exps.reduce((a, b) => a + b, 0),
        probs = exps.map((v) => v / sum),
        selectedIndex = probs.indexOf(Math.max(...probs));
      return {
        selected: options[selectedIndex]!,
        selectedIndex,
        probabilities: options.map((option, i) => ({option, probability: probs[i]!, logit: scores[i]!}))
      };
    })
  };
}
