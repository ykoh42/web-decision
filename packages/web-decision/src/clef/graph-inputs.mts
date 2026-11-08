// Choice-only packed protocol to the experimental Clef graph's 11 user tensors.
import {validatePacked} from './protocol.mjs';
import type {ClefChoiceGraph, ClefQuestion, GraphInput, GraphInputData, GraphInputDType} from './types.mjs';

const mappings = new WeakMap<ClefChoiceGraph, number[]>();
function item(name: string, dtype: GraphInputDType, shape: number[], data: GraphInputData): GraphInput {
  return {name, dtype, shape, data};
}

function collectSpans(ids: number[], questions: ClefQuestion[]) {
  const gather: number[] = [],
    seen = new Map<number, number>(),
    spans: Array<[number, number]> = [],
    owners: number[] = [],
    offsets = [0];
  for (const [index, question] of questions.entries()) {
    for (const span of question.option_spans) {
      spans.push(span);
      owners.push(index);
      for (let pos = span[0]; pos < span[1]; pos++) {
        const id = ids[pos];
        if (id === undefined) throw Error('Invalid choice option span');
        if (!seen.has(id)) {
          seen.set(id, gather.length);
          gather.push(id);
        }
      }
    }
    offsets.push(spans.length);
  }
  return {gather, seen, spans, owners, offsets};
}

export function buildChoiceInputs(value: unknown): ClefChoiceGraph {
  const packed = structuredClone(value);
  validatePacked(packed);
  const ids = packed.ids,
    questions = packed.questions,
    n = ids.length,
    q = questions.length;
  const o = questions.reduce((sum, question) => sum + question.option_spans.length, 0);
  if (n < 4 || n > 512 || q < 1 || q > 8 || o < 2 || o > 64) throw Error('Choice request exceeds graph bounds');
  const {gather, seen, spans, owners, offsets} = collectSpans(ids, questions);
  const k = gather.length;
  if (k < 1 || k > 512) throw Error('Choice request exceeds graph gather bounds');
  const positions = Int32Array.from({length: n}, (_, index) => index);
  const attention = new Uint16Array(n * n),
    valid = new Uint16Array(n).fill(0x3c00);
  for (let row = 0; row < n; row++) attention.fill(0xfbff, row * n + row + 1, (row + 1) * n);
  const counts = new Float32Array(o * k),
    qmask = new Float32Array(q * n),
    omask = new Float32Array(o * n);
  const owner = new Float32Array(o * q),
    types = new Int32Array(q);
  for (const [index, question] of questions.entries()) {
    qmask.fill(1, index * n + question.question_span[0], index * n + question.question_span[1]);
    types[index] = question.question_type;
  }
  for (const [index, [start, end]] of spans.entries()) {
    omask.fill(1, index * n + start, index * n + end);
    const questionOwner = owners[index];
    if (questionOwner === undefined) throw Error('Invalid choice option owner');
    owner[index * q + questionOwner] = 1;
    for (let pos = start; pos < end; pos++) {
      const id = ids[pos];
      const gatherIndex = id === undefined ? undefined : seen.get(id);
      if (gatherIndex === undefined) throw Error('Invalid choice gather mapping');
      const countIndex = index * k + gatherIndex;
      counts[countIndex] = (counts[countIndex] ?? 0) + 1;
    }
  }
  const graph: ClefChoiceGraph = {
    packed,
    offsets,
    inputs: [
      item('ids', 'int32', [1, n], Int32Array.from(ids)),
      item('positions', 'int32', [1, n], positions),
      item('attention', 'float16', [1, 1, n, n], attention),
      item('valid', 'float16', [1, n], valid),
      item('gather_ids', 'int32', [1, k], Int32Array.from(gather)),
      item('occurrence_counts', 'float32', [o, k], counts),
      item('question_mask', 'float32', [q, n], qmask),
      item('option_mask', 'float32', [o, n], omask),
      item('owner', 'float32', [o, q], owner),
      item('question_types', 'int32', [q], types),
      item('last_index', 'int32', [1], Int32Array.of(n - 1))
    ]
  };
  mappings.set(graph, [...offsets]);
  return graph;
}

export function splitChoiceLogits(graph: ClefChoiceGraph, logits: unknown): Float32Array[] {
  const offsets = mappings.get(graph);
  if (!offsets || graph.offsets.length !== offsets.length || !offsets.every((value, i) => graph.offsets[i] === value))
    throw Error('Invalid graph output mapping');
  if (!(logits instanceof Float32Array) || logits.length !== offsets.at(-1) || logits.some((v) => !Number.isFinite(v)))
    throw Error('Invalid graph FP32 output');
  return offsets.slice(0, -1).map((start, index) => logits.slice(start, offsets[index + 1] ?? start));
}
