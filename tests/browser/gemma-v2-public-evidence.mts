import assert from 'node:assert/strict';
import {sha256} from '../../scripts/release/curated-gpu-evidence.mjs';
import {
  GEMMA_V2_OFFICIAL_GOLDEN_SHA256,
  readPinnedGemma4V2Subset
} from '../../scripts/release/strands-gemma4-v2-conformance.mjs';

export const gemmaV2PublicSpec = 'tests/browser/gemma-v2-public-real-webgpu.spec.ts';
export const gemmaV2PublicTitle = 'the exact public Gemma v2 archive records official subset and six upstream cases';
export const gemmaV2PublicRunAttachment = 'gemma-v2-public-run.json';
export const gemmaV2PublicVectorsAttachment = 'gemma-v2-public-code-probabilities.jsonl';
export const gemmaV2PublicProbabilityEncoding = 'system-one-four-decimal';

type Question = {
  type: 'choice' | 'noul' | 'score';
  criteria: Record<string, unknown> | readonly string[] | null;
};
type GoldenRequest = {
  request: {questions: Record<string, Question>; images?: readonly string[]};
  variant: string;
  state_ids: number[];
  questions: Record<string, {codes: string[]; question_ids: number[]}>;
};
type PublicAnswer = {type: 'noul'; noul: number} | {type: 'choice' | 'score'; probabilities: Record<string, number>};
export type PublicGemmaV2Result = {
  model: string;
  answers: Record<string, PublicAnswer>;
  usage: {input_tokens: number; output_tokens: number};
};

export function readGemmaV2PublicGoldenRequests(golden: Uint8Array, subset: Uint8Array) {
  assert.equal(sha256(golden), GEMMA_V2_OFFICIAL_GOLDEN_SHA256, 'Official checkpoint golden bytes differ.');
  const lines = new TextDecoder().decode(golden).split(/\r?\n/u);
  return readPinnedGemma4V2Subset(subset).map((line): {line: number; golden: GoldenRequest} => {
    const text = lines[line];
    assert.ok(text, `Missing official golden row ${line}.`);
    const request = JSON.parse(text) as GoldenRequest;
    assert.equal(request.variant, 'strands-chat');
    return {line, golden: request};
  });
}

function labels(question: Question): string[] {
  if (question.type === 'noul') return ['false', 'true'];
  assert.ok(question.criteria);
  if (question.type === 'score') {
    assert.ok(Array.isArray(question.criteria));
    return question.criteria.map((_, index) => String(index));
  }
  assert.equal(question.type, 'choice');
  assert.ok(!Array.isArray(question.criteria));
  return Object.keys(question.criteria);
}

function codeProbabilities(question: Question, answer: PublicAnswer) {
  assert.equal(answer.type, question.type);
  const names = labels(question);
  if (answer.type !== 'noul') {
    assert.deepEqual(Object.keys(answer.probabilities), names, 'Public probability labels or order differ.');
    return names.map((name) => answer.probabilities[name]!);
  }
  // System One gives only the rounded true probability for a noul answer.
  // Recover the displayed complement, not the raw false probability.
  return [(10_000 - Math.round(answer.noul * 10_000)) / 10_000, answer.noul];
}

export function gemmaV2PublicCodeRow(line: number, golden: GoldenRequest, result: PublicGemmaV2Result) {
  const questions = Object.entries(golden.request.questions);
  const names = questions.map(([name]) => name);
  assert.deepEqual(Object.keys(golden.questions), names, 'Official question order differs.');
  assert.deepEqual(Object.keys(result.answers), names, 'Public question order differs.');
  const inputTokens = Object.values(golden.questions).reduce(
    (total, {question_ids}) => total + question_ids.length,
    golden.state_ids.length
  );
  assert.deepEqual(result.usage, {input_tokens: inputTokens, output_tokens: names.length});
  return {
    line,
    questions: Object.fromEntries(
      questions.map(([name, question]) => {
        const {codes} = golden.questions[name]!;
        const answer = result.answers[name];
        assert.ok(answer);
        const probabilities = codeProbabilities(question, answer);
        assert.equal(probabilities.length, codes.length);
        assert.ok(probabilities.every((value) => Number.isFinite(value) && value >= 0 && value <= 1));
        return [name, {codes, probabilities}];
      })
    )
  };
}
