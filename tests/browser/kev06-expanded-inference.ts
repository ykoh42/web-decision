import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import type {
  DecisionRequest,
  KevSystemOneQuestion,
  KevSystemOneResult,
  KevSystemOneAnswer
} from '../../packages/web-decision/src/index.d.ts';

export const EXPANDED_INFERENCE_FIXTURE_PATH = 'tests/fixtures/kev06-expanded-inference.json';
export const EXPANDED_INFERENCE_FIXTURE_SHA256 = 'e42c5882958ac78374f138a41f2eed78245810845d1f0fbf5af76188d669088b';
export const EXPANDED_DECISION_CASE_IDS = [
  'baseline-double-charge',
  'account-security',
  'delayed-package',
  'long-tracking-context'
] as const;
export const EXPANDED_TYPED_SYSTEM_ONE_ID = 'typed-system-one-probe';
export const EXPANDED_TYPED_QUESTION_NAMES = ['team', 'refund', 'sentiment'] as const;
export const EXPANDED_TYPED_QUESTION_TYPES = ['choice', 'noul', 'score'] as const;
export const EXPANDED_TOKENIZER_SHA256 = 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539';

const MODEL_ID = 'kev-0.6b-q4f16';
const TYPED_DECIMAL_PLACES = 4;
const TYPED_ROUNDING_UNIT = 10 ** -TYPED_DECIMAL_PLACES;

export interface ReferenceOutput {
  readonly options: readonly string[];
  readonly selected: string;
  readonly logits: readonly number[];
  readonly probabilities: readonly number[];
}

export interface RepresentativeDecisionCase {
  readonly id: string;
  readonly request: DecisionRequest;
  readonly tokenIds: readonly number[];
  readonly tokenCount: number;
  readonly optionIndices: readonly number[];
  readonly reference: ReferenceOutput;
}

export interface TypedReferenceAnswer extends ReferenceOutput {
  readonly optionIndices: readonly number[];
}

export interface TypedReferenceRun {
  readonly tokenIds: readonly number[];
  readonly tokenCount: number;
  readonly answers: Readonly<Record<string, TypedReferenceAnswer>>;
}

export interface ExpandedInferenceFixture {
  readonly schemaVersion: 1;
  readonly scope: string;
  readonly modelId: string;
  readonly tokenizerSha256: string;
  readonly sourceGraphSha256: string;
  readonly sourceWeightsSha256: string;
  readonly referenceRuntime: Readonly<{
    name: string;
    provider: readonly string[];
    graphOptimization: string;
    intraOpThreads: number;
  }>;
  readonly sourceEvidenceSha256: Readonly<Record<string, string>>;
  readonly decisionCases: readonly RepresentativeDecisionCase[];
  readonly typedSystemOne: Readonly<{
    id: string;
    state: string;
    questions: Readonly<Record<string, KevSystemOneQuestion>>;
    packed: TypedReferenceRun;
    single: Readonly<Record<string, TypedReferenceRun>>;
  }>;
}

export interface DecisionInferenceObservation {
  readonly id: string;
  readonly tokenCount: number;
  readonly selected: string;
  readonly selectedIndex: number;
  readonly probabilities: readonly Readonly<{option: string; logit: number; probability: number}>[];
}

export interface ExpandedInferenceObservation {
  readonly decisionRuns: readonly DecisionInferenceObservation[];
  readonly typedSystemOne: Readonly<{
    packed: KevSystemOneResult<'kev-0.6b-q4f16'>;
    single: Readonly<Record<string, KevSystemOneResult<'kev-0.6b-q4f16'>>>;
  }>;
}

export interface ExpandedInferencePolicy {
  readonly fixturePath: string;
  readonly fixtureSha256: string;
  readonly decisionCaseIds: readonly string[];
  readonly typedSystemOneId: string;
  readonly typedQuestionNames: readonly string[];
  readonly typedQuestionTypes: readonly string[];
}

export function validateExpandedInferencePolicy(value: unknown): ExpandedInferencePolicy {
  const policy = asRecord(value, 'Expanded inference policy');
  assert.equal(policy.fixturePath, EXPANDED_INFERENCE_FIXTURE_PATH, 'Expanded inference fixture path is not pinned.');
  assert.equal(
    policy.fixtureSha256,
    EXPANDED_INFERENCE_FIXTURE_SHA256,
    'Expanded inference fixture digest is not pinned.'
  );
  assert.deepEqual(policy.decisionCaseIds, EXPANDED_DECISION_CASE_IDS, 'Expanded decision case matrix is not pinned.');
  assert.equal(policy.typedSystemOneId, EXPANDED_TYPED_SYSTEM_ONE_ID, 'Typed System One case ID is not pinned.');
  assert.deepEqual(
    policy.typedQuestionNames,
    EXPANDED_TYPED_QUESTION_NAMES,
    'Typed System One question names are not pinned.'
  );
  assert.deepEqual(
    policy.typedQuestionTypes,
    EXPANDED_TYPED_QUESTION_TYPES,
    'Typed System One question types are not pinned.'
  );
  return {
    fixturePath: EXPANDED_INFERENCE_FIXTURE_PATH,
    fixtureSha256: EXPANDED_INFERENCE_FIXTURE_SHA256,
    decisionCaseIds: [...EXPANDED_DECISION_CASE_IDS],
    typedSystemOneId: EXPANDED_TYPED_SYSTEM_ONE_ID,
    typedQuestionNames: [...EXPANDED_TYPED_QUESTION_NAMES],
    typedQuestionTypes: [...EXPANDED_TYPED_QUESTION_TYPES]
  };
}

export function loadExpandedInferenceFixture(
  workspaceRoot: string,
  policy: ExpandedInferencePolicy
): {fixture: ExpandedInferenceFixture; fixtureSha256: string} {
  const path = resolve(workspaceRoot, policy.fixturePath);
  const bytes = readFileSync(path);
  const digest = sha256(bytes);
  assert.equal(digest, policy.fixtureSha256, 'Expanded inference fixture digest mismatch.');
  const fixture = JSON.parse(bytes.toString('utf8')) as ExpandedInferenceFixture;
  validateExpandedInferenceFixture(fixture);
  return {fixture, fixtureSha256: digest};
}

export function validateExpandedInferenceFixture(fixture: ExpandedInferenceFixture): void {
  assert.equal(fixture.schemaVersion, 1, 'Unsupported expanded inference fixture schema.');
  assert.equal(
    fixture.scope,
    'pinned-input-and-reference-observations; additional outputs are descriptive, not accuracy thresholds'
  );
  assert.equal(fixture.modelId, MODEL_ID, 'Expanded inference fixture model ID mismatch.');
  assert.equal(fixture.tokenizerSha256, EXPANDED_TOKENIZER_SHA256, 'Expanded inference tokenizer pin is invalid.');
  assert.match(fixture.sourceGraphSha256, /^[a-f0-9]{64}$/u, 'Expanded inference ONNX graph pin is invalid.');
  assert.match(fixture.sourceWeightsSha256, /^[a-f0-9]{64}$/u, 'Expanded inference ONNX weights pin is invalid.');
  assert.equal(fixture.referenceRuntime.name, 'onnxruntime 1.31.1 CPUExecutionProvider');
  assert.deepEqual(fixture.referenceRuntime.provider, ['CPUExecutionProvider']);
  assert.equal(fixture.referenceRuntime.graphOptimization, 'ORT_ENABLE_ALL');
  assert.equal(fixture.referenceRuntime.intraOpThreads, 4);
  for (const [name, digest] of Object.entries(fixture.sourceEvidenceSha256)) {
    assert.ok(name.length > 0 && /^[a-f0-9]{64}$/u.test(digest), `Invalid expanded evidence source hash: ${name}.`);
  }
  assert.deepEqual(Object.keys(fixture.sourceEvidenceSha256).sort(), [
    'representativeInputs',
    'stableAndRebuildBrowserObservations',
    'typedCpuReference',
    'typedWindowsGpuObservation'
  ]);
  assert.deepEqual(
    fixture.decisionCases.map(({id}) => id),
    EXPANDED_DECISION_CASE_IDS
  );
  for (const item of fixture.decisionCases) validateDecisionFixtureCase(item);
  validateTypedFixture(fixture.typedSystemOne);
}

export function createExpandedInferenceEvidence(
  fixture: ExpandedInferenceFixture,
  fixtureSha256: string,
  expectedManifestSha256: string,
  observed: ExpandedInferenceObservation
) {
  assert.equal(fixtureSha256, EXPANDED_INFERENCE_FIXTURE_SHA256);
  assert.equal(
    observed.decisionRuns.length,
    fixture.decisionCases.length,
    'Browser returned an incomplete decision matrix.'
  );
  assert.deepEqual(
    Object.keys(observed.typedSystemOne.single),
    EXPANDED_TYPED_QUESTION_NAMES,
    'Browser returned an incomplete typed single-question matrix.'
  );
  const decisionCases = fixture.decisionCases.map((item, index) => {
    const inference = observed.decisionRuns[index];
    assert.ok(inference, `Browser omitted expanded decision run ${item.id}.`);
    const logits = inference.probabilities.map(({logit}) => logit);
    const probabilities = inference.probabilities.map(({probability}) => probability);
    return {
      id: item.id,
      request: item.request,
      observed: {
        tokenCount: inference.tokenCount,
        selected: inference.selected,
        selectedIndex: inference.selectedIndex,
        optionOrder: inference.probabilities.map(({option}) => option),
        logits,
        probabilities
      },
      reference: decisionReference(item),
      comparison: {
        maxAbsoluteLogitDifference: maxAbsoluteDifference(logits, item.reference.logits),
        maxAbsoluteProbabilityDifference: maxAbsoluteDifference(probabilities, item.reference.probabilities)
      }
    };
  });
  const typedFixture = fixture.typedSystemOne;
  const typedObserved = observed.typedSystemOne;
  const packed = typedRunEvidence(typedFixture.packed, typedObserved.packed);
  const single = Object.fromEntries(
    EXPANDED_TYPED_QUESTION_NAMES.map((name) => {
      const expected = typedFixture.single[name];
      const actual = typedObserved.single[name];
      assert.ok(expected && actual, `Browser omitted typed single-question run ${name}.`);
      return [name, typedRunEvidence(expected, actual)];
    })
  );
  return {
    scope: fixture.scope,
    fixtureSha256,
    modelId: fixture.modelId,
    tokenizerSha256: fixture.tokenizerSha256,
    sourceGraphSha256: fixture.sourceGraphSha256,
    sourceWeightsSha256: fixture.sourceWeightsSha256,
    sourceEvidenceSha256: fixture.sourceEvidenceSha256,
    referenceRuntime: fixture.referenceRuntime,
    decisionCases,
    typedSystemOne: {
      id: typedFixture.id,
      modelId: MODEL_ID,
      backend: 'tvm-webgpu',
      assetManifestSha256: expectedManifestSha256,
      state: typedFixture.state,
      questions: typedFixture.questions,
      packed,
      single
    }
  };
}

export function validateExpandedInferenceEvidence(
  value: unknown,
  fixture: ExpandedInferenceFixture,
  fixtureSha256: string,
  expectedManifestSha256: string
): void {
  const evidence = asRecord(value, 'Expanded inference evidence');
  assert.equal(evidence.scope, fixture.scope, 'Expanded inference scope mismatch.');
  assert.equal(evidence.fixtureSha256, fixtureSha256, 'Expanded inference fixture pin mismatch.');
  assert.equal(evidence.modelId, MODEL_ID, 'Expanded inference model ID mismatch.');
  assert.equal(evidence.tokenizerSha256, fixture.tokenizerSha256, 'Expanded inference tokenizer mismatch.');
  assert.equal(evidence.sourceGraphSha256, fixture.sourceGraphSha256, 'Expanded inference graph pin mismatch.');
  assert.equal(evidence.sourceWeightsSha256, fixture.sourceWeightsSha256, 'Expanded inference weights pin mismatch.');
  assert.deepEqual(
    evidence.sourceEvidenceSha256,
    fixture.sourceEvidenceSha256,
    'Expanded source evidence hashes mismatch.'
  );
  assert.deepEqual(
    evidence.referenceRuntime,
    fixture.referenceRuntime,
    'Expanded inference runtime reference mismatch.'
  );
  validateDecisionEvidence(evidence.decisionCases, fixture.decisionCases);
  validateTypedEvidence(evidence.typedSystemOne, fixture.typedSystemOne, expectedManifestSha256);
}

function validateDecisionFixtureCase(item: RepresentativeDecisionCase): void {
  assert.ok(item.id.length > 0, 'Expanded inference case ID is empty.');
  assert.equal(item.request.questions.length, 1, `Expanded case ${item.id} must contain one decision question.`);
  const options = item.request.questions[0]?.options;
  assert.ok(options && options.length > 0, `Expanded case ${item.id} has no options.`);
  assert.equal(item.tokenCount, item.tokenIds.length, `Expanded case ${item.id} token count mismatch.`);
  assert.ok(item.tokenIds.every(Number.isSafeInteger), `Expanded case ${item.id} token IDs are invalid.`);
  assert.equal(item.optionIndices.length, options.length, `Expanded case ${item.id} option index count mismatch.`);
  assert.ok(item.optionIndices.every((index) => Number.isSafeInteger(index) && index >= 0 && index < item.tokenCount));
  assert.deepEqual(item.reference.options, options, `Expanded case ${item.id} reference option order mismatch.`);
  validateReferenceOutput(item.reference, item.id);
}

function validateTypedFixture(typed: ExpandedInferenceFixture['typedSystemOne']): void {
  assert.equal(typed.id, EXPANDED_TYPED_SYSTEM_ONE_ID, 'Typed System One fixture ID mismatch.');
  assert.deepEqual(Object.keys(typed.questions), EXPANDED_TYPED_QUESTION_NAMES);
  assert.deepEqual(
    EXPANDED_TYPED_QUESTION_NAMES.map((name) => typed.questions[name]?.type),
    EXPANDED_TYPED_QUESTION_TYPES
  );
  validateTypedReferenceRun(typed.packed, typed.questions, 'packed');
  assert.deepEqual(Object.keys(typed.single), EXPANDED_TYPED_QUESTION_NAMES);
  for (const name of EXPANDED_TYPED_QUESTION_NAMES) {
    const question = typed.questions[name];
    const single = typed.single[name];
    assert.ok(question && single, `Typed single-question fixture is missing ${name}.`);
    validateTypedReferenceRun(single, {[name]: question}, `single ${name}`);
    assert.deepEqual(Object.keys(single.answers), [name], `Typed single reference includes unrelated answer ${name}.`);
  }
}

function validateTypedReferenceRun(
  run: TypedReferenceRun,
  questions: Readonly<Record<string, KevSystemOneQuestion>>,
  description: string
): void {
  assert.equal(run.tokenCount, run.tokenIds.length, `${description} reference token count mismatch.`);
  assert.ok(run.tokenIds.every(Number.isSafeInteger), `${description} reference token IDs are invalid.`);
  const questionEntries = Object.entries(questions);
  assert.deepEqual(
    Object.keys(run.answers),
    questionEntries.map(([name]) => name)
  );
  for (const [name, question] of questionEntries) {
    const answer = run.answers[name];
    assert.ok(answer, `${description} reference answer ${name} is missing.`);
    const expectedOptions = optionsForQuestion(name, question);
    assert.deepEqual(answer.options, expectedOptions, `${description} reference options mismatch for ${name}.`);
    assert.equal(answer.optionIndices.length, expectedOptions.length);
    assert.ok(
      answer.optionIndices.every((index) => Number.isSafeInteger(index) && index >= 0 && index < run.tokenCount)
    );
    validateReferenceOutput(answer, `${description} ${name}`);
  }
}

function optionsForQuestion(name: string, question: KevSystemOneQuestion): readonly string[] {
  if (question.type === 'choice') return Object.keys(question.criteria);
  if (question.type === 'noul') return ['false', 'true'];
  return question.criteria.map((_, index) => String(index));
}

function validateReferenceOutput(reference: ReferenceOutput, description: string): void {
  const count = reference.options.length;
  assert.ok(count > 0, `${description} reference has no options.`);
  assert.equal(reference.logits.length, count, `${description} reference logit length mismatch.`);
  assert.equal(reference.probabilities.length, count, `${description} reference probability length mismatch.`);
  assert.ok(reference.logits.every(Number.isFinite), `${description} reference has a non-finite logit.`);
  validateProbabilityVector(reference.probabilities, false, `${description} reference probabilities`);
  assert.equal(reference.selected, selectMaximum(reference.options, reference.probabilities));
}

function validateDecisionEvidence(value: unknown, expected: readonly RepresentativeDecisionCase[]): void {
  assert.ok(Array.isArray(value), 'Expanded decision evidence is missing.');
  const observed = value as unknown[];
  assert.deepEqual(
    observed.map((item) => asRecord(item, 'Expanded decision case').id),
    expected.map(({id}) => id)
  );
  for (let index = 0; index < expected.length; index++) {
    const fixtureCase = expected[index];
    const evidenceCase = asRecord(observed[index], 'Expanded decision case');
    assert.ok(fixtureCase, `Expanded decision fixture case ${index} is missing.`);
    validateOneDecisionEvidence(evidenceCase, fixtureCase);
  }
}

function validateOneDecisionEvidence(evidence: Record<string, unknown>, fixture: RepresentativeDecisionCase): void {
  assert.equal(evidence.id, fixture.id, `Expanded decision ID mismatch for ${fixture.id}.`);
  assert.deepEqual(evidence.request, fixture.request, `Expanded decision request mismatch for ${fixture.id}.`);
  const expectedReference = decisionReference(fixture);
  assert.deepEqual(evidence.reference, expectedReference, `Expanded CPU reference mismatch for ${fixture.id}.`);
  const observed = asRecord(evidence.observed, `Observed decision ${fixture.id}`);
  assert.equal(observed.tokenCount, fixture.tokenCount, `Decision token count mismatch for ${fixture.id}.`);
  const options = fixture.request.questions[0]?.options ?? [];
  assert.deepEqual(observed.optionOrder, options, `Decision option order mismatch for ${fixture.id}.`);
  const logits = numberVector(observed.logits, `Decision logits for ${fixture.id}`, options.length);
  const probabilities = numberVector(
    observed.probabilities,
    `Decision probabilities for ${fixture.id}`,
    options.length
  );
  validateProbabilityVector(probabilities, false, `Decision probabilities for ${fixture.id}`);
  const selected = requiredText(observed.selected, `Decision selection for ${fixture.id}`);
  const selectedIndex = selectMaximumIndex(probabilities);
  assert.equal(observed.selectedIndex, selectedIndex, `Decision selected index is not argmax for ${fixture.id}.`);
  assert.equal(selected, options[selectedIndex], `Decision selected option is not argmax for ${fixture.id}.`);
  assert.deepEqual(
    evidence.comparison,
    {
      maxAbsoluteLogitDifference: maxAbsoluteDifference(logits, fixture.reference.logits),
      maxAbsoluteProbabilityDifference: maxAbsoluteDifference(probabilities, fixture.reference.probabilities)
    },
    `Decision comparison observation mismatch for ${fixture.id}.`
  );
}

function validateTypedEvidence(
  value: unknown,
  fixture: ExpandedInferenceFixture['typedSystemOne'],
  expectedManifestSha256: string
): void {
  const evidence = asRecord(value, 'Typed System One evidence');
  assert.equal(evidence.id, fixture.id, 'Typed System One evidence ID mismatch.');
  assert.equal(evidence.modelId, MODEL_ID, 'Typed System One model ID mismatch.');
  assert.equal(evidence.backend, 'tvm-webgpu', 'Typed System One evidence did not use TVM WebGPU.');
  assert.equal(evidence.assetManifestSha256, expectedManifestSha256, 'Typed System One asset manifest mismatch.');
  assert.equal(evidence.state, fixture.state, 'Typed System One state differs from the pinned input.');
  assert.deepEqual(evidence.questions, fixture.questions, 'Typed System One questions differ from the pinned input.');
  validateTypedRunEvidence(evidence.packed, fixture.packed, fixture.questions, 'packed');
  const single = asRecord(evidence.single, 'Typed single-question runs');
  assert.deepEqual(Object.keys(single), EXPANDED_TYPED_QUESTION_NAMES, 'Typed single-question runs are incomplete.');
  for (const name of EXPANDED_TYPED_QUESTION_NAMES) {
    const expectedRun = fixture.single[name];
    const question = fixture.questions[name];
    assert.ok(expectedRun && question, `Typed single-question fixture ${name} is missing.`);
    validateTypedRunEvidence(single[name], expectedRun, {[name]: question}, `single ${name}`);
  }
}

function validateTypedRunEvidence(
  value: unknown,
  reference: TypedReferenceRun,
  questions: Readonly<Record<string, KevSystemOneQuestion>>,
  description: string
): void {
  const evidence = asRecord(value, `Typed ${description} evidence`);
  assert.deepEqual(evidence.reference, reference, `Typed ${description} reference mismatch.`);
  const observed = asRecord(evidence.observed, `Typed ${description} observed output`);
  assert.equal(observed.model, MODEL_ID, `Typed ${description} model mismatch.`);
  assert.equal(observed.tokenCount, reference.tokenCount, `Typed ${description} token count mismatch.`);
  const answers = asRecord(observed.answers, `Typed ${description} answers`);
  const questionNames = Object.keys(questions);
  assert.deepEqual(Object.keys(answers), questionNames, `Typed ${description} answers are incomplete.`);
  const entries = asNamedAnswerEntries(observed.answerEntries, `Typed ${description} answerEntries`);
  assert.deepEqual(
    entries.map(([name]) => name),
    questionNames,
    `Typed ${description} answer order mismatch.`
  );
  const differences: Record<string, number> = {};
  for (const name of questionNames) {
    const question = questions[name];
    const expected = reference.answers[name];
    assert.ok(question && expected, `Typed ${description} fixture answer ${name} is missing.`);
    const answer = asRecord(answers[name], `Typed ${description} answer ${name}`);
    assert.deepEqual(entries.find(([entryName]) => entryName === name)?.[1], answers[name]);
    differences[name] = validateTypedAnswer(answer, question, expected, `${description} ${name}`);
  }
  assert.deepEqual(evidence.comparison, differences, `Typed ${description} comparison measurements mismatch.`);
}

function validateTypedAnswer(
  answer: Record<string, unknown>,
  question: KevSystemOneQuestion,
  reference: TypedReferenceAnswer,
  description: string
): number {
  assert.equal(answer.type, question.type, `${description} answer type mismatch.`);
  if (question.type === 'choice') return validateChoiceAnswer(answer, question, reference, description);
  if (question.type === 'noul') return validateNoulAnswer(answer, reference, description);
  return validateScoreAnswer(answer, question.criteria, reference, description);
}

function validateChoiceAnswer(
  answer: Record<string, unknown>,
  question: Extract<KevSystemOneQuestion, {type: 'choice'}>,
  reference: TypedReferenceAnswer,
  description: string
): number {
  const options = Object.keys(question.criteria);
  const distribution = numericRecord(answer.probabilities, `${description} probabilities`);
  assert.deepEqual(Object.keys(distribution), options, `${description} probability keys mismatch.`);
  const entries = typedProbabilityEntries(answer.probabilityEntries, options, distribution, description);
  assert.equal(answer.choice, selectMaximum(options, entries), `${description} choice is not the probability argmax.`);
  assertUnitInterval(answer.confidence, `${description} confidence`);
  return maxAbsoluteDifference(entries, reference.probabilities);
}

function validateNoulAnswer(
  answer: Record<string, unknown>,
  reference: TypedReferenceAnswer,
  description: string
): number {
  assert.deepEqual(reference.options, ['false', 'true'], `${description} reference must encode false/true order.`);
  assertUnitInterval(answer.noul, `${description} yes probability`);
  const yesProbability = answer.noul as number;
  const probabilities = [1 - yesProbability, yesProbability];
  validateProbabilityVector(probabilities, false, `${description} derived probabilities`);
  return maxAbsoluteDifference(probabilities, reference.probabilities);
}

function validateScoreAnswer(
  answer: Record<string, unknown>,
  criteria: readonly unknown[],
  reference: TypedReferenceAnswer,
  description: string
): number {
  const options = criteria.map((_, index) => String(index));
  const distribution = numericRecord(answer.probabilities, `${description} probabilities`);
  assert.deepEqual(Object.keys(distribution), options, `${description} score levels mismatch.`);
  const entries = typedProbabilityEntries(answer.probabilityEntries, options, distribution, description);
  assert.ok(Number.isFinite(answer.score), `${description} score must be finite.`);
  assert.ok((answer.score as number) >= 0 && (answer.score as number) <= criteria.length - 1);
  assertUnitInterval(answer.confidence, `${description} confidence`);
  assert.deepEqual(answer.legend, Object.fromEntries(criteria.map((label, index) => [String(index), label])));
  return maxAbsoluteDifference(entries, reference.probabilities);
}

function typedRunEvidence(reference: TypedReferenceRun, observed: KevSystemOneResult<'kev-0.6b-q4f16'>) {
  const comparison = Object.fromEntries(
    Object.keys(reference.answers).map((name) => {
      const expected = reference.answers[name];
      const answer = observed.answers[name];
      assert.ok(expected && answer, `Typed output is missing ${name}.`);
      const values = typedAnswerProbabilities(answer);
      return [name, maxAbsoluteDifference(values, expected.probabilities)];
    })
  );
  return {
    observed,
    reference,
    comparison
  };
}

function typedAnswerProbabilities(answer: KevSystemOneAnswer): readonly number[] {
  if (answer.type === 'choice') return answer.probabilityEntries.map(([, probability]) => probability);
  if (answer.type === 'score') return answer.probabilityEntries.map(([, probability]) => probability);
  return [1 - answer.noul, answer.noul];
}

function decisionReference(item: RepresentativeDecisionCase) {
  return {
    runtime: 'onnxruntime 1.31.1 CPUExecutionProvider',
    tokenizerSha256: EXPANDED_TOKENIZER_SHA256,
    tokenCount: item.tokenCount,
    tokenIds: item.tokenIds,
    optionIndices: item.optionIndices,
    options: item.reference.options,
    selected: item.reference.selected,
    logits: item.reference.logits,
    probabilities: item.reference.probabilities
  };
}

function validateProbabilityVector(values: readonly number[], rounded: boolean, description: string): void {
  assert.ok(
    values.every((value) => Number.isFinite(value) && value >= 0 && value <= 1),
    `${description} are invalid.`
  );
  const sum = values.reduce((total, value) => total + value, 0);
  const roundingAllowance = rounded ? (values.length * TYPED_ROUNDING_UNIT) / 2 : 0;
  const floatingPointAllowance = values.length * Number.EPSILON * 8;
  assert.ok(Math.abs(sum - 1) <= roundingAllowance + floatingPointAllowance, `${description} do not sum to one.`);
}

function typedProbabilityEntries(
  value: unknown,
  options: readonly string[],
  distribution: Readonly<Record<string, number>>,
  description: string
): number[] {
  assert.ok(Array.isArray(value), `${description} probability entries are missing.`);
  const entries = asEntryPairs(value, `${description} probability entries`);
  assert.deepEqual(
    entries.map(([key]) => key),
    options,
    `${description} probability entry order mismatch.`
  );
  const probabilities = entries.map(([key, probability]) => {
    assert.equal(probability, distribution[key], `${description} probability views disagree for ${key}.`);
    return probability;
  });
  validateProbabilityVector(probabilities, true, `${description} probabilities`);
  return probabilities;
}

function asEntryPairs(value: unknown, description: string): Array<[string, number]> {
  assert.ok(Array.isArray(value), `${description} must be an array.`);
  return (value as unknown[]).map((entry) => {
    assert.ok(Array.isArray(entry) && entry.length === 2, `${description} entries must be key/value pairs.`);
    const key = requiredText(entry[0], `${description} key`);
    assert.ok(
      typeof entry[1] === 'number' && Number.isFinite(entry[1]),
      `${description} values must be finite numbers.`
    );
    return [key, entry[1]];
  });
}

function asNamedAnswerEntries(value: unknown, description: string): Array<[string, unknown]> {
  assert.ok(Array.isArray(value), `${description} must be an array.`);
  return (value as unknown[]).map((entry) => {
    assert.ok(Array.isArray(entry) && entry.length === 2, `${description} entries must be name/value pairs.`);
    return [requiredText(entry[0], `${description} question name`), entry[1]];
  });
}

function numericRecord(value: unknown, description: string): Record<string, number> {
  const record = asRecord(value, description);
  const result: Record<string, number> = {};
  for (const [key, item] of Object.entries(record)) {
    assert.ok(typeof item === 'number' && Number.isFinite(item), `${description} value ${key} must be finite.`);
    result[key] = item;
  }
  return result;
}

function numberVector(value: unknown, description: string, expectedLength: number): number[] {
  assert.ok(Array.isArray(value), `${description} must be an array.`);
  assert.equal(value.length, expectedLength, `${description} has the wrong length.`);
  assert.ok(
    value.every((item) => typeof item === 'number' && Number.isFinite(item)),
    `${description} values must be finite.`
  );
  return value as number[];
}

function selectMaximum(options: readonly string[], probabilities: readonly number[]): string {
  return options[selectMaximumIndex(probabilities)] ?? '';
}

function selectMaximumIndex(probabilities: readonly number[]): number {
  assert.ok(probabilities.length > 0, 'Cannot select from an empty probability vector.');
  let selected = 0;
  for (let index = 1; index < probabilities.length; index++) {
    if (probabilities[index]! > probabilities[selected]!) selected = index;
  }
  return selected;
}

function maxAbsoluteDifference(left: readonly number[], right: readonly number[]): number {
  assert.equal(left.length, right.length, 'Compared vectors must have equal lengths.');
  return left.reduce((maximum, value, index) => Math.max(maximum, Math.abs(value - right[index]!)), 0);
}

function assertUnitInterval(value: unknown, description: string): asserts value is number {
  assert.ok(
    typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1,
    `${description} must be in [0, 1].`
  );
}

function requiredText(value: unknown, description: string): string {
  assert.ok(typeof value === 'string' && value.length > 0, `${description} is required.`);
  return value;
}

function asRecord(value: unknown, description: string): Record<string, unknown> {
  assert.ok(value !== null && typeof value === 'object' && !Array.isArray(value), `${description} must be an object.`);
  return value as Record<string, unknown>;
}

function sha256(value: Uint8Array): string {
  return createHash('sha256').update(value).digest('hex');
}
