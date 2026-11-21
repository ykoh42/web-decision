import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
import test from 'node:test';
import {Tokenizer} from '@huggingface/tokenizers';
import {
  MAPIKA2B_DECIDER_REVISION,
  MAPIKA2B_MAX_OPTIONS,
  MAPIKA2B_REPOSITORY,
  MAPIKA2B_REVISION,
  MAPIKA2B_TOKENIZER_CONFIG_SHA256,
  MAPIKA2B_TOKENIZER_SHA256
} from '../src/mapika2b/contract.mts';
import {
  createMapika2bLabels,
  prefixMapika2bImages,
  prepareMapika2bRequest,
  type Mapika2bImageGrid
} from '../src/mapika2b/protocol.mts';
import {finishMapika2bResult, type Mapika2bResult} from '../src/mapika2b/readout.mts';

interface SourcePin {
  readonly repository: string;
  readonly revision: string;
  readonly deciderRevision: string;
}

interface ReferenceCase {
  readonly id: string;
  readonly request: {readonly questions: Readonly<Record<string, Record<string, unknown>>>};
  readonly questionIndex: readonly (readonly [string, string, number, number])[];
  readonly renderedQuestions: Readonly<
    Record<string, {readonly kind?: string; readonly type: string; names: string[]}>
  >;
  readonly rows: readonly {
    readonly inputIds: readonly number[];
    readonly slots: readonly number[];
    readonly optionCounts: readonly number[];
    readonly temperatures: readonly number[];
  }[];
  readonly usage: {readonly input_tokens: number};
  readonly imageGridThw?: readonly Mapika2bImageGrid[];
}

interface ReferenceAnswer {
  readonly id: string;
  readonly result: Mapika2bResult;
  readonly rows: readonly {readonly logits: readonly (readonly number[])[]}[];
}

const oracleRoot = new URL('../../../scripts/mapika/', import.meta.url);

async function readTokenizerFile(snapshot: string, name: string, expected: string) {
  const buffer = await readFile(join(snapshot, name));
  assert.equal(createHash('sha256').update(buffer).digest('hex'), expected, 'Unpinned Mapika tokenizer data');
  return JSON.parse(buffer.toString('utf8')) as Record<string, unknown>;
}

function orderedRequest({request, questionIndex, renderedQuestions}: ReferenceCase) {
  const questions = new Map(
    questionIndex.map(([name]) => {
      const question = {...request.questions[name]!};
      const criteria = question.criteria;
      const rendered = renderedQuestions[name];
      if (
        (question.type ?? 'choice') === 'choice' &&
        criteria &&
        typeof criteria === 'object' &&
        !Array.isArray(criteria)
      ) {
        const record = criteria as Record<string, unknown>;
        question.criteria = new Map(rendered!.names.map((option) => [option, record[option]]));
      }
      return [name, question] as const;
    })
  );
  return {...request, questions};
}

function tensors(reference: ReferenceAnswer): Float32Array[] {
  return reference.rows.map(({logits}) => {
    const values = new Float32Array(logits.length * MAPIKA2B_MAX_OPTIONS).fill(-Infinity);
    for (const [slot, row] of logits.entries()) values.set(row, slot * MAPIKA2B_MAX_OPTIONS);
    return values;
  });
}

const snapshot = process.env.WEB_DECISION_MAPIKA2B_SNAPSHOT;
test('matches the pinned Mapika oracle token IDs, slot metadata and BF16 readout', {skip: !snapshot}, async () => {
  const preparedFile =
    process.env.WEB_DECISION_MAPIKA2B_PREPARED_ORACLE ?? new URL('reference-prepared.json', oracleRoot);
  const answersFile = process.env.WEB_DECISION_MAPIKA2B_ANSWER_ORACLE ?? new URL('reference-answers.json', oracleRoot);
  const [data, config, preparedText, answersText] = await Promise.all([
    readTokenizerFile(snapshot!, 'tokenizer.json', MAPIKA2B_TOKENIZER_SHA256),
    readTokenizerFile(snapshot!, 'tokenizer_config.json', MAPIKA2B_TOKENIZER_CONFIG_SHA256),
    readFile(preparedFile, 'utf8'),
    readFile(answersFile, 'utf8')
  ]);
  const prepared = JSON.parse(preparedText) as {source: SourcePin; cases: ReferenceCase[]};
  const official = JSON.parse(answersText) as {source: SourcePin; cases: ReferenceAnswer[]};
  for (const {source} of [prepared, official]) {
    assert.equal(source.repository, MAPIKA2B_REPOSITORY);
    assert.equal(source.revision, MAPIKA2B_REVISION);
    assert.equal(source.deciderRevision, MAPIKA2B_DECIDER_REVISION);
  }
  const tokenizer = new Tokenizer(data, config);
  const labels = createMapika2bLabels(tokenizer);
  const expectedAnswers = new Map(official.cases.map((item) => [item.id, item]));
  for (const item of prepared.cases) {
    let plan = prepareMapika2bRequest(orderedRequest(item), tokenizer, labels);
    if (item.imageGridThw) plan = prefixMapika2bImages(plan, item.imageGridThw);
    assert.deepEqual(
      plan.rows.map(({inputIds, slots}) => ({
        inputIds,
        slots: slots.map(({tokenIndex}) => tokenIndex),
        optionCounts: slots.map(({optionCount}) => optionCount),
        temperatures: slots.map(({temperature}) => temperature)
      })),
      item.rows.map(({inputIds, slots, optionCounts, temperatures}) => ({inputIds, slots, optionCounts, temperatures})),
      item.id
    );
    assert.equal(plan.inputTokenCount, item.usage.input_tokens, item.id);
    const expected = expectedAnswers.get(item.id);
    if (expected) {
      assert.deepEqual(finishMapika2bResult(plan, tensors(expected)), expected.result, item.id);
      expectedAnswers.delete(item.id);
    }
  }
  assert.equal(expectedAnswers.size, 0, 'Every pinned answer fixture must be checked.');
});
