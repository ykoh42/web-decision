import assert from 'node:assert/strict';
import test from 'node:test';
import {kananaFloat} from '../src/kanana/json-float.ts';
import {
  MAPIKA2B_LIMITS,
  MAPIKA2B_MAX_OPTIONS,
  MAPIKA2B_TEMPERATURES,
  MAPIKA2B_VISION,
  type Mapika2bLimits
} from '../src/mapika2b/contract.mts';
import {
  createMapika2bLabels,
  prefixMapika2bImages,
  prepareMapika2bRequest,
  resolveMapika2bLimits,
  resolveMapika2bTemperatures
} from '../src/mapika2b/protocol.mts';
import {finishMapika2bResult} from '../src/mapika2b/readout.mts';
import {createMapika2bSession} from '../src/mapika2b/session.mts';

const labelOffset = 10_000;
const contextPrefix = 'Context:\n';
const tokenizer = {
  encode(value: string, options: {add_special_tokens: false}) {
    assert.deepEqual(options, {add_special_tokens: false});
    if (/^[A-Z]{2}$/u.test(value))
      return {ids: [labelOffset + (value.charCodeAt(0) - 65) * 26 + value.charCodeAt(1) - 65]};
    return {ids: Array.from(value, (character) => character.codePointAt(0)!)};
  }
};
const labels = createMapika2bLabels(tokenizer);
const text = (ids: readonly number[]) => String.fromCodePoint(...ids);
const prepare = (request: unknown, limits: Mapika2bLimits = MAPIKA2B_LIMITS) =>
  prepareMapika2bRequest(request, tokenizer, labels, limits);

const choice = {instructions: 'Which queue?', criteria: ['billing', 'technical']};
const noul = {type: 'bool', criteria: {true: 'a refund is needed', false: 'no refund needed'}};
const score = {type: 'score', instructions: 'How urgent?', criteria: {10: '2: urgent', 1: 'low'}};

test('Noul criteria accept upstream boolean keys with string keys taking precedence', () => {
  const booleanKeys = new Map<unknown, unknown>([
    [false, 'no refund'],
    [true, 'refund']
  ]);
  const strings = {false: 'no refund', true: 'refund'};
  const request = (criteria: unknown) => ({state: '', questions: {q: {type: 'noul', criteria}}});
  assert.deepEqual(prepare(request(booleanKeys)).rows[0]!.inputIds, prepare(request(strings)).rows[0]!.inputIds);
  booleanKeys.set('true', 'replacement');
  assert.deepEqual(
    prepare(request(booleanKeys)).rows[0]!.inputIds,
    prepare(request({false: 'no refund', true: 'replacement'})).rows[0]!.inputIds
  );
  booleanKeys.set(3, 'ignored');
  assert.deepEqual(
    prepare(request(booleanKeys)).rows[0]!.inputIds,
    prepare(request({false: 'no refund', true: 'replacement'})).rows[0]!.inputIds
  );
  assert.deepEqual(
    prepare(
      request(
        new Map<unknown, unknown>([
          [0, 'no refund'],
          [1n, 'refund']
        ])
      )
    ).rows[0]!.inputIds,
    prepare(request(strings)).rows[0]!.inputIds
  );
});

test('serving limits can be raised or lowered while invalid limit names are rejected', () => {
  const limits = resolveMapika2bLimits({maxStateTokens: 40_000, maxImages: 6});
  assert.equal(limits.maxStateTokens, 40_000);
  assert.equal(limits.maxImages, 6);
  assert.equal(limits.maxRows, MAPIKA2B_LIMITS.maxRows);
  assert.equal(resolveMapika2bLimits({maxImages: 0}).maxImages, 0);
  assert.throws(() => resolveMapika2bLimits({maxRows: -1}), /nonnegative integers/u);
  assert.throws(() => resolveMapika2bLimits({maxRows: Number.POSITIVE_INFINITY}), /nonnegative integers/u);
  assert.throws(() => resolveMapika2bLimits({unknown: 5} as never), /Unknown Mapika serving limit/u);
});

test('preserves Decider scalar and per-type temperature overrides in every scoring row', () => {
  const request = {state: '', questions: {choice, noul, score}};
  const temperatures = resolveMapika2bTemperatures({temperature: 1.5, temperatureByType: {noul: 3.185}});
  const plan = prepareMapika2bRequest(request, tokenizer, labels, MAPIKA2B_LIMITS, temperatures);
  assert.deepEqual(
    plan.rows.map(({slots}) => slots[0]!.temperature),
    [1.5, 3.185, 1.5, 1.5]
  );
  assert.deepEqual(resolveMapika2bTemperatures({temperature: 2}), {choice: 2, noul: 2, score: 2});
  assert.deepEqual(resolveMapika2bTemperatures({temperatureByType: {noul: 3.185}}), {
    choice: 2.515,
    noul: 3.185,
    score: 2.515
  });
  assert.throws(() => resolveMapika2bTemperatures({temperature: Infinity}), /finite positive/u);
  assert.throws(() => resolveMapika2bTemperatures({temperatureByType: {noul: 0}}), /finite positive/u);
});

test('renders plain state and exact independent slots, with Score temperature on isolated yes/no rows', () => {
  const request = {
    state: {
      items: [{_index: 41}, true, null, ['ok'], 7n, kananaFloat(2), -0, 1e-5]
    },
    questions: {choice, noul, score}
  };
  const plan = prepare(request);
  const state =
    'Context:\n{"items": [{"_index": 41}, {"_index": 1, "value": true}, {"_index": 2, "value": null}, ' +
    '{"_index": 3, "value": ["ok"]}, {"_index": 4, "value": 7}, {"_index": 5, "value": 2.0}, ' +
    '{"_index": 6, "value": -0.0}, {"_index": 7, "value": 1e-05}]}';
  assert.deepEqual(
    plan.rows.map(({inputIds}) => text(inputIds.slice(state.length))),
    [
      '\n\nQuestion: Which queue?\nOptions:\n(A) billing\n(B) technical\nAnswer: (',
      '\n\nQuestion: Which answer fits the context?\nOptions:\n(A) no: no refund needed\n(B) yes: a refund is needed\nAnswer: (',
      '\n\nQuestion: How urgent?\nProposed answer: low\nDoes the proposed answer fit?\nOptions:\n(A) no\n(B) yes\nAnswer: (',
      '\n\nQuestion: How urgent?\nProposed answer: urgent\nDoes the proposed answer fit?\nOptions:\n(A) no\n(B) yes\nAnswer: ('
    ]
  );
  assert.equal(text(plan.rows[0]!.inputIds.slice(0, state.length)), state);
  assert.deepEqual(
    plan.rows.map(({inputIds, slots}) => [inputIds[slots[0]!.tokenIndex], slots[0]!.temperature]),
    [
      [40, 2.465],
      [40, 0.05],
      [40, 2.833],
      [40, 2.833]
    ]
  );
  const commonPrefix = state.length + '\n\nQuestion: '.length;
  assert.equal(
    plan.inputTokenCount,
    commonPrefix + plan.rows.reduce((sum, row) => sum + row.inputIds.length - commonPrefix, 0)
  );
  assert.deepEqual(
    plan.questions.map(({name, kind, firstSlot}) => [name, kind, firstSlot]),
    [
      ['choice', 'choice', 0],
      ['noul', 'noul', 1],
      ['score', 'score', 2]
    ]
  );
});

test('joint views use numbered question and answer pieces and one list-style Score slot', () => {
  const plan = prepare({
    state: 'A charged twice.',
    independent: false,
    questions: new Map<string, unknown>([
      ['01', choice],
      ['2', score]
    ])
  });
  assert.deepEqual(
    plan.questions.map(({name}) => name),
    ['01', '2']
  );
  assert.equal(plan.rows.length, 1);
  const row = plan.rows[0]!;
  const prompt = text(row.inputIds);
  assert.match(prompt, /\n\nQuestion 1: Which queue\?\nOptions:\n\(A\) billing\n\(B\) technical\nAnswer 1: \(/u);
  assert.match(prompt, /\n\nQuestion 2: How urgent\?\nOptions:\n\(A\) 0: low\n\(B\) 1: 2: urgent\nAnswer 2: \($/u);
  assert.deepEqual(
    row.slots.map(({tokenIndex, temperature}) => [prompt.slice(tokenIndex - 3, tokenIndex + 1), temperature]),
    [
      ['1: (', 2.465],
      ['2: (', 2.833]
    ]
  );
  assert.equal(plan.inputTokenCount, prompt.length);
});

test('uses the 255 original output letters and wide option token pieces', () => {
  assert.equal(labels.names.length, 255);
  assert.deepEqual(labels.names.slice(0, 28), [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'AA', 'AB']);
  const criteria = Array.from({length: 255}, (_, index) => 'option-' + index);
  const plan = prepare({state: '', questions: {wide: {instructions: 'Pick', criteria}}});
  const row = plan.rows[0]!;
  assert.equal(row.slots[0]!.optionCount, 255);
  const marker = row.inputIds.indexOf(labelOffset);
  assert.notEqual(marker, -1);
  assert.equal(text(row.inputIds.slice(marker - 2, marker)), '\n(');
  assert.equal(text(row.inputIds.slice(marker + 1, marker + 12)), ') option-26');
  assert.throws(
    () => prepare({state: '', questions: {wide: {instructions: 'Pick', criteria: [...criteria, 'last']}}}),
    /2 to 255 options/u
  );
});

test('state tokens truncate before appending questions and limits are checked on the complete rows', () => {
  const state = 'f'.repeat(MAPIKA2B_LIMITS.maxStateTokens);
  const plan = prepare({state, questions: {choice}});
  assert.equal(plan.stateTokenCount, MAPIKA2B_LIMITS.maxStateTokens);
  assert.ok(plan.rows[0]!.inputIds.length > MAPIKA2B_LIMITS.maxStateTokens);
  assert.equal(plan.rows[0]!.inputIds[MAPIKA2B_LIMITS.maxStateTokens], 10);
  assert.equal(
    plan.rows[0]!.inputIds.slice(0, contextPrefix.length)
      .map((id) => String.fromCodePoint(id))
      .join(''),
    contextPrefix
  );
  const exactly = {...MAPIKA2B_LIMITS, maxRowTokens: plan.rows[0]!.inputIds.length};
  assert.ok(prepare({state, questions: {choice}}, exactly));
  assert.throws(
    () => prepare({state, questions: {choice}}, {...exactly, maxRowTokens: exactly.maxRowTokens - 1}),
    /per-row token limit/u
  );
  assert.throws(
    () =>
      prepare(
        {state, questions: {choice, noul}},
        {...exactly, maxRowTokens: MAPIKA2B_LIMITS.maxRowTokens, maxRequestTokens: exactly.maxRowTokens}
      ),
    /total-request token limit/u
  );
  assert.equal(prepare({state, questions: {}}).inputTokenCount, 0);
  assert.equal(prepare({state, questions: {}, independent: false}).stateTokenCount, MAPIKA2B_LIMITS.maxStateTokens);
});

test('empty System One requests preserve native usage in both layouts without a text forward', async () => {
  const source = {format: 'png', data_base64: 'AA=='};
  const imageReferences: string[] = [];
  const session = createMapika2bSession({
    tokenizer,
    async encodeImage({reference}) {
      imageReferences.push(reference);
      return {grid: reference === 'img2' ? [1, 4, 2] : [1, 2, 2], embeddings: new Float32Array(0)};
    },
    async run() {
      assert.fail('An empty request must not run the text graph.');
    },
    async dispose() {}
  });
  const request = {state: 'img2 then img then img2', questions: {}};
  const assets = {img: source, img2: source};
  const {startToken, imageToken, endToken} = MAPIKA2B_VISION;
  const img2Block = [startToken, imageToken, imageToken, endToken];
  const prefix = [...img2Block, startToken, imageToken, endToken, ...img2Block];

  for (const independent of [true, false]) {
    const withoutImages = await session.evaluate({...request, independent});
    const withImages = await session.evaluate({...request, independent, assets});
    const textTokens = independent ? 0 : (contextPrefix + request.state).length;
    const imageTokens = independent ? 0 : (contextPrefix + 'image 1 then image 2 then image 3').length + prefix.length;
    for (const [evaluation, inputTokens] of [
      [withoutImages, textTokens],
      [withImages, imageTokens]
    ] as const) {
      assert.deepEqual(evaluation.result, {
        model: 'decider-2b-v11',
        answers: {},
        usage: {input_tokens: inputTokens, output_tokens: 0}
      });
      assert.deepEqual(evaluation.logitsByRow, independent ? [] : [new Float32Array(0)]);
    }
    assert.deepEqual(
      withImages.rows.map(({inputIds}) => inputIds.slice(0, prefix.length)),
      independent ? [] : [prefix]
    );
  }
  assert.deepEqual(imageReferences, ['img2', 'img', 'img2', 'img']);
  await assert.rejects(session.evaluate({state: 'missing', questions: {}, assets}), /not referenced/u);
  await session.dispose();
});

test('image references resolve by longest match and occurrence order before building the native vision prefix', () => {
  const one = {format: 'PNG', data_base64: 'AA=='};
  const two = {type: 'image', format: 'jpeg', data_base64: 'AQ=='};
  const plan = prepare({
    state: {photo: 'look at img-right then img then img-right'},
    assets: {img: one, 'img-right': two},
    questions: {choice, noul}
  });
  assert.deepEqual(
    plan.images.map(({reference, data}) => [reference, data[0]]),
    [
      ['img-right', 1],
      ['img', 0],
      ['img-right', 1]
    ]
  );
  assert.ok(
    text(plan.rows[0]!.inputIds).startsWith('Context:\n{"photo": "look at image 1 then image 2 then image 3"}')
  );
  const grid = [
    [1, 2, 4],
    [1, 2, 2],
    [1, 2, 4]
  ] as const;
  const prefixed = prefixMapika2bImages(plan, grid);
  const {startToken, endToken, imageToken} = MAPIKA2B_VISION;
  assert.deepEqual(prefixed.rows[0]!.inputIds.slice(0, prefixed.imagePrefixLength), [
    startToken,
    imageToken,
    imageToken,
    endToken,
    startToken,
    imageToken,
    endToken,
    startToken,
    imageToken,
    imageToken,
    endToken
  ]);
  assert.equal(
    prefixed.inputTokenCount,
    prefixed.rows.reduce((sum, row) => sum + row.inputIds.length, 0)
  );
  assert.equal(prefixed.rows[1]!.slots[0]!.tokenIndex, plan.rows[1]!.slots[0]!.tokenIndex + prefixed.imagePrefixLength);
  assert.throws(() => prefixMapika2bImages(plan, [[1, 2, 2]]), /one image grid/u);
  assert.throws(() => prefixMapika2bImages(prefixed, grid), /one vision prefix/u);
});

test('image checks reject question-only assets, excessive repeated references and malformed base64', () => {
  const asset = {format: 'png', data_base64: 'AA=='};
  assert.throws(
    () => prepare({state: 'none', questions: {choice: {...choice, instructions: 'photo'}}, assets: {photo: asset}}),
    /referenced from the state only/u
  );
  assert.throws(
    () => prepare({state: 'photo photo photo photo photo', questions: {choice}, assets: {photo: asset}}),
    /image reference limit/u
  );
  assert.throws(
    () => prepare({state: 'photo', questions: {choice}, assets: {photo: {...asset, data_base64: 'A'}}}),
    /invalid base64/u
  );
});

function logits(...values: readonly number[][]) {
  const result = new Float32Array(values.length * MAPIKA2B_MAX_OPTIONS).fill(-Infinity);
  for (const [index, row] of values.entries()) result.set(row, index * MAPIKA2B_MAX_OPTIONS);
  return result;
}

test('applies official calibrated readout over valid columns and assembles typed answers', () => {
  const plan = prepare({state: 'A charged twice.', questions: {choice, noul, score}});
  const result = finishMapika2bResult(plan, [
    logits([0, MAPIKA2B_TEMPERATURES.choice, 1000]),
    logits([0, MAPIKA2B_TEMPERATURES.noul]),
    logits([0, 0]),
    logits([0, MAPIKA2B_TEMPERATURES.score])
  ]);
  assert.deepEqual(result.answers, {
    choice: {
      type: 'choice',
      choice: 'technical',
      confidence: 0.4621,
      x_p_max: 0.7311,
      certainty: 0.1601,
      probabilities: {billing: 0.2689, technical: 0.7311}
    },
    noul: {type: 'noul', noul: 0.7311},
    score: {
      type: 'score',
      score: 0.59,
      confidence: 0.1877,
      x_p_max: 0.5938,
      certainty: 0.0256,
      legend: {0: 'low', 1: '2: urgent'},
      probabilities: {0: 0.4062, 1: 0.5938},
      level_fit: {0: 0.5, 1: 0.7311},
      fit_mass: 1.2311
    }
  });
  assert.deepEqual(result.usage, {input_tokens: plan.inputTokenCount, output_tokens: 0});
  assert.equal(result.model, 'decider-2b-v11');
  assert.throws(() => finishMapika2bResult(plan, []), /wrong number of scoring rows/u);
  assert.throws(
    () => finishMapika2bResult(plan, [new Float32Array(2), logits([0, 0]), logits([0, 0]), logits([0, 0])]),
    /255 logits per answer slot/u
  );
});

test('readout accepts every joint slot and preserves names safely including prototype-like keys', () => {
  const requests = new Map([
    ['2', choice],
    ['1', choice],
    ['__proto__', choice]
  ]);
  const plan = prepare({state: '', independent: false, questions: requests});
  const result = finishMapika2bResult(plan, [logits([0, 0], [0, 0], [0, 0])]);
  assert.deepEqual(Object.keys(result.answers), ['2', '1', '__proto__']);
  assert.equal(result.answers['__proto__']?.type, 'choice');
  assert.equal(Object.getPrototypeOf(result.answers), Object.prototype);
});
