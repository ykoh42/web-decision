import assert from 'node:assert/strict';
import test from 'node:test';
import {WebDecisionError} from '../src/kev.js';
import {
  finishStrandsGemma4Answer,
  packStrandsGemma4Request,
  STRANDS_GEMMA4_MAX_LENGTH,
  unpackStrandsGemma4Scores
} from '../src/strands/gemma4-protocol.mjs';

function tokenizer() {
  const pieces = new Map([[2, '<bos>']]);
  const ids = new Map();
  let nextId = 10;
  return {
    encode(text, {add_special_tokens}) {
      const encoded = Array.from(text, (piece) => {
        if (!ids.has(piece)) {
          ids.set(piece, nextId);
          pieces.set(nextId, piece);
          nextId++;
        }
        return ids.get(piece);
      });
      return {ids: add_special_tokens ? [2, ...encoded] : encoded};
    },
    decode(tokenIds) {
      return tokenIds.map((id) => pieces.get(id) ?? '').join('');
    }
  };
}

function lineTokenizer() {
  const pieces = new Map([[2, '<bos>']]);
  const ids = new Map();
  let nextId = 10;
  return {
    encode(text, {add_special_tokens}) {
      const encoded = (text.match(/[^\n]+|\n/gu) ?? []).map((piece) => {
        if (!ids.has(piece)) {
          ids.set(piece, nextId);
          pieces.set(nextId, piece);
          nextId++;
        }
        return ids.get(piece);
      });
      return {ids: add_special_tokens ? [2, ...encoded] : encoded};
    },
    decode(tokenIds) {
      return tokenIds.map((id) => pieces.get(id) ?? '').join('');
    }
  };
}

const config = {
  temperatureByKind: {choice: 1, score: 1, noul: 1},
  ordinalSmoothing: 0.1
};

test('renders Strands state and typed questions with ordered option spans', () => {
  const packed = packStrandsGemma4Request(
    tokenizer(),
    {text: '  hello  ', count: 2},
    {
      color: {
        type: 'choice',
        instructions: '  Pick a color.  ',
        criteria: {red: ' warm\n color ', blue: null}
      },
      truth: {type: 'noul', instructions: 'Is it true?', criteria: {true: ' yes ', false: null}},
      sentiment: {type: 'score', instructions: 'Rate it.', criteria: ['low', 'medium', 'high']}
    }
  );

  assert.equal(packed.rows.length, 3);
  assert.equal(packed.rows[0].stateText, '<state>\n{\n  "text": "  hello  ",\n  "count": 2\n}\n</state>\n');
  assert.equal(
    packed.rows[0].questionText,
    '<question type="choice">\nSelect exactly one option.\nPick a color.\n<options>\n1. red — warm color\n2. blue\n</options>\n</question>\n<answer>'
  );
  assert.deepEqual(packed.rows[0].options, ['red', 'blue']);
  assert.equal(packed.rows[1].questionText.includes('1. false\n2. true — yes'), true);
  assert.equal(packed.rows[2].questionText.includes('1. 0 — low\n2. 1 — medium\n3. 2 — high'), true);
  assert.deepEqual(
    packed.rows.map((row) => row.answerIndex),
    packed.rows.map((row) => row.inputIds.length - 1)
  );
  for (const row of packed.rows) {
    assert.equal(row.optionIndices.length, row.options.length);
    assert.equal(
      row.optionIndices.every((index) => index < row.answerIndex),
      true
    );
    assert.equal(row.inputIds[0], 2);
  }
  assert.equal(
    packed.tokenCount,
    packed.rows.reduce((sum, row) => sum + row.inputIds.length, 0)
  );
});

test('renders structured numbers as Python does after parsing the JSON wire', () => {
  const numbers = [1e-7, 1e-5, 1e-4, 1e16, -1e-7, 1e21, -0];
  const packed = packStrandsGemma4Request(
    tokenizer(),
    {numbers},
    {
      pick: {type: 'choice', instructions: {numbers}, criteria: {one: {numbers}, two: null}}
    }
  );
  const expected =
    '[\n    1e-07,\n    1e-05,\n    0.0001,\n    10000000000000000,\n    -1e-07,\n    1e+21,\n    0\n  ]';
  assert.ok(packed.rows[0].stateText.includes(`"numbers": ${expected}`));
  assert.ok(packed.rows[0].questionText.includes('1e-07'));
  assert.ok(packed.rows[0].questionText.includes('10000000000000000'));
  for (const value of [NaN, Infinity, -Infinity]) {
    assert.throws(
      () => packStrandsGemma4Request(tokenizer(), {value}, {pick: {type: 'noul', instructions: 'Valid?'}}),
      {code: 'INVALID_INPUT'}
    );
  }
});

test('accepts only pinned Strands content types for option descriptions', () => {
  const packed = packStrandsGemma4Request(tokenizer(), 'state', {
    pick: {
      type: 'choice',
      instructions: 'Pick one.',
      criteria: {
        structured: {enabled: true, count: 2},
        listed: [false, 3]
      }
    },
    truth: {
      type: 'noul',
      instructions: 'Is this true?',
      criteria: {true: {evidence: ['yes', 1]}, false: null}
    }
  });
  assert.equal(packed.rows[0].questionText.includes('"enabled": true'), true);
  assert.equal(packed.rows[0].questionText.includes('false'), true);
  assert.equal(packed.rows[1].questionText.includes('"evidence"'), true);

  for (const description of [1, true]) {
    assert.throws(
      () =>
        packStrandsGemma4Request(tokenizer(), 'state', {
          pick: {type: 'choice', instructions: 'Pick one.', criteria: {first: null, second: description}}
        }),
      {code: 'INVALID_INPUT'}
    );
    assert.throws(
      () =>
        packStrandsGemma4Request(tokenizer(), 'state', {
          truth: {type: 'noul', instructions: 'Is this true?', criteria: {true: description}}
        }),
      {code: 'INVALID_INPUT'}
    );
  }
});

test('keeps Unicode option offsets exact, including surrogate pairs', () => {
  const packed = packStrandsGemma4Request(tokenizer(), '상태 🙂', {
    language: {type: 'choice', instructions: 'Choose.', criteria: {日本語: '東アジア', '👩‍💻': 'developer'}}
  });
  const row = packed.rows[0];
  assert.equal(row.options[0], '日本語');
  assert.equal(row.options[1], '👩‍💻');
  const prefixLength = row.inputIds.length - Array.from(row.questionText).length;
  const expectedLastTokens = row.options.map((label, index) => {
    const description = row.descriptions[index];
    const line = `${index + 1}. ${label}${description ? ` — ${description}` : ''}`;
    const start = row.questionText.indexOf(line);
    return prefixLength + Array.from(row.questionText.slice(0, start + line.length)).length - 1;
  });
  assert.deepEqual(row.optionIndices, expectedLastTokens);
});

test('preserves Map and array order for numeric-looking question and option names', () => {
  const packed = packStrandsGemma4Request(
    tokenizer(),
    'state',
    new Map([
      [
        '10',
        {
          type: 'choice',
          instructions: 'Pick.',
          criteria: new Map([
            ['10', null],
            ['2', null]
          ])
        }
      ],
      ['2', {type: 'choice', instructions: 'Pick.', criteria: ['second', 'first']}]
    ])
  );
  assert.deepEqual(
    packed.rows.map((row) => row.name),
    ['10', '2']
  );
  assert.deepEqual(
    packed.rows.map((row) => row.options),
    [
      ['10', '2'],
      ['second', 'first']
    ]
  );
});

test('choice outputs keep the first option on a tie and derive normalized confidence', () => {
  const answer = finishStrandsGemma4Answer({type: 'choice', options: ['first', 'second', 'third']}, [1, 1, 0], config);
  assert.deepEqual(answer, {
    type: 'choice',
    choice: 'first',
    probabilities: {first: 0.4223, second: 0.4223, third: 0.1554},
    confidence: 0.1335
  });
});

test('noul reports probability true without inventing an abstention or confidence field', () => {
  const answer = finishStrandsGemma4Answer({type: 'noul', options: ['false', 'true']}, [0, Math.log(3)], config);
  assert.deepEqual(answer, {type: 'noul', noul: 0.75});
});

test('score returns the expected ordered level, legend, probabilities, and ordinal confidence', () => {
  const answer = finishStrandsGemma4Answer(
    {type: 'score', options: ['0', '1', '2'], descriptions: ['low', 'middle', 'high']},
    [0, 0, 0],
    config
  );
  assert.deepEqual(answer, {
    type: 'score',
    score: 1,
    legend: {0: 'low', 1: 'middle', 2: 'high'},
    probabilities: {0: 0.3333, 1: 0.3333, 2: 0.3333},
    confidence: 0.2684
  });
});

test('unpacks mixed typed answers by original question name and model id', () => {
  const packed = packStrandsGemma4Request(
    tokenizer(),
    'state',
    Object.fromEntries([
      ['__proto__', {type: 'choice', instructions: 'Pick.', criteria: {first: null, second: null}}],
      ['truth', {type: 'noul', instructions: 'True?', criteria: {true: 'yes', false: 'no'}}],
      ['rating', {type: 'score', instructions: 'Rate.', criteria: ['low', 'high']}]
    ])
  );
  const result = unpackStrandsGemma4Scores(
    packed,
    [
      [0, 1],
      [Math.log(3), 0],
      [0, 0]
    ],
    config,
    'experimental-strands-gemma4-e2b'
  );
  assert.equal(result.model, 'experimental-strands-gemma4-e2b');
  assert.equal(result.tokenCount, packed.tokenCount);
  assert.deepEqual(Object.keys(result.answers), ['__proto__', 'truth', 'rating']);
  assert.equal(result.answers.__proto__.choice, 'second');
  assert.deepEqual(result.answers.truth, {type: 'noul', noul: 0.25});
  assert.deepEqual(result.answers.rating, {
    type: 'score',
    score: 0.5,
    legend: {0: 'low', 1: 'high'},
    probabilities: {0: 0.5, 1: 0.5},
    confidence: 0
  });
});

test('never truncates a prompt to fit the pinned context window', () => {
  const input = {choice: {type: 'choice', instructions: 'Pick.', criteria: {yes: null, no: null}}};
  assert.equal(STRANDS_GEMMA4_MAX_LENGTH, 4096);
  assert.throws(
    () => packStrandsGemma4Request(tokenizer(), 'state', input, 1),
    (error) => error instanceof WebDecisionError && error.code === 'INVALID_INPUT'
  );
});

test('keeps choices above 255 options and bounds them by the tokenized context', () => {
  const choice = (count) => ({
    pick: {
      type: 'choice',
      instructions: 'Pick.',
      criteria: Object.fromEntries(Array.from({length: count}, (_, i) => [`option-${i}`, null]))
    }
  });
  assert.doesNotThrow(() => packStrandsGemma4Request(tokenizer(), '', choice(2)));
  assert.doesNotThrow(() => packStrandsGemma4Request(tokenizer(), '', choice(255)));
  assert.doesNotThrow(() => packStrandsGemma4Request(tokenizer(), '', choice(256)));
  const largeChoice = packStrandsGemma4Request(lineTokenizer(), '', choice(1000));
  assert.equal(largeChoice.rows[0].options.length, 1000);
  assert.ok(largeChoice.rows[0].inputIds.length < STRANDS_GEMMA4_MAX_LENGTH);
  assert.throws(() => packStrandsGemma4Request(tokenizer(), '', choice(1)), {code: 'INVALID_INPUT'});
  assert.throws(() => packStrandsGemma4Request(tokenizer(), '', choice(512)), {code: 'INVALID_INPUT'});
  const score = (count) => ({
    rate: {type: 'score', instructions: 'Rate.', criteria: Array.from({length: count}, (_, i) => `level-${i}`)}
  });
  assert.doesNotThrow(() => packStrandsGemma4Request(tokenizer(), '', score(2)));
  assert.doesNotThrow(() => packStrandsGemma4Request(tokenizer(), '', score(10)));
  assert.throws(() => packStrandsGemma4Request(tokenizer(), '', score(11)), {code: 'INVALID_INPUT'});
  assert.throws(() => packStrandsGemma4Request({encode: () => ({ids: [10]}), decode: () => 'x'}, '', choice(2)), {
    code: 'INVALID_ASSET'
  });
});

test('rejects malformed model logits and unsupported score shapes', () => {
  assert.throws(() => finishStrandsGemma4Answer({type: 'choice', options: ['a', 'b']}, [0, Number.NaN], config), {
    code: 'INFERENCE_ERROR'
  });
  assert.throws(
    () =>
      finishStrandsGemma4Answer(
        {type: 'score', options: Array(11).fill('x'), descriptions: Array(11).fill('x')},
        Array(11).fill(0),
        config
      ),
    {code: 'INFERENCE_ERROR'}
  );
  assert.doesNotThrow(() =>
    finishStrandsGemma4Answer(
      {type: 'choice', options: Array.from({length: 256}, (_, index) => `option-${index}`)},
      Array(256).fill(0),
      config
    )
  );
});
