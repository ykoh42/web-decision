import assert from 'node:assert/strict';
import test from 'node:test';
import {runInNewContext} from 'node:vm';
import {DecisionModel} from '../src/model.js';
import {KEV_MAX_BRANCH_TOKENS, KEV_MAX_OPTIONS, KEV_MAX_STATE_TOKENS, WebDecisionError} from '../src/kev.js';
import type {KevTokenizer} from '../src/kev.js';
import type {KevAnswer, KevResult} from '../src/curated/types.js';
import {finishKevSystemOne, prepareKevSystemOne} from '../src/kev/system-one.mjs';

type TestAnswer = KevAnswer & {
  readonly choice?: string;
  readonly score?: number;
  readonly noul?: number;
  readonly probabilities?: Readonly<Record<string, number>>;
  readonly probabilityEntries?: readonly (readonly [string, number])[];
  readonly legend?: Readonly<Record<string, string>>;
  readonly confidence?: number;
};
type TestAnswers<Questions> =
  Questions extends ReadonlyMap<infer Key, unknown>
    ? {[Name in Extract<Key, string>]: TestAnswer}
    : Questions extends object
      ? {[Name in keyof Questions]: TestAnswer}
      : Record<string, TestAnswer>;
type TestResult<Questions> = Omit<KevResult, 'answers' | 'answerEntries'> & {
  readonly answers: TestAnswers<Questions>;
  readonly answerEntries: readonly (readonly [question: string, answer: TestAnswer])[];
};
type TestModel = Omit<DecisionModel, 'systemOne' | 'limits'> & {
  limits: {
    maxInputTokens: number | undefined;
    maxStateTokens: number;
    maxBranchTokens: number;
    maxQuestions: number | undefined;
    maxOptions: number | undefined;
    maxInputCharacters: number | undefined;
  };
  systemOne<Questions>(
    state: unknown,
    questions: Questions,
    options?: {signal?: AbortSignal | undefined} | null
  ): Promise<TestResult<Questions>>;
};

const halfBits = new Map([
  [0, 0x0000],
  [1, 0x3c00],
  [2, 0x4000],
  [3, 0x4200]
]);

function createModel(
  logitsByQuestion = [
    [3, 0, 0],
    [0, 1, 2],
    [0, 2]
  ],
  supportsSystemOne = true,
  modelId = 'kev-0.6b-q4f16',
  maxInputCharacters?: number
) {
  const texts: string[] = [];
  let inferences = 0;
  const lifecycle = {loss: null, disposed: false};
  const tokenizer = {
    encode(text: string) {
      texts.push(text);
      return {ids: Array.from(text, (character) => character.codePointAt(0)! + 200)};
    }
  };
  const runtime = {
    beginScope() {},
    endScope() {},
    dispose() {
      lifecycle.disposed = true;
    },
    cpu() {
      return 'cpu';
    },
    empty(shape: number[], dtype: string) {
      let values: ArrayLike<number>;
      return {
        copyFrom(next: unknown) {
          values = next as ArrayLike<number>;
          return this;
        },
        toRawBytes() {
          assert.equal(dtype, 'float16');
          const bytes = new Uint8Array(shape[1]! * 2);
          const view = new DataView(bytes.buffer);
          for (let index = 0; index < shape[1]!; index++)
            view.setUint16(index * 2, halfBits.get(values[index]!)!, true);
          return bytes;
        },
        values: () => values
      };
    }
  };
  const device = {sync: async () => {}};
  const gpu = {pushErrorScope() {}, popErrorScope: async () => null};
  const model = new DecisionModel(
    runtime as unknown as ConstructorParameters<typeof DecisionModel>[0],
    device as unknown as ConstructorParameters<typeof DecisionModel>[1],
    gpu as unknown as ConstructorParameters<typeof DecisionModel>[2],
    (ids) => {
      inferences++;
      const scores = new Float32Array((ids as {values(): ArrayLike<number>}).values().length);
      const optionEnds = [];
      for (let index = 0; index < (ids as {values(): ArrayLike<number>}).values().length; index++)
        if ((ids as {values(): ArrayLike<number>}).values()[index] === 151649) optionEnds.push(index);
      let offset = 0;
      for (const group of logitsByQuestion) {
        for (const score of group) scores[optionEnds[offset++]!] = score;
      }
      return scores;
    },
    [] as ConstructorParameters<typeof DecisionModel>[4],
    tokenizer as unknown as KevTokenizer,
    {
      maxStateTokens: KEV_MAX_STATE_TOKENS,
      maxBranchTokens: KEV_MAX_BRANCH_TOKENS,
      maxOptions: KEV_MAX_OPTIONS,
      ...(maxInputCharacters === undefined ? {} : {maxInputCharacters})
    } as unknown as ConstructorParameters<typeof DecisionModel>[6],
    [] as ConstructorParameters<typeof DecisionModel>[7],
    lifecycle as unknown as ConstructorParameters<typeof DecisionModel>[8],
    supportsSystemOne,
    modelId
  ) as TestModel;
  return {
    model,
    texts,
    lifecycle,
    get inferences() {
      return inferences;
    }
  };
}

test('Kev systemOne preserves typed Choice, Noul, and Score values in one packed request', async () => {
  const context = createModel();
  const {model, texts} = context;
  const result = await model.systemOne(
    {ticket: 'refund', priority: 2, details: {source: 'email'}, items: ['overcharged', false]},
    {
      department: {
        type: 'choice',
        instructions: 'Which team should handle this?',
        criteria: {billing: 'Charges and refunds', technical: null, other: {owner: 'unknown'}}
      },
      urgency: {type: 'score', instructions: 'How urgent is this?', criteria: ['can wait', 'this week', 'today']},
      refund: {type: 'noul', instructions: null, criteria: {true: 'asks for money back', false: 'something else'}}
    }
  );

  assert.equal(result.model, 'kev-0.6b-q4f16');
  assert.equal(context.inferences, 1);
  assert.equal(result.answers.department.type, 'choice');
  assert.equal(result.answers.department.choice, 'billing');
  assert.deepEqual(result.answers.department.probabilities, {billing: 0.9094, technical: 0.0453, other: 0.0453});
  assert.equal(result.answers.department.confidence, 0.8642);
  assert.equal(result.answers.urgency.type, 'score');
  assert.equal(result.answers.urgency.score, 1.5752);
  assert.equal(result.answers.urgency.confidence, 0.3628);
  assert.deepEqual(result.answers.urgency.probabilities, {0: 0.09, 1: 0.2447, 2: 0.6652});
  assert.deepEqual(result.answers.urgency.probabilityEntries, [
    ['0', 0.09],
    ['1', 0.2447],
    ['2', 0.6652]
  ]);
  assert.deepEqual(result.answers.urgency.legend, {0: 'can wait', 1: 'this week', 2: 'today'});
  assert.equal(result.answers.refund.noul, 0.8808);
  assert.deepEqual(texts, [
    'ticket: refund\npriority: 2\ndetails:\n  source: email\nitems:\n  - overcharged\n  - False',
    'Which team should handle this?',
    'billing: Charges and refunds',
    'technical',
    'other: owner: unknown',
    'How urgent is this?',
    'can wait',
    'this week',
    'today',
    '',
    'no: something else',
    'yes: asks for money back'
  ]);
  await model.dispose();
});

test('Kev typed Maps preserve numeric-key order in rendered input and result entries', async () => {
  const {model, texts} = createModel([
    [2, 0],
    [0, 2]
  ]);
  const result = await model.systemOne(
    new Map<'2' | '1', unknown>([
      ['2', 'second'],
      [
        '1',
        new Map<'10' | '3', unknown>([
          ['10', 10n],
          ['3', 3.5]
        ])
      ]
    ]),
    new Map<'10' | '2', unknown>([
      [
        '10',
        {
          type: 'choice',
          instructions: new Map<'8' | '4', unknown>([
            ['8', 'context'],
            ['4', new Map<'2', unknown>([['2', 'deep']])]
          ]),
          criteria: new Map<'2' | '1', unknown>([
            [
              '2',
              new Map<'10' | '3', unknown>([
                ['10', 'ten'],
                ['3', 3n]
              ])
            ],
            ['1', 'first']
          ])
        }
      ],
      [
        '2',
        {
          type: 'noul',
          instructions: null,
          criteria: new Map<'true' | 'false' | 'ignored', unknown>([
            ['true', 'yes'],
            ['false', 'no'],
            ['ignored', 'skip']
          ])
        }
      ]
    ])
  );

  assert.deepEqual(
    result.answerEntries.map(([name]) => name),
    ['10', '2']
  );
  assert.deepEqual(Object.keys(result.answers), ['2', '10']);
  assert.deepEqual(result.answerEntries[0]![1].probabilityEntries, [
    ['2', 0.8808],
    ['1', 0.1192]
  ]);
  assert.deepEqual(Object.keys(result.answers['10']!.probabilities!), ['1', '2']);
  assert.deepEqual(result.answers['10']!.probabilities, {1: 0.1192, 2: 0.8808});
  assert.deepEqual(texts, [
    '2: second\n1:\n  10: 10\n  3: 3.5',
    '8: context\n4:\n  2: deep',
    '2: 10: ten\n3: 3',
    '1: first',
    '',
    'no: no',
    'yes: yes'
  ]);
  await model.dispose();
});

test('Kev typed input accepts cross-realm Maps and bypasses overridden Map methods', () => {
  const state: unknown = runInNewContext('new Map([["2", "second"], ["1", "first"]])');
  const questions: unknown = runInNewContext(
    'new Map([["route", {type: "choice", instructions: null, criteria: new Map([["2", "two"], ["1", "one"]])}]])'
  );
  const prepared = prepareKevSystemOne(state, questions);
  assert.equal(prepared.request.state, '2: second\n1: first');
  assert.deepEqual(
    prepared.metadata[0]!.options.map(({key}) => key),
    ['2', '1']
  );

  const criteria = new Map<string, unknown>([
    ['second', '2'],
    ['first', '1']
  ]);
  criteria.entries = () => {
    throw new Error('The caller-owned entries method must not run.');
  };
  criteria[Symbol.iterator] = () => {
    throw new Error('The caller-owned iterator must not run.');
  };
  const overridden = prepareKevSystemOne('', {
    route: {type: 'choice', instructions: null, criteria}
  });
  assert.deepEqual(
    overridden.metadata[0]!.options.map(({key}) => key),
    ['second', 'first']
  );

  class KevMapSubclass extends Map<string, unknown> {}
  const subclass = prepareKevSystemOne(new KevMapSubclass([['state', 'value']]), {
    route: {type: 'choice', instructions: null, criteria: new KevMapSubclass([['key', null]])}
  });
  assert.equal(subclass.request.state, 'state: value');
  assert.deepEqual(
    subclass.metadata[0]!.options.map(({key}) => key),
    ['key']
  );
});

test('Kev typed input rejects structural Map-like objects and Proxy-wrapped Maps', () => {
  const entries: [string, string][] = [
    ['second', '2'],
    ['first', '1']
  ];
  const structuralMap = {
    size: entries.length,
    get() {},
    has() {},
    entries() {
      return entries[Symbol.iterator]();
    },
    *[Symbol.iterator]() {
      yield* entries;
    }
  };
  const proxyMap = new Proxy(new Map<string, string>(entries), {});

  for (const criteria of [structuralMap, proxyMap])
    assert.throws(() => prepareKevSystemOne('', {route: {type: 'choice', instructions: null, criteria}}), {
      code: 'INVALID_INPUT'
    });
});

test('Kev typed questions ignore unknown fields while validating known accessors', async () => {
  const {model} = createModel([[0, 2]]);
  const result = await model.systemOne('state', {
    check: {
      type: 'noul',
      instructions: null,
      context: {source: 'ignored'},
      get ignoredAccessor() {
        throw new Error('Unknown accessors should not be read.');
      }
    }
  });

  assert.equal(result.answers.check.type, 'noul');
  await assert.rejects(
    model.systemOne('state', {
      check: {
        type: 'noul',
        get instructions() {
          return 'not safe to read';
        }
      }
    }),
    {code: 'INVALID_INPUT'}
  );
  await model.dispose();
});

test('Kev Choice accepts empty and whitespace option keys', async () => {
  const {model, texts} = createModel([[2, 0]]);
  const result = await model.systemOne('state', {
    route: {
      type: 'choice',
      instructions: null,
      criteria: new Map([
        ['', 'empty key'],
        [' ', 'space key']
      ])
    }
  });

  assert.equal(result.answers.route.choice, '');
  assert.deepEqual(result.answers.route.probabilityEntries, [
    ['', 0.8808],
    [' ', 0.1192]
  ]);
  assert.deepEqual(texts.slice(2, 4), [': empty key', ' : space key']);
  await model.dispose();
});

test('Kev renders bigint exactly and rejects unsafe integer Numbers', () => {
  const prepared = prepareKevSystemOne(
    {
      exact: 9007199254740993123456789n,
      positive: 42n,
      negative: -42n,
      zero: 0n,
      safe: 9007199254740991,
      onePointZero: 1.0,
      negativeZero: -0.0,
      smallFloat: 1e-6,
      million: 1e6
    },
    {check: {type: 'noul', instructions: null}}
  );
  assert.equal(
    prepared.request.state,
    'exact: 9007199254740993123456789\npositive: 42\nnegative: -42\nzero: 0\nsafe: 9007199254740991\nonePointZero: 1\nnegativeZero: 0\nsmallFloat: 0.000001\nmillion: 1000000'
  );

  assert.throws(() => prepareKevSystemOne({unsafe: 9007199254740992}, {check: {type: 'noul', instructions: null}}), {
    code: 'INVALID_INPUT'
  });
});

test('Kev BigInt rendering preserves exact character-cap boundaries', () => {
  const question = {check: {type: 'noul', instructions: null}};
  assert.equal(prepareKevSystemOne(42n, question, 7).request.state, '42');
  assert.throws(() => prepareKevSystemOne(42n, question, 6), {code: 'INVALID_INPUT'});
  assert.equal(prepareKevSystemOne(-42n, question, 8).request.state, '-42');
  assert.throws(() => prepareKevSystemOne(-42n, question, 7), {code: 'INVALID_INPUT'});
  assert.equal(prepareKevSystemOne(0n, question, 6).request.state, '0');
  assert.throws(() => prepareKevSystemOne(0n, question, 5), {code: 'INVALID_INPUT'});
});

test('Kev rejects over-budget BigInts before decimal conversion', () => {
  const huge = 10n ** 100_000n;
  const question = {check: {type: 'noul', instructions: null}};
  const choice = {
    check: {type: 'choice', instructions: null, criteria: {large: huge}}
  };
  const score = {
    check: {type: 'score', instructions: null, criteria: ['low', huge]}
  };
  const originalToString = BigInt.prototype.toString;
  let conversions = 0;
  BigInt.prototype.toString = function (...args) {
    conversions++;
    return Reflect.apply(originalToString, this, args);
  };

  try {
    assert.throws(() => prepareKevSystemOne(huge, question, 64), {code: 'INVALID_INPUT'});
    assert.throws(() => prepareKevSystemOne({nested: [huge]}, question, 64), {code: 'INVALID_INPUT'});
    assert.throws(() => prepareKevSystemOne('', choice, 64), {code: 'INVALID_INPUT'});
    assert.throws(() => prepareKevSystemOne('', score, 64), {code: 'INVALID_INPUT'});
  } finally {
    BigInt.prototype.toString = originalToString;
  }

  assert.equal(conversions, 0);
});

test('Kev rejects oversized BigInt inputs before tokenizing or inferring', async () => {
  const huge = 10n ** 100_000n;
  const context = createModel(undefined, true, 'kev-0.6b-q4f16', 64);
  const question = {check: {type: 'noul', instructions: null}};

  await assert.rejects(context.model.systemOne(huge, question), {code: 'INVALID_INPUT'});
  await assert.rejects(context.model.systemOne({nested: [huge]}, question), {code: 'INVALID_INPUT'});
  await assert.rejects(
    context.model.systemOne('', {
      check: {type: 'choice', instructions: null, criteria: {large: huge}}
    }),
    {code: 'INVALID_INPUT'}
  );
  await assert.rejects(
    context.model.systemOne('', {
      check: {type: 'score', instructions: null, criteria: ['low', huge]}
    }),
    {code: 'INVALID_INPUT'}
  );
  assert.deepEqual(context.texts, []);
  assert.equal(context.inferences, 0);
  await context.model.dispose();
});

test('Kev character budget rejects near-boundary BigInts before tokenizing or inferring', async () => {
  const context = createModel(undefined, true, 'kev-0.6b-q4f16', 3);
  const question = {check: {type: 'noul', instructions: null}};

  await assert.rejects(context.model.systemOne(1000n, question), {code: 'INVALID_INPUT'});
  await assert.rejects(context.model.systemOne(-100n, question), {code: 'INVALID_INPUT'});
  assert.deepEqual(context.texts, []);
  assert.equal(context.inferences, 0);
  await context.model.dispose();
});

test('Kev Noul ignores unrelated criteria keys without enumerating them', async () => {
  const criteria = new Proxy(
    {true: 'yes', false: 'no', ignored: 'not part of the Noul protocol'},
    {
      ownKeys() {
        throw new Error('Noul criteria extras should not be enumerated.');
      }
    }
  );
  const {model, texts} = createModel([[0, 2]]);
  const result = await model.systemOne('state', {check: {type: 'noul', instructions: null, criteria}});

  assert.equal(result.answers.check.noul, 0.8808);
  assert.deepEqual(texts, ['state', '', 'no: no', 'yes: yes']);
  await model.dispose();
});

test('Kev Reflexes systemOne reports the selected fine-tuned model identity', async () => {
  const {model} = createModel(undefined, true, 'kev-0.6b-reflexes-q4f16');
  const result = await model.systemOne('A customer asks for a refund.', {
    refund: {type: 'noul', instructions: 'Is the customer asking for a refund?'}
  });

  assert.equal(result.model, 'kev-0.6b-reflexes-q4f16');
  assert.equal(result.answers.refund.type, 'noul');
  await model.dispose();
});

test('Kev systemOne rejects unsupported typed schemas before inference', async () => {
  const context = createModel();
  const {model} = context;
  const invalidQuestions = [
    {},
    {question: {type: 'unknown', instructions: 'Choose.', criteria: ['a', 'b']}},
    {question: {type: 'noul', instructions: undefined}},
    {question: {type: 'choice', instructions: 'Choose.', criteria: {}}},
    {question: {type: 'score', instructions: 'Rate.', criteria: []}}
  ];

  for (const questions of invalidQuestions) {
    await assert.rejects(
      model.systemOne('state', questions),
      (error) => error instanceof WebDecisionError && error.code === 'INVALID_INPUT'
    );
  }
  const cycle: Record<string, unknown> = {};
  cycle.self = cycle;
  await assert.rejects(
    model.systemOne(cycle, {check: {type: 'noul', instructions: 'Is this true?'}}),
    (error) => error instanceof WebDecisionError && error.code === 'INVALID_INPUT'
  );
  assert.equal(context.inferences, 0);
  await model.dispose();
});

test('Kev typed questions default omitted instructions to null like the upstream API', async () => {
  const context = createModel([
    [0, 1],
    [0, 1],
    [0, 1]
  ]);
  const {model, texts} = context;
  const result = await model.systemOne('state', {
    choice: {type: 'choice', criteria: {no: null, yes: null}},
    noul: {type: 'noul'},
    score: {type: 'score', criteria: ['low', 'high']}
  });

  assert.equal(context.inferences, 1);
  assert.equal(result.answers.choice.choice, 'yes');
  assert.equal(result.answers.noul.noul, 0.7311);
  assert.equal(result.answers.score.score, 0.7311);
  assert.deepEqual(texts, ['state', '', 'no', 'yes', '', 'no', 'yes', '', 'low', 'high']);
  await model.dispose();
});

test('Kev systemOne preserves a one-option choice and rejects question fields it cannot interpret', async () => {
  const {model, texts} = createModel([[0]]);
  const result = await model.systemOne('state', {only: {type: 'choice', instructions: null, criteria: {only: null}}});
  assert.equal(result.answers.only.choice, 'only');
  assert.deepEqual(result.answers.only.probabilities, {only: 1});
  assert.equal(result.answers.only.confidence, 1);
  assert.equal(texts[1], '');
  await model.dispose();
  await assert.rejects(model.systemOne('state', {check: {type: 'noul', instructions: null}}), {
    code: 'SESSION_CLOSED'
  });
});

test('Kev typed Score accepts one level as the upstream contract permits', async () => {
  const {model} = createModel([[0]]);
  const result = await model.systemOne('state', {
    score: {type: 'score', instructions: null, criteria: ['only level']}
  });

  assert.equal(result.answers.score.score, 0);
  assert.deepEqual(result.answers.score.legend, {0: 'only level'});
  assert.deepEqual(result.answers.score.probabilities, {0: 1});
  assert.equal(result.answers.score.confidence, 1);
  await model.dispose();
});

test('Kev Choice maps colliding rendered labels back to distinct criteria keys', async () => {
  const {model, texts} = createModel([[0, 2]]);
  const result = await model.systemOne('state', {
    route: {type: 'choice', instructions: null, criteria: {plain: 'item', 'plain: item': null}}
  });

  assert.equal(result.answers.route.choice, 'plain: item');
  assert.deepEqual(result.answers.route.probabilities, {plain: 0.1192, 'plain: item': 0.8808});
  assert.deepEqual(texts, ['state', '', 'plain: item', 'plain: item']);
  await model.dispose();
});

test('Kev Score preserves repeated empty level labels from the typed contract', async () => {
  const {model} = createModel([[2, 2]]);
  const result = await model.systemOne('state', {
    score: {type: 'score', instructions: null, criteria: [null, '']}
  });
  assert.equal(result.answers.score.type, 'score');
  assert.equal(result.answers.score.score, 0.5);
  assert.deepEqual(result.answers.score.legend, {0: '', 1: ''});
  assert.deepEqual(result.answers.score.probabilities, {0: 0.5, 1: 0.5});
  assert.equal(result.answers.score.confidence, 0);
  await model.dispose();
});

test('Kev Score serializes typed results to four decimals like the upstream API', async () => {
  // oxlint-disable-next-line unicorn/no-new-array -- Keep the fixture at eight dense zero logits.
  const {model} = createModel([new Array(8).fill(0)]);
  const result = await model.systemOne('state', {
    score: {type: 'score', instructions: null, criteria: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']}
  });

  assert.equal(result.answers.score.score, 3.5);
  assert.equal(result.answers.score.confidence, 0);
  assert.deepEqual(result.answers.score.probabilities, {
    0: 0.125,
    1: 0.125,
    2: 0.125,
    3: 0.125,
    4: 0.125,
    5: 0.125,
    6: 0.125,
    7: 0.125
  });
  await model.dispose();
});

test('Kev typed Choice, Noul, and Score use Python-compatible four-decimal rounding', () => {
  const prepared = prepareKevSystemOne('state', {
    choice: {type: 'choice', instructions: null, criteria: {low: null, high: null}},
    noul: {type: 'noul', instructions: null},
    score: {type: 'score', instructions: null, criteria: ['low', 'high']}
  });
  const probability = [0.03125, 0.96875];
  const result = finishKevSystemOne(prepared, {
    tokenCount: 0,
    answers: [
      {selectedIndex: 1, probabilities: probability.map((value) => ({probability: value}))},
      {probabilities: probability.map((value) => ({probability: value}))},
      {probabilities: probability.map((value) => ({probability: value}))}
    ]
  } as Parameters<typeof finishKevSystemOne>[1]) as TestResult<{choice: unknown; noul: unknown; score: unknown}>;

  assert.deepEqual(result.answers.choice.probabilities, {low: 0.0312, high: 0.9688});
  assert.equal(result.answers.choice.confidence, 0.9375);
  assert.equal(result.answers.noul.noul, 0.9688);
  assert.equal(result.answers.score.score, 0.9688);
  assert.deepEqual(result.answers.score.probabilities, {0: 0.0312, 1: 0.9688});
  assert.equal(result.answers.score.confidence, 0.9375);
});

test('Kev Score confidence uses the current TypeSafe distance normalization', () => {
  const prepared = prepareKevSystemOne('state', {
    concentrated: {type: 'score', criteria: ['low', 'middle', 'high']},
    uniform: {type: 'score', criteria: ['low', 'middle', 'high']},
    oneHot: {type: 'score', criteria: ['low', 'middle', 'high']},
    tiedMode: {type: 'score', criteria: ['first', 'second', 'third', 'fourth']},
    slightlyUnnormalized: {type: 'score', criteria: ['low', 'middle', 'high']},
    oneLevel: {type: 'score', criteria: ['only']}
  });
  const result = finishKevSystemOne(prepared, {
    tokenCount: 0,
    answers: [
      {probabilities: [0.75, 0, 0.25].map((probability) => ({probability}))},
      {probabilities: [1 / 3, 1 / 3, 1 / 3].map((probability) => ({probability}))},
      {probabilities: [0, 1, 0].map((probability) => ({probability}))},
      {probabilities: [0.4, 0.4, 0, 0.2].map((probability) => ({probability}))},
      {probabilities: [0.75, 0, 0.26].map((probability) => ({probability}))},
      {probabilities: [{probability: 1}]}
    ]
  } as Parameters<typeof finishKevSystemOne>[1]) as TestResult<{
    concentrated: unknown;
    uniform: unknown;
    oneHot: unknown;
    tiedMode: unknown;
    slightlyUnnormalized: unknown;
    oneLevel: unknown;
  }>;

  assert.equal(result.answers.concentrated.confidence, 0.25);
  assert.equal(result.answers.uniform.confidence, 0);
  assert.equal(result.answers.oneHot.confidence, 1);
  assert.equal(result.answers.tiedMode.confidence, 0);
  assert.equal(result.answers.slightlyUnnormalized.confidence, 0.2277);
  assert.equal(result.answers.oneLevel.confidence, 1);
});

test('Kev Choice confidence normalizes probabilities before applying the upstream formula', () => {
  const prepared = prepareKevSystemOne('state', {
    choice: {type: 'choice', criteria: {first: null, second: null}}
  });
  const result = finishKevSystemOne(prepared, {
    tokenCount: 0,
    answers: [{selectedIndex: 1, probabilities: [0.75, 0.26].map((probability) => ({probability}))}]
  } as Parameters<typeof finishKevSystemOne>[1]) as TestResult<{choice: unknown}>;

  assert.equal(result.answers.choice.confidence, 0.4851);
});

test('Kev confidence normalizes a zero-sum probability vector like the upstream API', () => {
  const prepared = prepareKevSystemOne('state', {
    choice: {type: 'choice', criteria: {first: null, second: null}}
  });
  const result = finishKevSystemOne(prepared, {
    tokenCount: 0,
    answers: [{selectedIndex: 0, probabilities: [{probability: 0}, {probability: 0}]}]
  } as Parameters<typeof finishKevSystemOne>[1]) as TestResult<{choice: unknown}>;

  assert.equal(result.answers.choice.confidence, 0);
});

test('Kev typed Score rounds valid values through the maximum 254-level score', () => {
  const prepared = prepareKevSystemOne('state', {
    score: {
      type: 'score',
      instructions: null,
      criteria: Array.from({length: KEV_MAX_OPTIONS}, (_, index) => String(index))
    }
  });
  const probabilities = Array.from({length: KEV_MAX_OPTIONS}, (_, index) => ({
    probability: index === KEV_MAX_OPTIONS - 1 ? 1 : 0
  }));
  const result = finishKevSystemOne(prepared, {
    tokenCount: 0,
    answers: [{probabilities}]
  } as Parameters<typeof finishKevSystemOne>[1]) as TestResult<{score: unknown}>;

  assert.equal(result.answers.score.score, KEV_MAX_OPTIONS - 1);
  assert.ok(result.answers.score.score >= 0 && result.answers.score.score <= KEV_MAX_OPTIONS - 1);
  assert.equal(result.answers.score.confidence, 1);
  assert.equal(result.answers.score.probabilities![String(KEV_MAX_OPTIONS - 1)], 1);
  assert.equal(result.answers.score.probabilityEntries!.length, KEV_MAX_OPTIONS);
});

test('Kev typed Score accepts 255 ordered levels and rejects 256', () => {
  const accepted = prepareKevSystemOne('state', {
    score: {type: 'score', criteria: Array.from({length: KEV_MAX_OPTIONS}, (_, index) => String(index))}
  });
  assert.equal(accepted.metadata[0]!.options.length, KEV_MAX_OPTIONS);
  assert.throws(
    () =>
      prepareKevSystemOne('state', {
        score: {type: 'score', criteria: Array.from({length: KEV_MAX_OPTIONS + 1}, (_, index) => String(index))}
      }),
    {code: 'INVALID_INPUT'}
  );
});

test('Kev routes without typed System One support reject that input', async () => {
  const {model} = createModel([[0, 2]], false);
  await assert.rejects(model.systemOne('state', {question: {type: 'noul', instructions: null}}), {
    code: 'UNSUPPORTED_ENVIRONMENT'
  });
  await model.dispose();
});

test('Kev systemOne checks an aborted signal before rendering typed input', async () => {
  const context = createModel();
  const {model, texts} = context;
  const controller = new AbortController();
  controller.abort();
  let state: unknown = 'value';
  for (let depth = 0; depth < 2_000; depth++) state = {next: state};

  await assert.rejects(
    model.systemOne(state, {check: {type: 'noul', instructions: null}}, {signal: controller.signal}),
    (error) => (error as Error).name === 'AbortError'
  );
  assert.deepEqual(texts, []);
  assert.equal(context.inferences, 0);
  await model.dispose();
});

test('Kev systemOne accepts the upstream 255 ordered Choice criteria and rejects the 256th', async () => {
  const criteria = Object.fromEntries(
    Array.from({length: 255}, (_, index) => [String.fromCodePoint(0x4e00 + index), null])
  );
  // oxlint-disable-next-line unicorn/no-new-array -- Keep one dense zero logit for every option.
  const context = createModel([new Array(KEV_MAX_OPTIONS).fill(0)]);
  const result = await context.model.systemOne('s', {route: {type: 'choice', instructions: '', criteria}});
  assert.equal(result.answers.route.choice, Object.keys(criteria)[0]);
  assert.equal(Object.keys(result.answers.route.probabilities!).length, KEV_MAX_OPTIONS);
  assert.equal(result.tokenCount, 769);

  const tooMany = {...criteria, [String.fromCodePoint(0x4e00 + 255)]: null};
  await assert.rejects(context.model.systemOne('s', {route: {type: 'choice', instructions: '', criteria: tooMany}}), {
    code: 'INVALID_INPUT'
  });
  assert.equal(context.inferences, 1);
  await context.model.dispose();
});

test('Kev systemOne applies an opt-in packed-token cap without truncating typed input', async () => {
  const context = createModel();
  const {model, texts} = context;
  model.limits.maxInputTokens = 8;

  await assert.rejects(model.systemOne('long state', {check: {type: 'noul', instructions: 'Choose'}}), {
    code: 'INVALID_INPUT'
  });
  assert.deepEqual(texts, ['long state']);
  assert.equal(context.inferences, 0);
  await model.dispose();
});

test('Kev systemOne accepts more than 255 named questions without a fixed aggregate limit', async () => {
  const accepted = Object.fromEntries(
    Array.from({length: 600}, (_, index) => [`q${index}`, {type: 'noul', instructions: null}])
  );
  const prepared = prepareKevSystemOne('', accepted);
  assert.equal(prepared.request.questions.length, 600);
});

test('Kev character safety cap is opt-in and applies to rendered request text', async () => {
  const maxInputCharacters = 64;
  const boundaryState = 's'.repeat(maxInputCharacters - 5);
  const prepared = prepareKevSystemOne(boundaryState, {q: {type: 'noul', instructions: null}}, maxInputCharacters);
  assert.equal(prepared.request.state.length, maxInputCharacters - 5);
  assert.throws(
    () => prepareKevSystemOne(`${boundaryState}s`, {q: {type: 'noul', instructions: null}}, maxInputCharacters),
    {
      code: 'INVALID_INPUT'
    }
  );

  const aggregateState = 's'.repeat(maxInputCharacters - 5);
  const aggregateQuestions = {q: {type: 'choice', instructions: 'i', criteria: {a: 'd'}}};
  assert.doesNotThrow(() => prepareKevSystemOne(aggregateState, aggregateQuestions, maxInputCharacters));
  assert.throws(() => prepareKevSystemOne(`${aggregateState}s`, aggregateQuestions, maxInputCharacters), {
    code: 'INVALID_INPUT'
  });

  const context = createModel(undefined, true, 'kev-0.6b-q4f16', maxInputCharacters);
  const {model, texts} = context;
  const question = {check: {type: 'noul', instructions: null}};

  await assert.rejects(
    model.decide({
      state: 's'.repeat(maxInputCharacters + 1),
      questions: [{instructions: 'Choose', options: ['a', 'b']}]
    }),
    {code: 'INVALID_INPUT'}
  );
  await assert.rejects(model.systemOne('s'.repeat(maxInputCharacters + 1), question), {
    code: 'INVALID_INPUT'
  });
  let state: unknown = {value: 'x'};
  for (let depth = 0; depth < 20; depth++) state = {left: state, right: state};
  await assert.rejects(model.systemOne(state, question), {code: 'INVALID_INPUT'});
  assert.deepEqual(texts, []);
  assert.equal(context.inferences, 0);
  await model.dispose();
});

test('Kev character safety cap counts Python-stripped array prefixes after rendering', () => {
  const state = {items: ['\u0085visible']};
  const renderedState = 'items:\n  - visible';
  const maxInputCharacters = renderedState.length + 'no'.length + 'yes'.length;
  const questions = {check: {type: 'noul', instructions: null}};
  const prepared = prepareKevSystemOne(state, questions, maxInputCharacters);

  assert.equal(prepared.request.state, renderedState);
  assert.throws(() => prepareKevSystemOne(state, questions, maxInputCharacters - 1), {
    code: 'INVALID_INPUT'
  });
});

test('Kev character safety cap rejects oversized array output before building it', async () => {
  const question = {check: {type: 'noul', instructions: null}};
  assert.throws(() => prepareKevSystemOne(['x'.repeat(1_000_000)], question, 64), {
    code: 'INVALID_INPUT'
  });
  assert.throws(() => prepareKevSystemOne([' '.repeat(1_000_000) + 'x'.repeat(1_000_000)], question, 64), {
    code: 'INVALID_INPUT'
  });

  let nested: unknown = 'x';
  for (let depth = 0; depth < 10_000; depth++) nested = [nested];
  assert.throws(() => prepareKevSystemOne(nested, question, 64), {code: 'INVALID_INPUT'});

  const context = createModel(undefined, true, 'kev-0.6b-q4f16', 64);
  await assert.rejects(context.model.systemOne(['x'.repeat(1_000_000)], question), {code: 'INVALID_INPUT'});
  await assert.rejects(context.model.systemOne([' '.repeat(1_000_000) + 'x'.repeat(1_000_000)], question), {
    code: 'INVALID_INPUT'
  });
  assert.deepEqual(context.texts, []);
  assert.equal(context.inferences, 0);
  await context.model.dispose();
});

test('Kev character safety cap accepts a long array prefix that renders away', () => {
  const prepared = prepareKevSystemOne([' '.repeat(1_000_000)], {check: {type: 'noul', instructions: null}}, 7);
  assert.equal(prepared.request.state, '- ');
});

test('Kev has a configured character cap before tokenizing repeated shared JSON values', async () => {
  const maxInputCharacters = 4096;
  const context = createModel(undefined, true, 'kev-0.6b-q4f16', maxInputCharacters);
  const {model, texts} = context;
  const question = {check: {type: 'noul', instructions: null}};
  let state: unknown = {value: 'x'};
  for (let depth = 0; depth < 20; depth++) state = {left: state, right: state};
  await assert.rejects(model.systemOne(state, question), {code: 'INVALID_INPUT'});
  assert.deepEqual(texts, []);
  assert.equal(context.inferences, 0);
  await model.dispose();
});

test('Kev does not reject text by character count when no safety cap was configured', async () => {
  const context = createModel();
  const state = 's'.repeat(65_537);
  const checked = await context.model
    .systemOne(state, {check: {type: 'noul', instructions: null}})
    .catch((error) => error);
  assert.equal(checked.code, 'INVALID_INPUT');
  assert.equal(checked.message.includes('character'), false);
  assert.equal(context.texts.at(-1)!.length, 65_537);
  await context.model.dispose();
});
