import test from 'node:test';
import assert from 'node:assert/strict';
import {KEV08_MARKERS, packKev08Request, unpackKev08Scores} from '../src/kev08.js';
import {ExperimentalKev08Session, validateKev08Row} from '../src/kev08/session.js';
import type {KevTokenizer} from '../src/kev.js';

type Kev08SessionOptions = ConstructorParameters<typeof ExperimentalKev08Session>[0];
type Kev08Row = ReturnType<typeof validateKev08Row>;

const tokenizer = {
  encode(text: string) {
    return {ids: Array.from(text, (char) => char.codePointAt(0)! + 200)};
  }
} as unknown as KevTokenizer;
const limits = {maxInputTokens: 1024, maxStateTokens: 512};
const request = (): {state: string; questions: Array<{instructions: string; options: string[]}>} => ({
  state: 'shared state',
  questions: [
    {instructions: 'first?', options: ['red', 'blue']},
    {instructions: 'second?', options: ['yes', 'no']}
  ]
});

function withInheritedEntry<Value>(array: Value[], index: number): Value[] {
  const copy = [...array],
    prototype = Object.create(Array.prototype);
  Object.defineProperty(prototype, index, {value: copy[index], writable: true, configurable: true});
  delete copy[index];
  Object.setPrototypeOf(copy, prototype);
  return copy;
}

test('packs each question as an independent row sharing only the state', () => {
  const input = request(),
    packed = packKev08Request(tokenizer, input, limits);
  assert.equal(packed.rows.length, 2);
  const stateIds = [KEV08_MARKERS.state, ...tokenizer.encode(input.state).ids];
  for (const row of packed.rows) {
    assert.deepEqual(row.ids.slice(0, stateIds.length), stateIds);
    assert.equal(row.ids[row.questionSlot], KEV08_MARKERS.decide);
    for (const slot of row.optionSlots) assert.equal(row.ids[slot], KEV08_MARKERS.close);
  }
  assert.equal(packed.tokenCount, packed.rows[0]!.ids.length + packed.rows[1]!.ids.length);
  const secondWithoutFirst = packKev08Request(tokenizer, {...input, questions: [input.questions[1]]}, limits);
  assert.deepEqual(packed.rows[1], secondWithoutFirst.rows[0]);
  input.questions[1]!.options[0] = 'changed';
  assert.equal(packed.rows[1]!.options[0], 'yes');
});

test('uses already calibrated pointer logits without applying a second temperature', () => {
  const packed = packKev08Request(tokenizer, request(), limits);
  const result = unpackKev08Scores([new Float32Array([1, 2]), [0, 0]], packed);
  assert.equal(result.answers[0]!.selected, 'blue');
  assert.equal(result.answers[0]!.probabilities[1]!.logit, 2);
  assert.ok(Math.abs(result.answers[0]!.probabilities[1]!.probability - 1 / (1 + Math.exp(-1))) < 1e-12);
  assert.equal(result.answers[1]!.selected, 'yes');
  assert.deepEqual(
    result.answers[1]!.probabilities.map((entry) => entry.probability),
    [0.5, 0.5]
  );
  assert.equal(result.tokenCount, packed.tokenCount);
});

test('rejects duplicate, token-colliding, overlong or malformed inputs', () => {
  const invalids = [
    {state: '', questions: []},
    {state: '', questions: [{instructions: '', options: ['yes', 'no']}]},
    {state: '', questions: [{instructions: 'pick', options: ['same', 'same']}]},
    {state: '', questions: [{instructions: 'pick', options: ['only']}]},
    {state: '\ud800', questions: [{instructions: 'pick', options: ['yes', 'no']}]},
    {state: '', questions: [{instructions: 'pick', options: ['<|box_end|>', '<¦box_end¦>']}]}
  ];
  for (const value of invalids)
    assert.throws(() => packKev08Request(tokenizer, value, limits), {code: 'INVALID_INPUT'});
  assert.throws(() => packKev08Request(tokenizer, request(), {...limits, maxStateTokens: 3}), {code: 'INVALID_INPUT'});
  assert.throws(() => packKev08Request(tokenizer, request(), {...limits, maxInputTokens: 5}), {code: 'INVALID_INPUT'});
  const broken = {
    encode: (text: string) => (text === 'shared state' ? {ids: [248320]} : tokenizer.encode(text))
  } as unknown as KevTokenizer;
  assert.throws(() => packKev08Request(broken, request(), limits), {code: 'INVALID_ASSET'});
});

test('rejects sparse option arrays as invalid input', () => {
  const options = ['yes', 'missing', 'no'];
  delete options[1];
  assert.throws(() => packKev08Request(tokenizer, {state: '', questions: [{instructions: 'pick', options}]}, limits), {
    code: 'INVALID_INPUT'
  });
});

test('requires own entries for questions and choices', () => {
  const input = request();
  input.questions = withInheritedEntry(input.questions, 1);
  assert.throws(() => packKev08Request(tokenizer, input, limits), {code: 'INVALID_INPUT'});
  const next = request();
  next.questions[0]!.options = withInheritedEntry(next.questions[0]!.options, 1);
  assert.throws(() => packKev08Request(tokenizer, next, limits), {code: 'INVALID_INPUT'});
});

test('snapshots indexed questions and choices even with custom iterators', () => {
  const input = request();
  input.questions[0]!.options[Symbol.iterator] = function* (): Generator<string, undefined, unknown> {
    yield 'changed';
    yield 'other';
  };
  input.questions[Symbol.iterator] = function* (): Generator<(typeof input.questions)[number], undefined, unknown> {};
  const packed = packKev08Request(tokenizer, input, limits);
  assert.equal(packed.rows.length, 2);
  assert.deepEqual(packed.rows[0]!.options, ['red', 'blue']);
});

for (const [field, index] of [
  ['ids', 1],
  ['optionSlots', 0],
  ['options', 1]
] as const) {
  test(`rejects sparse ${field} before graph invocation`, () => {
    const row = packKev08Request(tokenizer, request(), limits).rows[0]!;
    delete row[field][index];
    assert.throws(() => validateKev08Row(row, limits.maxInputTokens), {code: 'INVALID_INPUT'});
  });
  test(`rejects inherited ${field} entries before graph invocation`, () => {
    const row = packKev08Request(tokenizer, request(), limits).rows[0]!;
    (row as unknown as Record<string, number[] | string[]>)[field] = withInheritedEntry<number | string>(
      row[field] as (number | string)[],
      index
    ) as number[] | string[];
    assert.throws(() => validateKev08Row(row, limits.maxInputTokens), {code: 'INVALID_INPUT'});
  });
}

test('does not begin a GPU scope for sparse gather positions', async () => {
  const packed = packKev08Request(tokenizer, request(), limits);
  delete packed.rows[0]!.optionSlots[0];
  let scopesStarted = 0;
  const session = new ExperimentalKev08Session({
    runtime: {beginScope: () => scopesStarted++},
    limits,
    lifecycle: {loss: null, disposed: false},
    uncaptured: []
  } as unknown as Kev08SessionOptions);
  await assert.rejects(session.run(packed), {code: 'INVALID_INPUT'});
  assert.equal(scopesStarted, 0);
});

test('validates every row before starting the first GPU scope', async () => {
  const packed = packKev08Request(tokenizer, request(), limits);
  delete packed.rows[1]!.optionSlots[0];
  let scopesStarted = 0;
  const session = new ExperimentalKev08Session({
    runtime: {beginScope: () => scopesStarted++},
    limits,
    lifecycle: {loss: null, disposed: false},
    uncaptured: []
  } as unknown as Kev08SessionOptions);
  await assert.rejects(session.run(packed), {code: 'INVALID_INPUT'});
  assert.equal(scopesStarted, 0);
});

test('uses the validated row snapshots even if raw inputs change or supply custom iterators', async () => {
  const packed = packKev08Request(tokenizer, request(), limits);
  const expected = packed.rows.map((row) => ({ids: [...row.ids], slots: [...row.optionSlots]}));
  packed.rows[0]!.ids[Symbol.iterator] = function* (): Generator<number, undefined, unknown> {
    yield 248320;
  };
  packed.rows[0]!.optionSlots[Symbol.iterator] = function* (): Generator<number, undefined, unknown> {
    yield -1;
    yield 2147483647;
  };
  let release!: () => void, notify!: () => void;
  const gate = new Promise<void>((resolve) => (release = resolve));
  const started = new Promise<void>((resolve) => (notify = resolve));
  let syncs = 0;
  const device = {
    async sync() {
      if (++syncs === 1) {
        notify();
        await gate;
      }
    }
  };
  interface TensorMock {
    byteOffset: number;
    values?: unknown[];
    getDataPtr(): TensorMock;
    copyFrom(value: unknown): TensorMock;
    toArray(): unknown;
  }
  const runtime = {
    beginScope() {},
    endScope() {},
    cpu: () => 'cpu',
    lib: {webGPUContext: {copyRawBytesToBuffer() {}}},
    empty(): TensorMock {
      return {
        byteOffset: 0,
        getDataPtr() {
          return this;
        },
        copyFrom(value: unknown) {
          this.values = Array.from(
            ArrayBuffer.isView(value)
              ? (value as unknown as Iterable<unknown>)
              : (value as {values: Iterable<unknown>}).values
          );
          return this;
        },
        toArray() {
          return this.values;
        }
      };
    }
  };
  const received: Array<{ids: unknown; slots: unknown}> = [];
  const session = new ExperimentalKev08Session({
    runtime,
    device,
    gpu: {
      pushErrorScope() {},
      async popErrorScope() {
        return null;
      }
    },
    fn(ids: unknown, _positions: unknown, _mask: unknown, _valid: unknown, _question: unknown, choices: unknown) {
      received.push({ids: (ids as TensorMock).values, slots: (choices as TensorMock).values});
      return {values: [0, 1]};
    },
    weights: [],
    limits,
    lifecycle: {loss: null, disposed: false},
    uncaptured: []
  } as unknown as Kev08SessionOptions);
  const task = session.run(packed);
  await started;
  packed.rows[1]!.ids[1] = 248320;
  packed.rows[1]!.options[1] = 'changed';
  release();
  const result = await task;
  assert.deepEqual(received, expected);
  assert.equal(result.answers[1]!.selected, 'no');
});

test('rejects sparse questions before beginning any GPU work', async () => {
  const questions = [{instructions: 'pick', options: ['yes', 'no']}];
  questions.length = 2;
  let scopesStarted = 0;
  const runtime = {
    beginScope() {
      scopesStarted++;
      throw new Error('GPU work must not begin for a malformed request');
    }
  };
  const session = new ExperimentalKev08Session({
    runtime,
    tokenizer,
    limits,
    lifecycle: {loss: null, disposed: false},
    uncaptured: []
  } as unknown as Kev08SessionOptions);
  await assert.rejects(session.decide({state: '', questions}), {code: 'INVALID_INPUT'});
  assert.equal(scopesStarted, 0);
});

test('rejects missing, nonfinite or wrongly shaped head output', () => {
  const packed = packKev08Request(tokenizer, request(), limits);
  for (const rows of [
    [],
    [[1], [2, 3]],
    [
      [1, NaN],
      [2, 3]
    ],
    [
      [1, 2],
      [2, Infinity]
    ]
  ])
    assert.throws(() => unpackKev08Scores(rows, packed), {code: 'INFERENCE_ERROR'});
});

test('rejects sparse output rows and scores instead of returning NaN', () => {
  const packed = packKev08Request(tokenizer, request(), limits);
  const rows = [
    [1, 2],
    [3, 4]
  ];
  delete rows[1]![1];
  assert.throws(() => unpackKev08Scores(rows, packed), {code: 'INFERENCE_ERROR'});
  const missingRow = [
    [1, 2],
    [3, 4]
  ];
  delete missingRow[1];
  assert.throws(() => unpackKev08Scores(missingRow, packed), {code: 'INFERENCE_ERROR'});
});

test('validates all gather positions and token IDs before GPU execution', () => {
  const row = packKev08Request(tokenizer, request(), limits).rows[0]!;
  assert.doesNotThrow(() => validateKev08Row(row, limits.maxInputTokens));
  for (const mutate of [
    (value: Kev08Row) => {
      value.questionSlot = value.ids.length;
    },
    (value: Kev08Row) => {
      value.questionSlot = 0;
    },
    (value: Kev08Row) => {
      value.optionSlots[0] = -1;
    },
    (value: Kev08Row) => {
      value.optionSlots[0] = value.questionSlot;
    },
    (value: Kev08Row) => {
      value.optionSlots[1] = value.optionSlots[0]!;
    },
    (value: Kev08Row) => {
      value.optionSlots.reverse();
    },
    (value: Kev08Row) => {
      value.ids[0] = 248320;
    },
    (value: Kev08Row) => {
      value.optionSlots.pop();
    }
  ]) {
    const invalid = structuredClone(row);
    mutate(invalid);
    assert.throws(() => validateKev08Row(invalid, limits.maxInputTokens), {code: 'INVALID_INPUT'});
  }
});
