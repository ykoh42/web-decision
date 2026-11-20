import assert from 'node:assert/strict';
import test from 'node:test';
import {readQwen35GraphProbabilityRows} from '../src/strands/qwen35-graph-output.mts';
import {finishQwen35Answers, snapshotQwen35Request} from '../src/strands/qwen35-protocol.mts';

type FakeTensor = Readonly<{
  dtype: string;
  shape: number[];
  device: Readonly<{deviceType: number}>;
  toArray?: () => unknown;
}>;

type FakeTvmTuple = Readonly<{size: () => number; get: (index: number) => unknown}>;

function fakeTensor(shape: number[], dtype: string, deviceType: number, values?: unknown): FakeTensor {
  return {
    dtype,
    shape,
    device: {deviceType},
    toArray: () => (typeof values === 'function' ? values() : values)
  };
}

function tuple(probabilities: FakeTensor, logits = fakeTensor([2, 3], 'float32', 15)): FakeTvmTuple {
  const values = [logits, probabilities];
  return {size: () => values.length, get: (index) => values[index]};
}

test('Qwen R6/R28 output decoder reads the masked-probability tensor and preserves padded mixed-K rows', () => {
  const logits = fakeTensor([2, 3], 'float32', 15, () => {
    throw new Error('The tuple decoder must not read raw logits.');
  });
  const probabilities = fakeTensor([2, 3], 'float32', 1, Float32Array.of(0.4, 0.6, 0, 0.1, 0.2, 0.7));
  const rows = readQwen35GraphProbabilityRows(tuple(probabilities, logits), [2, 3]);

  assert.deepEqual(rows, [Float32Array.of(0.4, 0.6, 0), Float32Array.of(0.1, 0.2, 0.7)]);
  const request = snapshotQwen35Request({
    state: 'x',
    questions: {
      two: {type: 'choice', instructions: 'Pick.', criteria: {first: null, second: null}},
      three: {type: 'choice', instructions: 'Pick.', criteria: {left: null, middle: null, right: null}}
    }
  });
  const answers = finishQwen35Answers(request, [rows]);
  assert.deepEqual(
    answers.map(({answer}) => (answer.type === 'choice' ? answer.choice : null)),
    ['second', 'right']
  );
  assert.deepEqual(
    answers.map(({answer}) => (answer.type === 'choice' ? answer.probabilities : null)),
    [
      {first: 0.4, second: 0.6},
      {left: 0.1, middle: 0.2, right: 0.7}
    ]
  );
});

test('Qwen graph output preserves K=2, K=10, and K=255 order and mixed-K probabilities', () => {
  const optionCounts = [2, 10, 255] as const;
  const width = 255;

  for (const optionCount of optionCounts) {
    const exactWidthValues = Float32Array.from({length: optionCount}, (_, index) => (index + 1) / 512);
    const exactWidthRows = readQwen35GraphProbabilityRows(
      tuple(
        fakeTensor([1, optionCount], 'float32', 1, exactWidthValues),
        fakeTensor([1, optionCount], 'float32', 15, new Float32Array(optionCount))
      ),
      [optionCount]
    );
    assert.deepEqual(exactWidthRows, [exactWidthValues]);
  }

  const rawLogits = Float32Array.from({length: optionCounts.length * width}, (_, index) => index / 16);
  const originalRawLogits = rawLogits.slice();
  const probabilityValues = new Float32Array(optionCounts.length * width);
  probabilityValues.set([0.25, 0.75], 0);
  probabilityValues.fill(Math.fround(0.01), width, width + optionCounts[1]);
  probabilityValues.fill(Math.fround(1 / width), 2 * width, 3 * width);
  const logits = fakeTensor([3, width], 'float32', 15, rawLogits);
  const probabilities = fakeTensor([3, width], 'float32', 1, probabilityValues);
  const rows = readQwen35GraphProbabilityRows(tuple(probabilities, logits), optionCounts);

  assert.deepEqual(
    rows.map((row) => row.length),
    [255, 255, 255]
  );
  assert.deepEqual(Array.from(rows[0]!.slice(0, 2)), [0.25, 0.75]);
  assert.ok(rows[0]!.slice(2).every((value) => value === 0));
  assert.ok(rows[1]!.slice(0, 10).every((value) => value === Math.fround(0.01)));
  assert.ok(rows[1]!.slice(10).every((value) => value === 0));
  assert.ok(rows[2]!.every((value) => value === Math.fround(1 / width)));
  assert.deepEqual(rawLogits, originalRawLogits);

  const twoLabels = ['first', 'second'];
  const scoreLabels = Array.from({length: 10}, (_, index) => `level-${index}`);
  const maxLabels = Array.from({length: 255}, (_, index) => `option-${String(index).padStart(3, '0')}`);
  const request = snapshotQwen35Request({
    state: 'x',
    questions: {
      two: {
        type: 'choice',
        instructions: 'Pick.',
        criteria: Object.fromEntries(twoLabels.map((label) => [label, null]))
      },
      ten: {type: 'score', instructions: 'Rate.', criteria: scoreLabels},
      max: {
        type: 'choice',
        instructions: 'Pick.',
        criteria: Object.fromEntries(maxLabels.map((label) => [label, null]))
      }
    }
  });
  const answers = finishQwen35Answers(request, [rows]);
  assert.deepEqual(
    answers.map(({name}) => name),
    ['two', 'ten', 'max']
  );
  assert.deepEqual(
    answers.map(({answer}) => answer.type),
    ['choice', 'score', 'choice']
  );
  assert.deepEqual(answers[0]!.answer, {
    type: 'choice',
    choice: 'second',
    probabilities: {first: 0.25, second: 0.75},
    confidence: 0.5
  });
  assert.deepEqual(
    Object.values(answers[1]!.answer.type === 'score' ? answers[1]!.answer.probabilities : {}),
    Array.from({length: 10}, () => 0.01)
  );
  assert.deepEqual(
    Object.keys(answers[2]!.answer.type === 'choice' ? answers[2]!.answer.probabilities : {}),
    maxLabels
  );
});

test('Qwen R6/R28 output decoder rejects malformed ABI metadata and nonzero masked columns', () => {
  const probabilities = fakeTensor([2, 3], 'float32', 1, Float32Array.of(0.4, 0.6, 0, 0.1, 0.2, 0.7));
  assert.throws(
    () => readQwen35GraphProbabilityRows([fakeTensor([2, 3], 'float32', 15), probabilities], [2]),
    /batch/u
  );
  assert.throws(
    () =>
      readQwen35GraphProbabilityRows(
        [fakeTensor([2, 3], 'float32', 15), fakeTensor([2, 3], 'float16', 1, new Float32Array(6))],
        [2, 3]
      ),
    /dtype float32/u
  );
  assert.throws(
    () =>
      readQwen35GraphProbabilityRows(
        tuple(fakeTensor([2, 3], 'float32', 1, Float32Array.of(0.4, 0.6, 0.1, 0.1, 0.2, 0.7))),
        [2, 3]
      ),
    /zero after n_slots/u
  );
  assert.throws(() => readQwen35GraphProbabilityRows([fakeTensor([2, 3], 'float32', 15)], [2, 3]), /two-tensor/u);
  assert.throws(
    () => readQwen35GraphProbabilityRows(tuple(fakeTensor([2, 3, 1], 'float32', 1, new Float32Array(6))), [2, 3]),
    /positive \[batch, options\] shape/u
  );
  assert.throws(
    () => readQwen35GraphProbabilityRows(tuple(probabilities, fakeTensor([2, 2], 'float32', 15)), [2, 3]),
    /matching shapes/u
  );
  assert.throws(
    () => readQwen35GraphProbabilityRows(tuple(fakeTensor([2, 3], 'float32', 15, new Float32Array(6))), [2, 3]),
    /CPU-readable/u
  );
  assert.throws(
    () => readQwen35GraphProbabilityRows(tuple(fakeTensor([2, 3], 'float32', 1, new Int32Array(6))), [2, 3]),
    /Float32Array/u
  );
  assert.throws(
    () =>
      readQwen35GraphProbabilityRows(
        tuple(fakeTensor([1, 256], 'float32', 1, new Float32Array(256)), fakeTensor([1, 256], 'float32', 15)),
        [2]
      ),
    /option width must be from 2 to 255/u
  );
  assert.throws(
    () =>
      readQwen35GraphProbabilityRows(
        tuple(fakeTensor([1, 2], 'float32', 1, new Float32Array(2)), fakeTensor([1, 2], 'float32', 15)),
        [1]
      ),
    /n_slots must contain option counts from 2 to 255/u
  );
});
