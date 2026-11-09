import assert from 'node:assert/strict';
import test from 'node:test';
import {WebDecisionError} from '../src/kev.js';
import {createOpenDeciderNanoTokenizer, OPENDECIDER_NANO_TOKENS} from '../src/opendecider-nano/tokenizer.mjs';

type EncodeOptions = Readonly<{add_special_tokens: false; split_special_tokens: true}>;
type EncodeBehavior = (text: string) => unknown;
type ConstructionObserver = (data: unknown, config: unknown) => void;

const firstSentinel = 'question: Which team should handle this?';
const secondSentinel = 'input: I was charged twice for order 1182.';
const firstSentinelIds = [19751, 27, 6758, 2285, 943, 6016, 436, 32];
const secondSentinelIds = [5423, 27, 309, 369, 6636, 7019, 323, 1340, 12643, 19, 15];

function validData(): unknown {
  return {model: {vocab: {last: 50366}}, added_tokens: [{id: 50367}]};
}

function tokenizerConstructor(encode: EncodeBehavior, observe?: ConstructionObserver) {
  return class StubTokenizer {
    constructor(data: unknown, config: unknown) {
      observe?.(data, config);
    }

    encode(text: string, options: EncodeOptions): {readonly ids: unknown} {
      assert.deepEqual(options, {add_special_tokens: false, split_special_tokens: true});
      return {ids: encode(text)};
    }
  };
}

function referenceEncoding(text: string): unknown {
  if (text === firstSentinel) return firstSentinelIds;
  if (text === secondSentinel) return secondSentinelIds;
  return [12, 13];
}

function assertInvalidAsset(operation: () => unknown): void {
  assert.throws(operation, (error: unknown) => error instanceof WebDecisionError && error.code === 'INVALID_ASSET');
}

test('constructs the pinned tokenizer, validates sentinels in order, and copies ordinary token IDs', () => {
  const data = validData();
  const config = {};
  const construction: unknown[][] = [];
  const encoded: string[] = [];
  const Constructor = tokenizerConstructor(
    (text) => {
      encoded.push(text);
      return referenceEncoding(text);
    },
    (actualData, actualConfig) => construction.push([actualData, actualConfig])
  );

  const tokenizer = createOpenDeciderNanoTokenizer(Constructor, data, config);

  assert.deepEqual(construction, [[data, config]]);
  assert.deepEqual(encoded, [firstSentinel, secondSentinel]);
  assert.deepEqual(tokenizer.encode('ordinary input'), [12, 13]);
  assert.equal(Object.isFrozen(tokenizer), true);
  assert.deepEqual(OPENDECIDER_NANO_TOKENS, {cls: 50281, sep: 50282, pad: 50283, mask: 50284});
});

test('rejects invalid constructor inputs, vocabulary IDs, and maximum vocabulary size', () => {
  const validConstructor = tokenizerConstructor(referenceEncoding);
  const invalidArguments: ReadonlyArray<readonly [unknown, unknown, unknown]> = [
    [null, validData(), {}],
    [validConstructor, null, {}],
    [validConstructor, {model: {vocab: {last: 50367}}}, {}],
    [validConstructor, {model: {vocab: {first: -1, last: 50367}}, added_tokens: []}, {}],
    [validConstructor, {model: {vocab: {first: 1}}, added_tokens: []}, {}],
    [validConstructor, validData(), null],
    [validConstructor, validData(), []]
  ];

  for (const [Constructor, data, config] of invalidArguments) {
    assertInvalidAsset(() => createOpenDeciderNanoTokenizer(Constructor, data, config));
  }
});

test('rejects a tokenizer whose first sentinel does not match without evaluating the next sentinel', () => {
  const calls: string[] = [];
  const Constructor = tokenizerConstructor((text) => {
    calls.push(text);
    return [1];
  });

  assertInvalidAsset(() => createOpenDeciderNanoTokenizer(Constructor, validData(), {}));
  assert.deepEqual(calls, [firstSentinel]);
});

test('rejects unsafe, out-of-vocabulary, and model control IDs from encode', () => {
  const invalidIds: readonly unknown[] = [
    [-1],
    [50368],
    [Number.NaN],
    ['12'],
    [OPENDECIDER_NANO_TOKENS.mask],
    new Uint32Array([12])
  ];

  for (const ids of invalidIds) {
    const Constructor = tokenizerConstructor((text) =>
      text === firstSentinel ? firstSentinelIds : text === secondSentinel ? secondSentinelIds : ids
    );
    const tokenizer = createOpenDeciderNanoTokenizer(Constructor, validData(), {});
    assertInvalidAsset(() => tokenizer.encode('ordinary input'));
  }
});

test('wraps encode errors as invalid assets and preserves tokenizer-constructor errors', () => {
  const encodeCause = new Error('tokenizer encode failed');
  const encodeFailure = tokenizerConstructor(() => {
    throw encodeCause;
  });
  assert.throws(
    () => createOpenDeciderNanoTokenizer(encodeFailure, validData(), {}),
    (error: unknown) =>
      error instanceof WebDecisionError && error.code === 'INVALID_ASSET' && error.cause === encodeCause
  );

  const constructorCause = new Error('tokenizer constructor failed');
  class ConstructorFailure {
    constructor() {
      throw constructorCause;
    }
  }
  assert.throws(
    () => createOpenDeciderNanoTokenizer(ConstructorFailure, validData(), {}),
    (error: unknown) => error === constructorCause
  );
});
