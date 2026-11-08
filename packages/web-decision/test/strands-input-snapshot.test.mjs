import assert from 'node:assert/strict';
import test from 'node:test';
import {Tokenizer} from './strands-fake-tokenizer.mjs';
import {STRANDS_CHECKPOINT_MAX_LENGTH, STRANDS_FUSED_GRAPH_MAX_TOKENS} from '../src/strands/contract.mjs';
import {packStrandsRequest, unpackStrandsScores} from '../src/strands/choice.mjs';
import {encodeChoice, renderChoice, scoreChoice} from '../src/strands/protocol.mjs';
import {ExperimentalFusedRunner} from '../src/strands/runner.mjs';
import {ExperimentalStrandsSession} from '../src/strands/session.mjs';
import {createCausalInputs} from '../src/gpu-session-utils.js';

const question = () => ({instructions: 'Pick', options: ['red', 'blue']});
const request = () => ({state: 's', questions: [question(), question()]});

function tokenizerForRowLength(length) {
  const tokenizer = new Tokenizer();
  const suffix = tokenizer.encode(renderChoice('s', question()).questionText, {add_special_tokens: false});
  const prefixLength = length - suffix.ids.length;
  if (prefixLength < 1) throw new RangeError('Requested test row is too short');
  const encode = tokenizer.encode.bind(tokenizer);
  tokenizer.encode = (input, options) =>
    options?.add_special_tokens ? {ids: Array(prefixLength).fill(1)} : encode(input, options);
  return tokenizer;
}

const ownHole = (values, index) => {
  const copy = values.slice();
  delete copy[index];
  return copy;
};
const inherited = (values, index) => {
  const copy = ownHole(values, index),
    prototype = Object.create(Array.prototype);
  Object.defineProperty(prototype, index, {value: values[index], configurable: true});
  Object.setPrototypeOf(copy, prototype);
  return copy;
};
function fakeSession(tokenizer = new Tokenizer()) {
  const calls = [];
  const runner = {
    async run(row) {
      calls.push(row);
      return [0, 1];
    },
    async releaseCachedMemory() {},
    async dispose() {}
  };
  return {calls, session: new ExperimentalStrandsSession({tokenizer, runner})};
}

test('packer accepts 128, 129, and 4096 token rows and rejects 4097 without truncation', () => {
  assert.equal(STRANDS_FUSED_GRAPH_MAX_TOKENS, 4096);
  for (const length of [128, 129, STRANDS_FUSED_GRAPH_MAX_TOKENS]) {
    const accepted = packStrandsRequest(tokenizerForRowLength(length), request());
    assert.deepEqual(
      accepted.rows.map((row) => row.inputIds.length),
      [length, length]
    );
    assert.equal(accepted.tokenCount, length * 2);
  }

  assert.throws(() => packStrandsRequest(tokenizerForRowLength(STRANDS_FUSED_GRAPH_MAX_TOKENS + 1), request()), {
    code: 'INVALID_INPUT'
  });
});

test('checkpoint protocol boundary accepts 4096 tokens and rejects 4097', () => {
  assert.equal(STRANDS_CHECKPOINT_MAX_LENGTH, 4096);
  const accepted = encodeChoice(tokenizerForRowLength(STRANDS_CHECKPOINT_MAX_LENGTH), 's', question());
  assert.equal(accepted.inputIds.length, STRANDS_CHECKPOINT_MAX_LENGTH);

  assert.throws(
    () => encodeChoice(tokenizerForRowLength(STRANDS_CHECKPOINT_MAX_LENGTH + 1), 's', question()),
    /Prompt exceeds 4096 tokens/
  );
});

test('session validates and passes 4096-token rows to its runner without truncation', async () => {
  const fake = fakeSession(tokenizerForRowLength(STRANDS_FUSED_GRAPH_MAX_TOKENS));
  await fake.session.decide(request());
  assert.deepEqual(
    fake.calls.map((row) => row.inputIds.length),
    [4096, 4096]
  );
  await fake.session.dispose();

  const tooLong = fakeSession(tokenizerForRowLength(STRANDS_FUSED_GRAPH_MAX_TOKENS + 1));
  await assert.rejects(tooLong.session.decide(request()), {code: 'INVALID_INPUT'});
  assert.equal(tooLong.calls.length, 0);
  await tooLong.session.dispose();
});

test('4096-token causal inputs preserve positions and attention-mask semantics', () => {
  const tensors = [];
  const runtime = {
    empty(shape, dtype) {
      const tensor = {
        shape,
        dtype,
        byteOffset: 0,
        copyFrom(value) {
          this.value = value;
          return this;
        },
        getDataPtr() {
          return this;
        }
      };
      tensors.push(tensor);
      return tensor;
    },
    lib: {
      webGPUContext: {
        copyRawBytesToBuffer(bytes, tensor) {
          tensor.raw = bytes;
        }
      }
    }
  };
  const inputIds = Array.from({length: STRANDS_FUSED_GRAPH_MAX_TOKENS}, (_, index) => index % 248320);
  const inputs = createCausalInputs({runtime, device: {}}, inputIds);
  const mask = new DataView(inputs.mask.raw.buffer, inputs.mask.raw.byteOffset, inputs.mask.raw.byteLength);
  const valid = new DataView(inputs.valid.raw.buffer, inputs.valid.raw.byteOffset, inputs.valid.raw.byteLength);

  assert.deepEqual(inputs.ids.shape, [1, STRANDS_FUSED_GRAPH_MAX_TOKENS]);
  assert.equal(inputs.ids.value.length, STRANDS_FUSED_GRAPH_MAX_TOKENS);
  assert.equal(inputs.positions.value[0], 0);
  assert.equal(inputs.positions.value.at(-1), STRANDS_FUSED_GRAPH_MAX_TOKENS - 1);
  assert.deepEqual(inputs.mask.shape, [1, 1, STRANDS_FUSED_GRAPH_MAX_TOKENS, STRANDS_FUSED_GRAPH_MAX_TOKENS]);
  assert.equal(inputs.mask.raw.byteLength, STRANDS_FUSED_GRAPH_MAX_TOKENS ** 2 * 2);
  assert.equal(mask.getUint16(0, true), 0);
  assert.equal(mask.getUint16(2, true), 0xfbff);
  assert.equal(mask.getUint16(STRANDS_FUSED_GRAPH_MAX_TOKENS * 2, true), 0);
  assert.equal(mask.getUint16((STRANDS_FUSED_GRAPH_MAX_TOKENS ** 2 - 1) * 2, true), 0);
  assert.equal(valid.getUint16(0, true), 0x3c00);
  assert.equal(valid.getUint16((STRANDS_FUSED_GRAPH_MAX_TOKENS - 1) * 2, true), 0x3c00);
  assert.equal(tensors.length, 4);
});

for (const makeHole of [ownHole, inherited]) {
  test(`missing question and option entries from ${makeHole.name} reject before any fake graph`, async () => {
    for (const field of ['questions', 'options']) {
      const input = request(),
        fake = fakeSession();
      if (field === 'questions') input.questions = makeHole(input.questions, 1);
      else input.questions[1].options = makeHole(input.questions[1].options, 1);
      await assert.rejects(fake.session.decide(input), {code: 'INVALID_INPUT'});
      assert.equal(fake.calls.length, 0);
      await fake.session.dispose();
    }
  });
}

test('request and tokenizer data use indexed snapshots, not user iterators or repeated getters', async () => {
  const fake = fakeSession(),
    input = request(),
    original = input.questions[0].options;
  let states = 0,
    choices = 0;
  Object.defineProperty(input, 'state', {
    get() {
      states++;
      return states === 1 ? 's' : 123;
    }
  });
  Object.defineProperty(input.questions[0], 'options', {
    get() {
      choices++;
      return choices === 1 ? original : [false];
    }
  });
  input.questions[Symbol.iterator] = function* () {
    yield null;
  };
  original[Symbol.iterator] = function* () {
    yield 'wrong';
    yield 'other';
  };
  const answers = await fake.session.decide(input);
  assert.equal(states, 1);
  assert.equal(choices, 1);
  assert.deepEqual(
    answers.answers.map((item) => item.selected),
    ['blue', 'blue']
  );
  assert.deepEqual(fake.calls[0].options, ['red', 'blue']);
  await fake.session.dispose();
  const trick = new Tokenizer(),
    encode = trick.encode;
  trick.encode = function (...args) {
    const result = encode.apply(this, args);
    result.ids[Symbol.iterator] = function* () {
      yield -1;
    };
    return result;
  };
  const expected = packStrandsRequest(new Tokenizer(), request());
  const actual = packStrandsRequest(trick, request());
  assert.deepEqual(
    actual.rows.map((row) => row.inputIds),
    expected.rows.map((row) => row.inputIds)
  );
});

test('every question is snapshotted before tokenization can mutate later questions', () => {
  const input = request(),
    tokenizer = new Tokenizer(),
    encode = tokenizer.encode;
  tokenizer.encode = function (...args) {
    input.questions[1].options[1] = 'changed';
    return encode.apply(this, args);
  };
  const packed = packStrandsRequest(tokenizer, input);
  assert.deepEqual(packed.rows[1].options, ['red', 'blue']);
});

function fusedRunner() {
  const captures = [],
    scope = {begun: 0};
  const context = {pendingGPUToCPUCopy: null, copyRawBytesToBuffer() {}};
  const runtime = {
    beginScope() {
      scope.begun++;
    },
    endScope() {},
    cpu() {
      return 'cpu';
    },
    lib: {webGPUContext: context},
    empty(_shape, _dtype, target) {
      return {
        byteOffset: 0,
        getDataPtr() {
          return 1;
        },
        copyFrom(value) {
          this.value = ArrayBuffer.isView(value) ? Array.from(value) : value.value;
          if (target === 'cpu') context.pendingGPUToCPUCopy = Promise.resolve();
          return this;
        },
        toArray() {
          return this.value;
        }
      };
    }
  };
  const gpu = {
    pushErrorScope() {},
    async popErrorScope() {
      return null;
    }
  };
  const runner = new ExperimentalFusedRunner({
    runtime,
    gpu,
    device: {
      async sync() {
        const pending = context.pendingGPUToCPUCopy;
        context.pendingGPUToCPUCopy = null;
        if (pending) await pending;
      }
    },
    weights: Array(328).fill(0),
    lifecycle: {loss: null, disposed: false},
    uncaptured: [],
    fn(ids, _pos, _mask, _valid, answer, options) {
      captures.push({ids: ids.value, answer: answer.value[0], options: options.value});
      return {value: [0, 1]};
    }
  });
  return {runner, captures, scope};
}
const row = () => ({inputIds: [5, 6, 7, 8], answerIndex: 3, optionIndices: [1, 2], options: ['red', 'blue']});

for (const field of ['inputIds', 'optionIndices', 'options']) {
  test(`runner rejects sparse and inherited ${field} before GPU scope`, async () => {
    for (const makeHole of [ownHole, inherited]) {
      const value = row(),
        fake = fusedRunner();
      value[field] = makeHole(value[field], 1);
      await assert.rejects(fake.runner.run(value), {code: 'INVALID_INPUT'});
      assert.equal(fake.scope.begun, 0);
      assert.deepEqual(fake.captures, []);
    }
  });
}

test('runner executes exactly its validated index snapshots despite custom iterators and getters', async () => {
  const value = row(),
    fake = fusedRunner(),
    ids = value.inputIds,
    options = value.optionIndices;
  ids[Symbol.iterator] = function* () {
    yield -1;
  };
  options[Symbol.iterator] = function* () {
    yield -1;
    yield 999;
  };
  let idReads = 0,
    optionReads = 0;
  Object.defineProperty(value, 'inputIds', {
    get() {
      idReads++;
      return idReads === 1 ? ids : [-1];
    }
  });
  Object.defineProperty(value, 'optionIndices', {
    get() {
      optionReads++;
      return optionReads === 1 ? options : [-1, 999];
    }
  });
  assert.deepEqual(await fake.runner.run(value), [0, 1]);
  assert.equal(idReads, 1);
  assert.equal(optionReads, 1);
  assert.deepEqual(fake.captures, [{ids: [5, 6, 7, 8], answer: 3, options: [1, 2]}]);
});

test('protocol and unpack reject sparse and nonfinite results while ignoring custom score iterators', () => {
  const packed = packStrandsRequest(new Tokenizer(), request()),
    scores = [
      [0, 1],
      [2, 3]
    ];
  for (const invalid of [
    ownHole(scores, 1),
    inherited(scores, 1),
    [[0, 1], ownHole([2, 3], 1)],
    [[0, 1], inherited([2, 3], 1)],
    [
      [0, 1],
      [2, NaN]
    ],
    [
      [0, 1],
      [2, Infinity]
    ]
  ])
    assert.throws(() => unpackStrandsScores(invalid, packed), {code: 'INFERENCE_ERROR'});
  assert.throws(() => scoreChoice(['red', 'blue'], ownHole([0, 1], 1)));
  assert.throws(() => scoreChoice(ownHole(['red', 'blue'], 1), [0, 1]));
  const tricky = [0, 1];
  tricky[Symbol.iterator] = function* () {
    yield 99;
    yield 0;
  };
  assert.equal(unpackStrandsScores([tricky, [2, 3]], packed).answers[0].selected, 'blue');
  const choices = ['red', 'blue'];
  choices[Symbol.iterator] = function* () {
    yield 'wrong';
    yield 'other';
  };
  assert.match(renderChoice('s', {instructions: 'Pick', options: choices}).questionText, /1\. red\n2\. blue/);
  assert.equal(scoreChoice(choices, [0, 1]).selected, 'blue');
  const values = [0, 1];
  let valueReads = 0,
    optionReads = 0;
  Object.defineProperty(values, 1, {
    get() {
      valueReads++;
      return valueReads === 1 ? 1 : NaN;
    }
  });
  Object.defineProperty(choices, 1, {
    get() {
      optionReads++;
      return optionReads === 1 ? 'blue' : 'wrong';
    }
  });
  assert.equal(scoreChoice(choices, values).selected, 'blue');
  assert.deepEqual([valueReads, optionReads], [1, 1]);
});
