import assert from 'node:assert/strict';
import test from 'node:test';
import {
  D1_OMNI_ASSETS_URL,
  D1_OMNI_FILES,
  D1_OMNI_MAX_TOKENS,
  D1_OMNI_MODEL,
  D1_OMNI_MODEL_MAX_TOKENS,
  D1_OMNI_TEMPERATURES,
  D1_OMNI_TOTAL_BYTES
} from '../src/d1-omni/contract.mjs';
import {loadD1OmniModel, validateD1OmniLoadOptions} from '../src/d1-omni/loader.mjs';
import {finishD1OmniSystemOne, prepareD1OmniSystemOne} from '../src/d1-omni/protocol.mjs';
import {createD1OmniRunner, D1_OMNI_INPUT_NAMES} from '../src/d1-omni/runner.mjs';
import {createD1OmniSession} from '../src/d1-omni/session.mjs';
import type {D1OmniTokenizer} from '../src/d1-omni/tokenizer.mjs';

type FakeOnnxSession = NonNullable<Parameters<typeof createD1OmniRunner>[1]>;
type FakeFeeds = Parameters<FakeOnnxSession['run']>[0];
type FakeTensorData = Float32Array | BigInt64Array;

interface FakeOrtCall {
  readonly feeds: FakeFeeds;
  readonly names: string[];
}

interface FakeTensor {
  readonly type: string;
  readonly data: FakeTensorData;
  readonly dims: number[];
  disposed?: boolean;
  dispose?(): void;
}

interface FakeOrtSession extends FakeOnnxSession {
  released: boolean;
}

interface FakeOrtRuntime {
  readonly calls: FakeOrtCall[];
  readonly session: FakeOrtSession;
  readonly Tensor: new (type: string, data: FakeTensorData, dims: number[]) => FakeTensor;
  readonly env: {readonly webgpu: Record<string, unknown>};
  readonly InferenceSession: {create(): Promise<FakeOrtSession>};
}

function fakeTokenizer(): D1OmniTokenizer & {readonly inputs: string[]} {
  const inputs: string[] = [];
  return {
    inputs,
    ids: {'<|startoftext|>': 1, '<|pad|>': 0, '<|mask|>': 16},
    encode(text: string) {
      inputs.push(text);
      return Array.from(text, (character) => 100 + character.codePointAt(0)!);
    }
  };
}

function fakeOrt(): FakeOrtRuntime {
  const calls: FakeOrtCall[] = [];
  class Tensor implements FakeTensor {
    declare readonly type: string;
    declare readonly data: FakeTensorData;
    declare readonly dims: number[];
    declare disposed: boolean;

    constructor(type: string, data: FakeTensorData, dims: number[]) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }

    dispose(): void {
      this.disposed = true;
    }
  }
  const session: FakeOrtSession = {
    inputNames: [...D1_OMNI_INPUT_NAMES],
    outputNames: ['logits'],
    released: false,
    async run(feeds: FakeFeeds, names: string[]): Promise<unknown> {
      calls.push({feeds, names});
      const [batch, width] = (feeds.marker_pos as FakeTensor).dims as [number, number];
      const data = new Float32Array(batch * width);
      for (let row = 0; row < batch; row++) {
        for (let option = 0; option < width; option++) data[row * width + option] = option;
      }
      return {
        logits: {
          type: 'float32',
          dims: [batch, width],
          data,
          dispose(this: FakeTensor) {
            this.disposed = true;
          }
        }
      };
    },
    async release() {
      this.released = true;
    }
  };
  return {
    calls,
    session,
    Tensor,
    env: {webgpu: {}},
    InferenceSession: {
      async create() {
        return session;
      }
    }
  };
}

test('d1-omni pins its source files and compact text Q8 asset set', () => {
  assert.equal(D1_OMNI_MODEL, 'experimental-d1-omni-600m-q8');
  assert.equal(D1_OMNI_MAX_TOKENS, 4096);
  assert.equal(D1_OMNI_MODEL_MAX_TOKENS, 16384);
  assert.deepEqual(Object.keys(D1_OMNI_FILES), [
    'config.json',
    'tokenizer_config.json',
    'tokenizer.json',
    'onnx/model_quantized.onnx',
    'onnx/model_quantized.onnx_data'
  ]);
  assert.equal(D1_OMNI_TOTAL_BYTES, 411_065_237);
  assert.equal(D1_OMNI_FILES['onnx/model_quantized.onnx_data']!.bytes, 405_883_392);
});

test('d1-omni preserves choice order, text escaping, and the model prompt structure', () => {
  const tokenizer = fakeTokenizer();
  const prepared = prepareD1OmniSystemOne(
    {account: 5, note: '<|reserved_7|>'},
    {
      route: {
        type: 'choice',
        instructions: 'Pick one <|mask|>',
        criteria: {billing: 'Invoices and refunds', technical: null}
      }
    },
    tokenizer
  );
  const [row] = prepared.rows as [NonNullable<(typeof prepared.rows)[number]>, ...typeof prepared.rows];
  assert.deepEqual(row.question.labels, ['billing', 'technical']);
  assert.deepEqual(row.question.options, ['billing: Invoices and refunds', 'technical']);
  assert.equal(row.question.qtype, 0);
  assert.equal(row.ids[0], 1);
  assert.equal(row.markerPositions.length, 2);
  assert.ok(row.markerPositions.every((position) => row.ids[position] === 16));
  assert.ok(tokenizer.inputs.includes('{"account": 5, "note": "<¦reserved_7¦>"}'));
  assert.ok(tokenizer.inputs.includes('Pick one <¦mask¦>'));
  assert.ok(tokenizer.inputs.includes(' billing: Invoices and refunds'));
  assert.equal(prepared.tokenCount, row.ids.length);
});

test('d1-omni supports score endpoints and reverses Noul false/true logits to yes/no probabilities', () => {
  const tokenizer = fakeTokenizer();
  const prepared = prepareD1OmniSystemOne(
    null,
    {
      low: {type: 'score', instructions: 'Rate it', criteria: ['low', 'high']},
      high: {
        type: 'score',
        instructions: 'Rate it',
        criteria: Array.from({length: 10}, (_, index) => `level ${index}`)
      },
      truth: {type: 'noul', instructions: 'Is this true?'}
    },
    tokenizer
  );
  assert.equal(tokenizer.inputs[0]!, '');
  assert.equal(prepared.rows[0]!.question.qtype, 1);
  assert.equal(prepared.rows[1]!.question.options.length, 10);
  assert.equal(prepared.rows[2]!.question.qtype, 2);
  assert.deepEqual(prepared.rows[2]!.question.options, [
    'false: no, the statement does not hold',
    'true: yes, the statement holds'
  ]);

  const result = finishD1OmniSystemOne(
    prepared,
    [
      [0, 1],
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      [0, 1]
    ],
    D1_OMNI_TEMPERATURES
  );
  assert.ok((result.answers.truth as Extract<NonNullable<typeof result.answers.truth>, {type: 'noul'}>).noul > 0.5);
  assert.ok(
    (result.answers.truth as Extract<NonNullable<typeof result.answers.truth>, {type: 'noul'}>).probabilities.true! >
      (result.answers.truth as Extract<NonNullable<typeof result.answers.truth>, {type: 'noul'}>).probabilities.false!
  );
  assert.equal(
    (result.answers.low as Extract<NonNullable<typeof result.answers.low>, {type: 'score'}>).score,
    (result.answers.low as Extract<NonNullable<typeof result.answers.low>, {type: 'score'}>).expected
  );
  assert.equal(
    (result.answers.low as Extract<NonNullable<typeof result.answers.low>, {type: 'score'}>).mostLikelyLevel,
    1
  );
  assert.equal(result.outputTokenCount, 0);
  assert.equal(result.model, D1_OMNI_MODEL);
});

test('d1-omni matches upstream choice-count temperature groups and marks state truncation', () => {
  const tokenizer = fakeTokenizer();
  const prepared = prepareD1OmniSystemOne(
    'x'.repeat(500),
    {q: {type: 'choice', instructions: 'choose', criteria: ['a', 'b', 'c']}},
    tokenizer,
    128
  );
  assert.equal(prepared.truncated, true);
  assert.equal(prepared.rows[0]!.ids.length, 128);
  const scaled = finishD1OmniSystemOne(
    prepareD1OmniSystemOne('x', {q: {type: 'choice', instructions: 'choose', criteria: ['a', 'b', 'c']}}, tokenizer),
    [[0, 0.5, 1]],
    D1_OMNI_TEMPERATURES
  );
  assert.equal(
    (scaled.answers.q as Extract<NonNullable<typeof scaled.answers.q>, {type: 'choice'}>).probabilities.a! <
      (scaled.answers.q as Extract<NonNullable<typeof scaled.answers.q>, {type: 'choice'}>).probabilities.c!,
    true
  );
});

test('d1-omni rejects unsupported typed input and model limits before inference', () => {
  const tokenizer = fakeTokenizer();
  assert.throws(() => prepareD1OmniSystemOne('', {}, tokenizer), {code: 'INVALID_INPUT'});
  assert.throws(
    () => prepareD1OmniSystemOne('', {q: {type: 'score', instructions: 'rate', criteria: ['a']}}, tokenizer),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(
    () =>
      prepareD1OmniSystemOne('', {q: {type: 'score', instructions: 'rate', criteria: Array(11).fill('x')}}, tokenizer),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(
    () =>
      prepareD1OmniSystemOne('', {q: {type: 'choice', instructions: 'pick', criteria: ['same', 'same']}}, tokenizer),
    {code: 'INVALID_INPUT'}
  );
  const valid = prepareD1OmniSystemOne(
    '',
    {q: {type: 'choice', instructions: 'pick', criteria: ['a', 'b']}},
    tokenizer
  );
  assert.throws(() => finishD1OmniSystemOne(valid, [[Number.NaN, 1]], D1_OMNI_TEMPERATURES), {code: 'INFERENCE_ERROR'});
});

test('d1-omni accepts dynamic question and choice axes above 255', async () => {
  const ort = fakeOrt();
  const session = createD1OmniSession({
    tokenizer: fakeTokenizer(),
    runner: createD1OmniRunner(ort, ort.session, 0)
  });
  const options = Array.from({length: 256}, (_, index) => `option-${String(index).padStart(3, '0')}`);
  const questions = Array.from({length: 257}, (_, index) => ({
    instructions: `Choose question ${index}`,
    options: index === 0 ? options : ['first', 'second']
  }));

  const generic = await session.decide({state: 'state', questions});
  assert.equal(generic.answers.length, 257);
  assert.equal(generic.answers[0]!.probabilities.length, 256);
  assert.equal(generic.answers[0]!.selectedIndex, 255);
  assert.equal(generic.answers[0]!.selected, options[255]!);
  assert.equal(ort.calls.length, 33);
  assert.deepEqual(
    ort.calls.slice(0, 2).map((call) => (call.feeds.input_ids as FakeTensor).dims[0]),
    [8, 8]
  );
  assert.equal((ort.calls[0]!.feeds.marker_pos as FakeTensor).dims[1], 256);
  assert.ok(
    ort.calls.every((call) => {
      const [rows, tokens] = (call.feeds.input_ids as FakeTensor).dims;
      if (rows === undefined || tokens === undefined) return false;
      return rows <= 8 && rows * tokens <= 65536;
    })
  );

  const beforeInvalidBatch = ort.calls.length;
  const invalidBatch = Array.from({length: 257}, (_, index) => ({
    instructions: `Choose question ${index}`,
    options: ['first', 'second']
  }));
  invalidBatch[256] = {instructions: 'Choose question 256', options: ['only one']};
  await assert.rejects(session.decide({state: 'state', questions: invalidBatch}), {code: 'INVALID_INPUT'});
  assert.equal(ort.calls.length, beforeInvalidBatch);

  const typedQuestions = Object.fromEntries(
    Array.from({length: 257}, (_, index) => [
      `question-${index}`,
      {type: 'choice', instructions: `Choose question ${index}`, criteria: ['first', 'second']}
    ])
  );
  const typed = await session.systemOne('state', typedQuestions);
  assert.equal(Object.keys(typed.answers).length, 257);
  assert.equal(
    (typed.answers['question-256'] as Extract<NonNullable<(typeof typed.answers)['question-256']>, {type: 'choice'}>)
      .choice,
    'second'
  );
  assert.equal(ort.calls.length, 66);
  assert.ok(
    ort.calls.slice(33).every((call) => {
      const [rows, tokens] = (call.feeds.input_ids as FakeTensor).dims;
      if (rows === undefined || tokens === undefined) return false;
      return rows <= 8 && rows * tokens <= 65536;
    })
  );
  await session.dispose();
});

test('d1-omni runner batches dynamic rows with explicit int64 inputs and preserves order', async () => {
  const ort = fakeOrt();
  const session = {...ort.session};
  const runner = createD1OmniRunner(ort, session, 0);
  const tokenizer = fakeTokenizer();
  const prepared = prepareD1OmniSystemOne(
    'state',
    Object.fromEntries(
      Array.from({length: 10}, (_, index) => [
        `q${index}`,
        {
          type: 'choice',
          instructions: `question ${index}`,
          criteria: ['first', 'second']
        }
      ])
    ),
    tokenizer
  );
  const rows = await runner.run(prepared.rows);
  assert.equal(rows.length, 10);
  assert.deepEqual(
    ort.calls.map((call) => (call.feeds.input_ids as FakeTensor).dims[0]),
    [8, 2]
  );
  assert.deepEqual(ort.calls[0]!.names, ['logits']);
  assert.equal((ort.calls[0]!.feeds.input_ids as FakeTensor).type, 'int64');
  assert.equal((ort.calls[0]!.feeds.marker_mask as FakeTensor).type, 'int64');
  assert.ok(((ort.calls[0]!.feeds.attention_mask as FakeTensor).data as BigInt64Array).every((value) => value === 1n));
  assert.deepEqual(rows[0]!, [0, 1]);
  await runner.dispose();
  assert.equal(session.released, true);
});

test('d1-omni load validation rejects malformed URLs and unsupported loader options', () => {
  assert.equal(validateD1OmniLoadOptions({model: D1_OMNI_MODEL}).base.href, D1_OMNI_ASSETS_URL);
  assert.equal(
    validateD1OmniLoadOptions({model: D1_OMNI_MODEL, assetsUrl: 'https://mirror.example.test/d1/'}).base.href,
    'https://mirror.example.test/d1/'
  );
  assert.throws(() => validateD1OmniLoadOptions({model: D1_OMNI_MODEL, assetsUrl: 'file:///model/'}), {
    code: 'INVALID_INPUT'
  });
  assert.throws(() => validateD1OmniLoadOptions({model: D1_OMNI_MODEL, assetsUrl: null}), {code: 'INVALID_INPUT'});
  assert.throws(
    () =>
      validateD1OmniLoadOptions({model: D1_OMNI_MODEL, assetsUrl: 'https://example.test/model/', maxPrefetchBytes: 1}),
    {code: 'INVALID_INPUT'}
  );
});

test('d1-omni loader runs the declared WebGPU route and releases its ONNX session', async () => {
  const ort = fakeOrt();
  const tokenizerIds: Record<string, number> = {
    '<|startoftext|>': 1,
    '<|pad|>': 0,
    '<|mask|>': 16,
    '<|reserved_7|>': 17,
    '<|reserved_8|>': 18,
    '<|reserved_9|>': 19,
    '<|reserved_10|>': 20,
    '<|reserved_11|>': 21
  };
  class Tokenizer {
    token_to_id(token: string): number | undefined {
      return tokenizerIds[token];
    }
    encode(text: string): {ids: number[]} {
      return {ids: Array.from(text, (character) => 100 + character.codePointAt(0)!)};
    }
  }
  const config = {
    model_type: 'd1_omni',
    max_length: 16384,
    bos_token_id: 1,
    pad_token_id: 0,
    'transformers.js_config': {dtype: 'q8'},
    d1_omni: {max_length: 16384},
    temperatures: D1_OMNI_TEMPERATURES
  };
  const values: Record<string, unknown> = {
    'config.json': config,
    'tokenizer_config.json': {
      bos_token: '<|startoftext|>',
      pad_token: '<|pad|>',
      mask_token: '<|mask|>',
      model_max_length: 16384
    },
    'tokenizer.json': {model: {vocab: {}}}
  };
  let sourceBase: string | undefined;
  type D1OmniLoaderDependencies = NonNullable<Parameters<typeof loadD1OmniModel>[1]>;
  const openSource: NonNullable<D1OmniLoaderDependencies['openSource']> = async ({base}) => {
    sourceBase = base.href;
    return {
      async read(name: string): Promise<ArrayBuffer> {
        // Preserve the JS fixture's byte-view return even though the production source type is ArrayBuffer.
        return new TextEncoder().encode(JSON.stringify(values[name] ?? {})) as unknown as ArrayBuffer;
      }
    };
  };
  const model = await loadD1OmniModel(
    {model: D1_OMNI_MODEL},
    {
      ort,
      // The fake tokenizer only implements the methods this loader path consumes.
      Tokenizer: Tokenizer as unknown as NonNullable<D1OmniLoaderDependencies['Tokenizer']>,
      openSource,
      gpuApi: {
        async requestAdapter() {
          return {info: {isFallbackAdapter: false}};
        }
      }
    }
  );
  assert.equal(sourceBase, D1_OMNI_ASSETS_URL);
  const result = await model.systemOne('state', {
    team: {type: 'choice', instructions: 'Select a team', criteria: ['billing', 'technical']}
  });
  assert.equal(
    (result.answers.team as Extract<NonNullable<typeof result.answers.team>, {type: 'choice'}>).choice,
    'technical'
  );
  assert.equal(result.outputTokenCount, 0);
  await model.dispose();
  assert.equal(ort.session.released, true);
});
