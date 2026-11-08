import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import test from 'node:test';
import {
  encodeRequest,
  TinyDecide,
  type TinyDecideMetadata,
  type TinyDecidePrototype,
  type TinyDecideRequest,
  type TinyDecideWeightsMetadata,
  Weights,
  WordPiece
} from '../src/tinydecide/engine.js';
import type {TinyDecideSessionEngine} from '../src/tinydecide/session.mjs';
import type {TinyDecideAssetName} from '../src/tinydecide/source.mjs';
import {WebDecisionError} from '../src/kev.js';
import {loadTinyDecide, validateTinyDecideLoadOptions} from '../src/tinydecide/loader.mjs';
import {TINYDECIDE_MODEL, TINYDECIDE_REPOSITORY, TINYDECIDE_REVISION} from '../src/tinydecide/contract.mjs';
import {openTinyDecideSource} from '../src/tinydecide/source.mjs';
import {createTinyDecideSession} from '../src/tinydecide/session.mjs';

const sourceHash = '577fb33b169b370711248b94bf4d9c6f61eda5f4f158710f177a96ffe74990d1';
const correctionsHash = 'ee0de6af2b114b2c665c07c1e7908646c7afb939f69839184f72bf0436f990b9';

type EnginePrototypeInput = Parameters<TinyDecide['answer']>[1];
type TestEngine = TinyDecideSessionEngine & {
  readonly cfg: TinyDecideMetadata['cfg'];
  readonly calls: TinyDecideRequest[];
  readonly prototypeCalls: Array<EnginePrototypeInput>;
};

function asArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const buffer = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buffer).set(bytes);
  return buffer;
}

function createEngine(): TestEngine {
  const tok = new WordPiece({
    vocab: ['[UNK]', 'x', 'y', 'question', 'choose', 'one'],
    unk: '[UNK]',
    prefix: '##',
    max_chars: 100
  });
  const meta: TinyDecideMetadata = {
    cfg: {
      arch: 'electra',
      d: 5,
      dh_head: 2,
      emb_dim: 5,
      layers: 0,
      heads: 1,
      ln_eps: 1e-5,
      fusion: 'all'
    },
    tokenizer: {
      kind: 'wordpiece',
      vocab: ['[UNK]', 'x', 'y', 'question', 'choose', 'one'],
      unk: '[UNK]',
      prefix: '##',
      max_chars: 100
    },
    tensors: [],
    temp: [1, 1, 1, 1],
    beta: [1, 1, 1, 1, 1],
    specials: {
      '<|state|>': 100,
      '<|choice|>': 101,
      '<|noul|>': 102,
      '<|score|>': 103,
      '<|span|>': 104,
      '<|sep|>': 105,
      '<|o|>': 106,
      '<|lv|>': 107,
      '<|ans|>': 108
    },
    format: {ts_max: 128, p_q: 128, q_max: 192, k_max: 32, span_max: 16}
  };
  const calls: TinyDecideRequest[] = [];
  const prototypeCalls: Array<EnginePrototypeInput> = [];
  return {
    calls,
    prototypeCalls,
    cfg: meta.cfg,
    tok,
    meta,
    answer(request: TinyDecideRequest, protos?: EnginePrototypeInput) {
      calls.push(request);
      prototypeCalls.push(protos);
      const encoded = encodeRequest(tok, meta, request);
      const answers: ReturnType<TinyDecide['answer']>['answers'] = request.questions.map((question) => {
        if (question.type === 'noul') return {type: 'noul', p: 0.75, qvec: [1], z0: [0, 1], proj: false};
        if (question.type === 'span')
          return {type: 'span', text: 'x', p_present: 0.8, p_span: 0.5, tok: [0, 0], char: [0, 1]};
        const options = question.options ?? [];
        const probs = options.map((_, index) => (index === 0 ? 0.75 : 0.25 / (options.length - 1)));
        const z0 = probs.map(Math.log);
        if (question.type === 'choice')
          return {type: 'choice', probs, pick: 0, confidence: 0.5, qvec: [1], z0, proj: false};
        return {type: 'score', probs, pick: 0, confidence: 0.5, score: 0.25, qvec: [1], z0, proj: false};
      });
      const stateTokens = encoded.stIdx.length + 1;
      return {
        answers,
        tokens: {state: stateTokens, questions: encoded.ids.length - stateTokens, total: encoded.ids.length},
        truncated: encoded.truncated,
        ms: 1,
        ids: encoded.ids
      };
    }
  };
}

const choiceRequest = {
  state: 'x',
  questions: [{type: 'choice', text: 'Choose one.', options: ['x', 'y']}]
};

type EngineAnswer = ReturnType<TinyDecide['answer']>['answers'][number];

function expectAnswer<Type extends EngineAnswer['type']>(
  answer: EngineAnswer | undefined,
  type: Type
): Extract<EngineAnswer, {type: Type}> {
  assert.ok(answer);
  if (answer.type !== type) throw new Error(`Expected ${type} answer, received ${answer.type}.`);
  return answer as Extract<EngineAnswer, {type: Type}>;
}

const upstreamEngineUrl = new URL('./fixtures/vendor/tinydecide-engine.js.txt', import.meta.url);

async function loadUpstreamEngine() {
  const directory = await mkdtemp(join(tmpdir(), 'tinydecide-upstream-'));
  const modulePath = join(directory, 'engine.mjs');
  const source = await readFile(upstreamEngineUrl);
  await writeFile(modulePath, source);
  const engine = await import(pathToFileURL(modulePath).href);
  return {
    engine,
    async cleanup() {
      await rm(directory, {recursive: true, force: true});
    }
  };
}

function createLoaderMetadata(): TinyDecideMetadata & {name: string} {
  return {
    name: 'TinyDecide',
    cfg: {
      arch: 'electra',
      d: 256,
      dh_head: 128,
      emb_dim: 256,
      layers: 12,
      heads: 2,
      ln_eps: 1e-5,
      fusion: 'all'
    },
    beta: [1, 1, 1, 1, 1],
    format: {ts_max: 128, p_q: 128, q_max: 192, k_max: 32, span_max: 16},
    tokenizer: {
      kind: 'wordpiece',
      vocab: Array.from({length: 16000}, (_, index) => (index === 0 ? '[UNK]' : null)),
      unk: '[UNK]',
      prefix: '##',
      max_chars: 100
    },
    specials: {
      '<|state|>': 100,
      '<|choice|>': 101,
      '<|noul|>': 102,
      '<|score|>': 103,
      '<|span|>': 104,
      '<|sep|>': 105,
      '<|o|>': 106,
      '<|lv|>': 107,
      '<|ans|>': 108
    },
    tensors: Array.from({length: 219}, (_, index) => ({
      name: `test-${index}`,
      dtype: 'f32',
      shape: [1],
      offset: index * 4,
      scale_offset: 0
    })),
    temp: [1, 1, 1, 1]
  };
}

test('the preserved TinyDecide upstream engine fixture matches its pinned source hash', async () => {
  const source = await readFile(upstreamEngineUrl);
  assert.equal(createHash('sha256').update(source).digest('hex'), sourceHash);
});

test('the TypeScript tokenizer preserves the pinned upstream output and offsets', async () => {
  const upstream = await loadUpstreamEngine();
  try {
    const metadata = {
      vocab: ['[UNK]', 'hello', 'world', '!', '中', '##e', 'a', 'ss', '😊'],
      unk: '[UNK]',
      prefix: '##',
      max_chars: 5
    };
    const sourceTokenizer = new upstream.engine.WordPiece(metadata);
    const portTokenizer = new WordPiece(metadata);
    for (const text of ['Hello world!', 'café', '中🙂', 'abcdef', '\0bad\uFFFD', 'a\u00a0world', 'Straße']) {
      for (const limit of [Infinity, 0, 1, 3]) {
        assert.deepEqual(portTokenizer.encodeUntil(text, limit), sourceTokenizer.encodeUntil(text, limit), text);
      }
    }
  } finally {
    await upstream.cleanup();
  }
});

test('the TypeScript weight reader preserves upstream f32, q4 and quantized decoding', async () => {
  const upstream = await loadUpstreamEngine();
  try {
    const metadata = {
      tensors: [
        {name: 'float', dtype: 'f32', shape: [2, 2], offset: 0, scale_offset: 0},
        {name: 'q4', dtype: 'q4', shape: [1, 32], offset: 16, scale_offset: 32},
        {name: 'q8', dtype: 'q8', shape: [2, 4], offset: 36, scale_offset: 44}
      ]
    };
    const buffer = new ArrayBuffer(52);
    new Float32Array(buffer, 0, 4).set([-1, 0.5, 2, -3]);
    new Uint8Array(buffer, 16, 16).set(Array.from({length: 16}, (_, index) => index * 17));
    new Uint16Array(buffer, 32, 1)[0] = 0x3f80;
    new Int8Array(buffer, 36, 8).set([-8, -4, -2, 1, 2, 4, 6, 8]);
    new Float32Array(buffer, 44, 2).set([0.5, 2]);
    const weightsMetadata: TinyDecideWeightsMetadata = metadata;
    const original = new upstream.engine.Weights(metadata, buffer);
    const port = new Weights(weightsMetadata, buffer);
    for (const name of ['float', 'q4', 'q8']) {
      assert.deepEqual(port.get(name).shape, original.get(name).shape, name);
      assert.deepEqual(port.get(name).data, original.get(name).data, name);
    }
  } finally {
    await upstream.cleanup();
  }
});

test('the bundled TinyDecide correction helper matches the pinned upstream source hash', async () => {
  const source = await readFile(new URL('../src/tinydecide/corrections.js', import.meta.url));
  assert.equal(createHash('sha256').update(source).digest('hex'), correctionsHash);
});

test('TinyDecide verifies and reports separately loaded model assets', async () => {
  const body = new TextEncoder().encode('pinned fixture');
  const digest = await crypto.subtle.digest('SHA-256', body);
  const sha256 = Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, '0')).join('');
  const spec = {
    files: {'meta.json': {bytes: body.length, sha256}},
    manifest: {files: {'meta.json': {bytes: body.length, sha256}}},
    totalBytes: body.length
  };
  const previousFetch = globalThis.fetch;
  const progress: Array<{loadedBytes: number; totalBytes: number; fraction: number}> = [];
  globalThis.fetch = async () => new Response(body);
  try {
    const source = await openTinyDecideSource({
      base: new URL('https://assets.example/tiny/'),
      cacheEnabled: false,
      offline: false,
      onProgress: (value: {loadedBytes: number; totalBytes: number; fraction: number}) => progress.push(value),
      spec
    });
    assert.deepEqual(new Uint8Array(await source.read('meta.json')), body);
  } finally {
    globalThis.fetch = previousFetch;
  }
  assert.deepEqual(progress, [{loadedBytes: body.length, totalBytes: body.length, fraction: 1}]);
});

test('TinyDecide load options reject unsafe or unsupported values', () => {
  const model = validateTinyDecideLoadOptions({model: TINYDECIDE_MODEL});
  assert.equal(model.base.href, `https://huggingface.co/${TINYDECIDE_REPOSITORY}/resolve/${TINYDECIDE_REVISION}/`);
  assert.equal(model.cacheEnabled, true);
  assert.equal(model.offline, false);
  assert.equal(
    validateTinyDecideLoadOptions({
      model: TINYDECIDE_MODEL,
      assetsUrl: 'https://assets.example/tiny/',
      cache: false
    }).base.href,
    'https://assets.example/tiny/'
  );
  assert.throws(() => validateTinyDecideLoadOptions({model: TINYDECIDE_MODEL, assetsUrl: null}), {
    code: 'INVALID_INPUT'
  });
  assert.throws(() => validateTinyDecideLoadOptions({model: TINYDECIDE_MODEL, assetsUrl: ''}), {
    code: 'INVALID_INPUT'
  });
  assert.throws(() => validateTinyDecideLoadOptions({model: 'other', assetsUrl: 'https://assets.example/'}), {
    code: 'INVALID_INPUT'
  });
  assert.throws(
    () =>
      validateTinyDecideLoadOptions({
        model: 'experimental-tinydecide-10m-q4',
        assetsUrl: 'https://user:pass@example.test/'
      }),
    {code: 'INVALID_INPUT'}
  );
});

test('TinyDecide loader checks pinned metadata and constructs the native engine', async () => {
  const metadata = createLoaderMetadata();
  const metadataBytes = asArrayBuffer(new TextEncoder().encode(JSON.stringify(metadata)));
  const modelBytes = new ArrayBuffer(219 * Float32Array.BYTES_PER_ELEMENT);
  const loaded: TinyDecideAssetName[] = [];
  const assetsBase = `https://huggingface.co/${TINYDECIDE_REPOSITORY}/resolve/${TINYDECIDE_REVISION}/`;
  const testEngine = createEngine();
  const model = await loadTinyDecide(
    {model: TINYDECIDE_MODEL, cache: false},
    {
      openSource: async (options) => {
        assert.equal(options.base.href, assetsBase);
        return {
          read: async (name) => {
            loaded.push(name);
            return name === 'meta.json' ? metadataBytes : modelBytes;
          }
        };
      },
      TinyDecideClass: class implements TinyDecideSessionEngine {
        readonly meta: TinyDecideMetadata;
        readonly cfg: TinyDecideMetadata['cfg'];
        readonly tok: WordPiece;
        readonly answer: TinyDecide['answer'];

        constructor(parsed: TinyDecideMetadata, bytes: ArrayBuffer) {
          assert.equal((parsed as TinyDecideMetadata & {name: string}).name, 'TinyDecide');
          assert.equal(bytes, modelBytes);
          this.meta = parsed;
          this.cfg = parsed.cfg;
          this.tok = new WordPiece(parsed.tokenizer);
          this.answer = testEngine.answer.bind(testEngine);
        }
      }
    }
  );
  assert.deepEqual(loaded.sort(), ['meta.json', 'model.bin']);
  const result = await model.decide({state: 'x', questions: [{instructions: 'Choose one.', options: ['x', 'y']}]});
  assert.equal(result.answers[0]!.selected, 'x');
  await model.dispose();
});

test('TinyDecide loader rejects mismatched metadata and wraps engine failures', async () => {
  const options = {model: 'experimental-tinydecide-10m-q4', assetsUrl: 'https://assets.example/tiny/', cache: false};
  const source = (bytes: ArrayBuffer) => async () => ({
    read: async (name: TinyDecideAssetName) =>
      name === 'meta.json' ? bytes : new ArrayBuffer(219 * Float32Array.BYTES_PER_ELEMENT)
  });
  await assert.rejects(loadTinyDecide(options, {openSource: source(asArrayBuffer(new TextEncoder().encode('{}')))}), {
    code: 'INVALID_ASSET'
  });
  const metadata = createLoaderMetadata();
  await assert.rejects(
    loadTinyDecide(options, {
      openSource: source(asArrayBuffer(new TextEncoder().encode(JSON.stringify(metadata)))),
      TinyDecideClass: class extends TinyDecide {
        constructor(metadata: TinyDecideMetadata, bytes: ArrayBuffer) {
          super(metadata, bytes);
          throw new Error('engine fixture failed');
        }
      }
    }),
    {code: 'MODEL_LOAD_FAILED'}
  );
});

test('TinyDecide systemOne preserves choice, Noul, score and span answers', async () => {
  const engine = createEngine();
  const model = createTinyDecideSession({engine, model: 'experimental-tinydecide-10m-q4'});
  const result = await model.systemOne('x', {
    route: {type: 'choice', instructions: 'Choose one.', options: ['x', 'y']},
    urgent: {type: 'noul', instructions: 'This message is urgent.'},
    tone: {type: 'score', instructions: 'How positive?', options: ['low', 'medium', 'high']},
    time: {type: 'span', instructions: 'Extract the time.'}
  });
  const route = expectAnswer(result.answers.route, 'choice');
  const urgent = expectAnswer(result.answers.urgent, 'noul');
  const tone = expectAnswer(result.answers.tone, 'score');
  const time = expectAnswer(result.answers.time, 'span');
  assert.equal(result.model, 'experimental-tinydecide-10m-q4');
  assert.equal(route.type, 'choice');
  assert.deepEqual(route.probs, [0.75, 0.25]);
  assert.equal(urgent.p, 0.75);
  assert.equal(tone.score, 0.25);
  assert.equal(time.text, 'x');
  assert.equal(time.p_present, 0.8);
  assert.equal(time.p_span, 0.5);
  await model.dispose();
});

test('TinyDecide decide maps ordered probabilities and native logits', async () => {
  const engine = createEngine();
  const model = createTinyDecideSession({engine, model: 'experimental-tinydecide-10m-q4'});
  const result = await model.decide({
    state: 'x',
    questions: [{instructions: 'Choose one.', options: ['first', 'second', 'third']}]
  });
  assert.equal(result.answers[0]!.selected, 'first');
  assert.equal(result.answers[0]!.selectedIndex, 0);
  assert.deepEqual(
    result.answers[0]!.probabilities.map(({option}) => option),
    ['first', 'second', 'third']
  );
  assert.deepEqual(
    result.answers[0]!.probabilities.map(({logit}) => logit),
    [Math.log(0.75), Math.log(0.125), Math.log(0.125)]
  );
  await model.dispose();
});

test('TinyDecide snapshots and forwards native prototypes by question order', async () => {
  const engine = createEngine();
  const model = createTinyDecideSession({engine, model: 'experimental-tinydecide-10m-q4'});
  const proto = {
    vec: new Float32Array([1, 2, 3, 4]),
    cnt: new Int32Array([1, 2]),
    center: new Float32Array([5, 6]),
    lam: 0.5
  };
  const result = model.answer(choiceRequest, {protos: [proto]});
  proto.vec[0] = 99;
  proto.cnt[0] = 9;
  proto.center[0] = 99;
  await result;
  assert.deepEqual(Array.from(engine.prototypeCalls[0]![0]!.vec), [1, 2, 3, 4]);
  assert.deepEqual(Array.from(engine.prototypeCalls[0]![0]!.cnt), [1, 2]);
  assert.deepEqual(Array.from(engine.prototypeCalls[0]![0]!.center!), [5, 6]);
  assert.equal(engine.prototypeCalls[0]![0]!.lam, 0.5);
  await assert.rejects(
    model.answer(choiceRequest, {protos: [{vec: new Float32Array(1), cnt: new Int32Array(2), center: null}]}),
    {code: 'INVALID_INPUT'}
  );
  await assert.rejects(
    model.answer(
      {state: 'x', questions: [{type: 'span', text: 'extract'}]},
      {
        protos: [{vec: new Float32Array(4), cnt: new Int32Array(2), center: new Float32Array(2)}]
      }
    ),
    {code: 'INVALID_INPUT'}
  );
  await model.dispose();
});

test('TinyDecide systemOne forwards prototypes in named question insertion order', async () => {
  const engine = createEngine();
  const model = createTinyDecideSession({engine, model: 'experimental-tinydecide-10m-q4'});
  const proto = {vec: new Float32Array([1, 2, 3, 4]), cnt: new Int32Array([1, 1]), center: null};
  await model.systemOne(
    'x',
    {
      first: {type: 'choice', instructions: 'Choose.', options: ['x', 'y']},
      second: {type: 'choice', instructions: 'Choose.', options: ['y', 'x']}
    },
    {protos: [proto, null]}
  );
  assert.equal(engine.prototypeCalls[0]!.length, 2);
  assert.deepEqual(Array.from(engine.prototypeCalls[0]![0]!.vec), [1, 2, 3, 4]);
  assert.equal(engine.prototypeCalls[0]![1], null);
  await model.dispose();
});

test('TinyDecide builds native prototypes from prior labeled answers', async () => {
  const model = createTinyDecideSession({engine: createEngine(), model: 'experimental-tinydecide-10m-q4'});
  const example = {v: [1, 0], z: [0, -1]};
  const protos = model.makeProtos('choice', [[example], []]);
  assert.ok(protos);
  assert.deepEqual(Array.from(protos.vec), [1, 0, 0, 0]);
  assert.deepEqual(Array.from(protos.cnt), [1, 0]);
  assert.deepEqual(Array.from(protos.center!), [1, 0]);
  assert.equal(protos.lam, 0.25);
  assert.equal(model.makeProtos('span', []), null);
  assert.equal(model.makeProtos('choice', [[], []]), null);
  assert.throws(() => model.makeProtos('choice', [[{v: [1], z: [0, -1]}], []]), {code: 'INVALID_INPUT'});
  assert.throws(() => model.makeProtos('choice', [[{v: [1, 0], z: [0]}], []]), {code: 'INVALID_INPUT'});
  assert.throws(() => model.makeProtos('choice', [[{v: [1, 0], z: [0, -1], extra: true}], []]), {
    code: 'INVALID_INPUT'
  });
  await model.dispose();
});

test('TinyDecide rejects malformed options and preserves inference errors', async () => {
  const engine = createEngine();
  const model = createTinyDecideSession({engine, model: 'experimental-tinydecide-10m-q4'});
  await assert.rejects(model.answer(choiceRequest, {protos: {}}), {code: 'INVALID_INPUT'});
  const unsafeProtos: Array<TinyDecidePrototype | null | undefined> = [];
  Object.defineProperty(unsafeProtos, '0', {
    get() {
      throw new Error('should not be invoked');
    }
  });
  unsafeProtos.length = 1;
  await assert.rejects(model.answer(choiceRequest, {protos: unsafeProtos}), {code: 'INVALID_INPUT'});
  await assert.rejects(model.answer(choiceRequest, {signal: {}}), {code: 'INVALID_INPUT'});
  const sparseProtos: Array<TinyDecidePrototype | null | undefined> = [];
  sparseProtos.length = 1;
  await model.answer(choiceRequest, {protos: sparseProtos});
  await model.answer(choiceRequest, null);

  const failure = new Error('engine failed');
  engine.answer = () => {
    throw failure;
  };
  await assert.rejects(model.answer(choiceRequest), {code: 'INFERENCE_ERROR', cause: failure});
  await model.dispose();
  await assert.rejects(model.releaseCachedMemory(), {code: 'SESSION_CLOSED'});
});

test('TinyDecide preserves abort and model errors from the native engine', async () => {
  const engine = createEngine();
  const model = createTinyDecideSession({engine, model: 'experimental-tinydecide-10m-q4'});
  const abort = new DOMException('cancelled', 'AbortError');
  engine.answer = () => {
    throw abort;
  };
  await assert.rejects(model.answer(choiceRequest), (cause) => cause === abort);
  const invalid = new WebDecisionError('INVALID_ASSET', 'invalid test input');
  engine.answer = () => {
    throw invalid;
  };
  await assert.rejects(model.answer(choiceRequest), (cause) => cause === invalid);
  await model.dispose();
});

test('TinyDecide accepts 127 state tokens and marks state truncation after that', async () => {
  const model = createTinyDecideSession({engine: createEngine(), model: 'experimental-tinydecide-10m-q4'});
  const atLimit = await model.answer({...choiceRequest, state: 'x '.repeat(127)});
  const overLimit = await model.answer({...choiceRequest, state: 'x '.repeat(128)});
  assert.equal(atLimit.truncated, false);
  assert.equal(overLimit.truncated, true);
  assert.equal(atLimit.tokens.state, 128);
  assert.equal(overLimit.tokens.state, 128);
  await model.dispose();
});

test('TinyDecide accepts a 192-token question row and rejects the next token', async () => {
  const engine = createEngine();
  const model = createTinyDecideSession({engine, model: 'experimental-tinydecide-10m-q4'});
  const atLimit = {...choiceRequest, questions: [{type: 'choice', text: 'x '.repeat(185), options: ['x', 'x']}]};
  const overLimit = {...choiceRequest, questions: [{type: 'choice', text: 'x '.repeat(186), options: ['x', 'x']}]};
  await model.answer(atLimit);
  await assert.rejects(model.answer(overLimit), {code: 'INVALID_INPUT'});
  assert.equal(engine.calls.length, 1);
  await model.dispose();
});

test('TinyDecide chunks uncapped aggregate requests without changing packed token counts', async () => {
  const engine = createEngine();
  const model = createTinyDecideSession({engine, model: 'experimental-tinydecide-10m-q4'});
  const names = Array.from({length: 257}, (_, index) => `question-${index}`);
  const namedQuestions = Object.fromEntries(names.map((name) => [name, {type: 'noul', instructions: 'x '.repeat(7)}]));
  const prototypes = names.map((_, index) => ({
    vec: new Float32Array([index, 0, 0, 0]),
    cnt: new Int32Array([1, 0]),
    center: null
  }));
  const namedResult = await model.systemOne('', namedQuestions, {protos: prototypes});
  assert.equal(Object.keys(namedResult.answers).length, 257);
  assert.ok(namedResult.tokenCount > 2048);
  assert.equal(engine.prototypeCalls.length, 33);
  assert.equal(engine.prototypeCalls[31]![0]!.vec[0], 248);
  assert.equal(engine.prototypeCalls[32]![0]!.vec[0], 256);

  const genericResult = await model.decide({
    state: '',
    questions: names.map(() => ({instructions: 'x x', options: ['x', 'y']}))
  });
  assert.equal(genericResult.answers.length, 257);
  assert.ok(genericResult.tokenCount > 2048);

  const questions = Array.from({length: 257}, () => ({
    type: 'noul',
    text: 'x '.repeat(6)
  }));
  const answer = await model.answer({state: '', questions});
  assert.equal(answer.answers.length, 257);
  assert.equal(answer.tokens.total, 2057);
  assert.equal(answer.ids.length, answer.tokens.total);
  assert.equal(engine.calls.length, 99);
  assert.ok(engine.calls.every(({questions: chunk}) => chunk.length <= 8));
  assert.deepEqual(
    engine.calls.slice(0, 33).map(({questions: chunk}) => chunk.length),
    [...Array(32).fill(8), 1]
  );
  await model.dispose();
});

test('TinyDecide rejects unsupported options and closes after disposal', async () => {
  const engine = createEngine();
  const model = createTinyDecideSession({engine, model: 'experimental-tinydecide-10m-q4'});
  await assert.rejects(
    model.answer({state: 'x', questions: [{type: 'choice', text: 'Pick.', options: Array(33).fill('x')}]}),
    {code: 'INVALID_INPUT'}
  );
  await assert.rejects(model.systemOne('x', {extract: {type: 'span', instructions: 'Extract', options: ['x']}}), {
    code: 'INVALID_INPUT'
  });
  await model.dispose();
  await assert.rejects(model.decide(choiceRequest), {code: 'SESSION_CLOSED'});
});

test('TinyDecide skips a queued request when its signal is aborted', async () => {
  const model = createTinyDecideSession({engine: createEngine(), model: 'experimental-tinydecide-10m-q4'});
  const first = model.answer(choiceRequest);
  const controller = new AbortController();
  const second = model.answer(choiceRequest, {signal: controller.signal});
  controller.abort();
  await first;
  await assert.rejects(second, {name: 'AbortError'});
  await model.dispose();
});

test('TinyDecide rejects malformed request shapes before the engine runs', async () => {
  const engine = createEngine();
  const model = createTinyDecideSession({engine, model: 'experimental-tinydecide-10m-q4'});
  const sparseQuestions: Array<TinyDecideRequest['questions'][number]> = [];
  sparseQuestions.length = 1;
  const badOptions = Array(1).fill('x');
  const invalidRequests = [
    {state: 1, questions: [choiceRequest.questions[0]]},
    {state: 'x', questions: {}},
    {state: 'x', questions: []},
    {state: 'x', questions: sparseQuestions},
    {state: 'x', questions: [{type: 'other', text: 'Choose'}]},
    {state: 'x', questions: [{type: 'choice', text: 'Choose', options: 'x'}]},
    {state: 'x', questions: [{type: 'choice', text: 'Choose', options: badOptions}]},
    {state: 'x', questions: [{type: 'choice', text: 'Choose', options: ['x']}]},
    {state: 'x', questions: [{type: 'span', text: 'Extract', options: ['x']}]}
  ];
  for (const request of invalidRequests) await assert.rejects(model.answer(request), {code: 'INVALID_INPUT'});
  assert.equal(engine.calls.length, 0);
  await model.dispose();
});

test('TinyDecide stops an uncapped request when abort arrives between chunks', async () => {
  const engine = createEngine();
  const answer = engine.answer.bind(engine);
  const controller = new AbortController();
  engine.answer = (...args) => {
    const result = answer(...args);
    controller.abort();
    return result;
  };
  const model = createTinyDecideSession({engine, model: 'experimental-tinydecide-10m-q4'});
  const questions = Array.from({length: 9}, () => ({type: 'noul', text: 'x'}));
  await assert.rejects(model.answer({state: '', questions}, {signal: controller.signal}), {name: 'AbortError'});
  assert.equal(engine.calls.length, 1);
  assert.equal(engine.calls[0]!.questions.length, 8);
  await model.dispose();
});
