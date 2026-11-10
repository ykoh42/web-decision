import assert from 'node:assert/strict';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import test from 'node:test';
import {scoring} from './kanana-core-modules.mts';
import type {KananaDecisionInferenceInput, KananaDecisionInferenceOutput} from '../src/kanana/decision-session.js';

const outputDirectory = process.env.KANANA_CORE_DIST;
if (!outputDirectory) throw new Error('Compile the Kanana decision session before running its focused tests.');
const {createKananaDecisionSession} = await import(pathToFileURL(join(outputDirectory, 'decision-session.js')).href);
const {KANANA_IMAGE_TOKEN_ID, KANANA_MODEL_MAX_LENGTH} = await import(
  pathToFileURL(join(outputDirectory, 'contract.mjs')).href
);
const {KANANA_DECISION_TEMPERATURE, scaleKananaLogits} = scoring;

type ChoiceAnswer = {
  type: 'choice';
  choice: string;
  probabilities: Record<string, number>;
  probabilityEntries: [string, number][];
};
type NoulAnswer = {type: 'noul'; noul: number};
type ScoreAnswer = {
  type: 'score';
  score: number;
  legend: Record<string, string>;
  probabilityEntries: [string, number][];
};

const SPECIAL_TOKENS = ['<|im_start|>', '<|im_end|>', '<|vision_start|>', '<|image_pad|>', '<|vision_end|>'];
const SPECIAL_IDS = new Map(SPECIAL_TOKENS.map((token, index) => [token, index + 10]));
SPECIAL_IDS.set('<|image_pad|>', KANANA_IMAGE_TOKEN_ID);
const TOKENS_BY_ID = new Map([...SPECIAL_IDS].map(([token, id]) => [id, token]));
const BOS_ID = 1;
TOKENS_BY_ID.set(BOS_ID, '<|bos|>');

function charTokenizer() {
  return {
    encode(text: string, options: {add_special_tokens?: boolean} = {}) {
      const ids = [];
      for (let offset = 0; offset < text.length;) {
        const special = SPECIAL_TOKENS.find((token) => text.startsWith(token, offset));
        if (special) {
          ids.push(SPECIAL_IDS.get(special)!);
          offset += special.length;
          continue;
        }

        const codePoint = text.codePointAt(offset)!;
        const character = String.fromCodePoint(codePoint);
        ids.push(0x110000 + codePoint);
        offset += character.length;
      }
      if (options.add_special_tokens !== false) ids.unshift(BOS_ID);
      return {ids};
    },
    decode(ids: number[], _options?: {skip_special_tokens?: boolean}) {
      return ids.map((id) => TOKENS_BY_ID.get(id) ?? String.fromCodePoint(id - 0x110000)).join('');
    }
  };
}

function decodeInput(tokenizer: ReturnType<typeof charTokenizer>, inputIds: ArrayLike<number>) {
  return tokenizer.decode(Array.from(inputIds), {skip_special_tokens: false});
}

function hiddenState(inputIds: ArrayLike<number>, vector: number[]) {
  const values = new Float32Array(inputIds.length * vector.length);
  values.set(vector, values.length - vector.length);
  return {shape: [1, inputIds.length, vector.length], dtype: 'float32' as const, values};
}

type InferenceCall = {
  attentionMask: Int32Array;
  image: readonly unknown[];
  imageTokenThw: readonly (readonly [number, number, number])[];
  imagePositions: Int32Array;
  imageEmbeddings: {shape: readonly [number, number]; values: ArrayLike<number>};
  inputIds: Int32Array;
  positionIds: Int32Array;
  prompt: string;
  signal: AbortSignal | undefined;
};

function vectorInference(vectors: number[][], calls: InferenceCall[] = []) {
  let index = 0;
  const tokenizer = charTokenizer();
  return {
    calls,
    tokenizer,
    infer(input: KananaDecisionInferenceInput): KananaDecisionInferenceOutput {
      const vector = vectors[index++];
      if (!vector) throw new Error(`Missing test vector for inference ${index}.`);
      calls.push({
        attentionMask: input.attentionMask.slice(),
        image: input.image,
        imageTokenThw: input.imageTokenThw,
        imagePositions: input.imagePositions.slice(),
        imageEmbeddings: input.imageEmbeddings,
        inputIds: input.inputIds.slice(),
        positionIds: input.positionIds.slice(),
        prompt: decodeInput(tokenizer, input.inputIds),
        signal: input.signal
      });
      return hiddenState(input.inputIds, vector);
    }
  };
}

function nativeRequest(temperature = 1) {
  return {
    state: 'The customer needs help.',
    temperature,
    questions: new Map([
      [
        'route',
        {
          type: 'choice',
          instructions: 'Choose a route.',
          criteria: new Map([
            ['billing', 'Billing request'],
            ['support', 'Product support']
          ])
        }
      ],
      ['truth', {type: 'noul', instructions: 'Is the statement true?'}],
      ['severity', {type: 'score', instructions: 'Rate urgency.', criteria: ['low', 'middle', 'high']}]
    ])
  };
}

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return {promise, resolve, reject};
}

test('runs ordered Choice, Noul and Score questions with pinned scoring and request temperature', async () => {
  const {calls, tokenizer, infer} = vectorInference([
    [1, 0],
    [1, 0],
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1]
  ]);
  const session = createKananaDecisionSession({tokenizer, infer});
  const result = (await session.systemOne(nativeRequest(0.001), {temperature: 100})) as {
    model: string;
    answerEntries: [string, unknown][];
    answers: {route: ChoiceAnswer; truth: NoulAnswer; severity: ScoreAnswer};
    usage: {input_tokens: number};
    inputTruncation?: unknown;
  };
  const route = result.answers.route;
  const truth = result.answers.truth;
  const severity = result.answers.severity;
  const expectedProbability = 1 / (1 + Math.exp(-scaleKananaLogits([1, 0], KANANA_DECISION_TEMPERATURE, 100)[0]!));
  const expectedLowProbability =
    Math.exp(scaleKananaLogits([1, 0, 0], KANANA_DECISION_TEMPERATURE, 100)[0]!) /
    (Math.exp(scaleKananaLogits([1, 0, 0], KANANA_DECISION_TEMPERATURE, 100)[0]!) + 2);

  assert.equal(result.model, 'kanana-2-v-decide-3b-2611');
  assert.deepEqual(
    result.answerEntries.map(([id]) => id),
    ['route', 'truth', 'severity']
  );
  assert.equal(route.choice, 'billing');
  assert.deepEqual(
    route.probabilityEntries.map(([key]) => key),
    ['billing', 'support']
  );
  assert.equal(result.inputTruncation, undefined);
  assert.ok(Math.abs((route.probabilities.billing ?? Number.NaN) - expectedProbability) < 1e-6);
  assert.ok(Math.abs(truth.noul - expectedProbability) < 1e-6);
  assert.deepEqual(
    severity.probabilityEntries.map(([key]) => key),
    ['0', '1', '2']
  );
  assert.deepEqual(severity.legend, {0: 'low', 1: 'middle', 2: 'high'});
  assert.ok(Math.abs(severity.score - 1.5 * (1 - expectedLowProbability)) < 1e-6);
  assert.match(calls[3]!.prompt, /billing: Billing request/);
  assert.match(calls[4]!.prompt, /support: Product support/);
  assert.ok(calls.every(({image}) => image.length === 0));
  assert.ok(calls.every(({attentionMask, inputIds}) => attentionMask.length === inputIds.length));
  assert.equal(
    result.usage.input_tokens,
    calls.reduce((total, {inputIds}) => total + inputIds.length, 0)
  );
  await session.dispose();
});

test('decide exposes the same pinned native answer semantics', async () => {
  const {tokenizer, infer} = vectorInference([
    [1, 0],
    [1, 0],
    [0, 1]
  ]);
  const session = createKananaDecisionSession({tokenizer, infer});
  const result = (await session.decide({
    state: 'Choose.',
    questions: {
      decision: {
        type: 'choice',
        criteria: new Map([
          ['first', 'First'],
          ['second', 'Second']
        ])
      }
    }
  })) as {answers: {decision: ChoiceAnswer}};

  assert.equal(result.answers.decision.choice, 'first');
  assert.deepEqual(
    result.answers.decision.probabilityEntries.map(([key]) => key),
    ['first', 'second']
  );
  await session.dispose();
});

test('reuses candidate embeddings across calls and counts only tokens embedded on the current call', async () => {
  const calls: number[][] = [];
  const tokenizer = charTokenizer();
  const session = createKananaDecisionSession({
    tokenizer,
    infer(input: KananaDecisionInferenceInput) {
      calls.push(Array.from(input.inputIds));
      return hiddenState(input.inputIds, [1, 0]);
    }
  });
  const request = {
    state: 'Reuse candidate embeddings.',
    questions: {
      route: {
        type: 'choice',
        criteria: new Map([
          ['first', 'First candidate'],
          ['second', 'Second candidate']
        ])
      }
    }
  };

  const first = (await session.systemOne(request)) as {usage: {input_tokens: number}};
  const firstCallCount = calls.length;
  const second = (await session.systemOne(request)) as {usage: {input_tokens: number}};

  assert.equal(firstCallCount, 3);
  assert.equal(calls.length, firstCallCount + 1);
  assert.equal(
    first.usage.input_tokens,
    calls.slice(0, firstCallCount).reduce((total, ids) => total + ids.length, 0)
  );
  assert.equal(second.usage.input_tokens, calls.at(-1)!.length);
  await session.dispose();
});

test('deduplicates candidate token IDs across questions while preserving query order and usage', async () => {
  const calls: number[][] = [];
  const tokenizer = charTokenizer();
  const criteria = new Map([
    ['yes', 'Yes'],
    ['no', 'No']
  ]);
  const session = createKananaDecisionSession({
    tokenizer,
    infer(input: KananaDecisionInferenceInput) {
      calls.push(Array.from(input.inputIds));
      return hiddenState(input.inputIds, [1, 0]);
    }
  });
  const result = (await session.systemOne({
    state: 'Two questions use the same candidate documents.',
    questions: new Map([
      ['first', {type: 'choice', criteria}],
      ['second', {type: 'choice', criteria: new Map(criteria)}]
    ])
  })) as {answerEntries: [string, unknown][]; usage: {input_tokens: number}};

  assert.deepEqual(
    result.answerEntries.map(([question]) => question),
    ['first', 'second']
  );
  assert.equal(calls.length, 4);
  assert.equal(
    result.usage.input_tokens,
    calls.reduce((total, ids) => total + ids.length, 0)
  );
  await session.dispose();
});

test('keeps the most recently used 4096 candidate embeddings and evicts the least recent', async () => {
  const calls: number[][] = [];
  const tokenizer = charTokenizer();
  const session = createKananaDecisionSession({
    tokenizer,
    infer(input: KananaDecisionInferenceInput) {
      calls.push(Array.from(input.inputIds));
      return hiddenState(input.inputIds, [1, 0]);
    }
  });
  const criteria = new Map(
    Array.from({length: 4096}, (_, index) => {
      const label = `candidate-${String(index).padStart(4, '0')}`;
      return [label, `Description ${String(index).padStart(4, '0')}`] as const;
    })
  );
  const first = (await session.systemOne({
    state: 'Fill candidate cache.',
    questions: {large: {type: 'choice', criteria}}
  })) as {usage: {input_tokens: number}};
  const firstCandidate = calls[1]!;
  const secondCandidate = calls[2]!;
  assert.equal(calls.length, 4097);
  assert.equal(
    first.usage.input_tokens,
    calls.reduce((total, ids) => total + ids.length, 0)
  );

  const stableCandidate = new Map([['candidate-0000', 'Description 0000']]);
  await session.systemOne({
    state: 'Promote the oldest candidate, then insert a new candidate.',
    questions: new Map([
      ['promote', {type: 'choice', criteria: stableCandidate}],
      ['insert', {type: 'choice', criteria: new Map([['candidate-new', 'Description new']])}]
    ])
  });
  const afterPromotion = calls.length;
  await session.systemOne({
    state: 'The least recent candidate should be absent.',
    questions: new Map([
      ['retained', {type: 'choice', criteria: stableCandidate}],
      ['evicted', {type: 'choice', criteria: new Map([['candidate-0001', 'Description 0001']])}]
    ])
  });

  assert.equal(afterPromotion, 4100);
  assert.equal(calls.length, afterPromotion + 3);
  assert.deepEqual(calls.at(-1), secondCandidate);
  assert.notDeepEqual(calls.at(-1), firstCandidate);
  await session.dispose();
});

test('passes original image sources in prompt order and expands their merged image token grids', async () => {
  const {calls, tokenizer, infer} = vectorInference([
    [1, 0],
    [1, 0],
    [0, 1]
  ]);
  const grids: [unknown, number][] = [];
  const encodedImages: unknown[][] = [];
  const images = [new Blob(['first']), new Blob(['second'])];
  const session = createKananaDecisionSession({
    tokenizer,
    infer,
    imageTokenThw(image: unknown, index: number) {
      grids.push([image, index]);
      return index === 0 ? [1, 1, 1] : [1, 1, 2];
    },
    prepareImageEmbeddings(sources: readonly unknown[], imageTokenThw: readonly (readonly [number, number, number])[]) {
      encodedImages.push([...sources]);
      assert.deepEqual(imageTokenThw, [
        [1, 1, 1],
        [1, 1, 2]
      ]);
      return {shape: [3, 2_560], values: new Float32Array(3 * 2_560)};
    }
  });
  const result = (await session.systemOne({
    state: [
      {
        role: 'user',
        content: [
          {type: 'text', text: 'Look at this.'},
          {type: 'image', image: images[1]}
        ]
      }
    ],
    stateFormat: 'messages',
    images: [images[0]],
    questions: {visible: {type: 'noul', instructions: 'Is anything visible?'}}
  })) as {answers: {visible: NoulAnswer}};

  assert.equal(result.answers.visible.type, 'noul');
  assert.deepEqual(grids, [
    [images[0], 0],
    [images[1], 1]
  ]);
  assert.deepEqual(encodedImages, [images]);
  assert.deepEqual(calls[0]!.imageEmbeddings.shape, [3, 2_560]);
  assert.equal(calls[0]!.image[0], images[0]);
  assert.equal(calls[0]!.image[1], images[1]);
  assert.deepEqual(calls[0]!.imageTokenThw, [
    [1, 1, 1],
    [1, 1, 2]
  ]);
  assert.deepEqual(
    [...calls[0]!.imagePositions],
    [...calls[0]!.inputIds].flatMap((id, index) => (id === KANANA_IMAGE_TOKEN_ID ? [index] : []))
  );
  assert.equal(calls[0]!.positionIds.length, calls[0]!.inputIds.length * 4);
  assert.equal((calls[0]!.prompt.match(/<\|image_pad\|>/g) ?? []).length, 3);
  assert.deepEqual(
    calls.slice(1).map(({image}) => image),
    [[], []]
  );
  assert.ok(calls.slice(1).every(({imageEmbeddings}) => imageEmbeddings.shape[0] === 0));
  await session.dispose();
});

test('forwards the active request signal to image grid resolution', async () => {
  const {calls, tokenizer, infer} = vectorInference([
    [1, 0],
    [0, 1]
  ]);
  const controller = new AbortController();
  const reason = new Error('image request cancelled');
  const image = new Blob(['image']);
  let receivedSignal: AbortSignal | undefined;
  const session = createKananaDecisionSession({
    tokenizer,
    infer,
    imageTokenThw(_source: unknown, _index: number, signal?: AbortSignal) {
      receivedSignal = signal;
      return new Promise((_, reject) => {
        signal?.addEventListener('abort', () => reject(signal.reason), {once: true});
      });
    }
  });
  const request = session.systemOne(
    {
      state: 'Look at this image.',
      images: [image],
      questions: {visible: {type: 'noul', instructions: 'Is anything visible?'}}
    },
    {signal: controller.signal}
  );

  for (let attempt = 0; attempt < 10 && !receivedSignal; attempt++)
    await new Promise((resolve) => setImmediate(resolve));
  assert.equal(receivedSignal, controller.signal);
  controller.abort(reason);
  await assert.rejects(request, (error: unknown) => error === reason);
  assert.equal(calls.length, 0);
  await session.dispose();
});

test('rejects a request maxLength above the pinned model limit before inference', async () => {
  const {calls, tokenizer, infer} = vectorInference([]);
  const session = createKananaDecisionSession({tokenizer, infer});

  await assert.rejects(session.systemOne(nativeRequest(), {maxLength: KANANA_MODEL_MAX_LENGTH + 1}), {
    name: 'RangeError',
    message: `Kanana maxLength must be a positive safe integer no greater than ${KANANA_MODEL_MAX_LENGTH}.`
  });
  assert.equal(calls.length, 0);
  await session.dispose();
});

test('applies one configured truncation boundary to query and candidate embeddings', async () => {
  const {calls, tokenizer, infer} = vectorInference([
    [1, 0],
    [1, 0],
    [0, 1]
  ]);
  const session = createKananaDecisionSession({tokenizer, infer});
  const criteria = new Map([
    ['first', 'A'.repeat(1_000)],
    ['second', 'B'.repeat(900)]
  ]);
  const result = (await session.systemOne(
    {
      state: 's'.repeat(1_800),
      questions: {route: {type: 'choice', instructions: 'Choose.', criteria}},
      querySchemas: {
        route: {
          type: 'choice',
          criteria: [
            ['first', null],
            ['second', null]
          ]
        }
      }
    },
    {maxLength: 600}
  )) as {
    answers: {route: ChoiceAnswer};
    inputTruncation?: {
      truncated: true;
      maxLength: number;
      queryQuestionIds: string[];
      candidateDocuments: {questionId: string; keys: string[]}[];
    };
  };

  assert.ok(calls.length === 3);
  assert.ok(calls.every(({inputIds}) => inputIds.length <= 600));
  assert.ok(!calls[0]!.prompt.includes('s'.repeat(1_800)));
  assert.ok(calls[1]!.prompt.includes('A'.repeat(80)));
  assert.ok(!calls[1]!.prompt.includes('A'.repeat(1_000)));
  assert.ok(!calls[2]!.prompt.includes('B'.repeat(900)));
  assert.equal(result.answers.route.choice, 'first');
  assert.deepEqual(result.inputTruncation, {
    truncated: true,
    maxLength: 600,
    queryQuestionIds: ['route'],
    candidateDocuments: [{questionId: 'route', keys: ['first', 'second']}]
  });
  await session.dispose();
});

test('serializes accepted work, aborts a queued request, and drains before disposal', async () => {
  const calls = [];
  const started = deferred();
  const gate = deferred();
  const tokenizer = charTokenizer();
  let active = 0;
  let maximumActive = 0;
  let disposed = 0;
  const session = createKananaDecisionSession({
    tokenizer,
    async infer(input: KananaDecisionInferenceInput): Promise<KananaDecisionInferenceOutput> {
      calls.push(input);
      active++;
      maximumActive = Math.max(maximumActive, active);
      try {
        if (calls.length === 1) {
          started.resolve();
          await gate.promise;
        }
        return hiddenState(input.inputIds, [1, 0]);
      } finally {
        active--;
      }
    },
    dispose() {
      disposed++;
    }
  });

  const first = session.systemOne({
    state: 'state',
    questions: {route: {type: 'choice', criteria: {a: 'A', b: 'B'}}}
  });
  await started.promise;
  const controller = new AbortController();
  const reason = new Error('queued request cancelled');
  const queued = session.decide(
    {state: 'state', questions: {route: {type: 'choice', criteria: {a: 'A', b: 'B'}}}},
    {signal: controller.signal}
  );
  controller.abort(reason);
  const disposal = session.dispose();

  assert.strictEqual(session.dispose(), disposal);
  await assert.rejects(
    session.decide({state: 'state', questions: {route: {type: 'choice', criteria: {a: 'A', b: 'B'}}}}),
    {code: 'SESSION_CLOSED'}
  );
  gate.resolve();
  await first;
  await assert.rejects(queued, (error) => error === reason);
  await disposal;

  assert.equal(calls.length, 3);
  assert.equal(maximumActive, 1);
  assert.equal(disposed, 1);
});

test('forwards an active AbortSignal to inference and rejects with its abort reason', async () => {
  const started = deferred();
  const controller = new AbortController();
  const reason = new Error('inference cancelled');
  const tokenizer = charTokenizer();
  let inferenceSignal;
  const session = createKananaDecisionSession({
    tokenizer,
    infer(input: KananaDecisionInferenceInput): Promise<KananaDecisionInferenceOutput> {
      inferenceSignal = input.signal;
      started.resolve();
      return new Promise((_resolve, reject) => {
        input.signal!.addEventListener('abort', () => reject(input.signal!.reason), {once: true});
      });
    }
  });

  const pending = session.systemOne(
    {state: 'state', questions: {route: {type: 'choice', criteria: {a: 'A', b: 'B'}}}},
    {signal: controller.signal}
  );
  await started.promise;
  assert.strictEqual(inferenceSignal, controller.signal);
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
  await session.dispose();
});
