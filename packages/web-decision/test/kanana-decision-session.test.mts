import assert from 'node:assert/strict';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import test from 'node:test';
import {scoring} from './kanana-core-modules.mts';
import type {KananaDecisionInferenceInput, KananaDecisionInferenceOutput} from '../src/kanana/decision-session.js';

const outputDirectory = process.env.KANANA_CORE_DIST;
if (!outputDirectory) throw new Error('Compile the Kanana decision session before running its focused tests.');
const {createKananaDecisionSession} = await import(pathToFileURL(join(outputDirectory, 'decision-session.js')).href);
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
  inputIds: Int32Array;
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
        inputIds: input.inputIds.slice(),
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
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
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
  assert.ok(Math.abs((route.probabilities.billing ?? Number.NaN) - expectedProbability) < 1e-6);
  assert.ok(Math.abs(truth.noul - expectedProbability) < 1e-6);
  assert.deepEqual(
    severity.probabilityEntries.map(([key]) => key),
    ['0', '1', '2']
  );
  assert.deepEqual(severity.legend, {0: 'low', 1: 'middle', 2: 'high'});
  assert.ok(Math.abs(severity.score - 1.5 * (1 - expectedLowProbability)) < 1e-6);
  assert.match(calls[1]!.prompt, /billing: Billing request/);
  assert.match(calls[2]!.prompt, /support: Product support/);
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

test('passes original image sources in prompt order and expands their merged image token grids', async () => {
  const {calls, tokenizer, infer} = vectorInference([
    [1, 0],
    [1, 0],
    [0, 1]
  ]);
  const grids: [unknown, number][] = [];
  const images = [new Blob(['first']), new Blob(['second'])];
  const session = createKananaDecisionSession({
    tokenizer,
    infer,
    imageTokenThw(image: unknown, index: number) {
      grids.push([image, index]);
      return index === 0 ? [1, 1, 1] : [1, 1, 2];
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
  assert.equal(calls[0]!.image[0], images[0]);
  assert.equal(calls[0]!.image[1], images[1]);
  assert.equal((calls[0]!.prompt.match(/<\|image_pad\|>/g) ?? []).length, 3);
  assert.deepEqual(
    calls.slice(1).map(({image}) => image),
    [[], []]
  );
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
  )) as {answers: {route: ChoiceAnswer}};

  assert.ok(calls.length === 3);
  assert.ok(calls.every(({inputIds}) => inputIds.length <= 600));
  assert.ok(!calls[0]!.prompt.includes('s'.repeat(1_800)));
  assert.ok(calls[1]!.prompt.includes('A'.repeat(80)));
  assert.ok(!calls[1]!.prompt.includes('A'.repeat(1_000)));
  assert.ok(!calls[2]!.prompt.includes('B'.repeat(900)));
  assert.equal(result.answers.route.choice, 'first');
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
