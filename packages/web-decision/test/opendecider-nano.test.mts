import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {createReadStream, existsSync} from 'node:fs';
import {readFile, stat} from 'node:fs/promises';
import {join, normalize, relative, sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import test from 'node:test';
import {Tokenizer} from '@huggingface/tokenizers';
import {WebDecisionError} from '../src/kev.js';
import {
  finishOpenDeciderNanoRequest,
  packOpenDeciderNanoRequest,
  readOpenDeciderNanoMarkerScores
} from '../src/opendecider-nano/protocol.mjs';
import {openOpenDeciderNanoSource} from '../src/opendecider-nano/source.mjs';
import {createOpenDeciderNanoTokenizer} from '../src/opendecider-nano/tokenizer.mjs';
import {createOpenDeciderNanoSession} from '../src/opendecider-nano/session.mjs';

const fixtures = fileURLToPath(new URL('./fixtures/', import.meta.url));
const assets = join(fixtures, 'opendecider-nano-assets');
const tokenizerPath = join(fixtures, 'opendecider-nano-tokenizer.web.json');
const casesPath = join(fixtures, 'opendecider-nano-cpu-cases.json');
const hasTokenizerFixtures = existsSync(tokenizerPath) && existsSync(casesPath);
const tokenizerSkip = hasTokenizerFixtures ? false : 'Local OpenDecider-nano tokenizer fixtures are not included.';
const tokenizerData = hasTokenizerFixtures ? JSON.parse(await readFile(tokenizerPath, 'utf8')) : null;
const tokenizer = tokenizerData ? createOpenDeciderNanoTokenizer(Tokenizer, tokenizerData, {}) : null;
const cases = hasTokenizerFixtures ? JSON.parse(await readFile(casesPath, 'utf8')).cases : [];
const sourceFiles = [
  'manifest.json',
  'model.wasm',
  'tokenizer.json',
  'tokenizer_config.json',
  'weights/tensor-cache.json'
].map((path) => join(assets, path));
const sourceSkip = sourceFiles.every(existsSync)
  ? false
  : 'Local OpenDecider-nano graph and tokenizer assets are not included.';

test('OpenDecider-nano tokens match the official README input fixture', {skip: tokenizerSkip}, () => {
  const packed = packOpenDeciderNanoRequest(tokenizer!, {
    state: 'I was charged twice for order 1182.',
    questions: [
      {
        instructions: 'Which team should handle this?',
        options: ['billing: charges, refunds', 'tech: bugs, outages']
      }
    ]
  });
  assert.deepEqual(packed.questions[0]!.ids, cases[0].inputIds);
  assert.deepEqual(packed.questions[0]!.markerPositions, cases[0].markerPositions);
  assert.equal(packed.tokenCount, 37);
});

test('OpenDecider-nano typed questions preserve choice, ordered score, and yes probability', async () => {
  const prompts: string[] = [];
  const logits = [Float32Array.of(3, 0, 0), Float32Array.of(0, 1, 2), Float32Array.of(2, 0)];
  const session = createOpenDeciderNanoSession({
    tokenizer: {encode: (text) => (prompts.push(text), [1])},
    runner: {
      run: async () => logits.shift()!,
      releaseCachedMemory: async () => {},
      dispose: async () => {}
    }
  });

  const result = await session.systemOne(
    {ticket: 'refund', attempt: 2},
    {
      department: {
        type: 'choice',
        instructions: 'Which team should handle this?',
        criteria: {billing: 'charges, refunds', technical: 'bugs', other: null}
      },
      urgency: {type: 'score', instructions: 'How urgent is this?', criteria: ['not urgent', 'soon', 'blocking']},
      churnRisk: {type: 'noul', instructions: 'Does the user threaten to cancel or leave?'}
    }
  );

  assert.equal(result.model, 'experimental-opendecider-nano-fp16');
  const {department, urgency, churnRisk} = result.answers;
  assert.ok(department?.type === 'choice' && urgency?.type === 'score' && churnRisk?.type === 'noul');
  assert.equal(department.type, 'choice');
  assert.equal(department.choice, 'billing');
  assert.deepEqual(Object.keys(department.probabilities), ['billing', 'technical', 'other']);
  assert.equal(urgency.type, 'score');
  assert.equal(urgency.score, 2);
  assert.equal(urgency.legend['0'], 'not urgent');
  assert.ok(Math.abs(urgency.expected - 1.57521) < 0.0001);
  assert.equal(churnRisk.type, 'noul');
  assert.ok(Math.abs(churnRisk.noul - 0.880797) < 0.00001);
  assert.ok(Math.abs(churnRisk.probabilities.true + churnRisk.probabilities.false - 1) < 1e-12);
  assert.equal(result.tokenCount, 12);
  assert.deepEqual(prompts, [
    'input: {"ticket": "refund", "attempt": 2}',
    'question: Which team should handle this?',
    ' billing: charges, refunds',
    ' technical: bugs',
    ' other',
    'question: How urgent is this?',
    ' 0: not urgent',
    ' 1: soon',
    ' 2: blocking',
    'question: Does the user threaten to cancel or leave?',
    ' yes: Yes',
    ' no: No'
  ]);
  await session.dispose();
});

test('OpenDecider-nano accepts more than 255 generic and named questions', async () => {
  let inferenceCalls = 0;
  const session = createOpenDeciderNanoSession({
    tokenizer: {encode: () => [1]},
    runner: {
      run: async () => {
        inferenceCalls++;
        return Float32Array.of(1, 0);
      },
      releaseCachedMemory: async () => {},
      dispose: async () => {}
    }
  });
  const genericQuestions = Array.from({length: 256}, () => ({instructions: 'Choose.', options: ['yes', 'no']}));
  const genericResult = await session.decide({state: 'state', questions: genericQuestions});
  assert.equal(genericResult.answers.length, 256);
  assert.equal(inferenceCalls, 256);

  const namedQuestions = Object.fromEntries(
    Array.from({length: 256}, (_, index) => [`question-${index}`, {type: 'noul', instructions: 'Is this true?'}])
  );
  const typedResult = await session.systemOne('state', namedQuestions);
  assert.equal(Object.keys(typedResult.answers).length, 256);
  assert.ok(Object.hasOwn(typedResult.answers, 'question-255'));
  assert.equal(inferenceCalls, 512);
  await session.dispose();
});

test('OpenDecider-nano accepts more than 255 generic and typed options within one row', async () => {
  const session = createOpenDeciderNanoSession({
    tokenizer: {encode: () => [1]},
    runner: {
      run: async (_ids, markerPositions) => Float32Array.from(markerPositions, (_, index) => index),
      releaseCachedMemory: async () => {},
      dispose: async () => {}
    }
  });
  const options = Array.from({length: 256}, (_, index) => `option-${index}`);
  const genericResult = await session.decide({
    state: 'state',
    questions: [{instructions: 'Choose.', options}]
  });
  assert.equal(genericResult.answers[0]!.probabilities.length, 256);
  assert.equal(genericResult.answers[0]!.selected, 'option-255');
  assert.equal(genericResult.tokenCount, 518);

  const typedResult = await session.systemOne('state', {
    choice: {type: 'choice', instructions: 'Choose.', criteria: options},
    score: {type: 'score', instructions: 'Rate this.', criteria: options}
  });
  const {choice, score} = typedResult.answers;
  assert.ok(choice?.type === 'choice' && score?.type === 'score');
  assert.equal(choice.choice, 'option-255');
  assert.equal(Object.keys(choice.probabilities).length, 256);
  assert.equal(score.score, 255);
  assert.equal(score.legend['255'], 'option-255');
  assert.equal(typedResult.tokenCount, 518);
  await session.dispose();
});

test('OpenDecider-nano typed questions reject unsupported schemas before inference', async () => {
  let inferenceCalls = 0;
  const session = createOpenDeciderNanoSession({
    tokenizer: {encode: () => [1]},
    runner: {
      run: async () => {
        inferenceCalls++;
        return Float32Array.of(1, 0);
      },
      releaseCachedMemory: async () => {},
      dispose: async () => {}
    }
  });
  const validNoul = {check: {type: 'noul', instructions: 'Is this true?'}};
  const cyclic: {self?: unknown} = {};
  cyclic.self = cyclic;

  for (const [state, questions] of [
    ['state', {}],
    ['state', {check: {type: 'choice', instructions: 'Choose.', criteria: ['only one']}}],
    ['state', {check: {type: 'choice', instructions: 'Choose.', criteria: ['valid', 3]}}],
    ['state', {check: {type: 'other', instructions: 'Choose.', criteria: ['yes', 'no']}}],
    ['state', {check: {type: 'noul', instructions: 'Is this true?', criteria: {maybe: 'unclear'}}}],
    [cyclic, validNoul]
  ]) {
    await assert.rejects(
      session.systemOne(state, questions),
      (error) => error instanceof WebDecisionError && error.code === 'INVALID_INPUT'
    );
  }
  assert.equal(inferenceCalls, 0);
  await session.dispose();
});

test('text that looks like a model marker remains ordinary state text', {skip: tokenizerSkip}, () => {
  const packed = packOpenDeciderNanoRequest(tokenizer!, {
    state: 'The log contained [MASK] and [SEP] as literal text.',
    questions: [{instructions: 'Route the event.', options: ['billing', 'tech']}]
  });
  assert.equal(packed.questions[0]!.markerPositions.length, 2);
  assert.equal(packed.questions[0]!.ids.filter((id) => id === 50282).length, 3);
});

test('state truncation preserves the question and every option marker', {skip: tokenizerSkip}, () => {
  const packed = packOpenDeciderNanoRequest(tokenizer!, {
    state: 'ticket details '.repeat(4000),
    questions: [{instructions: 'Choose a route.', options: ['billing', 'technical']}]
  });
  const question = packed.questions[0]!;
  assert.equal(question.ids.length, 2048);
  assert.equal(question.truncated, true);
  assert.deepEqual(
    question.markerPositions.map((position) => question.ids[position]),
    [50284, 50284]
  );
  const result = finishOpenDeciderNanoRequest(packed, [Float32Array.of(2, 0)]);
  assert.equal(result.truncated, true);
  assert.equal(result.answers[0]!.selected, 'billing');
  assert.deepEqual(
    result.answers[0]!.probabilities.map(({option}) => option),
    ['billing', 'technical']
  );
});

test('invalid question choices and oversized question text are rejected', {skip: tokenizerSkip}, () => {
  assert.throws(
    () =>
      packOpenDeciderNanoRequest(tokenizer!, {
        state: 'ticket',
        questions: [{instructions: 'Choose.', options: ['billing', 'billing']}]
      }),
    (error) => error instanceof WebDecisionError && error.code === 'INVALID_INPUT'
  );
  assert.throws(
    () =>
      packOpenDeciderNanoRequest(tokenizer!, {
        state: 'ticket',
        questions: [{instructions: 'Choose.', options: ['x '.repeat(3000), 'technical']}]
      }),
    (error) => error instanceof WebDecisionError && error.code === 'INVALID_INPUT'
  );
});

test('marker readback follows marker positions and validates their layout', () => {
  const ids = [50284, 12, 13, 50284];
  const positions = [0, 3];
  const logits = Float32Array.of(-10, 0, 0, -3);
  assert.deepEqual(readOpenDeciderNanoMarkerScores(ids, positions, logits), Float32Array.of(-10, -3));
  assert.throws(() => readOpenDeciderNanoMarkerScores(ids, [0, 0], logits), WebDecisionError);
});

test('pinned source metadata and tokenizer load from the generated TVM bundle', {skip: sourceSkip}, async (context) => {
  const server = createServer(async (request, response) => {
    const path = normalize(decodeURIComponent(new URL(request.url!, 'http://localhost').pathname.slice(1)));
    const file = join(assets, path);
    if (relative(assets, file).startsWith(`..${sep}`)) {
      response.writeHead(400).end();
      return;
    }
    try {
      response.writeHead(200, {'content-length': (await stat(file)).size});
      createReadStream(file).pipe(response);
    } catch {
      response.writeHead(404).end();
    }
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  context.after(
    () => new Promise<void>((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())))
  );
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  const base = `http://127.0.0.1:${address.port}/`;
  const progress: Parameters<
    NonNullable<NonNullable<Parameters<typeof openOpenDeciderNanoSource>[1]>['onProgress']>
  >[0][] = [];
  const loaded = await openOpenDeciderNanoSource(new URL(base), {onProgress: (item) => progress.push(item)});
  assert.equal(loaded.manifest.build!.parameterCount, 200);
  assert.equal(loaded.tensorCache.metadata!.numParams, 200);
  assert.equal(loaded.tensorCache.records.length, 29);
  assert.ok(progress.length >= 2);
  assert.ok(progress.at(-1)!.loadedBytes < progress.at(-1)!.totalBytes);
});
