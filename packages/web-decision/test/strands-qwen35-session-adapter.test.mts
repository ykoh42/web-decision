import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {evaluateQwen35Request, type Qwen35GraphRunner} from '../src/strands/qwen35-session-adapter.mts';
import type {Qwen35GraphInputChunk, Qwen35GraphInputRecord} from '../src/strands/qwen35-graph-input.mts';
import {QWEN35_GRAPH_MAX_BATCH} from '../src/strands/qwen35-protocol.mts';
import {createQwen35PreEmbedSourceResolver} from '../src/strands/qwen35-pre-embed-sources.mts';
import {prepareQwen35R13GraphInputTuple} from '../src/strands/qwen35-r13-pre-embed-bridge.mts';

const utf8 = new TextEncoder();
const specialTokens = [
  ['<|image_pad|>', 248056],
  ['<|vision_end|>', 248054],
  ['<|endoftext|>', 248044],
  ['<bos>', 248043]
] as const;

function plainJsonCopy<T>(value: T): T {
  if (Array.isArray(value)) return value.map(plainJsonCopy) as T;
  if (value === null || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, plainJsonCopy(entry)])) as T;
}

function tokenizerFixture(onEncode?: () => void) {
  const byteDecoder = Object.fromEntries(
    Array.from({length: 256}, (_, byte) => [String.fromCharCode(byte + 0x100), byte])
  );
  const addedTokenIds = new Set(specialTokens.map(([, id]) => id));
  return {
    decoder: {byte_decoder: byteDecoder},
    get_added_tokens_decoder: () => addedTokenIds,
    encode(text: string, {add_special_tokens}: {readonly add_special_tokens: boolean}) {
      onEncode?.();
      const ids: number[] = [];
      const tokens: string[] = [];
      if (add_special_tokens) {
        ids.push(248043);
        tokens.push('<bos>');
      }
      let cursor = 0;
      while (cursor < text.length) {
        const special = specialTokens.find(([token]) => text.startsWith(token, cursor));
        if (special) {
          tokens.push(special[0]);
          ids.push(special[1]);
          cursor += special[0].length;
          continue;
        }
        let end = cursor + 1;
        while (end < text.length && !specialTokens.some(([token]) => text.startsWith(token, end))) end += 1;
        for (const byte of utf8.encode(text.slice(cursor, end).normalize('NFC'))) {
          tokens.push(String.fromCharCode(byte + 0x100));
          ids.push(byte + 1);
        }
        cursor = end;
      }
      return {ids, tokens};
    }
  };
}

function noulQuestions(count: number) {
  return Object.fromEntries(
    Array.from({length: count}, (_, index) => [`q${index}`, {type: 'noul', instructions: `Is ${index} true?`}])
  );
}

function probabilitiesFor(chunk: Qwen35GraphInputChunk): readonly (readonly number[])[] {
  return chunk.questions.map(({type, options}) => {
    if (type === 'choice')
      return Array.from({length: options.length}, (_, index) => (index === 1 ? 0.75 : index === 0 ? 0.25 : 0));
    if (type === 'score') {
      const total = (options.length * (options.length + 1)) / 2;
      return Array.from({length: options.length}, (_, index) => (index + 1) / total);
    }
    return [0.1, 0.9];
  });
}

function graphDispatchSizes(sourceGroups: readonly number[]): number[] {
  const dispatches: number[] = [];
  for (let remaining of sourceGroups) {
    while (remaining > 0) {
      const size = Math.min(remaining, QWEN35_GRAPH_MAX_BATCH);
      dispatches.push(size);
      remaining -= size;
    }
  }
  return dispatches;
}

function runnerFixture(
  run: (chunk: Qwen35GraphInputChunk, input: Qwen35GraphInputRecord) => unknown | PromiseLike<unknown>
) {
  const calls: Array<{chunk: Qwen35GraphInputChunk; input: Qwen35GraphInputRecord}> = [];
  const runner: Qwen35GraphRunner = {
    async run(chunk, input) {
      calls.push({chunk, input});
      return run(chunk, input);
    }
  };
  return {runner, calls};
}

test('Qwen adapter preserves named typed request, structured content, output schema, and source usage', async () => {
  const fixture = runnerFixture((chunk) => probabilitiesFor(chunk));
  const result = await evaluateQwen35Request(
    tokenizerFixture(),
    fixture.runner,
    {
      model: 'strands-decider-latest',
      state: {subject: 'a river', facts: ['flows north', {source: 'map'}]},
      ignoredRequestField: 'Pydantic extra=ignore',
      questions: {
        route: {
          type: 'choice',
          instructions: {task: 'choose route'},
          criteria: {north: {description: 'to the valley'}, south: null},
          ignoredQuestionField: true
        },
        quality: {type: 'score', instructions: 'rate quality', criteria: ['low', 'medium', 'high']},
        valid: {type: 'noul', instructions: 'does the statement hold?'}
      }
    },
    {modelName: 'strands-decider-2B-qwen3.5-v1-2610'}
  );

  assert.equal(fixture.calls.length, 1);
  assert.deepEqual(JSON.parse(JSON.stringify(fixture.calls[0]!.input.request.state)), {
    subject: 'a river',
    facts: ['flows north', {source: 'map'}]
  });
  assert.deepEqual(
    fixture.calls[0]!.chunk.questions.map(({name, type, options}) => [name, type, options.map(({label}) => label)]),
    [
      ['route', 'choice', ['north', 'south']],
      ['quality', 'score', ['0', '1', '2']],
      ['valid', 'noul', ['false', 'true']]
    ]
  );
  assert.deepEqual(Object.keys(result), ['model', 'answers', 'usage']);
  assert.deepEqual(Object.keys(result.answers), ['route', 'quality', 'valid']);
  const route = result.answers.route;
  const quality = result.answers.quality;
  const valid = result.answers.valid;
  if (route?.type !== 'choice' || quality?.type !== 'score' || valid?.type !== 'noul')
    assert.fail('Qwen typed answers must be returned with their source answer variants.');
  assert.equal(route.choice, 'south');
  assert.deepEqual(Object.keys(route), ['type', 'choice', 'probabilities', 'confidence']);
  assert.deepEqual(Object.keys(route.probabilities), ['north', 'south']);
  assert.ok(Number.isFinite(route.confidence));
  assert.deepEqual(Object.keys(quality), ['type', 'score', 'legend', 'probabilities', 'confidence']);
  assert.ok(quality.score > 1);
  assert.deepEqual(quality.legend, {0: 'low', 1: 'medium', 2: 'high'});
  assert.ok(valid.noul > 0.5);
  assert.deepEqual(Object.keys(valid), ['type', 'noul']);
  assert.equal('abstain' in valid, false);
  assert.equal(result.model, 'strands-decider-2B-qwen3.5-v1-2610');
  assert.equal(result.usage.input_tokens, fixture.calls[0]!.input.inputTokenCount);
  assert.equal(result.usage.output_tokens, 3);
});

test('Qwen maxBatch 32 preserves every typed answer across three-row graph chunks', async () => {
  const request = {
    state: 'Keep the seven source questions in order.',
    questions: {
      q00: {type: 'choice', instructions: 'marker-q00', criteria: {alpha: null, beta: null}},
      q01: {type: 'noul', instructions: 'marker-q01'},
      q02: {type: 'score', instructions: 'marker-q02', criteria: ['low', 'medium', 'high']},
      q03: {type: 'choice', instructions: 'marker-q03', criteria: {first: null, second: null, third: null}},
      q04: {type: 'noul', instructions: 'marker-q04'},
      q05: {type: 'score', instructions: 'marker-q05', criteria: ['poor', 'fair', 'good', 'great']},
      q06: {type: 'choice', instructions: 'marker-q06', criteria: {red: null, green: null}}
    }
  };
  const fixture = runnerFixture((chunk) => probabilitiesFor(chunk));
  const result = await evaluateQwen35Request(tokenizerFixture(), fixture.runner, request, {
    modelName: 'configured-engine-name',
    maxBatch: 32
  });

  assert.deepEqual(
    fixture.calls.map(({chunk}) => chunk.questions.length),
    [QWEN35_GRAPH_MAX_BATCH, QWEN35_GRAPH_MAX_BATCH, 1]
  );
  assert.ok(fixture.calls.every(({chunk}) => chunk.questions.length <= QWEN35_GRAPH_MAX_BATCH));
  const dispatchedQuestions = fixture.calls.flatMap(({chunk}) => chunk.questions);
  assert.deepEqual(
    dispatchedQuestions.map(({name}) => name),
    Object.keys(request.questions),
    'graph microbatches retain every source question in insertion order'
  );
  assert.deepEqual(
    dispatchedQuestions.map(({questionText}) => questionText.match(/marker-q\d{2}/u)?.[0]),
    Object.keys(request.questions).map((name) => `marker-${name}`),
    'question content is preserved through packing and graph splitting'
  );

  assert.deepEqual(Object.keys(result.answers), Object.keys(request.questions));
  assert.deepEqual(
    Object.values(result.answers).map(({type}) => type),
    ['choice', 'noul', 'score', 'choice', 'noul', 'score', 'choice'],
    'typed answers are merged back in source order'
  );
  assert.equal(result.answers.q00?.type, 'choice');
  if (result.answers.q00?.type === 'choice') assert.equal(result.answers.q00.choice, 'beta');
  assert.equal(result.answers.q01?.type, 'noul');
  if (result.answers.q01?.type === 'noul') assert.equal(result.answers.q01.noul, 0.9);
  assert.equal(result.answers.q02?.type, 'score');
  if (result.answers.q02?.type === 'score') {
    assert.ok(Math.abs(result.answers.q02.score - 1.3333) < 0.001);
    assert.deepEqual(result.answers.q02.legend, {0: 'low', 1: 'medium', 2: 'high'});
  }
  assert.equal(result.answers.q03?.type, 'choice');
  if (result.answers.q03?.type === 'choice') assert.equal(result.answers.q03.choice, 'second');
  assert.equal(result.answers.q04?.type, 'noul');
  if (result.answers.q04?.type === 'noul') assert.equal(result.answers.q04.noul, 0.9);
  assert.equal(result.answers.q05?.type, 'score');
  if (result.answers.q05?.type === 'score') {
    assert.equal(result.answers.q05.score, 2);
    assert.deepEqual(result.answers.q05.legend, {0: 'poor', 1: 'fair', 2: 'good', 3: 'great'});
  }
  assert.equal(result.answers.q06?.type, 'choice');
  if (result.answers.q06?.type === 'choice') assert.equal(result.answers.q06.choice, 'green');
  assert.equal(result.usage.output_tokens, 7);
  const prefixLength = fixture.calls[0]!.chunk.prefixInputIds.length;
  const suffixTokens = dispatchedQuestions.reduce((sum, question) => sum + question.inputIds.length, 0);
  assert.equal(
    result.usage.input_tokens,
    prefixLength + suffixTokens,
    'maxBatch 32 keeps these questions in one source batch while graph dispatches use three-row chunks'
  );
});

test('Qwen adapter accepts numeric record keys, preserves Map order, and rejects unsafe integers', async () => {
  const recordFixture = runnerFixture((chunk) => probabilitiesFor(chunk));
  const recordResult = await evaluateQwen35Request(
    tokenizerFixture(),
    recordFixture.runner,
    {
      state: 'Numeric keys use ECMAScript own-key order.',
      questions: JSON.parse(
        '{"10":{"type":"choice","instructions":"Pick one.","criteria":{"10":null,"2":null}},"2":{"type":"noul","instructions":"Is it true?"}}'
      ) as Record<string, unknown>
    },
    {modelName: 'pinned-qwen'}
  );
  assert.deepEqual(
    recordFixture.calls[0]!.chunk.questions.map(({name}) => name),
    ['2', '10']
  );
  assert.deepEqual(
    recordFixture.calls[0]!.chunk.questions[1]!.options.map(({label}) => label),
    ['2', '10']
  );
  assert.deepEqual(plainJsonCopy(recordResult.answers['10']), {
    type: 'choice',
    choice: '10',
    probabilities: {2: 0.25, 10: 0.75},
    confidence: 0.5
  });

  const mapFixture = runnerFixture((chunk) => probabilitiesFor(chunk));
  const mapResult = await evaluateQwen35Request(
    tokenizerFixture(),
    mapFixture.runner,
    {
      state: 'Map keys keep explicit order.',
      questions: new Map([
        [
          '10',
          {
            type: 'choice',
            instructions: 'Pick one.',
            criteria: new Map([
              ['10', null],
              ['2', null]
            ])
          }
        ],
        ['2', {type: 'noul', instructions: 'Is it true?'}]
      ])
    },
    {modelName: 'pinned-qwen'}
  );
  assert.deepEqual(
    mapFixture.calls[0]!.chunk.questions.map(({name}) => name),
    ['10', '2']
  );
  assert.deepEqual(
    mapFixture.calls[0]!.chunk.questions[0]!.options.map(({label}) => label),
    ['10', '2']
  );
  assert.deepEqual(plainJsonCopy(mapResult.answers['10']), {
    type: 'choice',
    choice: '2',
    probabilities: {2: 0.75, 10: 0.25},
    confidence: 0.5
  });

  const unsafeFixture = runnerFixture((chunk) => probabilitiesFor(chunk));
  await assert.rejects(
    evaluateQwen35Request(
      tokenizerFixture(),
      unsafeFixture.runner,
      {
        state: {count: Number.MAX_SAFE_INTEGER + 1},
        questions: {truth: {type: 'noul', instructions: 'Is it true?'}}
      },
      {modelName: 'pinned-qwen'}
    ),
    /unsafe integer Number; use bigint/u
  );
  assert.equal(unsafeFixture.calls.length, 0);
});

test('Qwen preserves explicit Map order for numeric question and choice keys in response records', async () => {
  const fixture = runnerFixture((chunk) =>
    chunk.questions.map((question) => (question.type === 'choice' ? [0.9, 0.1] : [0.1, 0.9]))
  );
  const result = await evaluateQwen35Request(
    tokenizerFixture(),
    fixture.runner,
    {
      state: 'ordered state',
      questions: new Map([
        ['10', {type: 'noul', instructions: 'first question'}],
        ['2', {type: 'noul', instructions: 'second question'}],
        [
          'pick',
          {
            type: 'choice',
            instructions: 'pick by declared order',
            criteria: new Map([
              ['10', null],
              ['2', null]
            ])
          }
        ]
      ])
    },
    {modelName: 'configured-engine-name'}
  );

  assert.deepEqual(
    fixture.calls[0]!.chunk.questions.map(({name}) => name),
    ['10', '2', 'pick']
  );
  assert.deepEqual(Object.keys(result.answers), ['10', '2', 'pick']);
  assert.deepEqual(Object.keys(result), ['model', 'answers', 'usage']);
  const pick = result.answers.pick;
  if (pick?.type !== 'choice') assert.fail('Ordered choice row must remain a choice answer.');
  assert.equal(pick.choice, fixture.calls[0]!.chunk.questions[2]!.options[0]!.label);
  assert.deepEqual(Object.keys(pick), ['type', 'choice', 'probabilities', 'confidence']);
  assert.deepEqual(Object.keys(pick.probabilities), ['10', '2']);
  const serializedAnswers = JSON.stringify(result.answers);
  assert.ok(serializedAnswers.startsWith('{"10":'));
  assert.ok(serializedAnswers.indexOf('"2":') > serializedAnswers.indexOf('"10":'));
  assert.ok(serializedAnswers.includes('"probabilities":{"10":0.9,"2":0.1}'));
});

test('Qwen adapter accepts the pinned Strands schema slot boundaries and all Content forms', async () => {
  // Oracle: Strands schema.py at 13c05149265dc404c09d2469636f71b1320e1713.
  const choiceCriteria: Record<string, unknown> = Object.fromEntries(
    Array.from({length: 255}, (_, index) => [`o${index}`, null])
  );
  choiceCriteria.o0 = 'text description';
  choiceCriteria.o1 = {move: 'north'};
  choiceCriteria.o2 = ['list', 'description'];
  const questions = {
    choice_min: {
      type: 'choice',
      instructions: 'Pick one.',
      criteria: {first: null, second: null}
    },
    choice_max: {
      type: 'choice',
      instructions: {task: 'Choose from the complete set.'},
      criteria: choiceCriteria
    },
    score_min: {type: 'score', instructions: ['Rate the state.'], criteria: ['low', 'high']},
    score_max: {
      type: 'score',
      instructions: 'Use the full rubric.',
      criteria: Array.from({length: 10}, (_, index) => `level-${index}`)
    },
    noul_false: {
      type: 'noul',
      instructions: 'The statement is false.',
      criteria: {false: null, true: null}
    },
    noul_true: {
      type: 'noul',
      instructions: 'The statement is true.',
      criteria: {false: null, true: null}
    }
  };
  const fixture = runnerFixture((chunk) =>
    chunk.questions.map((question) => {
      if (question.name === 'noul_false') return [1, 0];
      if (question.name === 'noul_true') return [0, 1];
      return question.options.map((_, index) => (index === question.options.length - 1 ? 1 : 0));
    })
  );
  const result = await evaluateQwen35Request(
    tokenizerFixture(),
    fixture.runner,
    {state: ['state text', {source: 'structured state'}], questions},
    {modelName: 'configured-engine-name'}
  );

  const input = fixture.calls[0]!.input;
  assert.deepEqual(
    input.request.questions.map(({name, type, options}) => [name, type, options.length]),
    [
      ['choice_min', 'choice', 2],
      ['choice_max', 'choice', 255],
      ['score_min', 'score', 2],
      ['score_max', 'score', 10],
      ['noul_false', 'noul', 2],
      ['noul_true', 'noul', 2]
    ]
  );
  assert.deepEqual(
    input.request.questions[0]!.options.map(({label}) => label),
    ['first', 'second']
  );
  assert.deepEqual(
    input.request.questions[4]!.options.map(({label}) => label),
    ['false', 'true']
  );
  assert.deepEqual(
    input.request.questions[5]!.options.map(({label}) => label),
    ['false', 'true']
  );
  assert.deepEqual(JSON.parse(JSON.stringify(input.request.state)), ['state text', {source: 'structured state'}]);
  assert.match(input.request.questions[1]!.instructionText, /complete set/u);
  assert.match(fixture.calls[0]!.chunk.questions[2]!.questionText, /\[\n  "Rate the state\."\n\]/u);

  const choiceMin = result.answers.choice_min;
  const choiceMax = result.answers.choice_max;
  const scoreMin = result.answers.score_min;
  const scoreMax = result.answers.score_max;
  const noulFalse = result.answers.noul_false;
  const noulTrue = result.answers.noul_true;
  if (
    choiceMin?.type !== 'choice' ||
    choiceMax?.type !== 'choice' ||
    scoreMin?.type !== 'score' ||
    scoreMax?.type !== 'score' ||
    noulFalse?.type !== 'noul' ||
    noulTrue?.type !== 'noul'
  )
    assert.fail('Qwen adapter must preserve each pinned Strands answer variant.');
  assert.equal(Object.keys(choiceMin.probabilities).length, 2);
  assert.equal(Object.keys(choiceMax.probabilities).length, 255);
  assert.equal(choiceMax.choice, 'o254');
  assert.equal(Object.keys(scoreMin.legend).length, 2);
  assert.equal(Object.keys(scoreMax.legend).length, 10);
  assert.ok(noulFalse.noul < 0.01);
  assert.ok(noulTrue.noul > 0.99);
});

test('Qwen adapter rejects requests outside pinned Strands question cardinalities', async () => {
  // Oracle: ChoiceQuestion._check_size, ScoreQuestion._check_levels, and
  // SystemOneRequest.questions(min_length=1) in the pinned schema.py.
  const invalidRequests = [
    {
      request: {state: '', questions: {q: {type: 'choice', instructions: '', criteria: {only: null}}}},
      error: /choice requires 2 to 255 options/u
    },
    {
      request: {
        state: '',
        questions: {
          q: {
            type: 'choice',
            instructions: '',
            criteria: Object.fromEntries(Array.from({length: 256}, (_, index) => [`o${index}`, null]))
          }
        }
      },
      error: /choice requires 2 to 255 options/u
    },
    {
      request: {state: '', questions: {q: {type: 'score', instructions: '', criteria: ['only']}}},
      error: /score requires 2 to 10 levels/u
    },
    {
      request: {
        state: '',
        questions: {
          q: {
            type: 'score',
            instructions: '',
            criteria: Array.from({length: 11}, (_, index) => `level-${index}`)
          }
        }
      },
      error: /score requires 2 to 10 levels/u
    },
    {request: {state: '', questions: {}}, error: /at least one named question/u}
  ];

  for (const {request, error} of invalidRequests) {
    const fixture = runnerFixture((chunk) => probabilitiesFor(chunk));
    await assert.rejects(
      evaluateQwen35Request(tokenizerFixture(), fixture.runner, request, {modelName: 'configured-engine-name'}),
      error
    );
    assert.equal(fixture.calls.length, 0);
  }
});

test('Qwen adapter preserves source batch grouping across graph dispatches, order, usage, and strict-window behavior', async () => {
  const fixture = runnerFixture((chunk) => probabilitiesFor(chunk));
  const request = {state: 'shared state', questions: noulQuestions(33)};
  const result = await evaluateQwen35Request(tokenizerFixture(), fixture.runner, request, {
    modelName: 'configured-engine-name',
    maxBatch: 32
  });

  assert.deepEqual(
    fixture.calls.map(({chunk}) => chunk.questions.length),
    graphDispatchSizes([32, 1])
  );
  assert.deepEqual(Object.keys(result.answers), Object.keys(request.questions));
  const prefixLength = fixture.calls[0]!.chunk.prefixInputIds.length;
  const suffixTokens = fixture.calls.reduce(
    (sum, {chunk}) => sum + chunk.questions.reduce((rows, question) => rows + question.inputIds.length, 0),
    0
  );
  assert.equal(result.usage.input_tokens, 2 * prefixLength + suffixTokens);
  assert.equal(result.usage.output_tokens, 33);
  assert.equal(result.model, 'configured-engine-name');

  const batchEightQuestions = noulQuestions(17);
  const batchEightFixture = runnerFixture((chunk) => probabilitiesFor(chunk));
  const batchEightResult = await evaluateQwen35Request(
    tokenizerFixture(),
    batchEightFixture.runner,
    {state: 'shared state', questions: batchEightQuestions},
    {modelName: 'configured-engine-name', maxBatch: 8}
  );

  assert.deepEqual(
    batchEightFixture.calls.map(({chunk}) => chunk.questions.length),
    graphDispatchSizes([8, 8, 1])
  );
  assert.deepEqual(Object.keys(batchEightResult.answers), Object.keys(batchEightQuestions));
  assert.equal(batchEightResult.usage.output_tokens, 17);

  const tokenCapFixture = runnerFixture((chunk) => probabilitiesFor(chunk));
  await evaluateQwen35Request(
    tokenizerFixture(),
    tokenCapFixture.runner,
    {state: '', questions: noulQuestions(3)},
    {
      modelName: 'configured-engine-name',
      maxBatch: 3,
      maxBatchTokens: 1
    }
  );
  assert.deepEqual(
    tokenCapFixture.calls.map(({chunk}) => chunk.questions.length),
    [1, 1, 1]
  );

  const strictFixture = runnerFixture((chunk) => probabilitiesFor(chunk));
  await assert.rejects(
    evaluateQwen35Request(
      tokenizerFixture(),
      strictFixture.runner,
      {state: 's'.repeat(5000), questions: {q: {type: 'noul', instructions: 'Is this true?'}}},
      {modelName: 'configured-engine-name', strictWindow: true}
    ),
    /strict|context|4096/iu
  );
  assert.equal(strictFixture.calls.length, 0);
});

test('Qwen adapter splits configured batches above the graph size without capping request questions', async () => {
  const tokenizer = tokenizerFixture();
  const questions = Object.fromEntries(
    Array.from({length: 65}, (_, index) => [`q${index}`, {type: 'noul', instructions: 'Is this true?'}])
  );
  const request = {state: 'shared state', questions};
  const fixture33 = runnerFixture((chunk) => probabilitiesFor(chunk));
  const result33 = await evaluateQwen35Request(
    tokenizer,
    fixture33.runner,
    {state: 'shared state', questions: Object.fromEntries(Object.entries(questions).slice(0, 33))},
    {modelName: 'configured-engine-name', maxBatch: 64}
  );

  assert.deepEqual(
    fixture33.calls.map(({chunk}) => chunk.questions.length),
    graphDispatchSizes([33])
  );
  const prefixLength33 = fixture33.calls[0]!.chunk.prefixInputIds.length;
  const suffixTokens33 = fixture33.calls.reduce(
    (sum, {chunk}) => sum + chunk.questions.reduce((rows, question) => rows + question.inputIds.length, 0),
    0
  );
  assert.equal(result33.usage.input_tokens, prefixLength33 + suffixTokens33);
  assert.equal(result33.usage.output_tokens, 33);

  const fixture = runnerFixture((chunk) => probabilitiesFor(chunk));
  const result = await evaluateQwen35Request(tokenizer, fixture.runner, request, {
    modelName: 'configured-engine-name',
    maxBatch: 64
  });

  assert.deepEqual(
    fixture.calls.map(({chunk}) => chunk.questions.length),
    graphDispatchSizes([64, 1])
  );
  assert.deepEqual(Object.keys(result.answers), Object.keys(questions));
  assert.equal(result.usage.output_tokens, 65);
  const prefixLength = fixture.calls[0]!.chunk.prefixInputIds.length;
  const suffixTokens = fixture.calls.reduce(
    (sum, {chunk}) => sum + chunk.questions.reduce((rows, question) => rows + question.inputIds.length, 0),
    0
  );
  assert.equal(result.usage.input_tokens, 2 * prefixLength + suffixTokens);

  const rowCost = prefixLength + fixture.calls[0]!.chunk.questions[0]!.inputIds.length;
  const cappedFixture = runnerFixture((chunk) => probabilitiesFor(chunk));
  const cappedResult = await evaluateQwen35Request(tokenizer, cappedFixture.runner, request, {
    modelName: 'configured-engine-name',
    maxBatch: 64,
    maxBatchTokens: 40 * rowCost
  });
  assert.deepEqual(
    cappedFixture.calls.map(({chunk}) => chunk.questions.length),
    graphDispatchSizes([40, 24, 1])
  );
  assert.deepEqual(Object.keys(cappedResult.answers), Object.keys(questions));
  const cappedSuffixTokens = cappedFixture.calls.reduce(
    (sum, {chunk}) => sum + chunk.questions.reduce((rows, question) => rows + question.inputIds.length, 0),
    0
  );
  assert.equal(cappedResult.usage.input_tokens, 3 * prefixLength + cappedSuffixTokens);
});

test('Qwen adapter stays within the pinned context before tokenizing or starting the graph', async () => {
  const request = {
    state: 's'.repeat(6000),
    questions: {q: {type: 'noul', instructions: 'i'.repeat(5000)}}
  };
  const fixture = runnerFixture((chunk) => probabilitiesFor(chunk));
  const result = await evaluateQwen35Request(tokenizerFixture(), fixture.runner, request, {
    modelName: 'configured-engine-name',
    maxLength: 4096
  });

  assert.equal(fixture.calls[0]!.input.maxLength, 4096);
  assert.equal(fixture.calls[0]!.chunk.prefixInputShape[1] + fixture.calls[0]!.chunk.suffixInputShape[1], 4096);
  assert.equal(result.usage.input_tokens, 4096);

  const smallerFixture = runnerFixture((chunk) => probabilitiesFor(chunk));
  await evaluateQwen35Request(tokenizerFixture(), smallerFixture.runner, request, {
    modelName: 'configured-engine-name',
    maxLength: 2048
  });
  assert.equal(smallerFixture.calls[0]!.input.maxLength, 2048);
  assert.equal(
    smallerFixture.calls[0]!.chunk.prefixInputShape[1] + smallerFixture.calls[0]!.chunk.suffixInputShape[1],
    2048
  );

  let tokenizationCalls = 0;
  const noEncodeTokenizer = tokenizerFixture(() => {
    tokenizationCalls += 1;
  });
  const rejectedFixture = runnerFixture((chunk) => probabilitiesFor(chunk));
  await assert.rejects(
    evaluateQwen35Request(noEncodeTokenizer, rejectedFixture.runner, request, {
      modelName: 'configured-engine-name',
      maxLength: 4097
    }),
    (error: unknown) => error instanceof RangeError && /4096/u.test(error.message)
  );
  assert.equal(tokenizationCalls, 0);
  assert.equal(rejectedFixture.calls.length, 0);
});

test('Qwen adapter carries real image preprocessing data and matches VisionEngine batch grouping', async () => {
  const require = createRequire(import.meta.url);
  const magickWasm = await readFile(require.resolve('@imagemagick/magick-wasm/magick.wasm'));
  const image = Buffer.from(
    await readFile(new URL('./fixtures/qwen35-image-input/single-pixel.png', import.meta.url))
  ).toString('base64');
  const fixture = runnerFixture((chunk, input) => {
    assert.equal(input.imageCount, 1);
    assert.ok(input.imageInput!.pixelValues.length > 0);
    assert.deepEqual(Array.from(input.imageGridThwShape), [1, 3]);
    assert.ok(chunk.mmTokenTypeIds.includes(1));
    return probabilitiesFor(chunk);
  });
  const result = await evaluateQwen35Request(
    tokenizerFixture(),
    fixture.runner,
    {state: 'Inspect the image.', images: [image], questions: noulQuestions(2)},
    {modelName: 'configured-engine-name', maxBatch: 2, maxBatchTokens: 1, magickWasm}
  );

  assert.deepEqual(
    fixture.calls.map(({chunk}) => chunk.questions.length),
    [2]
  );
  assert.equal(result.usage.output_tokens, 2);
});

test('reuses verified source-order image features across the 3-row chunks of a 32-question request', async () => {
  const require = createRequire(import.meta.url);
  const magickWasm = await readFile(require.resolve('@imagemagick/magick-wasm/magick.wasm'));
  const image = Buffer.from(
    await readFile(new URL('./fixtures/qwen35-image-input/single-pixel.png', import.meta.url))
  ).toString('base64');
  const processedImageIndices: number[] = [];
  const visionBatches: Array<readonly Float32Array[]> = [];
  const resolver = createQwen35PreEmbedSourceResolver({
    embeddings: {
      async lookupTokenEmbeddings(tokenIds) {
        return new Float32Array(tokenIds.length * 2048);
      }
    },
    vision: {
      async run(input) {
        const features = input.visionGeometry.map(({featureRows}, imageIndex) => {
          processedImageIndices.push(imageIndex);
          return new Float32Array(featureRows * 2048).fill(imageIndex === 0 ? 4 : 8);
        });
        visionBatches.push(features);
        return features;
      }
    }
  });
  const resolvedFeatureBatches: Array<readonly Float32Array[]> = [];
  const composedImageFeatures: Array<readonly [number, number]> = [];
  const fixture = runnerFixture(async (chunk, input) => {
    const sources = await resolver(chunk, input);
    resolvedFeatureBatches.push(sources.imageFeatures);
    const [inputEmbeds] = await prepareQwen35R13GraphInputTuple(chunk, input, sources);
    const imageStarts: number[] = [];
    for (let index = 0; index < chunk.mmTokenTypeIds.length; index += 1)
      if (chunk.mmTokenTypeIds[index] === 1 && (index === 0 || chunk.mmTokenTypeIds[index - 1] === 0))
        imageStarts.push(index);
    assert.equal(imageStarts.length, 2);
    const firstImageStart = imageStarts[0]! * 2048;
    const secondImageStart = imageStarts[1]! * 2048;
    const observed = [inputEmbeds.values[firstImageStart]!, inputEmbeds.values[secondImageStart]!] as const;
    assert.deepEqual(observed, [4, 8], 'each R13 chunk keeps features in source image order');
    composedImageFeatures.push(observed);
    return probabilitiesFor(chunk);
  });

  const result = await evaluateQwen35Request(
    tokenizerFixture(),
    fixture.runner,
    {state: 'Inspect the images in order.', images: [image, image], questions: noulQuestions(32)},
    {modelName: 'configured-engine-name', maxBatch: 32, maxBatchTokens: 1, magickWasm}
  );

  assert.deepEqual(
    fixture.calls.map(({chunk}) => chunk.questions.length),
    Array.from({length: 10}, () => QWEN35_GRAPH_MAX_BATCH).concat([2])
  );
  assert.equal(fixture.calls.length, 11);
  assert.deepEqual(processedImageIndices, [0, 1], 'each source image is evaluated once for the prepared request');
  assert.equal(visionBatches.length, 1);
  assert.equal(resolvedFeatureBatches.length, 11);
  assert.ok(
    resolvedFeatureBatches.every((features) => features === visionBatches[0]),
    'all graph chunks receive the same verified feature array'
  );
  assert.deepEqual(
    composedImageFeatures,
    Array.from({length: 11}, () => [4, 8])
  );
  assert.deepEqual(
    Object.keys(result.answers),
    Array.from({length: 32}, (_, index) => `q${index}`)
  );
});

test('Qwen adapter rejects malformed masked-probability rows before formatting a response', async () => {
  const fixture = runnerFixture(() => [
    [0, 1],
    [0, 1]
  ]);
  await assert.rejects(
    evaluateQwen35Request(
      tokenizerFixture(),
      fixture.runner,
      {state: 'state', questions: {q: {type: 'noul', instructions: 'true?'}}},
      {modelName: 'configured-engine-name'}
    ),
    /one masked-probability row/u
  );
});

test('Qwen adapter preserves masked probabilities for mixed option counts without re-softmaxing', async () => {
  const fixture = runnerFixture(() => [
    // For the two-option row, the K=3 graph's -10000 padding sentinel can absorb the
    // normalization mass before n_slots masks it to zero.
    Float32Array.of(0, 0, 0),
    Float32Array.of(0.2, 0.3, 0.5)
  ]);
  const result = await evaluateQwen35Request(
    tokenizerFixture(),
    fixture.runner,
    {
      state: 'x',
      questions: {
        two: {type: 'choice', instructions: 'Pick.', criteria: {first: null, second: null}},
        three: {type: 'choice', instructions: 'Pick.', criteria: {left: null, middle: null, right: null}}
      }
    },
    {modelName: 'configured-engine-name'}
  );

  assert.deepEqual(plainJsonCopy(result.answers.two), {
    type: 'choice',
    choice: 'first',
    probabilities: {first: 0, second: 0},
    confidence: 0
  });
  assert.deepEqual(plainJsonCopy(result.answers.three), {
    type: 'choice',
    choice: 'right',
    probabilities: {left: 0.2, middle: 0.3, right: 0.5},
    confidence: 0.25
  });
});
