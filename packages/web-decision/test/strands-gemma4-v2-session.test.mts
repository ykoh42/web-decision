import assert from 'node:assert/strict';
import {test} from 'node:test';
import {kananaFloat} from '../src/kanana/json-float.ts';
import {createGemma4V2Session, type Gemma4V2RunnerInput} from '../src/strands/gemma4-v2-session.mjs';
import {renderStrandsV2} from '../src/strands/gemma4-v2-protocol.mjs';

const tokenizer = {
  encode(text: string) {
    return {ids: Array.from(new TextEncoder().encode(text), (x) => x + 1000)};
  }
};
const choice = {type: 'choice' as const, instructions: 'Choose', criteria: {red: 'red', blue: 'blue'}};
function create(
  runner: {
    run(input: Gemma4V2RunnerInput, signal?: AbortSignal | null): Promise<ArrayLike<number>>;
    releaseCachedMemory(): void | Promise<void>;
    dispose(): void | Promise<void>;
  },
  prepareImage = async (image: string) => ({tokens: 2, value: image})
) {
  return createGemma4V2Session({
    tokenizer,
    runner,
    prepareImage,
    modelName: 'e2b-v2',
    temperature: 1.5456,
    softcap: 30
  });
}

test('snapshots and evaluates text, images, and multiple question types', async () => {
  const calls: Gemma4V2RunnerInput[] = [];
  const session = create({
    async run(input) {
      calls.push(input);
      return [0, 0];
    },
    releaseCachedMemory() {},
    dispose() {}
  });
  const request = {state: 'old', images: ['image-a'], questions: {color: {...choice}, truth: {instructions: 'true?'}}};
  const resultPromise = session.systemOne(request);
  request.state = 'new';
  request.images[0] = 'image-b';
  request.questions.color.instructions = 'changed';
  const result = await resultPromise;
  assert.equal(result.model, 'e2b-v2');
  assert.equal(result.answers.color?.type, 'choice');
  assert.equal(result.answers.truth?.type, 'noul');
  assert.equal(result.usage.output_tokens, 2);
  assert.equal(calls.length, 2);
  assert.deepEqual(calls[0]!.images, ['image-a']);
  assert.equal(calls[0]!.imageSpans[0]!.end - calls[0]!.imageSpans[0]!.start, 2);
  const expected = renderStrandsV2(
    {state: 'old', questions: {color: choice, truth: {instructions: 'true?'}}},
    tokenizer,
    [2]
  );
  assert.deepEqual(calls[0]!.stateIds, expected.stateIds);
  assert.deepEqual(calls[0]!.questionIds, expected.questions[0]!.questionIds);
  assert.equal(result.usage.input_tokens, expected.inputTokens);
  await session.dispose();
});

test('checks every question against the requested context before invoking the runner', async () => {
  let calls = 0;
  const session = create({
    async run() {
      calls++;
      return [0, 0];
    },
    releaseCachedMemory() {},
    dispose() {}
  });
  const request = {state: 's', questions: {short: {...choice}, long: {...choice, instructions: 'x'.repeat(300)}}};
  const rendered = renderStrandsV2(request, tokenizer);
  const maxLength = rendered.stateIds.length + rendered.questions[0]!.questionIds.length;
  await assert.rejects(session.systemOne(request, {maxLength}), /context length/i);
  assert.equal(calls, 0);
  await session.systemOne(request, {maxLength: 1000});
  assert.equal(calls, 2);
  await session.dispose();
});

test('serializes GPU work, observes cancellation, and waits for active work before disposal', async () => {
  let unblock!: () => void;
  let started!: () => void;
  const entered = new Promise<void>((resolve) => {
    started = resolve;
  });
  const gate = new Promise<void>((resolve) => {
    unblock = resolve;
  });
  let calls = 0;
  let disposed = false;
  const session = create({
    async run() {
      calls++;
      started();
      await gate;
      return [0, 0];
    },
    releaseCachedMemory() {},
    dispose() {
      disposed = true;
    }
  });
  const controller = new AbortController();
  const active = session.systemOne({state: 's', questions: {q: choice}}, {signal: controller.signal});
  await entered;
  const queued = session.systemOne({state: 's', questions: {q: choice}});
  const queuedRejection = assert.rejects(queued, /disposed/i);
  controller.abort(new Error('cancelled'));
  await assert.rejects(active, /cancelled/i);
  const disposal = session.dispose();
  assert.equal(disposed, false);
  await assert.rejects(session.systemOne({state: 's', questions: {q: choice}}), /disposed/i);
  unblock();
  await queuedRejection;
  await disposal;
  assert.equal(calls, 1);
  assert.equal(disposed, true);
  assert.equal(session.dispose(), disposal);
});

test('rejects invalid audio and propagates image or runner failures', async () => {
  const session = create(
    {
      async run() {
        throw new Error('runner failed');
      },
      releaseCachedMemory() {},
      dispose() {}
    },
    async () => {
      throw new Error('image failed');
    }
  );
  await assert.rejects(session.systemOne({state: 's', questions: {q: choice}, audio: 'data'}), /audio/i);
  await assert.rejects(session.systemOne({state: 's', questions: {q: choice}, images: ['bad']}), /image failed/i);
  await assert.rejects(session.systemOne({state: 's', questions: {q: choice}}), /runner failed/i);
  await session.dispose();
});

test('preserves Python dictionary order for numeric Map keys through prompts and answers', async () => {
  const {readFile} = await import('node:fs/promises');
  const rows = JSON.parse(
    await readFile(new URL('./fixtures/strands-gemma4-v2-map-order.json', import.meta.url), 'utf8')
  ) as Array<{
    keys: string[];
    state_text: string;
    questions: Record<string, string>;
    answer: {choice: string; probabilities: Record<string, number>};
  }>;
  for (const row of rows) {
    const state = new Map<string, unknown>([
      [
        '10',
        new Map([
          ['20', 'twenty'],
          ['2', 'two']
        ])
      ],
      ['2', ['value']]
    ]);
    const criteria = new Map(row.keys.map((key) => [key, `description ${key}`]));
    const request = {
      state,
      questions: new Map([
        ['10', {type: 'choice' as const, instructions: 'Choose', criteria}],
        ['2', {type: 'noul' as const, instructions: 'Is it true?'}]
      ])
    };
    const rendered = renderStrandsV2(request, tokenizer);
    assert.deepEqual(
      rendered.questions.map((q) => q.name),
      ['10', '2']
    );
    assert.deepEqual(rendered.stateIds, [2, 105, ...tokenizer.encode(row.state_text.slice('<bos><|turn>'.length)).ids]);
    assert.equal(rendered.questions[0]!.questionText, row.questions['10']);
    assert.deepEqual(rendered.questions[0]!.options, row.keys);
    const logits = Array.from({length: row.keys.length}, (_, index) => (index === 1 ? 100 : -100));
    const session = create({
      async run(input) {
        return input.questionIds.length === rendered.questions[0]!.questionIds.length ? logits : [0, 0];
      },
      releaseCachedMemory() {},
      dispose() {}
    });
    const result = await session.systemOne(request);
    assert.deepEqual(Object.keys(result.answers), ['10', '2']);
    const answer = result.answers['10']!;
    if (answer.type !== 'choice') throw new Error('Expected a choice');
    assert.equal(answer.choice, row.answer.choice);
    assert.deepEqual(Object.keys(answer.probabilities), row.keys);
    assert.deepEqual(JSON.parse(JSON.stringify(answer.probabilities)), row.answer.probabilities);
    await session.dispose();
  }
});

test('accepts exact context boundaries including configured long contexts', async () => {
  let calls = 0;
  const session = create({
    async run() {
      calls++;
      return [0, 0];
    },
    releaseCachedMemory() {},
    dispose() {}
  });
  const request = (length: number) => ({state: 'x'.repeat(length), questions: {q: choice}});
  const rendered = renderStrandsV2(request(1), tokenizer);
  const overhead = rendered.stateIds.length + rendered.questions[0]!.questionIds.length - 1;
  for (const limit of [32768, 131072]) {
    await session.systemOne(request(limit - overhead), {maxLength: limit});
    await assert.rejects(session.systemOne(request(limit - overhead + 1), {maxLength: limit}), /context length/i);
  }
  assert.equal(calls, 2);
  await session.dispose();
});

test('accepts null call options like the existing Strands public API', async () => {
  const session = create({
    async run() {
      return [0, 0];
    },
    releaseCachedMemory() {},
    dispose() {}
  });
  const result = await session.systemOne({state: '', questions: {q: {instructions: 'yes?'}}}, null);
  assert.equal(result.answers.q?.type, 'noul');
  await session.dispose();
});

test('closes and keeps disposal idempotent even when runner disposal fails', async () => {
  let disposals = 0;
  const session = create({
    async run() {
      return [0, 0];
    },
    releaseCachedMemory() {},
    dispose() {
      disposals++;
      throw new Error('device disposal failed');
    }
  });
  const first = session.dispose();
  assert.equal(session.dispose(), first);
  await assert.rejects(first, /device disposal failed/);
  await assert.rejects(session.systemOne({state: '', questions: {q: {instructions: 'yes?'}}}), /disposed/);
  assert.equal(disposals, 1);
});

test('rejects a non-string model field as the upstream request does', async () => {
  const session = create({
    async run() {
      throw new Error('runner must not execute');
    },
    releaseCachedMemory() {},
    dispose() {}
  });
  await assert.rejects(
    session.systemOne({model: 123, questions: {q: {instructions: 'yes?'}}} as never),
    /model must be a string/i
  );
  await session.dispose();
});

test('reports the compiled graph limit even when the caller requests a longer context', async () => {
  let calls = 0;
  const session = create({
    async run() {
      calls++;
      return [0, 0];
    },
    releaseCachedMemory() {},
    dispose() {}
  });
  const request = (length: number) => ({state: 'x'.repeat(length), questions: {q: choice}});
  const rendered = renderStrandsV2(request(1), tokenizer);
  const overhead = rendered.stateIds.length + rendered.questions[0]!.questionIds.length - 1;
  await session.systemOne(request(131072 - overhead), {maxLength: 131073});
  await assert.rejects(session.systemOne(request(131073 - overhead), {maxLength: 131073}), /compiled.*131072/i);
  assert.equal(calls, 1);
  await session.dispose();
});

test('serializes concurrent requests and runs queued memory release in order', async () => {
  const events: string[] = [];
  let unblock!: () => void;
  let entered!: () => void;
  const gate = new Promise<void>((resolve) => {
    unblock = resolve;
  });
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  let count = 0;
  const session = create({
    async run() {
      const number = ++count;
      events.push(`start-${number}`);
      if (number === 1) {
        entered();
        await gate;
      }
      events.push(`end-${number}`);
      return [0, 0];
    },
    releaseCachedMemory() {
      events.push('release');
    },
    dispose() {
      events.push('dispose');
    }
  });
  const first = session.systemOne({state: 'one', questions: {q: {instructions: 'yes?'}}});
  await started;
  const second = session.systemOne({state: 'two', questions: {q: {instructions: 'yes?'}}});
  const release = session.releaseCachedMemory();
  assert.deepEqual(events, ['start-1']);
  unblock();
  await Promise.all([first, second, release]);
  assert.deepEqual(events, ['start-1', 'end-1', 'start-2', 'end-2', 'release']);
  await session.dispose();
  assert.equal(events.at(-1), 'dispose');
});

test('cancelled active work completes cleanup before a queued request starts', async () => {
  let unblock!: () => void;
  let entered!: () => void;
  const gate = new Promise<void>((resolve) => {
    unblock = resolve;
  });
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  const events: string[] = [];
  let count = 0;
  const session = create({
    async run() {
      const number = ++count;
      events.push(`start-${number}`);
      if (number === 1) {
        entered();
        await gate;
      }
      events.push(`end-${number}`);
      return [0, 0];
    },
    releaseCachedMemory() {},
    dispose() {
      events.push('dispose');
    }
  });
  const controller = new AbortController();
  const first = session.systemOne({state: 'one', questions: {q: {instructions: 'yes?'}}}, {signal: controller.signal});
  await started;
  const second = session.systemOne({state: 'two', questions: {q: {instructions: 'yes?'}}});
  controller.abort(new Error('cancelled active'));
  await assert.rejects(first, /cancelled active/);
  assert.deepEqual(events, ['start-1']);
  unblock();
  await second;
  assert.deepEqual(events, ['start-1', 'end-1', 'start-2', 'end-2']);
  await session.dispose();
});

test('cancelling during the first of several questions prevents later execution', async () => {
  const controller = new AbortController();
  let calls = 0;
  const session = create({
    async run() {
      calls++;
      controller.abort(new Error('stop questions'));
      return [0, 0];
    },
    releaseCachedMemory() {},
    dispose() {}
  });
  await assert.rejects(
    session.systemOne(
      {
        state: '',
        questions: {
          one: {instructions: 'first?'},
          two: {instructions: 'second?'}
        }
      },
      {signal: controller.signal}
    ),
    /stop questions/
  );
  assert.equal(calls, 1);
  await session.dispose();
});

test('disposal waits for image preparation and active inference before freeing the runner', async () => {
  let unblock!: () => void;
  let entered!: () => void;
  const gate = new Promise<void>((resolve) => {
    unblock = resolve;
  });
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  const events: string[] = [];
  const session = create(
    {
      async run() {
        events.push('run');
        return [0, 0];
      },
      releaseCachedMemory() {},
      dispose() {
        events.push('dispose');
      }
    },
    async () => {
      events.push('image-start');
      entered();
      await gate;
      events.push('image-end');
      return {tokens: 2, value: 'image'};
    }
  );
  const active = session.systemOne({state: '', images: ['encoded'], questions: {q: {instructions: 'yes?'}}});
  await started;
  const disposal = session.dispose();
  assert.deepEqual(events, ['image-start']);
  unblock();
  await active;
  await disposal;
  assert.deepEqual(events, ['image-start', 'image-end', 'run', 'dispose']);
});

test('rejects invalid or already-aborted signals before image preparation or inference', async () => {
  let prepared = 0;
  let calls = 0;
  const session = create(
    {
      async run() {
        calls++;
        return [0, 0];
      },
      releaseCachedMemory() {},
      dispose() {}
    },
    async () => {
      prepared++;
      return {tokens: 1, value: 'image'};
    }
  );
  const request = {state: '', images: ['encoded'], questions: {q: {instructions: 'yes?'}}};
  await assert.rejects(session.systemOne(request, {signal: {aborted: false} as AbortSignal}), /invalid abort signal/i);
  const controller = new AbortController();
  const reason = new Error('already cancelled');
  controller.abort(reason);
  await assert.rejects(session.systemOne(request, {signal: controller.signal}), (error) => error === reason);
  assert.equal(prepared, 0);
  assert.equal(calls, 0);
  await session.dispose();
  await assert.rejects(session.releaseCachedMemory(), /disposed/i);
});

test('preserves special object-property names in question and choice maps', async () => {
  const session = create({
    async run() {
      return [10, -10];
    },
    releaseCachedMemory() {},
    dispose() {}
  });
  const criteria = new Map([
    ['__proto__', 'first'],
    ['constructor', 'second']
  ]);
  const questions = new Map([
    ['__proto__', {type: 'choice' as const, instructions: 'pick', criteria}],
    ['constructor', {type: 'choice' as const, instructions: 'pick', criteria}]
  ]);
  const result = await session.systemOne({questions});
  assert.deepEqual(Object.keys(result.answers), ['__proto__', 'constructor']);
  for (const answer of Object.values(result.answers)) {
    if (answer.type !== 'choice') throw new Error('Expected a choice');
    assert.equal(answer.choice, '__proto__');
    assert.deepEqual(Object.keys(answer.probabilities), ['__proto__', 'constructor']);
    assert.equal(Object.hasOwn(JSON.parse(JSON.stringify(answer.probabilities)), '__proto__'), true);
  }
  await session.dispose();
});

test('preserves marked Python float values in snapshotted state and question content', async () => {
  const inputs: Gemma4V2RunnerInput[] = [];
  const session = create({
    async run(input) {
      inputs.push(input);
      return [0, 0];
    },
    releaseCachedMemory() {},
    dispose() {}
  });
  const state = {
    threshold: kananaFloat(1),
    signed: kananaFloat(-0),
    nested: [kananaFloat(0), kananaFloat(1e20)],
    ordered: new Map([
      ['10', kananaFloat(4)],
      ['2', kananaFloat(1e-7)]
    ]),
    ordinary: {value: 5}
  };
  const question = {
    type: 'choice' as const,
    instructions: {ratio: kananaFloat(2)},
    criteria: {a: {limit: kananaFloat(3)}, b: null}
  };
  try {
    const pending = session.systemOne({state, questions: {q: question}});
    state.nested[0] = kananaFloat(9);
    await pending;
    const decode = (ids: readonly number[]) =>
      new TextDecoder().decode(
        Uint8Array.from(
          ids.filter((id) => id >= 1000),
          (id) => id - 1000
        )
      );
    const stateText = decode(inputs[0]!.stateIds);
    const questionText = decode(inputs[0]!.questionIds);
    assert.match(stateText, /"threshold": 1\.0/u);
    assert.match(stateText, /"signed": -0\.0/u);
    assert.match(stateText, /"nested": \[\s+0\.0,/u);
    assert.doesNotMatch(stateText, /9\.0/u);
    assert.match(stateText, /"10": 4\.0,[\s]+"2": 1e-07/u);
    assert.match(stateText, /"ordinary": \{[\s]+"value": 5/u);
    assert.match(stateText, /1e\+20/u);
    assert.match(questionText, /"ratio": 2\.0/u);
    assert.match(questionText, /"limit": 3\.0/u);
  } finally {
    await session.dispose();
  }
});

test('preserves exact Python-sized integers in structured content', async () => {
  const inputs: Gemma4V2RunnerInput[] = [];
  const session = create({
    async run(input) {
      inputs.push(input);
      return [0, 0];
    },
    releaseCachedMemory() {},
    dispose() {}
  });
  try {
    await session.systemOne({state: {amount: 9007199254740993n}, questions: {q: {instructions: 'ok'}}});
    const text = new TextDecoder().decode(
      Uint8Array.from(
        inputs[0]!.stateIds.filter((id) => id >= 1000),
        (id) => id - 1000
      )
    );
    assert.match(text, /"amount": 9007199254740993/u);
  } finally {
    await session.dispose();
  }
});

test('cyclic snapshots are rejected without poisoning a later request', async () => {
  let calls = 0;
  const session = create({
    async run() {
      calls++;
      return [0, 0];
    },
    releaseCachedMemory() {},
    dispose() {}
  });
  const state: Record<string, unknown> = {};
  state.self = state;
  try {
    await assert.rejects(session.systemOne({state, questions: {q: {instructions: 'ok'}}}), /cycle/i);
    assert.equal(calls, 0);
    await session.systemOne({state: {value: kananaFloat(1)}, questions: {q: {instructions: 'ok'}}});
    assert.equal(calls, 1);
  } finally {
    await session.dispose();
  }
});

// Pinned upstream Request.model_validate_json accepts these tokens despite standard JSON excluding them.
test('preserves non-finite numbers supported by the pinned source JSON parser', async () => {
  const inputs: Gemma4V2RunnerInput[] = [];
  const session = create({
    async run(input) {
      inputs.push(input);
      return [0, 0];
    },
    releaseCachedMemory() {},
    dispose() {}
  });
  const state = {numbers: [NaN, Infinity, -Infinity], nested: new Map([['value', -Infinity]])};
  const question = {type: 'choice' as const, instructions: {value: NaN}, criteria: {yes: {value: Infinity}, no: null}};
  try {
    const pending = session.systemOne({state, questions: {q: question}});
    state.numbers[0] = 1;
    state.nested.set('value', 1);
    await pending;
    const decode = (ids: readonly number[]) =>
      new TextDecoder().decode(
        Uint8Array.from(
          ids.filter((id) => id >= 1000),
          (id) => id - 1000
        )
      );
    const stateText = decode(inputs[0]!.stateIds);
    const questionText = decode(inputs[0]!.questionIds);
    assert.match(stateText, /"numbers": \[\s+NaN,\s+Infinity,\s+-Infinity\s+\]/u);
    assert.match(stateText, /"value": -Infinity/u);
    assert.match(questionText, /"value": NaN/u);
    assert.match(questionText, /"value": Infinity/u);
  } finally {
    await session.dispose();
  }
});
