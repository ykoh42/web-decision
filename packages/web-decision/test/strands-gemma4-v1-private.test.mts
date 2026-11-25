import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {Tokenizer} from '@huggingface/tokenizers';
import {
  createGemma4V1Session,
  resolveGemma4V1ServingOptions,
  STRANDS_GEMMA4_V1_E2B_MODEL
} from '../src/strands/gemma4-v1-session.mjs';
import {packPinnedStrandsGemma4V1Request} from '../src/strands/gemma4-protocol.mjs';
import {gemma4V1CausalMask} from '../src/strands/gemma4-v1-decoder.mjs';
import {checkGemma4V1Buffer} from '../src/strands/gemma4-v1-runtime.mjs';
import {
  createGemma4V1Source,
  perLayerEmbedding,
  wordEmbedding,
  type Gemma4V1Inventory
} from '../src/strands/gemma4-v1-source.mjs';

const tokenizer = {
  encode: (text: string, {add_special_tokens}: {add_special_tokens: boolean}) => ({
    ids: [...(add_special_tokens ? [2] : []), ...Array.from(text, (char) => char.codePointAt(0)!)]
  }),
  decode: (ids: number[]) => String.fromCodePoint(...ids)
};
const digest = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');

test('accepts the pinned choice, ordinal and exact context boundaries on the private path', () => {
  const choice = (count: number) => ({
    state: '',
    questions: {
      choose: {
        type: 'choice',
        instructions: '',
        criteria: new Map(Array.from({length: count}, (_, index) => [String(index), null]))
      }
    }
  });
  assert.equal(packPinnedStrandsGemma4V1Request(tokenizer, choice(255)).rows[0]!.optionIndices.length, 255);
  assert.throws(() => packPinnedStrandsGemma4V1Request(tokenizer, choice(256)), /255/);
  const score = (count: number) => ({
    state: '',
    questions: {s: {instructions: 'Rate it', criteria: Array(count).fill('level')}}
  });
  assert.equal(packPinnedStrandsGemma4V1Request(tokenizer, score(10)).rows[0]?.type, 'score');
  assert.throws(() => packPinnedStrandsGemma4V1Request(tokenizer, score(11)), /10 levels/);
  const request = (length: number) => ({state: 'x'.repeat(length), questions: {q: {instructions: 'Yes?'}}});
  const overhead = packPinnedStrandsGemma4V1Request(tokenizer, request(0)).tokenCount;
  assert.equal(packPinnedStrandsGemma4V1Request(tokenizer, request(4096 - overhead)).tokenCount, 4096);
  assert.throws(() => packPinnedStrandsGemma4V1Request(tokenizer, request(4097 - overhead)), /4096-token/);
  assert.throws(
    () => packPinnedStrandsGemma4V1Request(tokenizer, {...request(0), images: ['encoded']}),
    /text state only/
  );
  assert.throws(() => packPinnedStrandsGemma4V1Request(tokenizer, {...request(0), images: null}), /text state only/);
  assert.equal(packPinnedStrandsGemma4V1Request(tokenizer, {...request(0), images: []}).rows[0]?.type, 'noul');
});

test('keeps the checkpoint default and permits the official 32K evaluation profile', async () => {
  const calls: number[] = [];
  const runner = {
    async run(state: readonly number[], questions: readonly {ids: readonly number[]}[]) {
      calls.push(state.length + questions[0]!.ids.length);
      return [Float32Array.from([0, 1])];
    },
    releaseCachedMemory() {},
    dispose() {}
  };
  const request = (length: number) => ({state: 'x'.repeat(length), questions: {q: {instructions: 'Yes?'}}});
  const overhead = packPinnedStrandsGemma4V1Request(tokenizer, request(0)).tokenCount;
  const enabled = createGemma4V1Session(tokenizer, runner, resolveGemma4V1ServingOptions({maxLength: 32768}));
  const tokens = 32768 - overhead;
  try {
    assert.deepEqual((await enabled.systemOne(request(tokens))).usage, {input_tokens: 32768, output_tokens: 1});
    await assert.rejects(enabled.systemOne(request(tokens + 1)), /32768-token/);
    assert.deepEqual(calls, [32768]);
  } finally {
    await enabled.dispose();
  }
});

test('checks the projection and compact attention allocation against actual device limits', () => {
  const projection = 75_497_472;
  checkGemma4V1Buffer({maxBufferSize: projection, maxStorageBufferBindingSize: projection}, 4096);
  assert.throws(
    () => checkGemma4V1Buffer({maxBufferSize: projection - 1, maxStorageBufferBindingSize: projection}, 4096),
    {code: 'DEVICE_RESOURCE_LIMIT'}
  );
  checkGemma4V1Buffer({maxBufferSize: projection, maxStorageBufferBindingSize: projection}, 32768);
  const bytes = 268_435_456;
  checkGemma4V1Buffer({maxBufferSize: bytes, maxStorageBufferBindingSize: bytes}, 131072);
  assert.throws(() => checkGemma4V1Buffer({maxBufferSize: bytes + 1, maxStorageBufferBindingSize: bytes - 1}, 131072), {
    code: 'DEVICE_RESOURCE_LIMIT'
  });
  assert.throws(() => checkGemma4V1Buffer({maxBufferSize: bytes - 1, maxStorageBufferBindingSize: bytes + 1}, 131072), {
    code: 'DEVICE_RESOURCE_LIMIT'
  });
});

test('matches padded token budgets and preserves answers with prefix reuse disabled', async () => {
  const request = {
    state: 'shared state',
    questions: new Map(Array.from({length: 5}, (_, i) => [String(i), {instructions: 'y'.repeat(i + 1)}]))
  };
  const packed = packPinnedStrandsGemma4V1Request(tokenizer, request);
  const base = packed.rows[0]!.inputIds.length;
  for (const policy of [
    {maxBatch: 3, maxBatchTokens: 0, usePrefixCache: true, batches: [3, 2]},
    {maxBatch: 3, maxBatchTokens: 2 * (base + 1), usePrefixCache: true, batches: [2, 1, 1, 1]},
    {maxBatch: 3, maxBatchTokens: 1, usePrefixCache: false, batches: [1, 1, 1, 1, 1]},
    {maxBatch: 5, maxBatchTokens: 0, usePrefixCache: false, batches: [5]}
  ]) {
    const actual: number[] = [];
    const session = createGemma4V1Session(
      tokenizer,
      {
        async run(state, questions, _signal, usePrefixCache) {
          assert.equal(usePrefixCache, policy.usePrefixCache);
          assert.deepEqual(state, packed.stateIds);
          actual.push(questions.length);
          return questions.map(() => new Float32Array([-1, 1]));
        },
        releaseCachedMemory() {},
        dispose() {}
      },
      resolveGemma4V1ServingOptions(policy)
    );
    try {
      const result = await session.systemOne(request);
      assert.deepEqual(actual, policy.batches);
      assert.deepEqual(Object.keys(result.answers), ['0', '1', '2', '3', '4']);
      assert.equal(result.usage.output_tokens, 5);
      const prefixCount = policy.usePrefixCache ? actual.length : 5;
      assert.equal(
        result.usage.input_tokens,
        packed.tokenCount - (packed.rows.length - prefixCount) * packed.stateIds.length
      );
    } finally {
      await session.dispose();
    }
  }
  for (const options of [{maxBatch: 0}, {maxBatchTokens: -1}, {usePrefixCache: 0}]) {
    assert.throws(() => resolveGemma4V1ServingOptions(options as Parameters<typeof resolveGemma4V1ServingOptions>[0]), {
      code: 'INVALID_INPUT'
    });
  }
});

test('passes absolute inputs as independent suffixes and counts one shared state per upstream batch', async () => {
  const sizes: number[] = [];
  let released = 0;
  const session = createGemma4V1Session(tokenizer, {
    async run(state, questions, signal) {
      signal.throwIfAborted();
      sizes.push(questions.length);
      assert.equal(state[0], 2);
      for (const q of questions) {
        assert.equal(q.answerIndex, q.ids.length - 1);
        assert.ok(q.optionIndices.every((index) => index >= 0 && index < q.answerIndex));
      }
      return questions.map(() => Float32Array.from([1, 0]));
    },
    releaseCachedMemory() {
      released++;
    },
    dispose() {}
  });
  const request = {
    state: {customer: 'billed twice'},
    questions: new Map(
      Array.from({length: 33}, (_, index) => [
        String(index),
        {
          instructions: 'Pick a team',
          criteria: new Map([
            ['2', null],
            ['1', null]
          ])
        }
      ])
    )
  };
  const packed = packPinnedStrandsGemma4V1Request(tokenizer, request);
  const result = await session.systemOne(request);
  assert.deepEqual(sizes, [32, 1]);
  assert.equal(result.model, STRANDS_GEMMA4_V1_E2B_MODEL);
  assert.deepEqual(result.usage, {input_tokens: packed.tokenCount - packed.stateIds.length * 31, output_tokens: 33});
  assert.equal(result.answers['0']?.type, 'choice');
  if (result.answers['0']?.type === 'choice') assert.equal(result.answers['0'].choice, '2');
  await session.releaseCachedMemory();
  assert.equal(released, 1);
  await session.dispose();
  await assert.rejects(session.systemOne(request), {code: 'SESSION_CLOSED'});
});

test('caller cancellation leaves the phase serialized and the next request usable', async () => {
  let endFirst!: () => void;
  let startFirst!: () => void;
  const first = new Promise<void>((resolve) => {
    endFirst = resolve;
  });
  const began = new Promise<void>((resolve) => {
    startFirst = resolve;
  });
  const caller = new AbortController();
  const seen: AbortSignal[] = [];
  const session = createGemma4V1Session(tokenizer, {
    async run(_state, _questions, signal) {
      seen.push(signal);
      if (seen.length === 1) {
        startFirst();
        await first;
        signal.throwIfAborted();
      }
      return [Float32Array.from([0, 1])];
    },
    releaseCachedMemory() {},
    dispose() {}
  });
  const request = {state: '', questions: {q: {instructions: 'Yes?'}}};
  const active = session.systemOne(request, {signal: caller.signal});
  await began;
  const next = session.systemOne(request);
  const sentinel = new Error('caller stopped');
  caller.abort(sentinel);
  await assert.rejects(active, sentinel);
  assert.equal(seen.length, 1);
  assert.equal(seen[0]?.reason, sentinel);
  endFirst();
  assert.deepEqual((await next).answers.q, {type: 'noul', noul: 0.7179});
  await session.dispose();
});

test('constructs absolute local causal masks across a 512-token slide and a last tile', () => {
  const local = gemma4V1CausalMask(600, 2, 88, 515, 512);
  const global = gemma4V1CausalMask(600, 2, 88, 515, null);
  const row = (data: Float32Array, i: number, j: number) => data[i * 515 + j]!;
  assert.deepEqual(local.shape, [1, 1, 2, 515]);
  assert.ok(row(local.data, 0, 0) < -1e38);
  assert.equal(row(local.data, 0, 1), 0);
  assert.equal(row(local.data, 0, 512), 0);
  assert.ok(row(local.data, 0, 513) < -1e38);
  assert.ok(row(local.data, 1, 1) < -1e38);
  assert.equal(row(local.data, 1, 513), 0);
  assert.equal(row(global.data, 0, 0), 0);
  assert.ok(row(global.data, 0, 513) < -1e38);
});

test('loads pinned indexed rows and pads only complete BF16 tensor bytes', async () => {
  const base = Buffer.from(new Uint16Array([0, 1, 10, 11, 20, 21, 42]).buffer);
  const adapter = Buffer.from(new Float32Array([0, 1.5]).buffer);
  const inventory: Gemma4V1Inventory = {
    format: 'web-decision-gemma4-e2b-v1-original-ranges-v1',
    base: {
      repository: 'vendor/base',
      revision: 'pinned',
      file: 'base.safetensors',
      bytes: base.length,
      sha256: digest(base),
      headerBytes: 4,
      chunks: [
        [0, 4, digest(base.subarray(0, 4))],
        [4, 4, digest(base.subarray(4, 8))],
        [8, 4, digest(base.subarray(8, 12))],
        [12, 2, digest(base.subarray(12, 14))]
      ],
      tensors: {
        [wordEmbedding]: {offset: 4, length: 8, shape: [2, 2], dtype: 'BF16'},
        [perLayerEmbedding]: {offset: 4, length: 8, shape: [2, 2], dtype: 'BF16'},
        scalar: {offset: 12, length: 2, shape: [], dtype: 'BF16'}
      }
    },
    adapter: {
      repository: 'vendor/adapter',
      revision: 'pinned',
      file: 'adapter.safetensors',
      bytes: adapter.length,
      sha256: digest(adapter),
      headerBytes: 4,
      chunks: [[0, 8, digest(adapter)]],
      tensors: {w: {offset: 4, length: 4, shape: [1], dtype: 'F32'}}
    }
  };
  const requests: number[] = [];
  const source = createGemma4V1Source(inventory, [], {}, async (origin, range) => {
    const data = origin.href.includes('/base/') ? base : adapter;
    const bytes = data.subarray(range.offset, range.offset + range.length);
    assert.equal(digest(bytes), range.sha256);
    requests.push(range.offset);
    return {bytes: new Uint8Array(bytes).buffer, cachedByHash: false};
  });
  assert.deepEqual([...(await source.rows(wordEmbedding, [1, 0, 1]))], [20, 21, 10, 11, 20, 21]);
  assert.deepEqual(requests, [8, 4]);
  assert.deepEqual([...(await source.tensor({source: 'base', tensor: 'scalar', shape: [1], dtype: 'uint32'}))], [42]);
  assert.deepEqual([...(await source.tensor({source: 'adapter', tensor: 'w', shape: [1], dtype: 'float32'}))], [1.5]);
  await assert.rejects(source.rows(wordEmbedding, [2]), /vocabulary/);
  await assert.rejects(source.tensor({source: 'adapter', tensor: 'w', shape: [2], dtype: 'float32'}), /binding/);
  assert.equal(await source.isOfflineReady(), false);
  await assert.rejects(source.prepareOffline(), /cache/);

  const stored = new Map<string, Response>();
  const cache = {
    async match(url: string) {
      return stored.get(url)?.clone();
    },
    async put(url: string, response: Response) {
      stored.set(url, response.clone());
    },
    async delete(url: string) {
      return stored.delete(url);
    }
  };
  const adapterBinding = {source: 'adapter', tensor: 'w', shape: [1], dtype: 'float32'} as const;
  const online = createGemma4V1Source(inventory, [adapterBinding], {cache});
  const native = globalThis.fetch;
  let adapterFailure = true;
  globalThis.fetch = async (url, init) => {
    const path = String(url);
    const bytes = path.includes('/base/') ? base : adapter;
    const range = new Headers(init?.headers).get('range');
    const match = /^bytes=(\d+)-(\d+)$/u.exec(range ?? '');
    assert.ok(match);
    const [start, end] = match.slice(1).map(Number) as [number, number];
    const body = new Uint8Array(bytes.subarray(start, end + 1));
    if (adapterFailure && bytes === adapter) body[0] = body[0]! + 1;
    return new Response(body, {status: 206, headers: {'Content-Range': `bytes ${start}-${end}/${bytes.length}`}});
  };
  try {
    await assert.rejects(online.prepareOffline(), /integrity/i);
    assert.equal(await online.isOfflineReady(), false);
    adapterFailure = false;
    await online.prepareOffline();
    assert.equal(await online.isOfflineReady(), true);
    const offline = createGemma4V1Source(inventory, [adapterBinding], {cache, offline: true});
    globalThis.fetch = async () => assert.fail('Network during offline restart');
    assert.equal(await offline.isOfflineReady(), true);
    await offline.prepareOffline();
    assert.deepEqual([...(await offline.tensor(adapterBinding))], [1.5]);
    const adapterDigest = inventory.adapter.chunks[0]![2];
    for (const key of stored.keys()) if (key.endsWith(adapterDigest)) stored.delete(key);
    assert.equal(await offline.isOfflineReady(), false);
    await assert.rejects(offline.prepareOffline(), /not available offline/);
  } finally {
    globalThis.fetch = native;
  }
});

test(
  'matches the pinned upstream tokenizer and recorded pointer-answer fixtures',
  {
    skip: !process.env.E2B_V1_OFFICIAL_FIXTURE || !process.env.E2B_V1_CHECKPOINT
  },
  async () => {
    const checkpoint = process.env.E2B_V1_CHECKPOINT!;
    const fixture = JSON.parse(await readFile(process.env.E2B_V1_OFFICIAL_FIXTURE!, 'utf8'));
    const raw = await readFile(checkpoint + '/tokenizer.json');
    const config = await readFile(checkpoint + '/tokenizer_config.json');
    assert.equal(digest(raw), '30383f3ce90a169cbabe9eb3da7bb77988bf906aad2e59184b378f5470ec4dbd');
    assert.equal(digest(config), '3ab5c7b94dc97d65ca7064496fa69b88ff875378e1cb7ee3e43070c3a8170999');
    const pinned = new Tokenizer(JSON.parse(raw.toString()), JSON.parse(config.toString()));
    for (const sample of fixture.cases) {
      const packed = packPinnedStrandsGemma4V1Request(pinned, sample.request);
      assert.deepEqual(packed.stateIds, sample.stateIds);
      for (const row of packed.rows) {
        const expected = sample.questions[row.name];
        assert.deepEqual(row.inputIds.slice(packed.stateIds.length), expected.ids);
        assert.deepEqual(
          row.optionIndices.map((index) => index - packed.stateIds.length),
          expected.positions
        );
        assert.equal(row.type, expected.type);
      }
      const session = createGemma4V1Session(pinned, {
        async run() {
          return packed.rows.map((row) => Float32Array.from(sample.questions[row.name].logits));
        },
        dispose() {},
        releaseCachedMemory() {}
      });
      const result = await session.systemOne(sample.request);
      assert.deepEqual(result.usage, sample.usage);
      assert.deepEqual(
        result.answers,
        Object.fromEntries(packed.rows.map((row) => [row.name, sample.questions[row.name].answer]))
      );
      await session.dispose();
    }
  }
);
