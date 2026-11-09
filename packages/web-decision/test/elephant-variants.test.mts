import assert from 'node:assert/strict';
import test from 'node:test';
import {ELEPHANT_MODEL, ELEPHANT_512_MODEL, ELEPHANT_LIMITS} from '../src/elephant/contract.mjs';
import {elephantFiles, openElephantSource} from '../src/elephant/source.mjs';
import {loadFakeElephantCandidate, validateElephantOptions} from '../src/elephant/loader.mjs';
import {createElephantSession} from '../src/elephant/session.mjs';
import {ElephantTvmRunner} from '../src/elephant/runner.mjs';
import {packElephantRequest} from '../src/elephant/protocol.mjs';
import type {ElephantPackedRequest, ElephantSlotTokenizer} from '../src/elephant/protocol.mjs';
import {graphInputs} from '../src/elephant/graph-inputs.mjs';
import {describeModel} from '../src/model-registry.js';

function deferred<Value = void>() {
  let settle!: (value: Value | PromiseLike<Value>) => void;
  const promise = new Promise<Value>((done) => {
    settle = done;
  });
  return {promise, resolve: (value?: Value) => settle(value as Value)};
}

test('each variant selects its own graph and reports its own branch limit', () => {
  const short = elephantFiles(ELEPHANT_MODEL),
    long = elephantFiles(ELEPHANT_512_MODEL);
  assert.deepEqual(short['model.wasm'], {
    bytes: 5093840,
    sha256: 'f7a8bb9cc2e5fe70c07edb05e678a854ebf83e9c78dc8bb1559185472c643181'
  });
  assert.deepEqual(long['model.wasm'], {
    bytes: 5094092,
    sha256: 'a7d9904be86cb1ab54a7778fe0c3315c90761e0f4760df1a04de83c88c1189c5'
  });
  assert.deepEqual(short['inventory.json'], long['inventory.json']);
  assert.deepEqual(short['tokenizer.json'], long['tokenizer.json']);
  const variants: Array<[typeof ELEPHANT_MODEL | typeof ELEPHANT_512_MODEL, number, ReturnType<typeof elephantFiles>]> =
    [
      [ELEPHANT_MODEL, 128, short],
      [ELEPHANT_512_MODEL, 512, long]
    ];
  for (const [model, max, graph] of variants) {
    const info = describeModel(model);
    assert.equal(info.checkpoint.graphSha256, graph['model.wasm']!.sha256);
    assert.equal(
      info.features.textChoice.limits.find((item) => item.resource === 'tokens' && item.scope === 'question')!.max,
      max
    );
    assert.equal(
      info.features.textChoice.limits.find((item) => item.resource === 'tokens' && item.scope === 'request'),
      undefined
    );
    assert.equal(validateElephantOptions({model, assetsUrl: 'https://example.test/'}).model, model);
  }
  assert.throws(() => elephantFiles('unknown'), {code: 'INVALID_ASSET'});
});

test('the loader passes the same validated limit to source, runner and session', async () => {
  type FakeDeps = Parameters<typeof loadFakeElephantCandidate>[1];
  for (const [model, limit] of [
    [ELEPHANT_MODEL, 128],
    [ELEPHANT_512_MODEL, 512]
  ] as const) {
    const seen: Array<[string, string | number | undefined]> = [];
    const owner = {
      lifecycle: {},
      uncaptured: [],
      disposalGuard: {blocked: false}
    } as unknown as ReturnType<FakeDeps['initialize']>;
    const session = 'session' as unknown as ReturnType<FakeDeps['session']>;
    const result = await loadFakeElephantCandidate(
      {model, assetsUrl: 'https://example.test/'},
      {
        fake: true,
        gpu: {
          async requestAdapter() {
            return {
              features: new Set(['shader-f16']),
              limits: ELEPHANT_LIMITS,
              async requestDevice() {
                return {lost: new Promise(() => {})};
              }
            };
          }
        },
        async openSource(_base, options) {
          seen.push(['source', options!.model]);
          return {wasm: {}, tokenizer: {}, tokenizerConfig: {}} as unknown as Awaited<
            ReturnType<FakeDeps['openSource']>
          >;
        },
        async prepareTokenizer() {
          return {tokenizer: {}, addedTokens: []} as unknown as Awaited<ReturnType<FakeDeps['prepareTokenizer']>>;
        },
        instantiate: async () => ({}) as Parameters<FakeDeps['initialize']>[0],
        initialize: () => owner,
        upload: async () => [],
        runner(args) {
          seen.push(['runner', args.maxBranchTokens]);
          return {} as ReturnType<FakeDeps['runner']>;
        },
        session(args) {
          seen.push(['session', args.maxBranchTokens]);
          return session;
        }
      }
    );
    assert.equal(result, session);
    assert.deepEqual(seen, [
      ['source', model],
      ['runner', limit],
      ['session', limit]
    ]);
  }
});

test('crossing the two graph assets fails closed before metadata finishes loading', async (t) => {
  const wrong128 = new Uint8Array(5094092),
    wrong512 = new Uint8Array(5093840);
  let graph!: Uint8Array;
  t.mock.method(globalThis, 'fetch', async (url: RequestInfo | URL) => {
    if (new URL(String(url)).pathname.endsWith('/model.wasm')) return new Response(graph as Uint8Array<ArrayBuffer>);
    return new Promise(() => {});
  });
  const wrongVariants: Array<[typeof ELEPHANT_MODEL | typeof ELEPHANT_512_MODEL, Uint8Array]> = [
    [ELEPHANT_MODEL, wrong128],
    [ELEPHANT_512_MODEL, wrong512]
  ];
  for (const [model, wrong] of wrongVariants) {
    graph = wrong;
    await assert.rejects(openElephantSource(new URL('https://example.test/'), {model}));
  }
  const sizedVariants: Array<[typeof ELEPHANT_MODEL | typeof ELEPHANT_512_MODEL, number]> = [
    [ELEPHANT_MODEL, 5093840],
    [ELEPHANT_512_MODEL, 5094092]
  ];
  for (const [model, size] of sizedVariants) {
    graph = new Uint8Array(size);
    await assert.rejects(openElephantSource(new URL('https://example.test/'), {model}), {code: 'INVALID_ASSET'});
  }
});

const tokenizer: ElephantSlotTokenizer = {
  encode: (text) => ({ids: [...text].filter((char) => '<Qx'.includes(char)).map((char) => char.charCodeAt(0))}),
  decode: (ids) => String.fromCharCode(...ids)
};
const request = (): {state: string; questions: Array<{instructions: string; options: string[]}>} => ({
  state: '',
  questions: [{instructions: 'choose', options: ['a', 'b']}]
});
function withLength(length: number): ReturnType<typeof request> {
  const value = request();
  const base = packElephantRequest(tokenizer, value, {addedTokens: [], maxBranchTokens: 512}).branches[0]!.ids.length;
  value.state = 'x'.repeat(length - base);
  return value;
}

test('variant sessions admit their exact boundary and reject the next token before inference', async () => {
  const limits: Array<[number, number]> = [
    [128, 129],
    [512, 513]
  ];
  for (const [limit, next] of limits) {
    let calls = 0;
    const session = createElephantSession({
      tokenizer,
      addedTokens: [],
      maxBranchTokens: limit,
      runner: {
        async run(packed: ElephantPackedRequest) {
          calls++;
          return packed.branches.map(() => new Float32Array([0, 1]));
        },
        async releaseCachedMemory() {},
        async dispose() {}
      }
    });
    assert.equal((await session.decide(withLength(limit))).answers[0]!.selected, 'b');
    await assert.rejects(session.decide(withLength(next)), {code: 'INVALID_INPUT'});
    assert.equal(calls, 1);
    assert.equal((await session.decide(withLength(limit))).answers[0]!.selected, 'b');
    await session.dispose();
  }
});

test('graph construction enforces the limit, block attention and final-token index', () => {
  const ids = Array.from({length: 512}, (_, i) => i);
  assert.throws(() => graphInputs(ids, 100));
  const inputs = graphInputs(ids, 100, 512);
  assert.deepEqual(inputs[2]!.shape, [1, 1, 512, 512]);
  assert.equal(inputs[2]!.data[99 * 512 + 100], 0xfbff);
  assert.equal(inputs[2]!.data[100 * 512 + 99], 0);
  assert.equal(inputs[2]!.data[100 * 512 + 511], 0);
  assert.equal(inputs[4]!.data[0], 511);
  assert.throws(() => graphInputs([...ids, 512], 100, 512));
});

function fakeRunner() {
  type DeferredSignal = ReturnType<typeof deferred<void>>;
  interface FakeTensor {
    readonly shape: number[];
    readonly dtype: string;
    readonly byteOffset: number;
    value?: number[];
    getDataPtr(): number;
    copyFrom(data: unknown): FakeTensor;
    toArray(): Float32Array;
  }
  const events: string[] = [],
    copies: Array<number[] | undefined> = [],
    gates: Array<{started: DeferredSignal; finish: DeferredSignal}> = [];
  const context: {pendingGPUToCPUCopy: Promise<void> | null; copyRawBytesToBuffer(): void} = {
    pendingGPUToCPUCopy: null,
    copyRawBytesToBuffer() {}
  };
  const device = {
    async sync() {
      const gate = gates.shift();
      if (gate) {
        gate.started.resolve();
        await gate.finish.promise;
      }
      const pending = context.pendingGPUToCPUCopy;
      if (pending) await pending;
      context.pendingGPUToCPUCopy = null;
    }
  };
  const runtime = {
    lib: {webGPUContext: context},
    cpu: () => 'cpu',
    beginScope() {},
    endScope() {},
    dispose() {},
    empty(shape: number[], dtype: string, where: unknown) {
      const tensor: FakeTensor = {
        shape,
        dtype,
        byteOffset: 0,
        getDataPtr: () => 0,
        copyFrom(data: unknown) {
          if (where === 'cpu') context.pendingGPUToCPUCopy = Promise.resolve();
          else if (dtype === 'int32') this.value = Array.from(data as ArrayLike<number>);
          return this;
        },
        toArray: () => new Float32Array(256)
      };
      return tensor;
    }
  };
  const runner = new ElephantTvmRunner({
    runtime,
    device,
    maxBranchTokens: 512,
    gpu: {
      pushErrorScope() {},
      async popErrorScope() {
        return null;
      }
    },
    fn(...args: unknown[]) {
      events.push('graph');
      copies.push((args[0] as {value?: number[]}).value);
      return {dtype: 'float32', shape: [1, 256]} as unknown as ReturnType<
        ConstructorParameters<typeof ElephantTvmRunner>[0]['fn']
      >;
    },
    weights: Array.from({length: 684}, () => ({})),
    lifecycle: {},
    disposalGuard: {},
    uncaptured: [],
    teardownObserver: {async beforeDispose() {}, async afterDispose() {}},
    teardownTimeoutMs: 100
  });
  return {runner, gates, events, copies};
}

test('all branches validate before GPU work; masks are allocated one at a time and IDs are snapshotted', async () => {
  const f = fakeRunner();
  const bad = {
    prefixIds: [1],
    branches: [
      {ids: Array(512).fill(1), options: ['a', 'b']},
      {ids: Array(513).fill(1), options: ['a', 'b']}
    ]
  } as unknown as ElephantPackedRequest;
  await assert.rejects(f.runner.run(bad), {code: 'INFERENCE_ERROR'});
  assert.deepEqual(f.events, []);
  const original = globalThis.Uint16Array;
  let masks = 0;
  const InstrumentedUint16Array = class extends original {};
  globalThis.Uint16Array = new Proxy(InstrumentedUint16Array, {
    construct(_target, args: unknown[], newTarget) {
      if (args[0] === 512 * 512) masks++;
      return Reflect.construct(original, args, newTarget);
    }
  }) as Uint16ArrayConstructor;
  try {
    const first = Array(512).fill(1),
      second = Array(512).fill(2);
    const gate = {started: deferred(), finish: deferred()};
    f.gates.push(gate);
    const result = f.runner.run({
      prefixIds: [1],
      branches: [
        {ids: first, options: ['a', 'b']},
        {ids: second, options: ['c', 'd']}
      ]
    } as unknown as ElephantPackedRequest);
    await gate.started.promise;
    assert.equal(masks, 1);
    second[0] = 99;
    gate.finish.resolve();
    assert.equal((await result).length, 2);
    assert.equal(masks, 2);
    assert.equal(f.copies[1]![0], 2);
    const cancellation = {started: deferred(), finish: deferred()};
    f.gates.push(cancellation);
    const controller = new AbortController();
    const cancelled = f.runner.run(
      {
        prefixIds: [1],
        branches: [
          {ids: first, options: ['a', 'b']},
          {ids: second, options: ['c', 'd']}
        ]
      } as unknown as ElephantPackedRequest,
      {signal: controller.signal}
    );
    const rejection = assert.rejects(cancelled, {name: 'AbortError'});
    await cancellation.started.promise;
    controller.abort();
    assert.equal(masks, 3);
    cancellation.finish.resolve();
    await rejection;
    assert.equal(masks, 3);
    assert.equal(f.events.length, 3);
  } finally {
    globalThis.Uint16Array = original;
  }
  await f.runner.dispose();
});

test('a later oversized branch is rejected before any inference', async () => {
  let calls = 0;
  const session = createElephantSession({
    tokenizer,
    addedTokens: [],
    maxBranchTokens: 512,
    runner: {
      async run(_packed: ElephantPackedRequest) {
        calls++;
        return [];
      },
      async releaseCachedMemory() {},
      async dispose() {}
    }
  });
  const value = request();
  value.questions.push({instructions: 'x'.repeat(600), options: ['c', 'd']});
  await assert.rejects(session.decide(value), {code: 'INVALID_INPUT'});
  assert.equal(calls, 0);
  await session.dispose();
});

test('512-token session cancellation skips later branches and disposal waits for the in-flight readback', async () => {
  const f = fakeRunner();
  const gate = {started: deferred(), finish: deferred()};
  f.gates.push(gate);
  const session = createElephantSession({tokenizer, addedTokens: [], runner: f.runner, maxBranchTokens: 512});
  const value = withLength(512);
  value.questions.push({...value.questions[0]!, options: ['c', 'd']});
  const controller = new AbortController();
  const decision = session.decide(value, {signal: controller.signal});
  await gate.started.promise;
  controller.abort();
  await assert.rejects(decision, {name: 'AbortError'});
  const disposal = session.dispose();
  await Promise.resolve();
  assert.deepEqual(f.events, ['graph']);
  gate.finish.resolve();
  await disposal;
  assert.deepEqual(f.events, ['graph']);
});
