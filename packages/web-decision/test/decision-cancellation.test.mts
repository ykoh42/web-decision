import test from 'node:test';
import assert from 'node:assert/strict';
import {DecisionModel} from '../src/model.js';
import {ExperimentalDebertaSession} from '../src/deberta/model.mjs';
import {ExperimentalMapikaSession} from '../src/mapika/session.js';
import {ExperimentalKev08Session} from '../src/kev08/session.js';
type Family = 'kev' | 'deberta' | 'mapika' | 'kev08';
type TestModel = {
  run: () => Promise<unknown>;
  decide(request: unknown, options?: unknown): Promise<unknown>;
  dispose(): Promise<void>;
};
const deferred = () => {
  let resolve!: () => void, reject!: (reason?: unknown) => void;
  const promise = new Promise<void>((a, b) => {
    resolve = a;
    reject = b;
  });
  return {promise, resolve, reject};
};
const tokenizer = {
  encode(text: string) {
    return {ids: text === '[STATE]' ? [128001] : Array.from(text, (c: string) => c.codePointAt(0)! + 200)};
  }
};
const request = () => ({state: 'state', questions: [{instructions: 'choose', options: ['a', 'b']}]});
function create(family: Family): {model: TestModel; events: string[]} {
  const lifecycle = {disposed: false, loss: null},
    events: string[] = [];
  const runtime = {
    endScope() {
      events.push('end');
    },
    dispose() {
      events.push('dispose');
      lifecycle.disposed = true;
    }
  };
  const mapika = () =>
    new ExperimentalMapikaSession({
      runtime,
      device: {},
      gpu: {},
      fn: null,
      weights: [],
      tokenizer,
      labels: {
        names: Array.from({length: 255}, (_, i) => String(i)),
        ids: Array.from({length: 255}, (_, i) => i),
        open: [3]
      },
      limits: {maxStateTokens: 384, maxInputTokens: 1024, controlIds: [999999], temperature: 1.03},
      uncaptured: [],
      lifecycle
    } as unknown as ConstructorParameters<typeof ExperimentalMapikaSession>[0]) as unknown as TestModel;
  let model;
  if (family === 'kev')
    model = new DecisionModel(
      runtime as unknown as ConstructorParameters<typeof DecisionModel>[0],
      {} as ConstructorParameters<typeof DecisionModel>[1],
      {} as ConstructorParameters<typeof DecisionModel>[2],
      null as unknown as ConstructorParameters<typeof DecisionModel>[3],
      [] as ConstructorParameters<typeof DecisionModel>[4],
      tokenizer as unknown as ConstructorParameters<typeof DecisionModel>[5],
      {maxStateTokens: 384, maxInputTokens: 1024} as ConstructorParameters<typeof DecisionModel>[6],
      [] as ConstructorParameters<typeof DecisionModel>[7],
      lifecycle as ConstructorParameters<typeof DecisionModel>[8]
    ) as unknown as TestModel;
  else if (family === 'mapika') model = mapika();
  else if (family === 'kev08')
    model = new ExperimentalKev08Session({
      runtime,
      device: {},
      gpu: {},
      fn: null,
      weights: [],
      tokenizer,
      limits: {maxStateTokens: 384, maxInputTokens: 1024},
      uncaptured: [],
      lifecycle
    } as unknown as ConstructorParameters<typeof ExperimentalKev08Session>[0]) as unknown as TestModel;
  else
    model = new ExperimentalDebertaSession({
      runtime,
      device: {},
      gpu: {},
      encoder: null,
      head: null,
      encoderWeights: [],
      headWeights: [],
      tokenizer,
      uncaptured: [],
      lifecycle
    } as unknown as ConstructorParameters<typeof ExperimentalDebertaSession>[0]) as unknown as TestModel;
  return {model, events};
}
for (const family of ['kev', 'deberta', 'mapika', 'kev08'] as const) {
  test(`${family}: already aborted request is not tokenized or enqueued`, async () => {
    const {model, events} = create(family),
      controller = new AbortController(),
      reason = Error('already cancelled');
    controller.abort(reason);
    let runs = 0;
    model.run = async () => {
      runs++;
    };
    await assert.rejects(model.decide(null, {signal: controller.signal}), (e: unknown) => e === reason);
    assert.equal(runs, 0);
    await model.dispose();
    assert.deepEqual(events, ['end', 'dispose']);
  });
  test(`${family}: queued cancellation skips execution and leaves later work intact`, async () => {
    const {model, events} = create(family),
      gate = deferred(),
      started = deferred(),
      controller = new AbortController(),
      reason = Error('cancel queued'),
      runs: number[] = [];
    model.run = async () => {
      runs.push(runs.length + 1);
      if (runs.length === 1) {
        started.resolve();
        await gate.promise;
      }
      return runs.length;
    };
    const first = model.decide(request());
    await started.promise;
    const cancelled = model.decide(request(), {signal: controller.signal}),
      next = model.decide(request());
    controller.abort(reason);
    await assert.rejects(cancelled, (e: unknown) => e === reason);
    assert.deepEqual(runs, [1]);
    const closing = model.dispose();
    gate.resolve();
    assert.equal(await first, 1);
    assert.equal(await next, 2);
    await closing;
    assert.deepEqual(runs, [1, 2]);
    assert.deepEqual(events, ['end', 'dispose']);
  });
  test(`${family}: active cancellation returns early while queue and dispose wait for GPU work`, async () => {
    const {model, events} = create(family),
      gate = deferred(),
      started = deferred(),
      controller = new AbortController(),
      reason = Error('cancel active'),
      runs: number[] = [];
    model.run = async () => {
      const id = runs.length + 1;
      runs.push(id);
      if (id === 1) {
        started.resolve();
        await gate.promise;
        throw Error('late GPU failure');
      }
      return id;
    };
    const active = model.decide(request(), {signal: controller.signal});
    await started.promise;
    const next = model.decide(request());
    controller.abort(reason);
    await assert.rejects(active, (e: unknown) => e === reason);
    const closing = model.dispose();
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(runs, [1]);
    assert.deepEqual(events, []);
    gate.resolve();
    assert.equal(await next, 2);
    await closing;
    assert.deepEqual(events, ['end', 'dispose']);
  });
}
for (const family of ['kev', 'deberta', 'mapika', 'kev08'] as const) {
  test(`${family}: decision removes its abort listener after completion or cancellation`, async () => {
    for (const mode of ['success', 'cancel']) {
      const {model} = create(family),
        controller = new AbortController(),
        signal = controller.signal,
        gate = deferred(),
        started = deferred();
      const add = signal.addEventListener.bind(signal),
        remove = signal.removeEventListener.bind(signal);
      let listeners = 0;
      signal.addEventListener = (...args: Parameters<AbortSignal['addEventListener']>) => {
        listeners++;
        return add(...args);
      };
      signal.removeEventListener = (...args: Parameters<AbortSignal['removeEventListener']>) => {
        listeners--;
        return remove(...args);
      };
      model.run = async () => {
        started.resolve();
        await gate.promise;
        return 'ok';
      };
      const promise = model.decide(request(), {signal});
      await started.promise;
      if (mode === 'cancel') {
        const reason = Error('cancel');
        controller.abort(reason);
        await assert.rejects(promise, (e) => e === reason);
        gate.resolve();
      } else {
        gate.resolve();
        assert.equal(await promise, 'ok');
      }
      assert.equal(listeners, 0);
      await model.dispose();
    }
  });
}
