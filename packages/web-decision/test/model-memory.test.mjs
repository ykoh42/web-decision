import test from 'node:test';
import assert from 'node:assert/strict';
import {DecisionModel} from '../src/model.js';
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((a, b) => {
    resolve = a;
    reject = b;
  });
  return {promise, resolve, reject};
};
const request = {state: 'x', questions: [{instructions: 'choose', options: ['a', 'b']}]};
function makeModel({sync = async () => {}, clear = () => {}, pop = async () => null} = {}) {
  const events = [],
    lifecycle = {loss: null, disposed: false};
  const runtime = {
    beginScope() {
      events.push('begin');
    },
    endScope() {
      events.push('end');
    },
    getGlobalFunc(name) {
      events.push(name);
      return () => {
        events.push('clear');
        clear();
      };
    },
    dispose() {
      events.push('dispose');
      lifecycle.disposed = true;
    }
  };
  const device = {
    async sync() {
      events.push('sync');
      await sync();
    }
  };
  const gpu = {
    pushErrorScope() {
      events.push('push');
    },
    async popErrorScope() {
      events.push('pop');
      return pop();
    }
  };
  const tokenizer = {
    encode(text) {
      return {ids: Array.from(text, (c) => c.codePointAt(0) + 200)};
    }
  };
  const model = new DecisionModel(
    runtime,
    device,
    gpu,
    null,
    [],
    tokenizer,
    {maxStateTokens: 384, maxInputTokens: 1024},
    [],
    lifecycle
  );
  return {model, events, lifecycle};
}
test('memory release is ordered between decisions and disposal waits for it', async () => {
  const waiting = deferred(),
    sync = deferred(),
    started = deferred(),
    {model, events} = makeModel({
      sync: () => {
        started.resolve();
        return sync.promise;
      }
    });
  let runs = 0;
  model.run = async () => {
    events.push(`run${++runs}`);
    if (runs === 1) await waiting.promise;
    return runs;
  };
  const first = model.decide(request),
    release = model.releaseCachedMemory(),
    second = model.decide(request);
  await Promise.resolve();
  assert.deepEqual(events, ['run1']);
  waiting.resolve();
  assert.equal(await first, 1);
  await started.promise;
  assert.deepEqual(events, ['run1', 'begin', 'push', 'vm.builtin.memory_manager.clear', 'clear', 'sync']);
  const close = model.dispose();
  await assert.rejects(model.releaseCachedMemory(), (e) => e.code === 'SESSION_CLOSED');
  assert.ok(!events.includes('run2'));
  assert.ok(!events.includes('dispose'));
  sync.resolve();
  await release;
  assert.equal(await second, 2);
  await close;
  assert.deepEqual(events, [
    'run1',
    'begin',
    'push',
    'vm.builtin.memory_manager.clear',
    'clear',
    'sync',
    'pop',
    'end',
    'run2',
    'end',
    'dispose'
  ]);
});
test('failed memory release preserves its cause and does not poison queued decisions', async () => {
  const problem = new Error('clear failed'),
    {model, events} = makeModel({
      clear() {
        throw problem;
      }
    });
  model.run = async () => 42;
  const release = model.releaseCachedMemory(),
    next = model.decide(request);
  await assert.rejects(release, (e) => e.code === 'INFERENCE_ERROR' && e.cause === problem);
  assert.equal(await next, 42);
  assert.deepEqual(events, ['begin', 'push', 'vm.builtin.memory_manager.clear', 'clear', 'pop', 'end']);
  await model.dispose();
});
test('memory release reports device loss and validation failures', async () => {
  const {model, lifecycle, events} = makeModel({pop: async () => new Error('invalid GPU operation')});
  await assert.rejects(
    model.releaseCachedMemory(),
    (e) => e.code === 'INFERENCE_ERROR' && e.cause.message === 'invalid GPU operation'
  );
  assert.ok(events.includes('end'));
  lifecycle.loss = {reason: 'unknown'};
  await assert.rejects(model.releaseCachedMemory(), (e) => e.code === 'INFERENCE_ERROR' && /lost/.test(e.message));
  await model.dispose();
});
test('device loss while memory release is synchronizing rejects without poisoning disposal', async () => {
  const wait = deferred(),
    started = deferred(),
    {model, lifecycle, events} = makeModel({
      sync: () => {
        started.resolve();
        return wait.promise;
      }
    });
  const release = model.releaseCachedMemory();
  await started.promise;
  lifecycle.loss = {reason: 'unknown'};
  wait.resolve();
  await assert.rejects(release, (e) => e.code === 'INFERENCE_ERROR' && /lost/.test(e.cause?.message));
  assert.deepEqual(events, ['begin', 'push', 'vm.builtin.memory_manager.clear', 'clear', 'sync', 'pop']);
  await model.dispose();
  assert.deepEqual(events, ['begin', 'push', 'vm.builtin.memory_manager.clear', 'clear', 'sync', 'pop', 'dispose']);
});
