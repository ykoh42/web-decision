import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {setImmediate as nextTurn} from 'node:timers/promises';
import test from 'node:test';
import {fileURLToPath} from 'node:url';
import {Tokenizer} from './strands-fake-tokenizer.mjs';
import {ExperimentalFusedRunner} from '../src/strands/runner.mjs';
import {ExperimentalStrandsSession} from '../src/strands/session.mjs';

function deferred() {
  let resolve, reject;
  const promise = new Promise((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return {promise, resolve, reject};
}

// Models the pinned TVM behavior: copyFrom stores a promise which performs
// the WASM write, and sync removes that reference before it awaits.
function fixture({
  firstFails = true,
  fallbackFails = true,
  missingReadback = false,
  replaceReadback = false,
  copyThrows = false,
  preexisting = false,
  graphReadback = false
} = {}) {
  let scope,
    graphRuns = 0,
    syncs = 0,
    writes = 0,
    disposals = 0;
  const firstReadback = deferred(),
    firstSync = deferred(),
    scopes = [];
  const context = {pendingGPUToCPUCopy: null, copyRawBytesToBuffer() {}};
  if (preexisting) context.pendingGPUToCPUCopy = Promise.resolve();
  const runtime = {
    lib: {webGPUContext: context},
    beginScope() {
      scope = {closed: false};
      scopes.push(scope);
    },
    endScope() {
      scope.closed = true;
    },
    cpu() {
      return 'cpu';
    },
    empty(_shape, _dtype, target) {
      const owner = scope;
      return {
        byteOffset: 0,
        getDataPtr() {
          return 1;
        },
        copyFrom(value) {
          this.value = ArrayBuffer.isView(value) ? Array.from(value) : value.value;
          if (target === 'cpu' && !missingReadback) {
            const source = graphRuns === 1 ? firstReadback.promise : Promise.resolve();
            context.pendingGPUToCPUCopy = source.then(() => {
              assert.equal(owner.closed, false, 'readback wrote after scope close');
              writes++;
            });
            if (copyThrows) throw Error('copy failed after submission');
          }
          return this;
        },
        toArray() {
          return this.value;
        }
      };
    },
    dispose() {
      disposals++;
    }
  };
  const device = {
    async sync() {
      syncs++;
      const pending = context.pendingGPUToCPUCopy;
      context.pendingGPUToCPUCopy = null;
      if (syncs === 1) {
        firstSync.resolve();
        if (firstFails) {
          if (replaceReadback) context.pendingGPUToCPUCopy = Promise.resolve();
          throw Error('initial sync failure');
        }
      }
      if (syncs === 2 && fallbackFails && firstFails) throw Error('fallback sync failure');
      if (pending) await pending;
    }
  };
  const runner = new ExperimentalFusedRunner({
    runtime,
    device,
    gpu: {
      pushErrorScope() {},
      async popErrorScope() {
        return null;
      }
    },
    fn() {
      graphRuns++;
      if (graphReadback) context.pendingGPUToCPUCopy = Promise.resolve();
      return {value: [0, 1]};
    },
    weights: Array(328).fill(0),
    lifecycle: {loss: null, disposed: false},
    uncaptured: []
  });
  return {
    session: new ExperimentalStrandsSession({tokenizer: new Tokenizer(), runner}),
    runner,
    firstReadback,
    firstSync,
    scopes,
    get graphRuns() {
      return graphRuns;
    },
    get syncs() {
      return syncs;
    },
    get writes() {
      return writes;
    },
    get disposals() {
      return disposals;
    }
  };
}

const request = {state: 's', questions: [{instructions: 'Pick', options: ['red', 'blue']}]};

test('failed sync and fallback retain the scope and poison queued work after readback settles', async () => {
  const f = fixture(),
    controller = new AbortController(),
    reason = new Error('cancel');
  const first = f.session.decide(request, {signal: controller.signal});
  await f.firstSync.promise;
  const queued = f.session.decide(request);
  const releasing = f.session.releaseCachedMemory();
  const disposing = f.session.dispose();
  controller.abort(reason);
  await assert.rejects(first, (error) => error === reason);
  await nextTurn();
  assert.equal(f.scopes[0].closed, false);
  assert.equal(f.graphRuns, 1);
  assert.equal(f.syncs, 1);
  f.firstReadback.resolve();
  await assert.rejects(queued, {code: 'INFERENCE_ERROR'});
  await assert.rejects(releasing, {code: 'INFERENCE_ERROR'});
  await assert.rejects(disposing, {code: 'INFERENCE_ERROR'});
  assert.equal(f.writes, 1);
  assert.equal(f.syncs, 2);
  assert.equal(f.scopes[0].closed, false);
  assert.equal(f.graphRuns, 1);
  assert.equal(f.disposals, 0);
});

test('a successful fallback cannot close the scope until the first readback settles', async () => {
  const f = fixture({fallbackFails: false});
  const first = f.session.decide(request);
  await f.firstSync.promise;
  const queued = f.session.decide(request);
  await nextTurn();
  assert.equal(f.scopes[0].closed, false);
  assert.equal(f.graphRuns, 1);
  f.firstReadback.resolve();
  await assert.rejects(first, {code: 'INFERENCE_ERROR'});
  assert.equal((await queued).answers[0].selected, 'blue');
  assert.equal(f.scopes[0].closed, true);
  assert.equal(f.graphRuns, 2);
  await f.session.dispose();
});

test('a rejected captured readback is observed before a successful fallback', async () => {
  const f = fixture({fallbackFails: false});
  const first = f.session.decide(request);
  await f.firstSync.promise;
  f.firstReadback.reject(Error('readback rejected'));
  await assert.rejects(first, {code: 'INFERENCE_ERROR'});
  assert.equal(f.writes, 0);
  assert.equal(f.syncs, 2);
  assert.equal(f.scopes[0].closed, true);
  await f.session.dispose();
});

test('successful inference uses one sync and closes its scope', async () => {
  const f = fixture({firstFails: false});
  const result = f.session.decide(request);
  await f.firstSync.promise;
  assert.equal(f.scopes[0].closed, false);
  f.firstReadback.resolve();
  assert.equal((await result).answers[0].selected, 'blue');
  assert.equal(f.syncs, 1);
  assert.equal(f.scopes[0].closed, true);
  await f.session.dispose();
});

test('missing readback promise retains an active scope and rejects reuse and disposal', async () => {
  const f = fixture({missingReadback: true});
  await assert.rejects(f.session.decide(request), {code: 'INFERENCE_ERROR'});
  assert.equal(f.scopes[0].closed, false);
  await assert.rejects(f.session.decide(request), {code: 'INFERENCE_ERROR'});
  await assert.rejects(f.session.dispose(), {code: 'INFERENCE_ERROR'});
  assert.equal(f.graphRuns, 1);
  assert.equal(f.disposals, 0);
});

test('a replaced readback field poisons the session instead of treating a later sync as a drain', async () => {
  const f = fixture({replaceReadback: true, fallbackFails: false});
  const first = f.session.decide(request);
  await f.firstSync.promise;
  f.firstReadback.resolve();
  await assert.rejects(first, {code: 'INFERENCE_ERROR'});
  assert.equal(f.syncs, 1);
  assert.equal(f.scopes[0].closed, false);
  await assert.rejects(f.session.dispose(), {code: 'INFERENCE_ERROR'});
});

test('a readback submitted before copy throws is settled before fallback and scope close', async () => {
  const f = fixture({copyThrows: true, firstFails: false, fallbackFails: false});
  const first = f.session.decide(request);
  await nextTurn();
  assert.equal(f.scopes[0].closed, false);
  assert.equal(f.syncs, 0);
  f.firstReadback.resolve();
  await assert.rejects(first, {code: 'INFERENCE_ERROR'});
  assert.equal(f.writes, 1);
  assert.equal(f.scopes[0].closed, true);
  await f.session.dispose();
});

test('a prior pending readback poisons before a new graph call', async () => {
  const f = fixture({preexisting: true});
  await assert.rejects(f.session.decide(request), {code: 'INFERENCE_ERROR'});
  assert.equal(f.graphRuns, 0);
  await assert.rejects(f.session.dispose(), {code: 'INFERENCE_ERROR'});
});

test('an unexpected rejected readback is observed without treating it as safe', () => {
  for (const mode of ['preexisting', 'graph', 'replacement', 'nonwritable']) {
    execFileSync(process.execPath, [fileURLToPath(new URL('./strands-unhandled-fixture.mjs', import.meta.url)), mode], {
      env: {PATH: '/usr/bin:/bin', LANG: 'C'},
      timeout: 2000,
      stdio: 'pipe'
    });
  }
});

test('an additional readback from the graph poisons before the tracked copy', async () => {
  const f = fixture({graphReadback: true});
  await assert.rejects(f.session.decide(request), {code: 'INFERENCE_ERROR'});
  assert.equal(f.graphRuns, 1);
  assert.equal(f.scopes[0].closed, false);
  await assert.rejects(f.session.dispose(), {code: 'INFERENCE_ERROR'});
});

test('device loss while readback is pending prevents this helper from closing the scope', async () => {
  const f = fixture({fallbackFails: false});
  const first = f.session.decide(request);
  await f.firstSync.promise;
  f.runner.lifecycle.loss = {reason: 'fixture-loss'};
  f.firstReadback.resolve();
  await assert.rejects(first, {code: 'INFERENCE_ERROR'});
  assert.equal(f.scopes[0].closed, false);
  assert.equal(f.disposals, 0);
  await assert.rejects(f.session.dispose(), {code: 'INFERENCE_ERROR'});
});
