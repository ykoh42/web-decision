import assert from 'node:assert/strict';
import test from 'node:test';
import {Tokenizer} from './strands-fake-tokenizer.mjs';
import {ExperimentalFusedRunner} from '../src/strands/runner.mjs';
import {ExperimentalStrandsSession} from '../src/strands/session.mjs';

// Model an outstanding CPU readback whose sync and fallback sync both reject.
// This fake does not establish how often the condition occurs on a real GPU.
function fixture() {
  let scope,
    syncs = 0,
    graphRuns = 0;
  const scopes = [],
    pendingReadbacks = [];
  const runtime = {
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
    lib: {webGPUContext: {copyRawBytesToBuffer() {}}},
    empty(_shape, _dtype, device) {
      const owner = scope;
      return {
        byteOffset: 0,
        getDataPtr() {
          return 1;
        },
        copyFrom(value) {
          if (device === 'cpu') pendingReadbacks.push(owner);
          this.value = ArrayBuffer.isView(value) ? Array.from(value) : value.value;
          return this;
        },
        toArray() {
          return this.value;
        }
      };
    },
    dispose() {}
  };
  const runner = new ExperimentalFusedRunner({
    runtime,
    gpu: {
      pushErrorScope() {},
      async popErrorScope() {
        return null;
      }
    },
    device: {
      async sync() {
        if (++syncs <= 2) throw new Error('Fake sync failed');
      }
    },
    weights: Array(328).fill(0),
    lifecycle: {loss: null, disposed: false},
    uncaptured: [],
    fn() {
      graphRuns++;
      return {value: [0, 1]};
    }
  });
  return {
    session: new ExperimentalStrandsSession({tokenizer: new Tokenizer(), runner}),
    request: {state: 's', questions: [{instructions: 'Pick', options: ['red', 'blue']}]},
    scopes,
    pendingReadbacks,
    graphRuns: () => graphRuns
  };
}

test('a missing pinned TVM pending-readback field fails before graph submission', async () => {
  const fake = fixture();
  await assert.rejects(fake.session.decide(fake.request), {code: 'INFERENCE_ERROR'});
  assert.equal(fake.pendingReadbacks.length, 0);
  assert.equal(fake.scopes.length, 0);
  assert.equal(fake.graphRuns(), 0);
});

test('reject later decisions when the pinned TVM readback state is unavailable', async () => {
  const fake = fixture();
  await assert.rejects(fake.session.decide(fake.request), {code: 'INFERENCE_ERROR'});
  await assert.rejects(fake.session.decide(fake.request));
  assert.equal(fake.graphRuns(), 0);
});
