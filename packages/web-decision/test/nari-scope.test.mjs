import assert from 'node:assert/strict';
import test from 'node:test';
import {withNariScope} from '../src/nari/scope.mjs';

function fixture() {
  const events = [],
    context = {pendingGPUToCPUCopy: null};
  const runtime = {
    lib: {webGPUContext: context},
    beginScope() {
      events.push('begin');
    },
    endScope() {
      events.push('end');
    }
  };
  const gpu = {
    pushErrorScope() {
      events.push('push');
    },
    async popErrorScope() {
      events.push('pop');
      return null;
    }
  };
  const device = {
    async sync() {
      events.push('sync');
      context.pendingGPUToCPUCopy = null;
    }
  };
  const runner = {
    runtime,
    gpu,
    device,
    lifecycle: {loss: null, disposed: false},
    uncaptured: [],
    disposalGuard: {blocked: false}
  };
  return {runner, runtime, gpu, device, context, events};
}

test('missing readback state prevents graph work and poisons future use', async () => {
  const f = fixture();
  delete f.context.pendingGPUToCPUCopy;
  let submitted = false;
  await assert.rejects(
    withNariScope(f.runner, 'failed', async () => {
      submitted = true;
    }),
    {code: 'INFERENCE_ERROR'}
  );
  assert.equal(submitted, false);
  assert.equal(f.runner.poisoned, true);
  await assert.rejects(
    withNariScope(f.runner, 'failed', async () => {}),
    {code: 'INFERENCE_ERROR'}
  );
});

test('failed sync waits for its captured readback before releasing the scope', async () => {
  const f = fixture();
  let resolveReadback,
    started,
    syncCount = 0,
    finished = false;
  const entered = new Promise((resolve) => {
    started = resolve;
  });
  f.device.sync = async () => {
    if (++syncCount === 1) {
      started();
      throw Error('sync failed');
    }
    f.context.pendingGPUToCPUCopy = null;
  };
  const readback = new Promise((resolve) => {
    resolveReadback = resolve;
  });
  const task = withNariScope(f.runner, 'failed', async (lease) => {
    lease.prepareReadback();
    f.context.pendingGPUToCPUCopy = readback;
    lease.captureReadback();
    await lease.sync();
  }).finally(() => {
    finished = true;
  });
  await entered;
  await Promise.resolve();
  assert.equal(finished, false);
  assert.equal(f.events.includes('end'), false);
  resolveReadback();
  await assert.rejects(task, {code: 'INFERENCE_ERROR'});
  assert.equal(f.events.includes('end'), true);
  assert.notEqual(f.runner.poisoned, true);
});

test('changed readback identity poisons the runner and retains the scope', async () => {
  const f = fixture();
  await assert.rejects(
    withNariScope(f.runner, 'failed', async (lease) => {
      lease.prepareReadback();
      f.context.pendingGPUToCPUCopy = Promise.resolve();
      lease.captureReadback();
      f.context.pendingGPUToCPUCopy = Promise.resolve();
      await lease.sync();
    }),
    {code: 'INFERENCE_ERROR'}
  );
  assert.equal(f.runner.poisoned, true);
  assert.equal(f.events.includes('end'), false);
});

test('scope closure failure poisons the runner', async () => {
  for (const target of ['runtime', 'gpu']) {
    const f = fixture();
    if (target === 'runtime')
      f.runtime.endScope = () => {
        throw Error('scope close failed');
      };
    else
      f.gpu.popErrorScope = async () => {
        throw Error('scope close failed');
      };
    await assert.rejects(
      withNariScope(f.runner, 'failed', async (lease) => lease.syncWithoutReadback()),
      {code: 'INFERENCE_ERROR'}
    );
    assert.equal(f.runner.poisoned, true);
  }
});
