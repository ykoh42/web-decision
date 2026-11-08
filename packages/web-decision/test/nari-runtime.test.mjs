import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {setImmediate as nextTurn} from 'node:timers/promises';
import {fixedRotary, runtimeOwnership, makeUploadGuard} from '../src/nari/runtime.mjs';

test('two separate rotary vectors have the recorded CPU float32 bytes', async () => {
  const r = await fixedRotary(),
    a = r['body.rotary_emb.inv_freq'],
    b = r['body.rotary_emb.original_inv_freq'];
  assert.equal(a.length, 32);
  assert.notEqual(a.buffer, b.buffer);
  assert.deepEqual([...a], [...b]);
  const raw = new Uint8Array(128),
    v = new DataView(raw.buffer);
  a.forEach((x, i) => v.setFloat32(i * 4, x, true));
  assert.equal(
    createHash('sha256').update(raw).digest('hex'),
    'ec4437c15ead01576c3e6c7412c11daba363d17cee8bfce5c5d1c54e2e09d2d0'
  );
});

test('rotary digest cancellation preserves the reason and observes late failure', async (t) => {
  let rejectHash, entered;
  const started = new Promise((resolve) => {
    entered = resolve;
  });
  t.mock.method(crypto.subtle, 'digest', () => {
    entered();
    return new Promise((_resolve, reject) => {
      rejectHash = reject;
    });
  });
  const controller = new AbortController();
  const reason = Error('cancel rotary');
  const pending = fixedRotary(controller.signal);
  await started;
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
  rejectHash(Error('late rotary hash failure'));
  await nextTurn();
});

test('partial GPU runtime initialization retains runtime and device on failure', () => {
  for (const failure of ['listener', 'initWebGPU', 'beginScope', 'webgpu', 'vm']) {
    let disposed = 0,
      destroyed = 0;
    const gpu = {
      lost: new Promise(() => {}),
      destroy() {
        destroyed++;
      },
      addEventListener() {
        if (failure === 'listener') throw Error('injected');
      }
    };
    const runtime = {
      dispose() {
        disposed++;
      },
      initWebGPU() {
        if (failure === 'initWebGPU') throw Error('injected');
      },
      beginScope() {
        if (failure === 'beginScope') throw Error('injected');
      },
      webgpu() {
        if (failure === 'webgpu') throw Error('injected');
        return {};
      },
      createVirtualMachine() {
        if (failure === 'vm') throw Error('injected');
        return {
          getFunction() {
            return () => {};
          }
        };
      }
    };
    const ownership = runtimeOwnership();
    assert.throws(() => ownership.initialize(runtime, gpu), /injected/);
    ownership.fail(runtime, gpu);
    assert.equal(disposed, 0, failure);
    assert.equal(destroyed, 0, failure);
    runtime.dispose();
    assert.equal(disposed, 0, failure);
  }
});

test('before initialization, ordinary runtime and device cleanup still occurs', () => {
  let disposed = 0,
    destroyed = 0;
  runtimeOwnership().fail(
    {
      dispose() {
        disposed++;
      }
    },
    {
      destroy() {
        destroyed++;
      }
    }
  );
  assert.equal(disposed, 1);
  assert.equal(destroyed, 1);
});

test('failure before disposal guard registration also retains both resources', () => {
  let disposed = 0,
    destroyed = 0;
  const runtime = {
      dispose() {
        disposed++;
      }
    },
    gpu = {
      destroy() {
        destroyed++;
      }
    };
  const ownership = runtimeOwnership();
  assert.throws(() => ownership.initialize(runtime, gpu));
  ownership.fail(runtime, gpu);
  assert.equal(disposed, 0);
  assert.equal(destroyed, 0);
});

test('failure after initialization retains the established owner', () => {
  let disposed = 0,
    destroyed = 0;
  const gpu = {
    lost: new Promise(() => {}),
    destroy() {
      destroyed++;
    },
    addEventListener() {}
  };
  const runtime = {
    dispose() {
      disposed++;
    },
    initWebGPU() {},
    beginScope() {},
    webgpu() {
      return {};
    },
    createVirtualMachine() {
      return {
        getFunction() {
          return () => {};
        }
      };
    }
  };
  const ownership = runtimeOwnership(),
    owner = ownership.initialize(runtime, gpu);
  ownership.fail(runtime, gpu);
  assert.equal(owner.blocked, true);
  runtime.dispose();
  assert.equal(disposed, 0);
  assert.equal(destroyed, 0);
});

test('upload boundary checks use running phase at five-second intervals', async () => {
  let now = 0;
  const calls = [];
  const check = makeUploadGuard(
    async (phase) => calls.push(phase),
    undefined,
    () => now
  );
  await check();
  now = 4999;
  await check();
  assert.deepEqual(calls, ['running']);
  now = 5000;
  await check();
  assert.deepEqual(calls, ['running', 'running']);
});
