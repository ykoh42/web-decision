import test from 'node:test';
import assert from 'node:assert/strict';
import {setImmediate as nextTurn} from 'node:timers/promises';
import {DEBERTA_REQUIRED_LIMITS, detectDebertaGPUDevice} from '../src/deberta/device.mjs';

function adapter({limits = DEBERTA_REQUIRED_LIMITS, features = ['shader-f16', 'subgroups'], failure} = {}) {
  const result = {calls: 0, descriptor: undefined, device: {label: 'device'}};
  result.value = {
    limits,
    features: new Set(features),
    info: {vendor: 'test'},
    async requestDevice(descriptor) {
      result.calls++;
      result.descriptor = descriptor;
      if (failure) throw failure;
      return result.device;
    }
  };
  return result;
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((accept, deny) => {
    resolve = accept;
    reject = deny;
  });
  return {promise, resolve, reject};
}
test('requests the pinned model limits without optional subgroups', async () => {
  const a = adapter(),
    options = [],
    result = await detectDebertaGPUDevice({
      requestAdapter: async (value) => {
        options.push(value);
        return a.value;
      }
    });
  assert.equal(result.device, a.device);
  assert.deepEqual(options, [{powerPreference: 'high-performance'}]);
  assert.deepEqual(a.descriptor.requiredLimits, DEBERTA_REQUIRED_LIMITS);
  assert.deepEqual(a.descriptor.requiredFeatures, ['shader-f16']);
});
test('never requests a device when shader-f16 is absent', async () => {
  const a = adapter({features: []});
  const result = await detectDebertaGPUDevice({requestAdapter: async () => a.value});
  assert.equal(result.adapter, a.value);
  assert.equal(result.device, undefined);
  assert.equal(a.calls, 0);
});
test('rejects every insufficient or invalid required limit before device creation', async () => {
  for (const [name, value] of Object.entries(DEBERTA_REQUIRED_LIMITS)) {
    for (const bad of [value - 1, NaN, undefined]) {
      const a = adapter({limits: {...DEBERTA_REQUIRED_LIMITS, [name]: bad}});
      await assert.rejects(detectDebertaGPUDevice({requestAdapter: async () => a.value}), new RegExp(name));
      assert.equal(a.calls, 0);
    }
  }
});
test('handles an unavailable adapter and preserves device-request failures', async () => {
  assert.equal(await detectDebertaGPUDevice(null), undefined);
  await assert.rejects(detectDebertaGPUDevice({requestAdapter: async () => null}), /No compatible/);
  const failure = Error('device denied'),
    a = adapter({failure});
  await assert.rejects(detectDebertaGPUDevice({requestAdapter: async () => a.value}), (error) => error === failure);
});

test('adapter info failure destroys the acquired device once and preserves the error', async () => {
  for (const destroyFails of [false, true]) {
    const failure = Error('adapter info failed'),
      a = adapter();
    let destroys = 0;
    a.device.destroy = () => {
      destroys++;
      if (destroyFails) throw Error('destroy failed');
    };
    Object.defineProperty(a.value, 'info', {
      get() {
        throw failure;
      }
    });
    await assert.rejects(detectDebertaGPUDevice({requestAdapter: async () => a.value}), (error) => error === failure);
    assert.equal(destroys, 1);
  }
});

test('rejects an already cancelled request without requesting an adapter', async () => {
  const controller = new AbortController(),
    reason = Error('cancelled');
  controller.abort(reason);
  await assert.rejects(
    detectDebertaGPUDevice({requestAdapter: () => assert.fail('adapter must not be requested')}, controller.signal),
    (error) => error === reason
  );
});

test('cancellation while waiting for an adapter prevents a late device request', async () => {
  const controller = new AbortController(),
    reason = Error('cancelled'),
    pending = deferred(),
    a = adapter(),
    request = detectDebertaGPUDevice({requestAdapter: () => pending.promise}, controller.signal);
  controller.abort(reason);
  await assert.rejects(request, (error) => error === reason);
  pending.resolve(a.value);
  await nextTurn();
  assert.equal(a.calls, 0);

  const another = new AbortController(),
    late = deferred(),
    rejected = detectDebertaGPUDevice({requestAdapter: () => late.promise}, another.signal);
  another.abort(reason);
  await assert.rejects(rejected, (error) => error === reason);
  late.reject(Error('late adapter failure'));
  await nextTurn();
});

test('cancellation destroys a device that arrives late and consumes late failures', async () => {
  const controller = new AbortController(),
    reason = Error('cancelled'),
    pending = deferred(),
    a = adapter(),
    device = {
      destroyCalls: 0,
      destroy() {
        this.destroyCalls++;
      }
    };
  a.value.requestDevice = () => pending.promise;
  const request = detectDebertaGPUDevice({requestAdapter: async () => a.value}, controller.signal);
  await nextTurn();
  controller.abort(reason);
  await assert.rejects(request, (error) => error === reason);
  pending.resolve(device);
  await nextTurn();
  assert.equal(device.destroyCalls, 1);

  const another = new AbortController(),
    late = deferred();
  a.value.requestDevice = () => late.promise;
  const rejected = detectDebertaGPUDevice({requestAdapter: async () => a.value}, another.signal);
  await nextTurn();
  another.abort(reason);
  await assert.rejects(rejected, (error) => error === reason);
  late.reject(Error('late failure'));
  await nextTurn();
});

test('cancellation at the device handoff destroys it once and preserves the reason', async () => {
  for (const destroyFails of [false, true]) {
    const controller = new AbortController(),
      reason = Error('handoff cancelled'),
      native = controller.signal,
      a = adapter();
    let deviceRequested = false,
      checksAfterRequest = 0,
      destroys = 0;
    const signal = {
      get aborted() {
        return native.aborted;
      },
      get reason() {
        return native.reason;
      },
      addEventListener: native.addEventListener.bind(native),
      removeEventListener: native.removeEventListener.bind(native),
      throwIfAborted() {
        if (deviceRequested && ++checksAfterRequest === 2) controller.abort(reason);
        native.throwIfAborted();
      }
    };
    a.value.requestDevice = async () => {
      deviceRequested = true;
      return {
        destroy() {
          destroys++;
          if (destroyFails) throw Error('destroy failed');
        }
      };
    };
    await assert.rejects(
      detectDebertaGPUDevice({requestAdapter: async () => a.value}, signal),
      (error) => error === reason
    );
    assert.equal(checksAfterRequest, 2);
    assert.equal(destroys, 1);
  }
});
