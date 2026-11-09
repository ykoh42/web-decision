import assert from 'node:assert/strict';
import test from 'node:test';
import {WebDecisionError} from '../src/kev.js';
import {requestStrandsAdapter} from '../src/strands/preflight.mjs';

const LARGEST_BINDING = 1_017_118_720;

function validAdapter(events: string[], onFeatures?: () => void): object {
  return {
    get features() {
      events.push('adapter.features');
      onFeatures?.();
      return new Set(['shader-f16']);
    },
    get limits() {
      events.push('adapter.limits');
      return {
        maxBufferSize: LARGEST_BINDING,
        maxStorageBufferBindingSize: LARGEST_BINDING,
        maxStorageBuffersPerShaderStage: 8
      };
    }
  };
}

test('reads the adapter getter twice and validates the returned adapter before resolving', async () => {
  const events: string[] = [];
  const adapter = validAdapter(events);
  let getterCalls = 0;
  const gpuApi = Object.defineProperty({}, 'requestAdapter', {
    get() {
      getterCalls++;
      events.push(`gpu.get:${getterCalls}`);
      return async () => {
        events.push('gpu.call');
        return adapter;
      };
    }
  });

  assert.equal(await requestStrandsAdapter(gpuApi), adapter);
  assert.deepEqual(events.slice(0, 3), ['gpu.get:1', 'gpu.get:2', 'gpu.call']);
  assert.equal(getterCalls, 2);
  assert.ok(events.indexOf('gpu.call') < events.indexOf('adapter.features'));
  assert.ok(events.includes('adapter.limits'));
});

test('preserves an already-aborted caller reason before reading GPU getters', async () => {
  const controller = new AbortController();
  const reason = new Error('cancel before adapter preflight');
  let getterCalls = 0;
  const gpuApi = Object.defineProperty({}, 'requestAdapter', {
    get() {
      getterCalls++;
      throw new Error('GPU getter should not be observed after cancellation.');
    }
  });
  controller.abort(reason);

  await assert.rejects(requestStrandsAdapter(gpuApi, controller.signal), (error: unknown) => error === reason);
  assert.equal(getterCalls, 0);
});

test('wraps adapter request failures with their original cause', async () => {
  const cause = new Error('adapter request failed');
  const events: string[] = [];
  const gpuApi = Object.defineProperty({}, 'requestAdapter', {
    get() {
      events.push('gpu.get');
      return () => {
        events.push('gpu.call');
        return Promise.reject(cause);
      };
    }
  });

  await assert.rejects(requestStrandsAdapter(gpuApi), (error: unknown) => {
    assert.ok(error instanceof WebDecisionError);
    assert.equal(error.code, 'MODEL_LOAD_FAILED');
    assert.equal(error.message, 'Could not request a WebGPU adapter.');
    assert.equal(error.cause, cause);
    return true;
  });
  assert.deepEqual(events, ['gpu.get', 'gpu.get', 'gpu.call']);
});

test('settles an in-flight adapter request with the caller reason and ignores its late adapter', async () => {
  const controller = new AbortController();
  const reason = new Error('cancel pending adapter request');
  const events: string[] = [];
  let resolveAdapter!: (adapter: object) => void;
  let signalReady!: () => void;
  const started = new Promise<void>((resolve) => {
    signalReady = resolve;
  });
  const pendingAdapter = new Promise<object>((resolve) => {
    resolveAdapter = resolve;
  });
  const adapter = validAdapter(events);
  const gpuApi = {
    requestAdapter() {
      events.push('gpu.call');
      signalReady();
      return pendingAdapter;
    }
  };
  const pending = requestStrandsAdapter(gpuApi, controller.signal);
  await started;
  controller.abort(reason);

  await assert.rejects(pending, (error: unknown) => error === reason);
  resolveAdapter(adapter);
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.deepEqual(events, ['gpu.call']);
});

test('checks cancellation again after validating a just-resolved adapter', async () => {
  const controller = new AbortController();
  const reason = new Error('cancel during adapter validation');
  const events: string[] = [];
  const adapter = validAdapter(events, () => controller.abort(reason));
  const gpuApi = {requestAdapter: async () => adapter};

  await assert.rejects(requestStrandsAdapter(gpuApi, controller.signal), (error: unknown) => error === reason);
  assert.ok(events.includes('adapter.features'));
  assert.ok(events.includes('adapter.limits'));
});
