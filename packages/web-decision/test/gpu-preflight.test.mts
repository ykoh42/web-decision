import assert from 'node:assert/strict';
import test from 'node:test';
import {requestF16Adapter} from '../src/gpu-preflight.js';
import {detectKevGPUDevice, KEV_REQUIRED_LIMITS} from '../src/kev-gpu.js';
import {detectDebertaGPUDevice, DEBERTA_REQUIRED_LIMITS} from '../src/deberta/device.mjs';

test('rejects a real adapter without shader-f16 before creating a device', async () => {
  let calls = 0,
    deviceCalls = 0;
  const adapter = {
    features: new Set(),
    requestDevice() {
      deviceCalls++;
    }
  };
  const gpu = {
    requestAdapter(options: unknown) {
      calls++;
      assert.deepEqual(options, {powerPreference: 'high-performance'});
      return adapter;
    }
  };
  await assert.rejects(
    requestF16Adapter(gpu as unknown as GPU),
    (error: {code: string; message: string}) =>
      error.code === 'UNSUPPORTED_ENVIRONMENT' && /shader-f16/.test(error.message)
  );
  assert.equal(calls, 1);
  assert.equal(deviceCalls, 0);
});

test('accepts a supported adapter without creating a device', async () => {
  const adapter = {
    features: new Set(['shader-f16']),
    requestDevice() {
      throw Error('device requested too early');
    }
  };
  await requestF16Adapter({requestAdapter: async () => adapter} as unknown as GPU);
});

test('both model loaders obtain a fresh adapter before requesting a device', async () => {
  for (const [detect, limits] of [
    [detectKevGPUDevice, KEV_REQUIRED_LIMITS],
    [detectDebertaGPUDevice, DEBERTA_REQUIRED_LIMITS]
  ] as const) {
    const device = {},
      first = {
        features: new Set(['shader-f16']),
        requestDevice() {
          throw Error('expired adapter reused');
        }
      };
    const fresh = {features: new Set(['shader-f16']), limits, requestDevice: async () => device};
    let calls = 0;
    const gpu = {requestAdapter: async () => (++calls === 1 ? first : fresh)};
    await requestF16Adapter(gpu as unknown as GPU);
    assert.equal((await detect(gpu as unknown as GPU))!.device, device);
    assert.equal(calls, 2);
  }
});

test('maps missing and rejected adapters to model initialization errors', async () => {
  await assert.rejects(
    requestF16Adapter({requestAdapter: async () => null} as unknown as GPU),
    (e: {code: string; cause?: {message: string}}) =>
      e.code === 'MODEL_LOAD_FAILED' && /No compatible/.test(e.cause?.message as string)
  );
  const failure = Error('adapter failed');
  await assert.rejects(
    requestF16Adapter({
      requestAdapter: async () => {
        throw failure;
      }
    } as unknown as GPU),
    (e: {code: string; cause?: unknown}) => e.code === 'MODEL_LOAD_FAILED' && e.cause === failure
  );
});

test('honors cancellation before and during a non-abortable adapter request', async () => {
  const already = new AbortController(),
    reason = Error('cancelled');
  already.abort(reason);
  await assert.rejects(
    requestF16Adapter(
      {
        requestAdapter() {
          throw Error('must not start');
        }
      } as unknown as GPU,
      already.signal
    ),
    (e) => e === reason
  );
  let rejectLate!: (reason?: unknown) => void;
  const controller = new AbortController();
  const request = requestF16Adapter(
    {
      requestAdapter: () =>
        new Promise((_, reject) => {
          rejectLate = reject;
        })
    } as unknown as GPU,
    controller.signal
  );
  controller.abort(reason);
  await assert.rejects(request, (e) => e === reason);
  rejectLate(Error('late adapter failure'));
  await new Promise((resolve) => setTimeout(resolve, 0));
});
