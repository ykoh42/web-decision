import assert from 'node:assert/strict';
import {test} from 'node:test';
import {withGemma4V2GpuErrors} from '../src/strands/gemma4-v2-gpu-errors.mjs';

function device(
  errors: Array<{message: string} | null>,
  limits = {maxBufferSize: 2147483644, maxStorageBufferBindingSize: 2147483644}
) {
  const pushed: string[] = [];
  let pops = 0;
  return {
    pushed,
    get pops() {
      return pops;
    },
    device: {
      limits,
      pushErrorScope(filter: string) {
        pushed.push(filter);
      },
      async popErrorScope() {
        return errors[pops++] ?? null;
      }
    } as unknown as GPUDevice
  };
}

test('checks both attention buffer limits before running the TVM graph', async () => {
  const {guardGemma4V2Runner, watchGemma4V2Device} = await import('../src/strands/gemma4-v2-gpu-errors.mjs');
  const input = {stateIds: [2, 3], questionIds: [4], imageSpans: [], images: []};
  const lifecycle = watchGemma4V2Device({lost: new Promise(() => {})});
  const runner = {
    async run() {
      return [7];
    },
    releaseCachedMemory() {},
    dispose() {}
  };
  for (const [maxBufferSize, maxStorageBufferBindingSize] of [
    [49148, 49152],
    [49152, 49148]
  ] as const) {
    const gpu = device([], {maxBufferSize, maxStorageBufferBindingSize});
    await assert.rejects(guardGemma4V2Runner(gpu.device, lifecycle, runner).run(input), {
      code: 'DEVICE_RESOURCE_LIMIT',
      message: /49152-byte/
    });
    assert.deepEqual(gpu.pushed, []);
  }
  const gpu = device([], {maxBufferSize: 49152, maxStorageBufferBindingSize: 49152});
  assert.deepEqual(await guardGemma4V2Runner(gpu.device, lifecycle, runner).run(input), [7]);
  assert.equal(gpu.pops, 3);
  lifecycle.close();
});

test('the tiled decoder checks the largest unpacked matrix and full hidden buffer', async () => {
  const {guardGemma4V2Runner, watchGemma4V2Device} = await import('../src/strands/gemma4-v2-gpu-errors.mjs');
  const lifecycle = watchGemma4V2Device({lost: new Promise(() => {})});
  const runner = {
    async run() {
      return [7];
    },
    releaseCachedMemory() {},
    dispose() {}
  };
  for (const [tokens, bytes] of [
    [3, 75_497_472],
    [32_768, 201_326_592],
    [131_072, 805_306_368]
  ] as const) {
    const input = {stateIds: Array<number>(tokens - 1).fill(2), questionIds: [4], imageSpans: [], images: []};
    for (const [maxBufferSize, maxStorageBufferBindingSize] of [
      [bytes - 4, bytes],
      [bytes, bytes - 4]
    ] as const) {
      const gpu = device([], {maxBufferSize, maxStorageBufferBindingSize});
      await assert.rejects(guardGemma4V2Runner(gpu.device, lifecycle, runner, true).run(input), {
        code: 'DEVICE_RESOURCE_LIMIT',
        message: new RegExp(`${bytes}-byte`, 'u')
      });
      assert.deepEqual(gpu.pushed, []);
    }
    const gpu = device([], {maxBufferSize: bytes, maxStorageBufferBindingSize: bytes});
    assert.deepEqual(await guardGemma4V2Runner(gpu.device, lifecycle, runner, true).run(input), [7]);
  }
  lifecycle.close();
});

test('rejects attention allocations outside the TVM WebGPU runtime signed range', async () => {
  const {guardGemma4V2Runner, watchGemma4V2Device} = await import('../src/strands/gemma4-v2-gpu-errors.mjs');
  const lifecycle = watchGemma4V2Device({lost: new Promise(() => {})});
  const gpu = device([], {maxBufferSize: 4294967292, maxStorageBufferBindingSize: 4294967292});
  const guarded = guardGemma4V2Runner(gpu.device, lifecycle, {
    async run() {
      return [7];
    },
    releaseCachedMemory() {},
    dispose() {}
  });
  const stateIds = Array.from({length: 131070}, () => 2);
  const input = {stateIds, questionIds: [4], imageSpans: [], images: []};
  assert.deepEqual(await guarded.run(input), [7]);
  await assert.rejects(guarded.run({...input, questionIds: [3, 4]}), {
    code: 'DEVICE_RESOURCE_LIMIT',
    message: /2147483648-byte.*signed 32-bit/
  });
  assert.equal(gpu.pops, 3);
  lifecycle.close();
});

test('returns results only after all GPU error scopes are clear', async () => {
  const gpu = device([null, null, null]);
  assert.equal(await withGemma4V2GpuErrors(gpu.device, async () => 7), 7);
  assert.deepEqual(gpu.pushed, ['internal', 'out-of-memory', 'validation']);
  assert.equal(gpu.pops, 3);
});

test('rejects GPU errors and closes every scope', async () => {
  for (const [errors, code] of [
    [[{message: 'validation error'}, null, null], 'INFERENCE_ERROR'],
    [[null, {message: 'out of memory'}, null], 'DEVICE_RESOURCE_LIMIT'],
    [[null, null, {message: 'internal failure'}], 'INFERENCE_ERROR']
  ] as const) {
    const gpu = device([...errors]);
    await assert.rejects(
      withGemma4V2GpuErrors(gpu.device, async () => 7),
      (error: unknown) => error instanceof Error && 'code' in error && error.code === code
    );
    assert.equal(gpu.pops, 3);
  }
});

test('preserves operation failure while closing every scope', async () => {
  const gpu = device([{message: 'secondary gpu error'}, null, null]);
  const reason = new Error('cancelled');
  await assert.rejects(
    withGemma4V2GpuErrors(gpu.device, async () => {
      throw reason;
    }),
    (error) => error === reason
  );
  assert.equal(gpu.pops, 3);
});

test('attempts to close remaining scopes when a scope pop fails', async () => {
  const reason = new Error('device scope failure');
  let pops = 0;
  const gpu = {
    pushErrorScope() {},
    async popErrorScope() {
      if (++pops === 1) throw reason;
      return null;
    }
  } as unknown as GPUDevice;
  await assert.rejects(
    withGemma4V2GpuErrors(gpu, async () => 7),
    (error) => error === reason
  );
  assert.equal(pops, 3);
});

test('attempts all runtime cleanup and preserves the first cleanup failure', async () => {
  const {disposeGemma4V2Runtime} = await import('../src/strands/gemma4-v2-gpu-errors.mjs');
  const events: string[] = [];
  const reason = new Error('scope cleanup failed');
  assert.throws(
    () =>
      disposeGemma4V2Runtime(
        {
          endScope() {
            events.push('scope');
            throw reason;
          },
          dispose() {
            events.push('runtime');
            throw new Error('runtime cleanup failed');
          }
        },
        {
          destroy() {
            events.push('device');
          }
        },
        true
      ),
    (error) => error === reason
  );
  assert.deepEqual(events, ['scope', 'runtime', 'device']);
});

test('tracks unexpected device loss while ignoring deliberate close', async () => {
  const {watchGemma4V2Device} = await import('../src/strands/gemma4-v2-gpu-errors.mjs');
  let lose!: (reason: unknown) => void;
  const pending = new Promise<unknown>((resolve) => {
    lose = resolve;
  });
  const watcher = watchGemma4V2Device({lost: pending});
  const reason = {message: 'GPU device lost'};
  lose(reason);
  await pending;
  assert.equal(watcher.lost, true);
  assert.equal(watcher.cause, reason);
  watcher.close();
  let closeDevice!: (reason: unknown) => void;
  const normal = new Promise<unknown>((resolve) => {
    closeDevice = resolve;
  });
  const closed = watchGemma4V2Device({lost: normal});
  closed.close();
  closeDevice({message: 'destroyed'});
  await normal;
  assert.equal(closed.lost, false);
});

test('does not dispose the TVM runtime twice after device-loss disposal', async () => {
  const {disposeGemma4V2Runtime, watchGemma4V2Device} = await import('../src/strands/gemma4-v2-gpu-errors.mjs');
  let lose!: (reason: unknown) => void;
  const lost = new Promise<unknown>((resolve) => {
    lose = resolve;
  });
  const watcher = watchGemma4V2Device({lost});
  lose({message: 'GPU lost'});
  await lost;
  const events: string[] = [];
  watcher.close();
  disposeGemma4V2Runtime(
    {
      endScope() {
        events.push('scope');
        throw new Error('already freed');
      },
      dispose() {
        events.push('runtime');
        throw new Error('already freed');
      }
    },
    {
      destroy() {
        events.push('device');
      }
    },
    true,
    watcher.lost
  );
  assert.deepEqual(events, ['device']);
});

test('tracks a rejected device-lost promise as a loss', async () => {
  const {watchGemma4V2Device} = await import('../src/strands/gemma4-v2-gpu-errors.mjs');
  const reason = new Error('device loss rejected');
  const promise = Promise.reject(reason);
  const watcher = watchGemma4V2Device({lost: promise});
  await promise.catch(() => {});
  assert.equal(watcher.lost, true);
  assert.equal(watcher.cause, reason);
  watcher.close();
});

test('device loss cancels an in-flight source read and preserves user cancellation', async () => {
  const {combineGemma4V2Signals, watchGemma4V2Device} = await import('../src/strands/gemma4-v2-gpu-errors.mjs');
  let lose!: (reason: unknown) => void;
  const lost = new Promise<unknown>((resolve) => {
    lose = resolve;
  });
  const watcher = watchGemma4V2Device({lost});
  const controller = new AbortController();
  const first = combineGemma4V2Signals(watcher.signal, controller.signal);
  const callerReason = new Error('caller cancelled');
  controller.abort(callerReason);
  assert.equal(first.signal.reason, callerReason);
  first.close();
  assert.equal(watcher.signal.aborted, false);
  const second = combineGemma4V2Signals(watcher.signal);
  const read = new Promise((_resolve, reject) => {
    second.signal.addEventListener('abort', () => reject(second.signal.reason), {once: true});
  });
  const deviceReason = {message: 'GPU gone'};
  lose(deviceReason);
  await assert.rejects(read, /GPU device was lost/);
  assert.equal(watcher.lost, true);
  assert.equal(watcher.cause, deviceReason);
  second.close();
  watcher.close();
});

test('combining already-cancelled signals keeps the caller reason first', async () => {
  const {combineGemma4V2Signals} = await import('../src/strands/gemma4-v2-gpu-errors.mjs');
  const device = new AbortController();
  const caller = new AbortController();
  const reason = new Error('caller first');
  device.abort(new Error('lost'));
  caller.abort(reason);
  const combined = combineGemma4V2Signals(device.signal, caller.signal);
  assert.equal(combined.signal.reason, reason);
  combined.close();
});

test('guarded runner cancels source work on loss and allows retry after caller-only cancellation', async () => {
  const {guardGemma4V2Runner, watchGemma4V2Device} = await import('../src/strands/gemma4-v2-gpu-errors.mjs');
  let lose!: (reason: unknown) => void;
  const lost = new Promise<unknown>((resolve) => {
    lose = resolve;
  });
  const lifecycle = watchGemma4V2Device({lost});
  const gpu = device([null, null, null, null, null, null]);
  let calls = 0;
  const runner = {
    async run(_input: unknown, signal?: AbortSignal | null): Promise<number[]> {
      calls++;
      return new Promise((_resolve, reject) => {
        signal!.addEventListener('abort', () => reject(signal!.reason), {once: true});
      });
    },
    releaseCachedMemory() {},
    dispose() {}
  };
  const guarded = guardGemma4V2Runner(gpu.device, lifecycle, runner);
  const input = {stateIds: [2], questionIds: [3], imageSpans: [], images: []};
  const controller = new AbortController();
  const reason = new Error('caller cancelled');
  const first = guarded.run(input, controller.signal);
  controller.abort(reason);
  await assert.rejects(first, (error) => error === reason);
  assert.equal(lifecycle.lost, false);
  const second = guarded.run(input);
  lose({message: 'device lost'});
  await assert.rejects(second, /GPU device was lost/);
  assert.equal(lifecycle.lost, true);
  await assert.rejects(guarded.run(input), /GPU device was lost/);
  assert.equal(calls, 2);
  assert.equal(gpu.pops, 6);
  lifecycle.close();
});

test('device loss during inference preserves loss through caller finally disposal', async () => {
  const {guardGemma4V2Runner, watchGemma4V2Device, disposeGemma4V2Runtime} =
    await import('../src/strands/gemma4-v2-gpu-errors.mjs');
  const {withGemma4V2Cleanup} = await import('../src/strands/gemma4-v2-cleanup.mjs');
  let lose!: (reason: unknown) => void;
  const lost = new Promise<unknown>((resolve) => {
    lose = resolve;
  });
  const lifecycle = watchGemma4V2Device({lost});
  const gpu = device([null, null, null]);
  const events: string[] = [];
  const runtime = {
    endScope() {
      events.push('scope');
      throw new Error('unreachable cleanup');
    },
    dispose() {
      events.push('runtime');
      throw new Error('already disposed');
    }
  };
  const deviceHandle = {
    destroy() {
      events.push('destroy');
    }
  };
  const runner = {
    run(_input: unknown, signal?: AbortSignal | null): Promise<number[]> {
      return withGemma4V2Cleanup(
        () =>
          new Promise((_resolve, reject) => {
            signal!.addEventListener('abort', () => reject(signal!.reason), {once: true});
          }),
        () => runtime.endScope()
      );
    },
    releaseCachedMemory() {},
    dispose() {
      lifecycle.close();
      disposeGemma4V2Runtime(runtime, deviceHandle, true, lifecycle.lost);
    }
  };
  const guarded = guardGemma4V2Runner(gpu.device, lifecycle, runner);
  const call = async () => {
    try {
      await guarded.run({stateIds: [2], questionIds: [3], imageSpans: [], images: []});
    } finally {
      await guarded.dispose();
    }
  };
  const pending = call();
  lose({message: 'device lost'});
  await assert.rejects(pending, /GPU device was lost/);
  assert.deepEqual(events, ['scope', 'destroy']);
});
