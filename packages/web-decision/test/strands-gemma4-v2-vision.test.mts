import assert from 'node:assert/strict';
import {test} from 'node:test';
import type {DLDevice, Instance, Tensor} from '../src/runtime/tvm-types.js';
import {
  gemma4V2VisionGeometry,
  insertGemma4V2VisionEmbedding,
  runGemma4V2Vision
} from '../src/strands/gemma4-v2-vision.mts';

function input(width: number, height: number) {
  const positions = new Int32Array(2520 * 2).fill(-1);
  for (let y = 0; y < height * 3; y++)
    for (let x = 0; x < width * 3; x++) {
      const index = y * width * 3 + x;
      positions[index * 2] = x;
      positions[index * 2 + 1] = y;
    }
  return {pixels: new Float32Array(2520 * 768), positions, tokens: width * height};
}

test('pooling geometry supports minimum, rectangular and maximum images', () => {
  for (const [width, height] of [
    [1, 1],
    [1, 2],
    [4, 5],
    [16, 16],
    [1, 280],
    [14, 20]
  ] as const) {
    const image = input(width, height);
    const result = gemma4V2VisionGeometry(image);
    assert.equal(result.indices.length, width * height * 9);
    for (let group = 0; group < width * height; group++) {
      const expected: number[] = [];
      for (let y = Math.floor(group / width) * 3; y < Math.floor(group / width) * 3 + 3; y++)
        for (let x = (group % width) * 3; x < (group % width) * 3 + 3; x++) expected.push(y * width * 3 + x);
      assert.deepEqual(Array.from(result.indices.subarray(group * 9, group * 9 + 9)), expected);
    }
    assert.equal(result.mask[width * height * 9 - 1], 0);
    if (width * height < 280) assert.equal(result.mask[width * height * 9], Math.fround(-1e30));
    assert.equal(result.cos[0], 1);
    assert.equal(result.sin[0], 0);
  }
});

test('inserts successive GPU image tensors at validated text positions', () => {
  const device = {deviceType: 15, deviceId: 0, async sync() {}} satisfies DLDevice;
  const calls: number[] = [];
  const gpu = (shape: number[], dtype = 'float32') => ({shape, dtype, device}) as Tensor;
  const runtime = {
    empty(shape: number[], dtype: string) {
      assert.deepEqual(shape, [1]);
      assert.equal(dtype, 'int32');
      return {
        copyFrom(values: Int32Array) {
          calls.push(values[0]!);
          return gpu(shape, dtype);
        }
      };
    }
  } as unknown as Instance;
  const hidden = gpu([1, 12, 1536]);
  const first = gpu([2, 1536]);
  const second = gpu([3, 1536]);
  const insert = (text: Tensor, image: Tensor, offset: Tensor) => {
    assert.equal(offset.dtype, 'int32');
    assert.equal(text.device, device);
    assert.equal(image.device, device);
    return gpu([1, 12, 1536]);
  };
  const once = insertGemma4V2VisionEmbedding(runtime, device, insert, hidden, first, 0);
  insertGemma4V2VisionEmbedding(runtime, device, insert, once, second, 9);
  assert.deepEqual(calls, [0, 9]);
  for (const start of [-1, 10, 0.5])
    assert.throws(() => insertGemma4V2VisionEmbedding(runtime, device, insert, hidden, second, start), /do not fit/);
  assert.deepEqual(calls, [0, 9]);
});

test('rejects invalid patch geometry before GPU allocation', () => {
  const image = input(1, 1);
  image.positions[2] = 0;
  image.positions[3] = 0;
  assert.throws(() => gemma4V2VisionGeometry(image), /Duplicate/);
});

test('runs all vision stages and returns a copy of the synchronized output', async () => {
  const events: string[] = [];
  const device = {
    deviceType: 1,
    deviceId: 0,
    async sync() {
      events.push('sync');
    }
  } satisfies DLDevice;
  let output = new Float32Array(1536).fill(7);
  const tensor = (shape: number[], dtype: string): Tensor => ({
    shape,
    dtype,
    device,
    copyFrom() {
      return this;
    },
    copyFromRawBytes() {
      return this;
    },
    toArray() {
      return output;
    },
    toRawBytes() {
      return new Uint8Array();
    }
  });
  const runtime = {
    beginScope() {
      events.push('begin');
    },
    endScope() {
      events.push('end');
    },
    empty(shape: number[] | number, dtype: string) {
      return tensor(typeof shape === 'number' ? [shape] : shape, dtype);
    },
    cpu() {
      return device;
    }
  } as unknown as Instance;
  const stage =
    (name: string, shape: number[]) =>
    (..._args: Tensor[]) => {
      events.push(name);
      return tensor(shape, 'float32');
    };
  const graph = {
    patch: stage('patch', [1, 2520, 768]),
    layer: stage('layer', [1, 2520, 768]),
    pool: stage('pool', [1, 768]),
    projection: stage('projection', [1, 1536])
  };
  const result = await runGemma4V2Vision(
    runtime,
    device,
    graph,
    {patch: [], layers: Array.from({length: 16}, () => []), projection: []},
    input(1, 1)
  );
  assert.deepEqual(events, [
    'begin',
    'patch',
    ...Array.from({length: 8}, () => ['layer', 'layer', 'sync']).flat(),
    'pool',
    'projection',
    'sync',
    'end'
  ]);
  output[0] = 99;
  assert.equal(result[0], 7);
  output = new Float32Array(1536);
  graph.projection = stage('bad', [2, 1536]);
  await assert.rejects(
    runGemma4V2Vision(
      runtime,
      device,
      graph,
      {patch: [], layers: Array.from({length: 16}, () => []), projection: []},
      input(1, 1)
    ),
    /output shape/
  );
  assert.equal(events.at(-1), 'end');
  const controller = new AbortController();
  controller.abort();
  const eventsBeforeAbort = events.length;
  await assert.rejects(
    runGemma4V2Vision(
      runtime,
      device,
      graph,
      {patch: [], layers: Array.from({length: 16}, () => []), projection: []},
      input(1, 1),
      controller.signal
    ),
    {name: 'AbortError'}
  );
  assert.equal(events.length, eventsBeforeAbort);
});

test('vision inference preserves its failure when scope cleanup also fails', async () => {
  const primary = new Error('vision GPU failure');
  const cleanup = new Error('scope cleanup failure');
  let ended = 0;
  const device = {deviceType: 15, deviceId: 0, async sync() {}} satisfies DLDevice;
  const runtime = {
    beginScope() {},
    endScope() {
      ended++;
      throw cleanup;
    },
    empty() {
      return {
        copyFrom() {
          return this;
        }
      };
    }
  } as unknown as Instance;
  const fail = () => {
    throw primary;
  };
  const graph = {patch: fail, layer: fail, pool: fail, projection: fail};
  const weights = {patch: [], layers: Array.from({length: 16}, () => []), projection: []};
  await assert.rejects(runGemma4V2Vision(runtime, device, graph, weights, input(1, 1)), (error) => error === primary);
  assert.equal(ended, 1);
});

test('vision synchronization failure remains primary and cleanup-only errors surface', async () => {
  const primary = new Error('device sync failed');
  const cleanup = new Error('scope cleanup failed');
  let failSync = true;
  const device = {
    deviceType: 15,
    deviceId: 0,
    async sync() {
      if (failSync) throw primary;
    }
  } satisfies DLDevice;
  const tensor = (shape: number[], dtype = 'float32') =>
    ({
      shape,
      dtype,
      device,
      copyFrom() {
        return this;
      },
      toArray() {
        return new Float32Array(1536);
      }
    }) as unknown as Tensor;
  const runtime = {
    beginScope() {},
    endScope() {
      throw cleanup;
    },
    empty(shape: number[] | number, dtype: string) {
      return tensor(typeof shape === 'number' ? [shape] : shape, dtype);
    },
    cpu() {
      return device;
    }
  } as unknown as Instance;
  const graph = {
    patch: () => tensor([1, 2520, 768]),
    layer: () => tensor([1, 2520, 768]),
    pool: () => tensor([1, 768]),
    projection: () => tensor([1, 1536])
  };
  const weights = {patch: [], layers: Array.from({length: 16}, () => []), projection: []};
  await assert.rejects(runGemma4V2Vision(runtime, device, graph, weights, input(1, 1)), (error) => error === primary);
  failSync = false;
  await assert.rejects(runGemma4V2Vision(runtime, device, graph, weights, input(1, 1)), (error) => error === cleanup);
});

test('cancellation after a vision synchronization stops further layers and releases the scope', async () => {
  const controller = new AbortController();
  let layers = 0;
  let ends = 0;
  const device = {
    deviceType: 15,
    deviceId: 0,
    async sync() {
      controller.abort();
    }
  } satisfies DLDevice;
  const tensor = (shape: number[], dtype = 'float32') =>
    ({
      shape,
      dtype,
      device,
      copyFrom() {
        return this;
      }
    }) as unknown as Tensor;
  const runtime = {
    beginScope() {},
    endScope() {
      ends++;
    },
    empty(shape: number[] | number, dtype: string) {
      return tensor(typeof shape === 'number' ? [shape] : shape, dtype);
    }
  } as unknown as Instance;
  const graph = {
    patch: () => tensor([1, 2520, 768]),
    layer: () => {
      layers++;
      return tensor([1, 2520, 768]);
    },
    pool: () => {
      throw new Error('unexpected pool');
    },
    projection: () => {
      throw new Error('unexpected projection');
    }
  };
  const weights = {patch: [], layers: Array.from({length: 16}, () => []), projection: []};
  await assert.rejects(runGemma4V2Vision(runtime, device, graph, weights, input(1, 1), controller.signal), {
    name: 'AbortError'
  });
  assert.equal(layers, 2);
  assert.equal(ends, 1);
});
