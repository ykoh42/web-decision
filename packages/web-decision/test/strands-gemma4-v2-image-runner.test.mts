import assert from 'node:assert/strict';
import {test} from 'node:test';
import type {DLDevice, Instance, Tensor, VirtualMachine} from '../src/runtime/tvm-types.js';
import {
  createGemma4V2ImageInserter,
  createGemma4V2ImageInserterWithWeights
} from '../src/strands/gemma4-v2-image-runner.mts';
import type {Gemma4V2VisionWeights} from '../src/strands/gemma4-v2-vision.mts';

function prepared() {
  const positions = new Int32Array(2520 * 2).fill(-1);
  for (let y = 0; y < 3; y++)
    for (let x = 0; x < 3; x++) {
      positions[(y * 3 + x) * 2] = x;
      positions[(y * 3 + x) * 2 + 1] = y;
    }
  return {pixels: new Float32Array(2520 * 768), positions, tokens: 1};
}

test('multiple images release intermediate GPU tensors after synchronization', async () => {
  const events: string[] = [];
  const scopes: FakeTensor[][] = [[]];
  const device = {
    deviceType: 15,
    deviceId: 0,
    async sync() {
      events.push('sync');
    }
  } satisfies DLDevice;
  type FakeTensor = Tensor & {disposed: boolean; dispose(): void};
  const tensor = (shape: number[], dtype = 'float32'): FakeTensor => {
    const value = {
      shape,
      dtype,
      device,
      disposed: false,
      copyFrom() {
        return this;
      },
      copyFromRawBytes() {
        return this;
      },
      toArray() {
        throw new Error('Unexpected CPU read');
      },
      toRawBytes() {
        throw new Error('Unexpected CPU read');
      },
      dispose() {
        this.disposed = true;
      }
    } satisfies FakeTensor;
    scopes.at(-1)!.push(value);
    return value;
  };
  const runtime = {
    beginScope() {
      scopes.push([]);
      events.push('begin');
    },
    endScope() {
      for (const value of scopes.pop()!) value.dispose();
      events.push('end');
    },
    moveToParentScope(value: FakeTensor) {
      const current = scopes.at(-1)!;
      current.splice(current.indexOf(value), 1);
      scopes.at(-2)!.push(value);
      return value;
    },
    empty(shape: number[] | number, dtype: string) {
      return tensor(typeof shape === 'number' ? [shape] : shape, dtype);
    }
  } as unknown as Instance;
  const insertions: FakeTensor[] = [];
  const vm = {
    getFunction(name: string) {
      return (..._args: unknown[]) => {
        if (name === 'vision_pool_main') return tensor([1, 768]);
        if (name === 'vision_projection_main') return tensor([1, 1536]);
        if (name === 'image_insert_main') {
          const result = tensor([1, 5, 1536]);
          insertions.push(result);
          return result;
        }
        return tensor([1, 2520, 768]);
      };
    }
  } satisfies VirtualMachine;
  const weights = {patch: [], layers: Array.from({length: 16}, () => []), projection: []};
  const insert = createGemma4V2ImageInserterWithWeights(runtime, device, vm, weights);
  const word = tensor([1, 5, 1536]);
  const result = await insert(word, {
    stateIds: [258880, 1, 258880, 2],
    questionIds: [3],
    imageSpans: [
      {start: 0, end: 1},
      {start: 2, end: 3}
    ],
    images: [prepared(), prepared()]
  });
  assert.equal(result, insertions[1]);
  assert.equal(insertions[0]!.disposed, true);
  assert.equal(insertions[1]!.disposed, false);
  assert.equal(word.disposed, false);
  assert.deepEqual(events, ['begin', ...Array(9).fill('sync'), 'end', 'begin', ...Array(9).fill('sync'), 'end']);
  assert.equal(scopes.length, 1);
  assert.equal(scopes[0]!.filter((value) => !value.disposed).length, 2);
});

test('lazy vision weights retry failures, respect cancellation and reject concurrent use', async () => {
  const device = {deviceType: 15, deviceId: 0, async sync() {}} satisfies DLDevice;
  type FakeTensor = Tensor & {disposed: boolean; dispose(): void};
  const scopes: FakeTensor[][] = [[]];
  const tensor = (shape: number[], dtype = 'float32'): FakeTensor => {
    const value = {
      shape,
      dtype,
      device,
      disposed: false,
      copyFrom() {
        return this;
      },
      copyFromRawBytes() {
        return this;
      },
      toArray() {
        throw new Error('Unexpected CPU read');
      },
      toRawBytes() {
        throw new Error('Unexpected CPU read');
      },
      dispose() {
        this.disposed = true;
      }
    } satisfies FakeTensor;
    scopes.at(-1)!.push(value);
    return value;
  };
  const runtime = {
    beginScope() {
      scopes.push([]);
    },
    endScope() {
      for (const value of scopes.pop()!) value.dispose();
    },
    moveToParentScope(value: FakeTensor) {
      const current = scopes.at(-1)!;
      const index = current.indexOf(value);
      if (index < 0 || scopes.length < 2) throw new Error('Wrong tensor scope');
      current.splice(index, 1);
      scopes.at(-2)!.push(value);
      return value;
    },
    detachFromCurrentScope(value: FakeTensor) {
      const current = scopes.at(-1)!;
      const index = current.indexOf(value);
      assert.notEqual(index, -1);
      current.splice(index, 1);
      return value;
    },
    empty(shape: number[] | number, dtype: string) {
      return tensor(typeof shape === 'number' ? [shape] : shape, dtype);
    }
  } as unknown as Instance;
  const vm = {
    getFunction(name: string) {
      return () => {
        if (name === 'vision_pool_main') return tensor([1, 768]);
        if (name === 'vision_projection_main') return tensor([1, 1536]);
        if (name === 'image_insert_main') return tensor([1, 2, 1536]);
        return tensor([1, 2520, 768]);
      };
    }
  } satisfies VirtualMachine;
  let attempts = 0;
  let finishLoad: ((weights: Gemma4V2VisionWeights) => void) | undefined;
  const loader = async (): Promise<Gemma4V2VisionWeights> => {
    attempts++;
    if (attempts === 1) throw new Error('network failed');
    return new Promise((resolve) => {
      finishLoad = resolve;
    });
  };
  const insert = await createGemma4V2ImageInserter(runtime, device, vm, {} as never, undefined, loader);
  const word = tensor([1, 2, 1536]);
  const textOnly = {stateIds: [1], questionIds: [2], imageSpans: [], images: []};
  assert.equal(await insert(word, textOnly), word);
  assert.equal(attempts, 0);
  const input = {stateIds: [258880], questionIds: [2], imageSpans: [{start: 0, end: 1}], images: [prepared()]};
  runtime.beginScope();
  await assert.rejects(insert(word, input), /network failed/);
  runtime.endScope();
  runtime.beginScope();
  const controller = new AbortController();
  const cancelled = insert(word, input, controller.signal);
  controller.abort();
  finishLoad!({patch: [tensor([1])], layers: Array.from({length: 16}, () => []), projection: []});
  await assert.rejects(cancelled, {name: 'AbortError'});
  runtime.endScope();
  runtime.beginScope();
  const first = insert(word, input);
  await assert.rejects(insert(word, input), /already in use/);
  const weight = tensor([1]);
  finishLoad!({patch: [weight], layers: Array.from({length: 16}, () => []), projection: []});
  await first;
  assert.equal(attempts, 3);
  runtime.endScope();
  assert.equal(weight.disposed, false);
  runtime.beginScope();
  await insert(word, input);
  runtime.endScope();
  assert.equal(attempts, 3);
  assert.equal(weight.disposed, false);
  insert.releaseCachedMemory();
  assert.equal(weight.disposed, true);
  insert.releaseCachedMemory();
  runtime.beginScope();
  const reloaded = insert(word, input);
  const nextWeight = tensor([1]);
  finishLoad!({patch: [nextWeight], layers: Array.from({length: 16}, () => []), projection: []});
  await reloaded;
  runtime.endScope();
  assert.equal(attempts, 4);
  assert.equal(nextWeight.disposed, false);
  insert.releaseCachedMemory();
  assert.equal(nextWeight.disposed, true);
});

test('image inference preserves its failure when scope cleanup also fails', async () => {
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
  const vm = {
    getFunction() {
      return () => {
        throw primary;
      };
    }
  } satisfies VirtualMachine;
  const weights = {patch: [], layers: Array.from({length: 16}, () => []), projection: []};
  const insert = createGemma4V2ImageInserterWithWeights(runtime, device, vm, weights);
  const request = {stateIds: [258880], questionIds: [2], imageSpans: [{start: 0, end: 1}], images: [prepared()]};
  await assert.rejects(insert({} as Tensor, request), (error) => error === primary);
  assert.equal(ended, 1);
});

test('image synchronization failure remains primary and cleanup-only errors surface', async () => {
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
      }
    }) as unknown as Tensor;
  const runtime = {
    beginScope() {},
    endScope() {
      throw cleanup;
    },
    moveToParentScope(value: Tensor) {
      return value;
    },
    empty(shape: number[] | number, dtype: string) {
      return tensor(typeof shape === 'number' ? [shape] : shape, dtype);
    }
  } as unknown as Instance;
  const vm = {
    getFunction(name: string) {
      return () => tensor(name === 'vision_projection_main' ? [1, 1536] : [1, 2, 1536]);
    }
  } satisfies VirtualMachine;
  const insert = createGemma4V2ImageInserterWithWeights(runtime, device, vm, {
    patch: [],
    layers: Array.from({length: 16}, () => []),
    projection: []
  });
  const request = {stateIds: [258880], questionIds: [2], imageSpans: [{start: 0, end: 1}], images: [prepared()]};
  const word = tensor([1, 2, 1536]);
  await assert.rejects(insert(word, request), (error) => error === primary);
  failSync = false;
  await assert.rejects(insert(word, request), (error) => error === cleanup);
});
