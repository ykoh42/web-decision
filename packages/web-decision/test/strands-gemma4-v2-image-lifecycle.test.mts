import assert from 'node:assert/strict';
import {test} from 'node:test';
import type {DLDevice, Instance, Tensor, VirtualMachine} from '../src/runtime/tvm-types.js';
import {createGemma4V2ImageInserter} from '../src/strands/gemma4-v2-image-runner.mts';
import {createGemma4V2Session, type Gemma4V2RunnerInput} from '../src/strands/gemma4-v2-session.mts';

function image() {
  const positions = new Int32Array(2520 * 2).fill(-1);
  for (let y = 0; y < 3; y++)
    for (let x = 0; x < 3; x++) {
      positions[(y * 3 + x) * 2] = x;
      positions[(y * 3 + x) * 2 + 1] = y;
    }
  return {pixels: new Float32Array(2520 * 768), positions, tokens: 1};
}

test('multi-image session releases cancelled work, retries, and disposes retained weights', async () => {
  type FakeTensor = Tensor & {disposed: boolean; dispose(): void};
  const scopes: FakeTensor[][] = [[]];
  const live = new Set<FakeTensor>();
  let syncs = 0;
  let releaseSecond!: () => void;
  let reachedSecond!: () => void;
  const secondStarted = new Promise<void>((resolve) => {
    reachedSecond = resolve;
  });
  const secondGate = new Promise<void>((resolve) => {
    releaseSecond = resolve;
  });
  const device = {
    deviceType: 15,
    deviceId: 0,
    async sync() {
      if (++syncs === 2) {
        reachedSecond();
        await secondGate;
      }
    }
  } satisfies DLDevice;
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
        throw new Error('Unexpected CPU copy');
      },
      toRawBytes() {
        throw new Error('Unexpected CPU copy');
      },
      dispose() {
        this.disposed = true;
        live.delete(this);
      }
    } satisfies FakeTensor;
    live.add(value);
    scopes.at(-1)!.push(value);
    return value;
  };
  const runtime = {
    beginScope() {
      scopes.push([]);
    },
    endScope() {
      for (const item of scopes.pop()!) item.dispose();
    },
    moveToParentScope(value: FakeTensor) {
      const current = scopes.at(-1)!;
      const index = current.indexOf(value);
      if (index < 0 || scopes.length < 2) throw new Error('Incorrect scope');
      current.splice(index, 1);
      scopes.at(-2)!.push(value);
      return value;
    },
    empty(shape: number[] | number, dtype: string) {
      return tensor(typeof shape === 'number' ? [shape] : shape, dtype);
    }
  } as unknown as Instance;
  const vm = {
    getFunction(name: string) {
      return (...args: unknown[]) => {
        if (name === 'vision_pool_main') return tensor([1, 768]);
        if (name === 'vision_projection_main') return tensor([1, 1536]);
        if (name === 'image_insert_main') return tensor([...(args[0] as Tensor).shape]);
        return tensor([1, 2520, 768]);
      };
    }
  } satisfies VirtualMachine;
  let loads = 0;
  let weight: FakeTensor | undefined;
  const loader = async () => {
    loads++;
    weight = tensor([1]);
    return {patch: [weight], layers: Array.from({length: 16}, () => []), projection: []};
  };
  const insertImages = await createGemma4V2ImageInserter(runtime, device, vm, {} as never, undefined, loader);
  let runs = 0;
  const runner = {
    async run(input: Gemma4V2RunnerInput, signal?: AbortSignal | null) {
      runs++;
      runtime.beginScope();
      try {
        const word = tensor([1, input.stateIds.length + input.questionIds.length, 1536]);
        await insertImages(word, input, signal);
        return [0, 0];
      } finally {
        runtime.endScope();
      }
    },
    releaseCachedMemory() {},
    dispose() {
      runtime.endScope();
    }
  };
  const session = createGemma4V2Session({
    tokenizer: {
      encode(text) {
        return {ids: Array.from(new TextEncoder().encode(text))};
      }
    },
    runner,
    prepareImage: async () => ({tokens: 1, value: image()}),
    modelName: 'e2b-v2',
    temperature: 1.5456,
    softcap: 30
  });
  const request = {state: 's', images: ['first', 'second'], questions: {q: {instructions: 'yes?'}}};
  const controller = new AbortController();
  const cancelled = session.systemOne(request, {signal: controller.signal});
  await secondStarted;
  controller.abort();
  await assert.rejects(cancelled, {name: 'AbortError'});
  const retry = session.systemOne(request);
  assert.equal(runs, 1, 'retry waits until the cancelled GPU scope is cleaned up');
  releaseSecond();
  const result = await retry;
  assert.equal(result.answers.q?.type, 'noul');
  assert.equal(runs, 2);
  assert.equal(loads, 1, 'verified weights are reused after cancellation');
  assert.equal(live.size, 1, 'only the session-owned weight survives requests');
  assert.equal(weight?.disposed, false);
  await session.dispose();
  assert.equal(live.size, 0);
  assert.equal(weight?.disposed, true);
  await assert.rejects(session.systemOne(request), /disposed|closed/i);
});
