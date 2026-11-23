import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createGemma4V2Runner} from '../src/strands/gemma4-v2-runner.mjs';
import type {Gemma4V2RunnerInput} from '../src/strands/gemma4-v2-session.mjs';

function setup() {
  const scopes: Array<Set<FakeTensor>> = [];
  const calls: Array<{name: string; args: unknown[]}> = [];
  const tupleOutputs: FakeTensor[][] = [];
  let releases = 0;
  let disposed = 0;
  let cachedReleases = 0;
  const embeddingIds: number[][] = [];
  const device = {deviceType: 15, deviceId: 0, async sync() {}};
  class FakeTensor {
    readonly device = device;
    disposed = false;
    constructor(
      readonly shape: number[],
      readonly dtype: string
    ) {
      scopes.at(-1)?.add(this);
    }
    copied: unknown;
    copyFrom(data?: unknown) {
      this.copied = data;
      return this;
    }
    toArray() {
      return new Float32Array(255).fill(7);
    }
    dispose() {
      assert.equal(this.disposed, false);
      this.disposed = true;
      releases++;
    }
  }
  const runtime = {
    beginScope() {
      scopes.push(new Set<FakeTensor>());
    },
    endScope() {
      for (const value of scopes.pop()!) value.dispose();
    },
    detachFromCurrentScope(value: FakeTensor) {
      assert.ok(scopes.at(-1)?.delete(value));
      return value;
    },
    empty(shape: number[], dtype: string) {
      return new FakeTensor(shape, dtype);
    },
    cpu() {
      return device;
    }
  };
  const vm = {
    getFunction(name: string) {
      return (...args: unknown[]) => {
        calls.push({name, args});
        if (name.startsWith('layer')) {
          for (const arg of args) if (arg instanceof FakeTensor) assert.equal(arg.disposed, false);
        }
        if (name === 'layer0_packed_main' || name === 'layer4_packed_main') {
          const values = Array.from({length: 3}, () => new FakeTensor([1], 'float32'));
          tupleOutputs.push(values);
          return {size: () => 3, get: (index: number) => values[index]!};
        }
        return new FakeTensor(name === 'readout_packed_main' ? [1, 255] : [1], 'float32');
      };
    }
  };
  const source = {
    releaseCachedMemory() {
      cachedReleases++;
    },
    async readEmbeddingRows(ids: readonly number[]) {
      embeddingIds.push([...ids]);
      return new Uint32Array(1);
    },
    async readTensorRange() {
      return new Uint8Array(4);
    },
    async readTensor(name: string) {
      return new Uint8Array(name.endsWith('layer_scalar') ? [0, 128] : [0, 0, 0, 0]);
    }
  };
  const deps = {
    runtime,
    device,
    vm,
    source,
    readout: new Uint8Array(255 * 1536 * 2),
    localFrequency: new Float32Array(128),
    globalFrequency: new Float32Array(256),
    async insertImages(word: FakeTensor) {
      return word;
    },
    dispose() {
      disposed++;
    }
  };
  return {
    deps,
    calls,
    tupleOutputs,
    scopes,
    embeddingIds,
    get disposed() {
      return disposed;
    },
    get releases() {
      return releases;
    },
    get cachedReleases() {
      return cachedReleases;
    }
  };
}
const request: Gemma4V2RunnerInput = {stateIds: [2, 4, 2], questionIds: [7], imageSpans: [], images: []};

test('releases source and image caches without closing the runner', async () => {
  const state = setup();
  let imageReleases = 0;
  const insertImages = Object.assign(state.deps.insertImages, {
    releaseCachedMemory() {
      imageReleases++;
    }
  });
  const runner = await createGemma4V2Runner({...state.deps, insertImages} as never);
  await runner.releaseCachedMemory();
  assert.equal(state.cachedReleases, 1);
  assert.equal(imageReleases, 1);
  await runner.dispose();
  await assert.rejects(async () => runner.releaseCachedMemory(), /disposed/i);
});

test('runs all 35 layers with shared attention state and readout, and releases TVM scopes', async () => {
  const fake = setup();
  const runner = await createGemma4V2Runner(fake.deps as never);
  const logits = await runner.run(request);
  assert.equal(logits.length, 255);
  assert.equal(logits[0], 7);
  const layers = fake.calls.filter((call) => call.name.startsWith('layer'));
  assert.equal(layers.length, 35);
  assert.deepEqual(
    layers.map((call) => call.name),
    Array.from({length: 35}, (_, i) => `layer${i < 15 ? (i % 5 === 4 ? 4 : 0) : i % 5 === 4 ? 19 : 15}_packed_main`)
  );
  assert.equal(layers[15]!.args.length, 9 + 13);
  assert.equal(layers[19]!.args.length, 9 + 13);
  for (let index = 15; index < 35; index++) {
    const saved = fake.tupleOutputs[index % 5 === 4 ? 14 : 13]!;
    assert.equal(layers[index]!.args[7], saved[1]);
    assert.equal(layers[index]!.args[8], saved[2]);
  }
  assert.equal(fake.calls.at(-1)!.name, 'readout_packed_main');
  assert.equal(fake.scopes.length, 0);
  assert.ok(fake.releases > 35);
  await runner.dispose();
  await runner.dispose();
  assert.equal(fake.disposed, 1);
  await assert.rejects(runner.run(request), /disposed/);
});

test('releases scoped resources after a source failure', async () => {
  const fake = setup();
  let reads = 0;
  fake.deps.source.readTensor = async (name: string) => {
    if (++reads === 30) throw new Error('source failed');
    return new Uint8Array(name.endsWith('layer_scalar') ? [0, 128] : [0, 0, 0, 0]);
  };
  const runner = await createGemma4V2Runner(fake.deps as never);
  await assert.rejects(runner.run(request), /source failed/);
  assert.equal(fake.scopes.length, 0);
  await runner.dispose();
});

test('uses pad token identity and image-inserted context for PLE', async () => {
  const fake = setup();
  let word: unknown;
  let merged: unknown;
  fake.deps.insertImages = async (tensor) => {
    word = tensor;
    merged = fake.deps.runtime.empty([1, 6, 1536], 'float32');
    return merged as typeof tensor;
  };
  const runner = await createGemma4V2Runner(fake.deps as never);
  await runner.run({
    stateIds: [2, 258880, 258880, 3],
    questionIds: [7, 2],
    imageSpans: [{start: 1, end: 3}],
    images: [{}]
  });
  assert.deepEqual(fake.embeddingIds[0], [2, 0, 3, 7]);
  for (const ids of fake.embeddingIds) assert.deepEqual(ids, [2, 0, 3, 7]);
  const ple = fake.calls.filter((call) => call.name === 'ple_packed_main');
  assert.equal(ple.length, 35);
  assert.notEqual(merged, word);
  for (const call of ple) assert.equal(call.args[0], merged);
  assert.equal(fake.calls.find((call) => call.name === 'layer0_packed_main')!.args[0], merged);
  await runner.dispose();
});

test('cancels between decoder layers and releases retained GPU resources', async () => {
  const fake = setup();
  const controller = new AbortController();
  let reads = 0;
  fake.deps.source.readTensor = async (name: string) => {
    if (++reads === 30) controller.abort(new Error('cancelled by caller'));
    return new Uint8Array(name.endsWith('layer_scalar') ? [0, 128] : [0, 0, 0, 0]);
  };
  const runner = await createGemma4V2Runner(fake.deps as never);
  await assert.rejects(runner.run(request, controller.signal), /cancelled by caller/);
  assert.equal(fake.scopes.length, 0);
  const layers = fake.calls.filter((call) => call.name.startsWith('layer'));
  assert.ok(layers.length > 0 && layers.length < 35);
  await runner.dispose();
});

test('rejects an invalid image span before loading embedding rows', async () => {
  const fake = setup();
  const runner = await createGemma4V2Runner(fake.deps as never);
  await assert.rejects(
    runner.run({stateIds: [2, 42], questionIds: [7], imageSpans: [{start: 1, end: 2}], images: [{}]}),
    /image placeholders/i
  );
  assert.equal(fake.embeddingIds.length, 0);
  assert.equal(fake.scopes.length, 0);
  await runner.dispose();
});

test('refuses lengths beyond the compiled TVM graph before GPU allocation', async () => {
  const fake = setup();
  const runner = await createGemma4V2Runner(fake.deps as never);
  await assert.rejects(
    runner.run({stateIds: Array(131073).fill(1), questionIds: [], imageSpans: [], images: []}),
    /compiled.*131072/i
  );
  assert.equal(fake.embeddingIds.length, 0);
  assert.equal(fake.calls.length, 0);
  assert.equal(fake.scopes.length, 0);
  await runner.dispose();
});

test('releases retained layer outputs when GPU synchronization fails', async () => {
  const fake = setup();
  let syncs = 0;
  fake.deps.device.sync = async () => {
    if (++syncs === 15) throw new Error('device lost');
  };
  const runner = await createGemma4V2Runner(fake.deps as never);
  await assert.rejects(runner.run(request), /device lost/);
  assert.equal(syncs, 15);
  assert.equal(fake.scopes.length, 0);
  assert.ok(fake.tupleOutputs[13]![1]!.disposed);
  assert.ok(fake.tupleOutputs[14]![1]!.disposed);
  await runner.dispose();
});

test('preserves the original GPU error when cleanup also throws', async () => {
  const fake = setup();
  const reason = new Error('original device lost');
  let syncs = 0;
  fake.deps.device.sync = async () => {
    if (++syncs === 15) throw reason;
  };
  const endScope = fake.deps.runtime.endScope;
  fake.deps.runtime.endScope = () => {
    endScope();
    if (syncs === 15) throw new Error('cleanup unreachable');
  };
  const runner = await createGemma4V2Runner(fake.deps as never);
  await assert.rejects(runner.run(request), (error) => error === reason);
  assert.equal(fake.scopes.length, 0);
  await runner.dispose();
});

test('reports a cleanup failure when computation itself succeeded', async () => {
  const fake = setup();
  const reason = new Error('scope cleanup failed');
  const endScope = fake.deps.runtime.endScope;
  fake.deps.runtime.endScope = () => {
    endScope();
    throw reason;
  };
  const runner = await createGemma4V2Runner(fake.deps as never);
  await assert.rejects(runner.run(request), (error) => error === reason);
  assert.equal(fake.scopes.length, 0);
  await runner.dispose();
});

test('uploads the prefetched layer slices and uses the per-layer fallback independently on the next run', async () => {
  const fake = setup();
  const layers = Array.from({length: 35}, (_, layer) => new Uint32Array(3 * 128).fill(layer + 1));
  const prefetches: number[][] = [];
  const source = {
    ...fake.deps.source,
    async readAllLayerEmbeddingRows(ids: readonly number[], budget: number) {
      prefetches.push([...ids]);
      assert.equal(budget, 32 * 1024 * 1024);
      return prefetches.length === 1 ? layers : undefined;
    }
  };
  const runner = await createGemma4V2Runner({...fake.deps, source} as never);
  try {
    await runner.run(request);
    assert.deepEqual(prefetches, [[2, 4, 7]]);
    assert.deepEqual(fake.embeddingIds, [[2, 4, 7]]);
    const uploads = fake.calls.filter(({name}) => name === 'ple_packed_main');
    assert.deepEqual(
      uploads.map(({args}) => (args[2] as {copied: unknown}).copied),
      layers
    );

    await runner.run(request);
    assert.deepEqual(prefetches, [
      [2, 4, 7],
      [2, 4, 7]
    ]);
    assert.equal(fake.embeddingIds.length, 37);
    assert.equal(fake.scopes.length, 0);
  } finally {
    await runner.dispose();
  }
});

test('releases scopes and propagates a prefetch failure without silently retrying per layer', async () => {
  const fake = setup();
  const error = new Error('PLE range unavailable');
  const source = {
    ...fake.deps.source,
    async readAllLayerEmbeddingRows() {
      throw error;
    }
  };
  const runner = await createGemma4V2Runner({...fake.deps, source} as never);
  try {
    await assert.rejects(runner.run(request), (reason) => reason === error);
    assert.deepEqual(fake.embeddingIds, [[2, 4, 7]]);
    assert.equal(
      fake.calls.some(({name}) => name === 'ple_packed_main'),
      false
    );
    assert.equal(fake.scopes.length, 0);
  } finally {
    await runner.dispose();
  }
});

test('does not upload prefetched rows when cancellation arrives as the projection read completes', async () => {
  const fake = setup();
  const controller = new AbortController();
  const cancelled = new Error('cancelled after projection read');
  const source = {
    ...fake.deps.source,
    async readAllLayerEmbeddingRows() {
      return Array.from({length: 35}, () => new Uint32Array(3 * 128));
    },
    async readTensorRange() {
      queueMicrotask(() => controller.abort(cancelled));
      return new Uint8Array(4);
    },
    async readTensor(name: string, signal?: AbortSignal | null) {
      signal?.throwIfAborted();
      return fake.deps.source.readTensor(name);
    }
  };
  const runner = await createGemma4V2Runner({...fake.deps, source} as never);
  try {
    await assert.rejects(runner.run(request, controller.signal), (reason) => reason === cancelled);
    assert.equal(
      fake.calls.some(({name}) => name === 'ple_packed_main'),
      false
    );
    assert.equal(fake.scopes.length, 0);
  } finally {
    await runner.dispose();
  }
});
