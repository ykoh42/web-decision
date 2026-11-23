import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test, {type TestContext} from 'node:test';
import type {AssetCache} from '../src/assets.js';
import {createGemma4V2OfflinePreparation} from '../src/strands/gemma4-v2-offline.mjs';
import {openGemma4V2SourceFromInventory, openPinnedGemma4V2Source} from '../src/strands/gemma4-v2-source.mjs';

const base = new URL('https://example.test/resolve/');
const hash = (bytes: Uint8Array | string) => createHash('sha256').update(bytes).digest('hex');

function fixture() {
  const headerJson = Buffer.from(
    JSON.stringify({
      'model.language_model.first': {dtype: 'BF16', shape: [3], data_offsets: [0, 6]},
      'model.vision_tower.second': {dtype: 'BF16', shape: [2], data_offsets: [6, 10]},
      'model.audio_tower.unused': {dtype: 'BF16', shape: [2], data_offsets: [10, 14]}
    })
  );
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(headerJson.length));
  const header = Buffer.concat([prefix, headerJson]);
  const bytes = Buffer.concat([header, Buffer.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])]);
  const boundaries = [0, header.length - 3, header.length + 4, header.length + 10, bytes.length];
  const chunks: Array<[number, number, string]> = [];
  for (let index = 0; index < boundaries.length - 1; index++) {
    const start = boundaries[index]!;
    const end = boundaries[index + 1]!;
    chunks.push([start, end - start, hash(bytes.subarray(start, end))]);
  }
  const inventory = {
    format: 'web-decision-strands-gemma4-v2-source-v1',
    repository: 'StrandsAgents/strands-decider-E2B-gemma4-v2-2611',
    revision: 'a5f32ea1bffe2e91c25c0c641d444e8115e28212',
    file: 'model.safetensors',
    bytes: bytes.length,
    sha256: hash(bytes),
    headerBytes: header.length,
    chunks,
    tensors: {
      'model.language_model.first': {offset: header.length, length: 6, shape: [3], dtype: 'BF16'},
      'model.vision_tower.second': {offset: header.length + 6, length: 4, shape: [2], dtype: 'BF16'},
      'model.audio_tower.unused': {offset: header.length + 10, length: 4, shape: [2], dtype: 'BF16'}
    }
  };
  return {bytes, inventory, digest: hash(JSON.stringify(inventory) + '\n')};
}

class MemoryCache implements AssetCache {
  readonly entries = new Map<string, Response>();
  async match(key: string) {
    return this.entries.get(key)?.clone();
  }
  async put(key: string, response: Response) {
    this.entries.set(key, response.clone());
  }
}

test('pinned weight requests stay under the immutable revision directory', async (t) => {
  let requested = '';
  t.mock.method(globalThis, 'fetch', async (input: string) => {
    requested = String(input);
    throw new Error('stop before downloading');
  });
  const source = await openPinnedGemma4V2Source();
  await assert.rejects(source.readTensor('model.language_model.norm.weight'));
  assert.equal(
    requested,
    'https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v2-2611/resolve/a5f32ea1bffe2e91c25c0c641d444e8115e28212/model.safetensors'
  );
  source.dispose();
});

function transport(t: TestContext, bytes: Buffer) {
  const requested: string[] = [];
  t.mock.method(globalThis, 'fetch', async (_url: string, init?: RequestInit) => {
    const range = new Headers(init?.headers).get('range');
    const match = /^bytes=(\d+)-(\d+)$/u.exec(range ?? '');
    assert.ok(match);
    const start = Number(match[1]);
    const end = Number(match[2]);
    requested.push(range!);
    return new Response(Uint8Array.from(bytes.subarray(start, end + 1)), {
      status: 206,
      headers: {'content-range': `bytes ${start}-${end}/${bytes.length}`}
    });
  });
  return requested;
}

test('reads verified tensors across chunk and header boundaries and reuses cached bytes offline', async (t) => {
  const data = fixture();
  const cache = new MemoryCache();
  const requested = transport(t, data.bytes);
  const source = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base, {cache});
  assert.deepEqual([...(await source.readTensor('model.language_model.first'))], [1, 2, 3, 4, 5, 6]);
  assert.deepEqual([...(await source.readTensor('model.vision_tower.second'))], [7, 8, 9, 10]);
  assert.deepEqual([...(await source.readTensorRange('model.language_model.first', 1, 4))], [2, 3, 4, 5]);
  await assert.rejects(source.readTensorRange('model.language_model.first', 5, 2), /range/i);
  assert.equal(requested.length, 3);
  const offline = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base, {cache, offline: true});
  assert.deepEqual([...(await offline.readTensor('model.language_model.first'))], [1, 2, 3, 4, 5, 6]);
  assert.equal(requested.length, 3);
  cache.entries.set([...cache.entries.keys()][0]!, new Response(Uint8Array.of(0)));
  const corrupted = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base, {cache, offline: true});
  await assert.rejects(corrupted.readTensor('model.language_model.first'), /offline/i);
  assert.equal(requested.length, 3);
});

test('releases resident chunks without removing the persistent cache', async (t) => {
  const data = fixture();
  const cache = new MemoryCache();
  const requested = transport(t, data.bytes);
  const source = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base, {cache});
  await source.readTensor('model.language_model.first');
  const count = requested.length;
  const cacheReads: string[] = [];
  const match = cache.match.bind(cache);
  t.mock.method(cache, 'match', async (key: string) => {
    cacheReads.push(key);
    return match(key);
  });
  await source.readTensor('model.language_model.first');
  assert.equal(cacheReads.length, 0);
  source.releaseCachedMemory();
  assert.deepEqual([...(await source.readTensor('model.language_model.first'))], [1, 2, 3, 4, 5, 6]);
  assert.ok(cacheReads.length > 0);
  assert.equal(requested.length, count);
  source.dispose();
  assert.throws(() => source.releaseCachedMemory(), /closed/i);
});

test('a pending fetch does not repopulate memory after release', async (t) => {
  const data = fixture();
  transport(t, data.bytes);
  const source = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base);
  await source.readTensor('model.language_model.first');
  source.releaseCachedMemory();
  t.mock.restoreAll();
  let finish!: () => void;
  let started!: () => void;
  const pending = new Promise<void>((resolve) => (finish = resolve));
  const ready = new Promise<void>((resolve) => (started = resolve));
  let requests = 0;
  t.mock.method(globalThis, 'fetch', async (_url: string, init?: RequestInit) => {
    const match = /^bytes=(\d+)-(\d+)$/u.exec(new Headers(init?.headers).get('range') ?? '');
    assert.ok(match);
    const start = Number(match[1]);
    const end = Number(match[2]);
    requests++;
    if (requests === 1) {
      started();
      await pending;
    }
    return new Response(Uint8Array.from(data.bytes.subarray(start, end + 1)), {
      status: 206,
      headers: {'content-range': `bytes ${start}-${end}/${data.bytes.length}`}
    });
  });
  const read = source.readTensorRange('model.language_model.first', 0, 1);
  await ready;
  source.releaseCachedMemory();
  finish();
  assert.deepEqual([...(await read)], [1]);
  assert.deepEqual([...(await source.readTensorRange('model.language_model.first', 0, 1))], [1]);
  assert.equal(requests, 2);
  source.dispose();
});

test('prepares only required weights for offline use and detects eviction', async (t) => {
  const data = fixture();
  const cache = new MemoryCache();
  const requests = transport(t, data.bytes);
  const source = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base, {cache});
  assert.equal(await source.hasCompleteOfflineWeights(), false);
  const progress: number[] = [];
  await source.prepareWeightsOffline({
    onProgress: (current, total) => {
      assert.ok(current <= total);
      progress.push(current);
    }
  });
  assert.equal(progress.length, 3);
  assert.equal(requests.length, 3);
  assert.equal(await source.hasCompleteOfflineWeights(), true);
  const offline = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base, {cache, offline: true});
  assert.deepEqual([...(await offline.readTensor('model.vision_tower.second'))], [7, 8, 9, 10]);
  await assert.rejects(offline.prepareWeightsOffline(), /network/i);
  const rangeKey = [...cache.entries.keys()].find((key) => key.includes('sha256'))!;
  const saved = cache.entries.get(rangeKey)!.clone();
  const corrupted = new Uint8Array(await saved.clone().arrayBuffer());
  corrupted[0] = corrupted[0]! ^ 1;
  cache.entries.set(rangeKey, new Response(corrupted));
  assert.equal(await source.hasCompleteOfflineWeights(), true);
  assert.equal(await source.verifyOfflineWeights(), false);
  const damagedOffline = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base, {
    cache,
    offline: true
  });
  await assert.rejects(damagedOffline.readTensor('model.language_model.first'), /offline/i);
  cache.entries.set(rangeKey, saved);
  assert.equal(await source.hasCompleteOfflineWeights(), true);
  assert.equal(await source.verifyOfflineWeights(), true);
  cache.entries.delete(rangeKey);
  assert.equal(await source.hasCompleteOfflineWeights(), false);
});

test('rejects altered inventory, inconsistent headers, and corrupted network ranges', async (t) => {
  const data = fixture();
  await assert.rejects(openGemma4V2SourceFromInventory(data.inventory, '0'.repeat(64), base), /digest/i);
  const altered = structuredClone(data.inventory);
  altered.tensors['model.language_model.first'].shape = [2];
  const alteredHash = hash(JSON.stringify(altered) + '\n');
  transport(t, data.bytes);
  const mismatched = await openGemma4V2SourceFromInventory(altered, alteredHash, base);
  await assert.rejects(mismatched.readTensor('model.language_model.first'), /header|tensor/i);
  t.mock.restoreAll();
  const bad = Buffer.from(data.bytes);
  bad[0] = bad[0]! ^ 1;
  transport(t, bad);
  const source = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base);
  await assert.rejects(source.readTensor('model.language_model.first'), /integrity/i);
});

test('cancels pending range reads when the request, source signal, or source is closed', async (t) => {
  for (const action of ['request', 'source', 'dispose'] as const) {
    const data = fixture();
    const sourceController = new AbortController();
    const requestController = new AbortController();
    let started!: () => void;
    const waiting = new Promise<void>((resolve) => {
      started = resolve;
    });
    t.mock.method(globalThis, 'fetch', async (_url: string, init?: RequestInit) => {
      const signal = init?.signal;
      assert.ok(signal);
      started();
      return new Promise<Response>((_resolve, reject) => {
        if (signal.aborted) reject(signal.reason);
        else signal.addEventListener('abort', () => reject(signal.reason), {once: true});
      });
    });
    const source = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base, {
      signal: sourceController.signal
    });
    const pending = source.readTensor('model.language_model.first', requestController.signal);
    await waiting;
    if (action === 'request') requestController.abort(new Error('request cancelled'));
    else if (action === 'source') sourceController.abort(new Error('source cancelled'));
    else source.dispose();
    await assert.rejects(pending, /cancelled|closed/i);
    if (action === 'dispose') await assert.rejects(source.readTensor('model.language_model.first'), /closed/i);
    source.dispose();
    t.mock.restoreAll();
  }
});

test('offline preparation never commits a partial or rejected cache', async (t) => {
  const data = fixture();
  const cache = new MemoryCache();
  const requested = transport(t, data.bytes);
  const source = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base, {cache});
  const controller = new AbortController();
  await assert.rejects(
    source.prepareWeightsOffline({
      signal: controller.signal,
      onProgress: () => controller.abort(new Error('cancelled preparation'))
    }),
    /cancelled preparation/i
  );
  assert.equal(await source.hasCompleteOfflineWeights(), false);
  assert.ok(requested.length < data.inventory.chunks.length);

  const denied: AssetCache = {
    match: async () => undefined,
    put: async () => {
      throw new Error('storage full');
    }
  };
  const noStorage = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base, {cache: denied});
  await assert.rejects(noStorage.prepareWeightsOffline(), /cached/i);
  assert.equal(await noStorage.hasCompleteOfflineWeights(), false);
});

test('explicit offline verification can be cancelled while a cached range read is pending', async (t) => {
  const data = fixture();
  const cache = new MemoryCache();
  transport(t, data.bytes);
  const source = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base, {cache});
  await source.prepareWeightsOffline();
  const originalMatch = cache.match.bind(cache);
  const targetHash = data.inventory.chunks[0]![2];
  let requests = 0;
  let started!: () => void;
  let release!: () => void;
  const waiting = new Promise<void>((resolve) => {
    started = resolve;
  });
  const blocked = new Promise<void>((resolve) => {
    release = resolve;
  });
  cache.match = async (key: string) => {
    if (key.includes(targetHash) && ++requests === 2) {
      started();
      await blocked;
    }
    return originalMatch(key);
  };
  const controller = new AbortController();
  const verifying = source.verifyOfflineWeights(controller.signal);
  await waiting;
  controller.abort(new Error('verification cancelled'));
  await assert.rejects(verifying, /verification cancelled/i);
  release();
  assert.equal(await source.verifyOfflineWeights(), true);
  source.dispose();
});

test('offline coordinator prepares the actual range source and detects missing ranges', async (t) => {
  const data = fixture();
  const cache = new MemoryCache();
  const contents = new Map<string, Uint8Array>();
  const asset = (name: string) => {
    const bytes = new TextEncoder().encode(name);
    contents.set(new URL(name, base).href, bytes);
    return {base, name, metadata: {bytes: bytes.length, sha256: hash(bytes)}};
  };
  const assets = {
    tokenizer: asset('tokenizer.json'),
    tokenizerConfig: asset('tokenizer_config.json'),
    readout: asset('readout.safetensors'),
    modelConfig: asset('config.json'),
    decisionConfig: asset('decision_config.json'),
    processorConfig: asset('processor_config.json'),
    artifacts: [asset('model.wasm')]
  };
  let requests = 0;
  t.mock.method(globalThis, 'fetch', async (input: string, init?: RequestInit) => {
    requests++;
    if (!String(input).endsWith('/model.safetensors')) {
      const bytes = contents.get(String(input));
      assert.ok(bytes);
      return new Response(Uint8Array.from(bytes));
    }
    const range = new Headers(init?.headers).get('range');
    const match = /^bytes=(\d+)-(\d+)$/u.exec(range ?? '');
    assert.ok(match);
    const start = Number(match[1]);
    const end = Number(match[2]);
    return new Response(Uint8Array.from(data.bytes.subarray(start, end + 1)), {
      status: 206,
      headers: {'content-range': `bytes ${start}-${end}/${data.bytes.length}`}
    });
  });
  let imageReady = false;
  const images = {
    async isOfflineReady() {
      return imageReady;
    },
    async prepareOffline() {
      imageReady = true;
    }
  };
  const source = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base, {cache});
  const preparation = createGemma4V2OfflinePreparation(source, cache, assets, images);
  await preparation.prepareOffline();
  assert.equal(await preparation.isOfflineReady(), true);
  assert.equal(await source.verifyOfflineWeights(), true);
  const onlineRequests = requests;
  const offline = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base, {cache, offline: true});
  assert.deepEqual([...(await offline.readTensor('model.language_model.first'))], [1, 2, 3, 4, 5, 6]);
  assert.equal(requests, onlineRequests);
  const key = [...cache.entries.keys()].find((item) => item.includes(data.inventory.chunks[0]![2]));
  assert.ok(key);
  cache.entries.delete(key);
  assert.equal(await preparation.isOfflineReady(), false);
  preparation.dispose();
  source.dispose();
  offline.dispose();
});

test('quota and marker write failures preserve verified ranges and permit retry', async (t) => {
  for (const failedWrite of ['range', 'marker'] as const) {
    const data = fixture();
    const cache = new MemoryCache();
    const requested = transport(t, data.bytes);
    const normalPut = cache.put.bind(cache);
    let failing = true;
    const deniedDigest = data.inventory.chunks[2]![2];
    cache.put = async (key, response) => {
      if (failing && (failedWrite === 'range' ? key.includes(deniedDigest) : key.endsWith('weights-offline-v1.json')))
        throw new DOMException('Insufficient storage', 'QuotaExceededError');
      return normalPut(key, response);
    };
    const source = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base, {cache});
    await assert.rejects(source.prepareWeightsOffline(), /cached|committed/i);
    assert.equal(await source.hasCompleteOfflineWeights(), false);
    const retained = [...cache.entries.keys()];
    assert.ok(retained.length >= 2);
    assert.equal(
      retained.some((key) => key.endsWith('weights-offline-v1.json')),
      false
    );
    const originalRanges = requested.slice();
    failing = false;
    await source.prepareWeightsOffline();
    assert.equal(await source.hasCompleteOfflineWeights(), true);
    assert.equal(await source.verifyOfflineWeights(), true);
    for (const key of retained) assert.ok(cache.entries.has(key));
    for (const chunk of data.inventory.chunks.slice(0, 2)) {
      const expected = `bytes=${chunk[0]}-${chunk[0] + chunk[1] - 1}`;
      assert.equal(requested.filter((range) => range === expected).length, 1);
    }
    if (failedWrite === 'marker') assert.deepEqual(requested, originalRanges);
    source.dispose();
    t.mock.restoreAll();
  }
});

test('device loss cancels an in-flight verified range fetch without retrying, and a new device can retry', async (t) => {
  const {guardGemma4V2Runner, watchGemma4V2Device} = await import('../src/strands/gemma4-v2-gpu-errors.mjs');
  const data = fixture();
  let pending = true;
  let requests = 0;
  let started!: () => void;
  const entered = new Promise<void>((resolve) => {
    started = resolve;
  });
  t.mock.method(globalThis, 'fetch', async (_url: string, init?: RequestInit) => {
    requests++;
    const match = /^bytes=(\d+)-(\d+)$/u.exec(new Headers(init?.headers).get('range') ?? '');
    assert.ok(match);
    if (pending) {
      started();
      return new Promise<Response>((_resolve, reject) => {
        init!.signal!.addEventListener('abort', () => reject(init!.signal!.reason), {once: true});
      });
    }
    const start = Number(match[1]);
    const end = Number(match[2]);
    return new Response(Uint8Array.from(data.bytes.subarray(start, end + 1)), {
      status: 206,
      headers: {'content-range': `bytes ${start}-${end}/${data.bytes.length}`}
    });
  });
  const source = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base);
  let lose!: (reason: unknown) => void;
  const lost = new Promise<unknown>((resolve) => {
    lose = resolve;
  });
  const gpu = {
    lost,
    limits: {maxBufferSize: 65_536, maxStorageBufferBindingSize: 65_536},
    pushErrorScope() {},
    async popErrorScope() {
      return null;
    }
  } as unknown as GPUDevice;
  const lifecycle = watchGemma4V2Device(gpu);
  const runner = {
    async run(_input: unknown, signal?: AbortSignal | null) {
      await source.readTensor('model.language_model.first', signal);
      return [0, 0];
    },
    releaseCachedMemory() {},
    dispose() {}
  };
  const input = {stateIds: [2], questionIds: [3], imageSpans: [], images: []};
  const guarded = guardGemma4V2Runner(gpu, lifecycle, runner);
  const active = guarded.run(input);
  await entered;
  lose({message: 'device lost'});
  await assert.rejects(active, /GPU device was lost/);
  assert.equal(requests, 1);
  assert.equal(lifecycle.lost, true);
  lifecycle.close();
  source.dispose();

  pending = false;
  const secondSource = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base);
  assert.deepEqual([...(await secondSource.readTensor('model.language_model.first'))], [1, 2, 3, 4, 5, 6]);
  secondSource.dispose();
});

const embeddingName = 'model.language_model.embed_tokens_per_layer.weight';
const embeddingRowBytes = 35 * 256 * 2;

function embeddingFixture(splitRow = false) {
  const shape = [262144, 35 * 256];
  const length = shape[0]! * embeddingRowBytes;
  const json = Buffer.from(JSON.stringify({[embeddingName]: {dtype: 'BF16', shape, data_offsets: [0, length]}}));
  const header = Buffer.alloc(json.length + 8);
  header.writeBigUInt64LE(BigInt(json.length));
  json.copy(header, 8);
  const selected = Buffer.alloc(6 * embeddingRowBytes);
  for (let word = 0; word < selected.length / 4; word++)
    selected.writeUInt32LE((0x01020304 + word * 0x1234567) >>> 0, word * 4);
  const ends = splitRow
    ? [512, embeddingRowBytes, selected.length]
    : [2, 3, 4, 5, 6].map((rows) => rows * embeddingRowBytes);
  const blocks = new Map([[0, header]]);
  let offset = 0;
  for (const end of ends) {
    blocks.set(header.length + offset, selected.subarray(offset, end));
    offset = end;
  }
  const chunks: Array<[number, number, string]> = [...blocks].map(([start, bytes]) => [
    start,
    bytes.length,
    hash(bytes)
  ]);
  const bytes = header.length + length;
  // The rest of the logical tensor is indexed only. The test transport rejects any read outside the materialized rows.
  for (let start = header.length + selected.length; start < bytes;) {
    const size = Math.min(4 * 1024 * 1024, bytes - start);
    chunks.push([start, size, '0'.repeat(64)]);
    start += size;
  }
  const {inventory: defaults} = fixture();
  const inventory = {
    ...defaults,
    bytes,
    headerBytes: header.length,
    chunks,
    tensors: {[embeddingName]: {offset: header.length, length, shape, dtype: 'BF16'}}
  };
  return {inventory, digest: hash(JSON.stringify(inventory) + '\n'), blocks};
}

function embeddingTransport(
  t: TestContext,
  data: ReturnType<typeof embeddingFixture>,
  onRead?: (start: number) => void
) {
  const ranges: number[] = [];
  t.mock.method(globalThis, 'fetch', async (_url: string, init?: RequestInit) => {
    const match = /^bytes=(\d+)-(\d+)$/u.exec(new Headers(init?.headers).get('range') ?? '');
    assert.ok(match);
    const start = Number(match[1]);
    const bytes = data.blocks.get(start);
    assert.ok(bytes);
    const end = start + bytes.length - 1;
    assert.equal(Number(match[2]), end);
    ranges.push(start);
    onRead?.(start);
    return new Response(Uint8Array.from(bytes), {
      status: 206,
      headers: {'content-range': `bytes ${start}-${end}/${data.inventory.bytes}`}
    });
  });
  return ranges;
}

test('reads each verified PLE range once and preserves every word and row order across all layers', async (t) => {
  const data = embeddingFixture();
  const reads = embeddingTransport(t, data);
  const source = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base);
  const ids = Int32Array.of(5, 0, 3, 2, 1, 4, 5);
  const budget = ids.length * embeddingRowBytes;
  try {
    assert.equal(await source.readAllLayerEmbeddingRows(ids, budget - 1), undefined);
    assert.deepEqual(reads, []);
    const prefetched = await source.readAllLayerEmbeddingRows(ids, budget);
    assert.ok(prefetched);
    assert.equal(prefetched.length, 35);
    assert.deepEqual(new Set(reads), new Set(data.blocks.keys()));
    assert.equal(reads.length, data.blocks.size);
    assert.equal(prefetched[0]!.buffer.byteLength, budget);
    assert.equal(prefetched[34]!.buffer, prefetched[0]!.buffer);

    source.releaseCachedMemory();
    for (let layer = 0; layer < 35; layer++)
      assert.deepEqual(prefetched[layer], await source.readEmbeddingRows(ids, layer));
    assert.equal(reads.length - data.blocks.size, 35 * 5);
  } finally {
    source.dispose();
  }
});

test('snapshots token IDs within the prefetch budget before reading the header', async (t) => {
  const data = embeddingFixture();
  const ids = [0];
  const reads = embeddingTransport(t, data, (start) => {
    if (start === 0) ids.splice(0, 1, 2, 3);
  });
  const source = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base);
  try {
    const rows = await source.readAllLayerEmbeddingRows(ids, embeddingRowBytes);
    assert.ok(rows);
    assert.deepEqual(ids, [2, 3]);
    assert.equal(rows[0]!.buffer.byteLength, embeddingRowBytes);
    assert.deepEqual(rows[34], await source.readEmbeddingRows([0], 34));
    assert.deepEqual(reads, [0, data.inventory.headerBytes]);
  } finally {
    source.dispose();
  }
});

test('falls back when valid layer slices for a row span different source ranges', async (t) => {
  const data = embeddingFixture(true);
  const reads = embeddingTransport(t, data);
  const source = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base);
  try {
    assert.equal(await source.readAllLayerEmbeddingRows([0], embeddingRowBytes), undefined);
    assert.deepEqual(reads, [0]);
    assert.equal((await source.readEmbeddingRows([0], 0))[0], 0x01020304);
    assert.equal((await source.readEmbeddingRows([0], 34)).length, 128);
  } finally {
    source.dispose();
  }
});

test('propagates range integrity failure and cancellation during prefetch and allows a verified retry', async (t) => {
  const data = embeddingFixture();
  const controller = new AbortController();
  const cancelled = new Error('prefetch cancelled');
  const firstRange = data.inventory.headerBytes;
  const failedRange = firstRange + 2 * embeddingRowBytes;
  let abort = false;
  const reads = embeddingTransport(t, data, (start) => {
    if (abort && start === failedRange) controller.abort(cancelled);
  });
  const source = await openGemma4V2SourceFromInventory(data.inventory, data.digest, base);
  const ids = [0, 2];
  const budget = ids.length * embeddingRowBytes;
  const damaged = data.blocks.get(failedRange)!;
  const firstByte = damaged[0]!;
  try {
    damaged[0] = firstByte ^ 1;
    await assert.rejects(source.readAllLayerEmbeddingRows(ids, budget), /integrity/i);
    damaged[0] = firstByte;
    abort = true;
    await assert.rejects(
      source.readAllLayerEmbeddingRows(ids, budget, controller.signal),
      (reason) => reason === cancelled
    );
    assert.deepEqual(reads, [0, firstRange, failedRange, failedRange]);
    abort = false;
    const retry = await source.readAllLayerEmbeddingRows(ids, budget);
    assert.deepEqual(retry?.[34], await source.readEmbeddingRows(ids, 34));
    assert.deepEqual(reads, [0, firstRange, failedRange, failedRange, failedRange]);
  } finally {
    source.dispose();
  }
});
