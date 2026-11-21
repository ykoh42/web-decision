import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test, {type TestContext} from 'node:test';
import type {AssetCache} from '../src/assets.js';
import {MAPIKA2B_REPOSITORY, MAPIKA2B_REVISION} from '../src/mapika2b/contract.mts';
import {openMapika2bSourceFromInventory, openPinnedMapika2bSource} from '../src/mapika2b/source.mts';

const base = new URL('https://mapika.example/resolve/');
const embeddingName = 'model.language_model.embed_tokens.weight';
const rangeBytes = 16;

function digest(bytes: Uint8Array | string): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function safetensors(file: string, records: readonly {name: string; shape: number[]; words: number[]}[]) {
  const descriptors: Record<string, {dtype: string; shape: number[]; data_offsets: number[]}> = {};
  const tensors = records.map(({name, shape, words}) => {
    const payload = Buffer.alloc(words.length * 2);
    for (const [index, word] of words.entries()) payload.writeUInt16LE(word, index * 2);
    return {name, shape, payload};
  });
  let payloadOffset = 0;
  for (const {name, shape, payload} of tensors) {
    descriptors[name] = {dtype: 'BF16', shape, data_offsets: [payloadOffset, payloadOffset + payload.length]};
    payloadOffset += payload.length;
  }
  const json = Buffer.from(JSON.stringify(descriptors));
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(json.length));
  const header = Buffer.concat([prefix, json]);
  const bytes = Buffer.concat([header, ...tensors.map(({payload}) => payload)]);
  return {
    bytes,
    asset: {
      file,
      bytes: bytes.length,
      headerRangeBytes: header.length,
      headerRangeSha256: digest(header),
      tensorCount: tensors.length,
      tensors: tensors.map(({name, shape, payload}) => {
        const offset = header.length + descriptors[name]!.data_offsets[0]!;
        const chunks = [];
        for (let start = 0; start < payload.length; start += rangeBytes) {
          const chunk = payload.subarray(start, start + rangeBytes);
          chunks.push({offset: offset + start, length: chunk.length, sha256: digest(chunk)});
        }
        return {name, shape, dtype: 'BF16', offset, length: payload.length, sha256: digest(payload), chunks};
      })
    }
  };
}

function fixture() {
  const text = safetensors('model.safetensors', [
    {
      name: embeddingName,
      shape: [4, 4],
      words: [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43]
    },
    {name: 'decoder.weight', shape: [2, 2], words: [101, 102, 103, 104]}
  ]);
  const vision = safetensors('vision/visual.safetensors', [
    {name: 'model.visual.weight', shape: [2], words: [201, 202]}
  ]);
  const inventory = {
    format: 'web-decision-mapika-2b-v12-source-inventory-v1',
    repository: MAPIKA2B_REPOSITORY,
    revision: MAPIKA2B_REVISION,
    maxRangeBytes: rangeBytes,
    weights: {text: text.asset, vision: vision.asset}
  };
  return {text, vision, inventory, sha: digest(JSON.stringify(inventory) + '\n')};
}

function rangeTransport(t: TestContext, {text, vision}: ReturnType<typeof fixture>) {
  const urls = new Map([
    [new URL(text.asset.file, base).href, text.bytes],
    [new URL(vision.asset.file, base).href, vision.bytes]
  ]);
  const ranges: string[] = [];
  t.mock.method(globalThis, 'fetch', async (url: string, init?: RequestInit) => {
    const data = urls.get(String(url));
    const range = new Headers(init?.headers).get('range');
    const match = /^bytes=(\d+)-(\d+)$/.exec(range ?? '');
    assert.ok(data && match);
    ranges.push(range!);
    const start = Number(match[1]);
    const end = Number(match[2]);
    const body = Uint8Array.from(data.subarray(start, end + 1));
    return new Response(body, {
      status: 206,
      headers: {'Content-Range': 'bytes ' + start + '-' + end + '/' + data.length}
    });
  });
  return ranges;
}

class MemoryCache implements AssetCache {
  readonly #entries = new Map<string, Response>();

  async match(key: string) {
    return this.#entries.get(key)?.clone();
  }

  async put(key: string, response: Response) {
    const bytes = await response.arrayBuffer();
    this.#entries.set(key, new Response(bytes));
  }
}

test('opens the bundled authoritative source inventory without fetching model data', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch', () => assert.fail('Source initialization fetched weights.'));
  const source = await openPinnedMapika2bSource();
  t.after(() => source.dispose());
  assert.equal(source.requiredRangeBytes, 3_763_692_048 + 662_866_152);
  assert.equal(fetch.mock.callCount(), 0);
});

test('gathers BF16 embedding rows by verified chunks in request order', async (t) => {
  const assets = fixture();
  const requests = rangeTransport(t, assets);
  const source = await openMapika2bSourceFromInventory(assets.inventory, assets.sha, base);
  t.after(() => source.dispose());

  await assert.rejects(source.readEmbeddingRows([4]), /outside the pinned embedding table/u);
  assert.deepEqual(requests, []);
  assert.deepEqual(
    await source.readEmbeddingRows(Int32Array.of(3, 0, 3)),
    Uint16Array.of(40, 41, 42, 43, 10, 11, 12, 13, 40, 41, 42, 43)
  );
  const header = assets.text.asset.headerRangeBytes;
  assert.deepEqual(requests, [
    'bytes=0-' + (header - 1),
    'bytes=' + (header + rangeBytes) + '-' + (header + 2 * rangeBytes - 1),
    'bytes=' + header + '-' + (header + rangeBytes - 1)
  ]);
  await assert.rejects(source.readParameter({name: 'decoder.weight', shape: [3]}), /packed weight shape differs/u);
  const bytes = await source.readParameter({name: 'decoder.weight', shape: [2, 1]});
  assert.deepEqual(bytes, Uint8Array.of(101, 0, 102, 0, 103, 0, 104, 0));
  await assert.rejects(source.readTensor('text', embeddingName), /selected rows/u);
});

test('requires complete verified ranges for offline mode', async (t) => {
  const assets = fixture();
  rangeTransport(t, assets);
  const cache = new MemoryCache();
  const open = (offline: boolean) =>
    openMapika2bSourceFromInventory(assets.inventory, assets.sha, base, {cache, offline});
  const online = await open(false);
  t.after(() => online.dispose());
  await online.readEmbeddingRows([0]);
  assert.equal(await online.commit(), false);
  await assert.rejects(open(true), /complete verified model/u);
  const downloaded: number[] = [];
  await online.prefetchRequiredRanges({onProgress: (bytes) => downloaded.push(bytes)});
  assert.equal(
    downloaded.reduce((sum, bytes) => sum + bytes, 0),
    online.requiredRangeBytes
  );
  assert.equal(await online.commit(), true);

  const fetch = t.mock.method(globalThis, 'fetch', () => assert.fail('Offline source tried the network.'));
  const offline = await open(true);
  t.after(() => offline.dispose());
  assert.deepEqual(await offline.readEmbeddingRows([2]), Uint16Array.of(30, 31, 32, 33));
  assert.deepEqual(await offline.readTensor('vision', 'model.visual.weight'), Uint8Array.of(201, 0, 202, 0));
  assert.equal(fetch.mock.callCount(), 0);
});

test('rejects changed inventory and mismatched byte-range contents', async (t) => {
  const assets = fixture();
  await assert.rejects(
    openMapika2bSourceFromInventory({...assets.inventory, maxRangeBytes: 8}, assets.sha, base),
    /inventory digest differs/u
  );
  const firstChunk = assets.text.asset.tensors[0]!.chunks[0]!;
  assets.text.bytes[firstChunk.offset] = 0xff;
  rangeTransport(t, assets);
  const source = await openMapika2bSourceFromInventory(assets.inventory, assets.sha, base);
  t.after(() => source.dispose());
  await assert.rejects(source.readEmbeddingRows([0]), /range integrity check failed/iu);
});

test('a failed concurrent prefetch cannot commit and can be retried', async (t) => {
  const assets = fixture();
  const firstChunk = assets.text.asset.tensors[0]!.chunks[0]!;
  const original = assets.text.bytes[firstChunk.offset]!;
  assets.text.bytes[firstChunk.offset] = original ^ 0xff;
  rangeTransport(t, assets);
  const cache = new MemoryCache();
  const source = await openMapika2bSourceFromInventory(assets.inventory, assets.sha, base, {cache});
  t.after(() => source.dispose());
  await assert.rejects(source.prefetchRequiredRanges(), /range integrity check failed/iu);
  assert.equal(await source.commit(), false);
  assets.text.bytes[firstChunk.offset] = original;
  await source.prefetchRequiredRanges();
  assert.equal(await source.commit(), true);
  const offline = await openMapika2bSourceFromInventory(assets.inventory, assets.sha, base, {cache, offline: true});
  t.after(() => offline.dispose());
  assert.deepEqual(await offline.readEmbeddingRows([0]), Uint16Array.of(10, 11, 12, 13));
});

test('storage quota failure keeps online reads usable without marking offline data complete', async (t) => {
  const assets = fixture();
  rangeTransport(t, assets);
  const cache: AssetCache = {
    async match() {
      return undefined;
    },
    async put() {
      throw new DOMException('Quota exceeded', 'QuotaExceededError');
    }
  };
  const source = await openMapika2bSourceFromInventory(assets.inventory, assets.sha, base, {cache});
  t.after(() => source.dispose());
  await source.prefetchRequiredRanges();
  assert.equal(await source.commit(), false);
  assert.deepEqual(await source.readEmbeddingRows([0]), Uint16Array.of(10, 11, 12, 13));
  await assert.rejects(
    openMapika2bSourceFromInventory(assets.inventory, assets.sha, base, {cache, offline: true}),
    /complete verified model/u
  );
});
