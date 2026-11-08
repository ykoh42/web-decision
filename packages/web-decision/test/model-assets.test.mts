import assert from 'node:assert/strict';
import test, {type TestContext} from 'node:test';
import {sha256} from '../src/assets.js';
import {createModelAssetReader, uploadModelShards} from '../src/model-assets.js';

type ManifestFile = {bytes: number; sha256: string};
type TestManifest = {files: Record<string, ManifestFile>};
type AssetProgress = {fraction: number; loadedBytes: number; phase?: string; totalBytes: number};
type TestCache = {
  match(key: string): Promise<Response | undefined>;
  put(key: string, response: Response): Promise<void>;
};
type ModelAssetReaderOptions = {
  base: URL;
  cache: TestCache;
  fetchImpl?: typeof fetch;
  manifest: TestManifest;
  options: {offline?: boolean; onProgress?: (progress: AssetProgress) => void};
  progressPhase?: string;
  signal?: AbortSignal;
};
type ModelAssetReader = {isCacheComplete(): boolean; read(name: string): Promise<ArrayBuffer>};
type AssetFixture = {bytes: Uint8Array; metadata: ManifestFile};
type TensorRecord = {byteOffset: number; dtype: string; name: string; nbytes: number; shape: number[]};
type ModelShard = {dataPath: string; nbytes: number; records: TensorRecord[]};
type TestTensor = {copyFromRawBytes(bytes: Uint8Array): void};
type TestDevice = {sync(): Promise<void>};
type TestRuntime = {empty(shape: number[], dtype: string, device: TestDevice): TestTensor};
type UploadShardsOptions = {
  device: TestDevice;
  maxPrefetchBytes?: number;
  read(name: string, signal?: AbortSignal): Promise<ArrayBuffer>;
  records: ModelShard[];
  runtime: TestRuntime;
  signal?: AbortSignal;
  uncaptured: string[];
  weights: TestTensor[];
};
type UploadFixture = {
  device: TestDevice;
  events: string[];
  read(name: string, signal?: AbortSignal): Promise<ArrayBuffer>;
  runtime: TestRuntime;
  weights: TestTensor[];
};

const encode = (value: string): Uint8Array => new TextEncoder().encode(value);

function responseFromBytes(bytes: Uint8Array): Response {
  const body = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(body).set(bytes);
  return new Response(body);
}

async function file(value: string): Promise<AssetFixture> {
  const bytes = encode(value);
  return {bytes, metadata: {bytes: bytes.byteLength, sha256: (await sha256(bytes)) as string}};
}

function createReader(options: ModelAssetReaderOptions): ModelAssetReader {
  return createModelAssetReader({
    ...options,
    fetchImpl: options.fetchImpl,
    progressPhase: options.progressPhase,
    signal: options.signal
  }) as ModelAssetReader;
}

function uploadShards(options: UploadShardsOptions): Promise<void> {
  return uploadModelShards({
    ...options,
    maxPrefetchBytes: options.maxPrefetchBytes,
    signal: options.signal
  });
}

test('model asset reader reports verified byte progress and incomplete cache status', async (t: TestContext) => {
  const first = await file('abc');
  const second = await file('de');
  const manifest: TestManifest = {files: {'first.bin': first.metadata, 'second.bin': second.metadata}};
  const progress: AssetProgress[] = [];
  const fetched: string[] = [];
  const cache: TestCache = {
    async match(key) {
      return key.includes(first.metadata.sha256) ? responseFromBytes(first.bytes) : undefined;
    },
    async put() {
      throw Error('synthetic cache write failure');
    }
  };
  t.mock.method(globalThis, 'fetch', async (input: RequestInfo | URL, _init?: RequestInit) => {
    const url = input instanceof Request ? input.url : input;
    const name = new URL(url).pathname.split('/').at(-1);
    if (!name) throw Error('The mocked model asset URL must have a path segment.');
    fetched.push(name);
    return responseFromBytes(second.bytes);
  });

  const reader = createReader({
    base: new URL('https://example.test/model/'),
    manifest,
    cache,
    options: {
      onProgress(value) {
        progress.push(value);
      }
    }
  });

  assert.deepEqual(new Uint8Array(await reader.read('first.bin')), first.bytes);
  assert.deepEqual(new Uint8Array(await reader.read('second.bin')), second.bytes);
  assert.deepEqual(fetched, ['second.bin']);
  assert.deepEqual(progress, [
    {loadedBytes: 3, totalBytes: 5, fraction: 3 / 5},
    {loadedBytes: 5, totalBytes: 5, fraction: 1}
  ]);
  assert.equal(reader.isCacheComplete(), false);

  const cachedReader = createReader({
    base: new URL('https://example.test/model/'),
    manifest: {files: {'first.bin': first.metadata}},
    cache,
    options: {}
  });
  await cachedReader.read('first.bin');
  assert.equal(cachedReader.isCacheComplete(), true);
});

function shards(): ModelShard[] {
  return [
    {
      dataPath: 'first.bin',
      nbytes: 2,
      records: [{name: 'param_0', shape: [2], dtype: 'int8', byteOffset: 0, nbytes: 2}]
    },
    {
      dataPath: 'second.bin',
      nbytes: 1,
      records: [{name: 'param_1', shape: [1], dtype: 'int8', byteOffset: 0, nbytes: 1}]
    }
  ];
}

function uploadFixture(): UploadFixture {
  const events: string[] = [];
  const weights: TestTensor[] = [];
  const runtime: TestRuntime = {
    empty(shape: number[], dtype: string, device: TestDevice): TestTensor {
      const name = `param_${weights.length}`;
      events.push(`empty:${name}:${shape.join('x')}:${dtype}:${Boolean(device)}`);
      return {
        copyFromRawBytes(bytes: Uint8Array): void {
          events.push(`copy:${name}:${[...bytes].join(',')}`);
        }
      };
    }
  };
  const device: TestDevice = {
    async sync() {
      events.push('sync');
    }
  };
  const read = async (name: string): Promise<ArrayBuffer> => {
    events.push(`read:${name}`);
    return (name === 'weights/first.bin' ? Uint8Array.of(4, 5).buffer : Uint8Array.of(6).buffer) as ArrayBuffer;
  };
  return {device, events, read, runtime, weights};
}

test('model shard upload consumes serial reads in manifest order', async () => {
  const fixture = uploadFixture();
  await uploadShards({
    records: shards(),
    read: fixture.read,
    runtime: fixture.runtime,
    device: fixture.device,
    weights: fixture.weights,
    uncaptured: []
  });
  assert.deepEqual(fixture.events, [
    'read:weights/first.bin',
    'empty:param_0:2:int8:true',
    'copy:param_0:4,5',
    'sync',
    'read:weights/second.bin',
    'empty:param_1:1:int8:true',
    'copy:param_1:6',
    'sync'
  ]);
  assert.equal(fixture.weights.length, 2);
});

test('prefetched shard reads still upload in manifest order', async () => {
  const fixture = uploadFixture();
  await uploadShards({
    records: shards(),
    read: fixture.read,
    runtime: fixture.runtime,
    device: fixture.device,
    weights: fixture.weights,
    maxPrefetchBytes: 2,
    uncaptured: []
  });
  assert.deepEqual(
    fixture.events.filter((event) => event.startsWith('copy:')),
    ['copy:param_0:4,5', 'copy:param_1:6']
  );
  assert.equal(fixture.events.filter((event) => event.startsWith('read:')).length, 2);
});

test('an uncaptured GPU error stops shard upload after sync', async () => {
  const fixture = uploadFixture();
  await assert.rejects(
    uploadShards({
      records: shards(),
      read: fixture.read,
      runtime: fixture.runtime,
      device: fixture.device,
      weights: fixture.weights,
      uncaptured: ['synthetic GPU validation failure']
    }),
    {code: 'INFERENCE_ERROR', message: 'synthetic GPU validation failure'}
  );
  assert.equal(fixture.events.at(-1), 'sync');
  assert.equal(
    fixture.events.some((event) => event === 'read:weights/second.bin'),
    false
  );
});
