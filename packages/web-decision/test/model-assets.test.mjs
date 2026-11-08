import assert from 'node:assert/strict';
import test from 'node:test';
import {sha256} from '../src/assets.js';
import {createModelAssetReader, uploadModelShards} from '../src/model-assets.js';

const encode = (value) => new TextEncoder().encode(value);

async function file(value) {
  const bytes = encode(value);
  return {bytes, metadata: {bytes: bytes.byteLength, sha256: await sha256(bytes)}};
}

test('model asset reader reports verified byte progress and incomplete cache status', async (t) => {
  const first = await file('abc');
  const second = await file('de');
  const manifest = {files: {'first.bin': first.metadata, 'second.bin': second.metadata}};
  const progress = [];
  const fetched = [];
  const cache = {
    async match(key) {
      return key.includes(first.metadata.sha256) ? new Response(first.bytes) : undefined;
    },
    async put() {
      throw Error('synthetic cache write failure');
    }
  };
  t.mock.method(globalThis, 'fetch', async (url) => {
    const name = new URL(url).pathname.split('/').at(-1);
    fetched.push(name);
    return new Response(second.bytes);
  });

  const reader = createModelAssetReader({
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

  const cachedReader = createModelAssetReader({
    base: new URL('https://example.test/model/'),
    manifest: {files: {'first.bin': first.metadata}},
    cache,
    options: {}
  });
  await cachedReader.read('first.bin');
  assert.equal(cachedReader.isCacheComplete(), true);
});

function shards() {
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

function uploadFixture() {
  const events = [];
  const weights = [];
  const runtime = {
    empty(shape, dtype, device) {
      const name = `param_${weights.length}`;
      events.push(`empty:${name}:${shape.join('x')}:${dtype}:${Boolean(device)}`);
      return {
        copyFromRawBytes(bytes) {
          events.push(`copy:${name}:${[...bytes].join(',')}`);
        }
      };
    }
  };
  const device = {
    async sync() {
      events.push('sync');
    }
  };
  const read = async (name) => {
    events.push(`read:${name}`);
    return name === 'weights/first.bin' ? Uint8Array.of(4, 5).buffer : Uint8Array.of(6).buffer;
  };
  return {device, events, read, runtime, weights};
}

test('model shard upload consumes serial reads in manifest order', async () => {
  const fixture = uploadFixture();
  await uploadModelShards({
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
  await uploadModelShards({
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
    uploadModelShards({
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
