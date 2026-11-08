import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MAPIKA_MAX_SHARD_BYTES,
  MAPIKA_PARAMETER_BYTES,
  readMapikaManifestResponse,
  validateMapikaWeightIndex
} from '../src/mapika/contract.js';

function fixture() {
  const smallBytes = 321 * 4 + 255 * 4;
  const records = [];
  let offset = 0;
  for (let i = 0; i < 323; i++) {
    const nbytes = i === 0 ? MAPIKA_PARAMETER_BYTES - smallBytes : i === 320 ? 255 * 4 : 4;
    const shape = [nbytes / 4];
    records.push({index: i, shape, dtype: 'int32', offset, nbytes});
    offset += nbytes;
  }
  const shards = [],
    files = {};
  let remaining = MAPIKA_PARAMETER_BYTES;
  for (let i = 0; i < 180; i++) {
    const nbytes = Math.min(remaining, MAPIKA_MAX_SHARD_BYTES);
    const path = `weights/shard-${String(i).padStart(4, '0')}.bin`;
    files[path] = {bytes: nbytes};
    shards.push({path, nbytes});
    remaining -= nbytes;
  }
  assert.equal(remaining, 0);
  assert.equal(offset, MAPIKA_PARAMETER_BYTES);
  return {manifest: {files}, index: {records, shards, totalBytes: offset}};
}

test('checks all parameter and shard boundaries before allocation', () => {
  const {manifest, index} = fixture();
  assert.equal(validateMapikaWeightIndex(manifest, index), index);
  for (const mutate of [
    (value) => {
      value.records[5].offset += 4;
    },
    (value) => {
      value.records[320].shape = [254];
    },
    (value) => {
      value.records[1].dtype = 'float64';
    },
    (value) => {
      value.shards[1].path = '../other.bin';
    },
    (value) => {
      value.shards[1].nbytes += 4;
    },
    (value) => {
      value.records.pop();
    }
  ]) {
    const changed = structuredClone(index);
    mutate(changed);
    assert.throws(() => validateMapikaWeightIndex(manifest, changed), {code: 'INVALID_ASSET'});
  }
});

test('rejects unrecognized and damaged manifests', async () => {
  for (const value of ['{}', '{bad', JSON.stringify({model: 'mapika-0.8b-f16'})])
    await assert.rejects(readMapikaManifestResponse(new Response(value)), {code: 'INVALID_ASSET'});
  const controller = new AbortController(),
    reason = Error('cancel');
  controller.abort(reason);
  await assert.rejects(readMapikaManifestResponse(new Response('{}'), controller.signal), (error) => error === reason);
});
