import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {selectModelFamily} from '../src/model-family.js';
import {validateTensorIndex} from '../src/tensor-index.js';
import {
  MAPIKA_Q4_MAX_SHARD_BYTES,
  MAPIKA_Q4_MODEL,
  MAPIKA_Q4_PARAMETER_BYTES,
  readMapikaQ4ManifestResponse,
  validateMapikaQ4WeightIndex
} from '../src/mapika-q4/contract.js';

const fixture = new URL('./fixtures/mapika-q4/', import.meta.url);

async function assets() {
  const manifestBytes = await readFile(new URL('manifest.json', fixture));
  const manifest = JSON.parse(manifestBytes),
    index = JSON.parse(await readFile(new URL('index.json', fixture)));
  return {manifestBytes, manifest, index};
}

test('routes only the explicit Q4 model while preserving the default and FP16 routes', () => {
  assert.equal(selectModelFamily({model: MAPIKA_Q4_MODEL}), 'mapika-q4');
  assert.equal(selectModelFamily({model: 'mapika-0.8b-f16'}), 'mapika');
  assert.equal(selectModelFamily({}), 'kev');
  assert.throws(() => selectModelFamily({model: 'mapika-0.8b-q4'}), {code: 'INVALID_INPUT'});
});

test('accepts the pinned Q4 manifest and rejects damaged or cancelled reads', async () => {
  const {manifestBytes, manifest} = await assets();
  assert.equal((await readMapikaQ4ManifestResponse(new Response(manifestBytes))).model, MAPIKA_Q4_MODEL);
  manifest.maxInputTokens = 512;
  await assert.rejects(readMapikaQ4ManifestResponse(new Response(JSON.stringify(manifest))), {code: 'INVALID_ASSET'});
  const controller = new AbortController(),
    reason = Error('stop');
  controller.abort(reason);
  await assert.rejects(
    readMapikaQ4ManifestResponse(new Response(manifestBytes), controller.signal),
    (error) => error === reason
  );
});

test('validates 510 ordered Q4 tensors, labels, tied embedding and bounded shards', async () => {
  const {manifest, index} = await assets();
  assert.equal(validateMapikaQ4WeightIndex(manifest, index), index);
  for (const mutate of [
    (item) => {
      item.records[1].offset += 4;
    },
    (item) => {
      item.records[133].shape = [254];
    },
    (item) => {
      item.records[506].dtype = 'int32';
    },
    (item) => {
      item.records[507].shape[1] = 31;
    },
    (item) => {
      item.records[0].dtype = 'float64';
    },
    (item) => {
      item.shards[1].path = '../other.bin';
    },
    (item) => {
      item.shards[1].nbytes += 4;
    },
    (item) => {
      item.records.pop();
    }
  ]) {
    const changed = structuredClone(index);
    mutate(changed);
    assert.throws(() => validateMapikaQ4WeightIndex(manifest, changed), {code: 'INVALID_ASSET'});
  }
});

test('uint32 remains disallowed by the shared validator unless explicitly enabled', async () => {
  const {manifest, index} = await assets();
  const limits = {
    parameterCount: 510,
    totalBytes: MAPIKA_Q4_PARAMETER_BYTES,
    shardCount: 51,
    maxShardBytes: MAPIKA_Q4_MAX_SHARD_BYTES
  };
  assert.throws(() => validateTensorIndex(manifest, index, limits), {code: 'INVALID_ASSET'});
  assert.equal(
    validateTensorIndex(manifest, index, {...limits, allowedDtypes: ['float16', 'float32', 'int32', 'uint32']}),
    index
  );
});
