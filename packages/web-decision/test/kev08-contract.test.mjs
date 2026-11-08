import test from 'node:test';
import assert from 'node:assert/strict';
import {
  KEV08_PARAMETER_BYTES,
  readKev08ManifestResponse,
  validateKev08TokenizerData,
  validateKev08WeightIndex
} from '../src/kev08/contract.js';

function fixture() {
  const head = new Map([
    [320, [256, 1024]],
    [321, [256]],
    [322, [256, 1024]],
    [323, [256]]
  ]);
  const headBytes = 2 * 256 * 1024 * 4 + 2 * 256 * 4;
  const records = [];
  let offset = 0;
  for (let i = 0; i < 326; i++) {
    const nbytes =
      i === 0 ? KEV08_PARAMETER_BYTES - headBytes - 321 * 4 : head.has(i) ? head.get(i).reduce((a, b) => a * b, 4) : 4;
    records.push({index: i, shape: head.get(i) ?? [nbytes / 4], dtype: 'float32', offset, nbytes});
    offset += nbytes;
  }
  const shards = [],
    files = {};
  let remaining = KEV08_PARAMETER_BYTES;
  for (let i = 0; i < 404; i++) {
    const nbytes = i < 179 ? 8 * 1024 * 1024 : i < 403 ? 4 : remaining;
    const path = `weights/shard-${String(i).padStart(4, '0')}.bin`;
    shards.push({path, nbytes});
    files[path] = {bytes: nbytes};
    remaining -= nbytes;
  }
  assert.equal(remaining, 0);
  assert.equal(offset, KEV08_PARAMETER_BYTES);
  return {manifest: {files}, index: {records, shards, totalBytes: offset}};
}

test('validates all shard boundaries and the four FP32 pointer-head shapes', () => {
  const {manifest, index} = fixture();
  assert.equal(validateKev08WeightIndex(manifest, index), index);
  for (const mutate of [
    (value) => {
      value.records[320].dtype = 'float16';
    },
    (value) => {
      value.records[322].shape = [1024, 256];
    },
    (value) => {
      value.records[10].offset += 4;
    },
    (value) => {
      value.shards[1].path = '../other.bin';
    },
    (value) => {
      value.shards.pop();
    }
  ]) {
    const invalid = structuredClone(index);
    mutate(invalid);
    assert.throws(() => validateKev08WeightIndex(manifest, invalid), {code: 'INVALID_ASSET'});
  }
});

test('requires the correct tokenizer marker IDs and rejects duplicates', () => {
  const data = {
    added_tokens: [
      {content: '<|fim_prefix|>', id: 248060},
      {content: '<|fim_middle|>', id: 248061},
      {content: '<|box_start|>', id: 248049},
      {content: '<|box_end|>', id: 248050},
      {content: '<|fim_suffix|>', id: 248062}
    ]
  };
  assert.doesNotThrow(() => validateKev08TokenizerData(data));
  const changed = structuredClone(data);
  changed.added_tokens[0].id++;
  assert.throws(() => validateKev08TokenizerData(changed), {code: 'INVALID_ASSET'});
  changed.added_tokens = [...data.added_tokens, data.added_tokens[0]];
  assert.throws(() => validateKev08TokenizerData(changed), {code: 'INVALID_ASSET'});
  assert.throws(() => validateKev08TokenizerData(null), {code: 'INVALID_ASSET'});
});

test('rejects an unrecognized manifest and preserves cancellation', async () => {
  await assert.rejects(readKev08ManifestResponse(new Response('{}')), {code: 'INVALID_ASSET'});
  const controller = new AbortController(),
    reason = Error('cancel');
  controller.abort(reason);
  await assert.rejects(readKev08ManifestResponse(new Response('{}'), controller.signal), (error) => error === reason);
});
