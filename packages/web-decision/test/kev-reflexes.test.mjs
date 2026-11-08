import test from 'node:test';
import assert from 'node:assert/strict';
import pinned from '../src/kev-reflexes-contract.json' with {type: 'json'};
import {
  REFLEXES_HEADS,
  REFLEXES_ID,
  readReflexesManifestResponse,
  uploadReflexesHeads,
  validateReflexesManifest
} from '../src/kev-reflexes.js';
import {selectModelFamily} from '../src/model-family.js';
import {responseFromReader} from './stream-helpers.mjs';

test('explicit reflexes selection accepts only its pinned model asset manifest', () => {
  assert.equal(selectModelFamily({model: REFLEXES_ID}), 'kev');
  assert.equal(selectModelFamily({}), 'kev');
  assert.equal(validateReflexesManifest(structuredClone(pinned)).model, REFLEXES_ID);
  assert.equal(Object.hasOwn(pinned, 'maxInputTokens'), false);
  assert.equal(pinned.maxStateTokens, 8192);
  assert.equal(pinned.maxBranchTokens, 8192);
  for (const mutate of [
    (m) => {
      m.model = 'kev-0.6b-q4f16';
    },
    (m) => {
      m.checkpoint.revision = '0'.repeat(40);
    },
    (m) => {
      m.parameterCount = 394;
    },
    (m) => {
      m.files['heads/param_394.bin'].sha256 = '0'.repeat(64);
    },
    (m) => {
      m.files['heads/param_397.bin'].bytes--;
    },
    (m) => {
      delete m.files['heads/param_395.bin'];
    },
    (m) => {
      m.files['other'] = {bytes: 0, sha256: '0'.repeat(64)};
    }
  ]) {
    const value = structuredClone(pinned);
    mutate(value);
    assert.throws(() => validateReflexesManifest(value), {code: 'INVALID_ASSET'});
  }
});

test('pinned Reflexes contract declares native state and branch limits', () => {
  assert.equal(validateReflexesManifest(structuredClone(pinned)).model, REFLEXES_ID);
  assert.equal(Object.hasOwn(pinned, 'maxInputTokens'), false);
  assert.equal(pinned.maxStateTokens, 8192);
  assert.equal(pinned.maxBranchTokens, 8192);
});

test('invalid or cancelled reflexes manifest preserves the correct error', async () => {
  await assert.rejects(readReflexesManifestResponse(new Response('{bad')), {code: 'INVALID_ASSET'});
  const controller = new AbortController(),
    reason = new Error('stop');
  await assert.rejects(
    readReflexesManifestResponse(
      responseFromReader(async () => {
        controller.abort(reason);
        throw Error('bad');
      }),
      controller.signal
    ),
    (e) => e === reason
  );
});

function fakeUpload({afterSync = () => {}, uncaptured = []} = {}) {
  const calls = [],
    weights = new Array(398),
    device = {
      async sync() {
        calls.push('sync');
        afterSync();
      }
    },
    runtime = {
      empty(shape, dtype, dev) {
        assert.equal(dev, device);
        const record = {shape: [...shape], dtype};
        calls.push(record);
        return {
          copyFromRawBytes(bytes) {
            record.bytes = bytes.byteLength;
            record.first = bytes[0];
          }
        };
      }
    };
  return {
    calls,
    weights,
    device,
    runtime,
    uncaptured,
    read: async (path) => {
      const head = REFLEXES_HEADS.find((h) => h.path === path);
      assert(head);
      calls.push(path);
      return new Uint8Array(head.bytes).fill(head.index - 393).buffer;
    }
  };
}

test('uploads all four distinct verified heads in parameter order', async () => {
  const f = fakeUpload();
  await uploadReflexesHeads(f);
  assert.deepEqual(
    f.calls.filter((x) => typeof x === 'string' && x !== 'sync'),
    REFLEXES_HEADS.map((x) => x.path)
  );
  assert.equal(f.calls.filter((x) => x === 'sync').length, 4);
  const uploads = f.calls.filter((x) => typeof x === 'object');
  assert.deepEqual(
    uploads.map(({shape, dtype, bytes, first}) => ({shape, dtype, bytes, first})),
    REFLEXES_HEADS.map((h) => ({shape: [...h.shape], dtype: 'float32', bytes: h.bytes, first: h.index - 393}))
  );
  assert(f.weights.slice(394).every(Boolean));
});

test('head upload stops on cancellation, bad length or a GPU error', async () => {
  const controller = new AbortController(),
    reason = new Error('stop'),
    f = fakeUpload({afterSync: () => controller.abort(reason)});
  await assert.rejects(uploadReflexesHeads({...f, signal: controller.signal}), (e) => e === reason);
  assert.equal(f.calls.filter((x) => x === 'sync').length, 1);
  assert.equal(f.weights[395], undefined);
  const bad = fakeUpload();
  bad.read = async () => new ArrayBuffer(1);
  await assert.rejects(uploadReflexesHeads(bad), {code: 'INVALID_ASSET'});
  assert.equal(bad.calls.length, 0);
  const gpu = fakeUpload({uncaptured: ['GPU failed']});
  await assert.rejects(uploadReflexesHeads(gpu), {code: 'INFERENCE_ERROR'});
  assert.equal(gpu.calls.filter((x) => x === 'sync').length, 1);
});
