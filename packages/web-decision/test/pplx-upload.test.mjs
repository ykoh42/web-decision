import assert from 'node:assert/strict';
import {createHash, webcrypto} from 'node:crypto';
import test from 'node:test';
import {uploadFlashParameters} from '../src/pplx/upload-parameters.mjs';

if (!globalThis.crypto) globalThis.crypto = webcrypto;
const data = Uint8Array.of(1, 2, 3, 4);
const sha = createHash('sha256').update(data).digest('hex');

function fixture() {
  return {
    state: 'pplx-q4-joint-assets-pending',
    entries: Array.from({length: 682}, (_, index) =>
      index < 680
        ? {
            position: index + 5,
            target: `file-${index}`,
            shape: [1],
            dtype: 'uint32',
            bindingKind: 'q4-candidate',
            outputBytes: 4,
            ranges: [{offset: 0, bytes: 4, sha256: sha}]
          }
        : {
            position: index + 5,
            target: `rotary-${index}`,
            shape: [1],
            dtype: 'float32',
            bindingKind: 'derived-rotary',
            values: [0.5]
          }
    )
  };
}

test('verifies ranges and awaits writes in graph order', async () => {
  const writes = [];
  const tensors = await uploadFlashParameters(fixture(), {
    createTensor: (entry) => entry.position,
    readRange: () => data.slice().buffer,
    writeRange: async (tensor, offset, bytes) => {
      writes.push({tensor, offset, bytes: [...bytes]});
    }
  });
  assert.equal(tensors.length, 682);
  assert.deepEqual(
    tensors,
    Array.from({length: 682}, (_, i) => i + 5)
  );
  assert.deepEqual(writes[0], {tensor: 5, offset: 0, bytes: [1, 2, 3, 4]});
  assert.deepEqual(writes[680], {tensor: 685, offset: 0, bytes: [0, 0, 0, 63]});
});

test('rejects malformed plans before creating tensors', async () => {
  const manifest = fixture();
  manifest.entries[500].ranges[0].offset = 4;
  let created = false;
  await assert.rejects(
    uploadFlashParameters(manifest, {
      createTensor: () => {
        created = true;
      },
      readRange() {},
      writeRange() {}
    }),
    /range/
  );
  assert.equal(created, false);
});

test('never writes corrupt ranges and stops after cancellation', async () => {
  let writes = 0;
  await assert.rejects(
    uploadFlashParameters(fixture(), {
      createTensor: () => ({}),
      readRange: () => Uint8Array.of(4, 3, 2, 1).buffer,
      writeRange: () => {
        writes++;
      }
    }),
    /hash mismatch/
  );
  assert.equal(writes, 0);
  const controller = new AbortController();
  await assert.rejects(
    uploadFlashParameters(fixture(), {
      createTensor: () => ({}),
      readRange: () => data.slice().buffer,
      writeRange: () => {
        writes++;
        controller.abort();
      },
      signal: controller.signal
    }),
    {name: 'AbortError'}
  );
  assert.equal(writes, 1);
});
