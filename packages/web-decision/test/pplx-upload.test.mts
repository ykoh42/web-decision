import assert from 'node:assert/strict';
import {createHash, webcrypto} from 'node:crypto';
import test from 'node:test';
import {uploadFlashParameters} from '../src/pplx/upload-parameters.mjs';
import type {FlashManifestEntry, FlashTensor} from '../src/pplx/types.js';

interface FixtureRange {
  offset: number;
  bytes: number;
  sha256: string;
}

interface FixtureQ4Entry {
  position: number;
  target: string;
  shape: number[];
  dtype: 'uint32';
  bindingKind: 'q4-candidate';
  outputBytes: number;
  ranges: FixtureRange[];
}

interface FixtureRotaryEntry {
  position: number;
  target: string;
  shape: number[];
  dtype: 'float32';
  bindingKind: 'derived-rotary';
  values: number[];
}

interface UploadFixture {
  state: 'pplx-q4-joint-assets-pending';
  entries: Array<FixtureQ4Entry | (FixtureRotaryEntry & {ranges?: never})>;
}

if (!globalThis.crypto) globalThis.crypto = webcrypto as unknown as Crypto;
const data = Uint8Array.of(1, 2, 3, 4);
const sha = createHash('sha256').update(data).digest('hex');

function fixture(): UploadFixture {
  return {
    state: 'pplx-q4-joint-assets-pending',
    entries: Array.from({length: 682}, (_, index): FixtureQ4Entry | FixtureRotaryEntry =>
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
  const writes: Array<{tensor: unknown; offset: number; bytes: number[]}> = [];
  const tensors = await uploadFlashParameters(fixture() as unknown as Parameters<typeof uploadFlashParameters>[0], {
    createTensor: (entry: FlashManifestEntry) => entry.position as unknown as FlashTensor,
    readRange: () => data.slice().buffer,
    writeRange: async (tensor: unknown, offset: number, bytes: Uint8Array) => {
      writes.push({tensor, offset, bytes: [...bytes]});
    }
  });
  assert.equal(tensors.length, 682);
  assert.deepEqual(
    tensors,
    Array.from({length: 682}, (_, i) => i + 5)
  );
  assert.deepEqual(writes[0]!, {tensor: 5, offset: 0, bytes: [1, 2, 3, 4]});
  assert.deepEqual(writes[680]!, {tensor: 685, offset: 0, bytes: [0, 0, 0, 63]});
});

test('rejects malformed plans before creating tensors', async () => {
  const manifest = fixture();
  manifest.entries[500]!.ranges![0]!.offset = 4;
  let created = false;
  await assert.rejects(
    uploadFlashParameters(
      manifest as unknown as Parameters<typeof uploadFlashParameters>[0],
      {
        createTensor: () => {
          created = true;
        },
        readRange() {},
        writeRange() {}
      } as unknown as Parameters<typeof uploadFlashParameters>[1]
    ),
    /range/
  );
  assert.equal(created, false);
});

test('never writes corrupt ranges and stops after cancellation', async () => {
  let writes = 0;
  await assert.rejects(
    uploadFlashParameters(fixture() as unknown as Parameters<typeof uploadFlashParameters>[0], {
      createTensor: () => ({}) as unknown as FlashTensor,
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
    uploadFlashParameters(fixture() as unknown as Parameters<typeof uploadFlashParameters>[0], {
      createTensor: () => ({}) as unknown as FlashTensor,
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
