import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {setImmediate as nextTurn} from 'node:timers/promises';
import test from 'node:test';
import {uploadParameterEntries} from '../src/parameter-upload.mjs';

const bytes = Uint8Array.of(1, 2, 3, 4);
const sha256 = createHash('sha256').update(bytes).digest('hex');
const entries = (count: number) =>
  Array.from({length: count}, (_, index) => ({
    position: index + 5,
    target: `weight-${index}`,
    shape: [1],
    dtype: 'float32',
    bindingKind: 'q4-candidate',
    outputBytes: 4,
    ranges: [{offset: 0, bytes: 4, sha256}]
  }));

type TestUpload = <Tensor>(
  input: ReturnType<typeof entries>,
  options: {
    createTensor: (entry: ReturnType<typeof entries>[number]) => Tensor;
    readRange: () => ArrayBuffer;
    writeRange: (tensor: Tensor, offset: number, data: Uint8Array) => unknown;
    signal?: AbortSignal | undefined;
    expectedCount?: number | undefined;
  }
) => Promise<Tensor[]>;

test('uploads 684 ordered Elephant tensors without changing the 682-tensor default', async () => {
  const writes: number[][] = [];
  const callbacks = {
    createTensor: (entry: {position: number}) => entry.position,
    readRange: () => bytes.slice().buffer,
    writeRange: (tensor: number, offset: number, data: Uint8Array) => writes.push([tensor, offset, ...data])
  };
  const output = await (uploadParameterEntries as unknown as TestUpload)(entries(684), {
    ...callbacks,
    expectedCount: 684
  });
  assert.equal(output.length, 684);
  assert.deepEqual(writes.at(-1), [688, 0, 1, 2, 3, 4]);
  await assert.rejects((uploadParameterEntries as unknown as TestUpload)(entries(684), callbacks), /manifest/);
  await assert.rejects(
    (uploadParameterEntries as unknown as TestUpload)(entries(683), {...callbacks, expectedCount: 683}),
    /manifest/
  );
});

test('abort during a range digest observes late rejection without writing', async (t) => {
  let entered!: () => void, rejectHash!: (reason?: unknown) => void;
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  t.mock.method(crypto.subtle, 'digest', () => {
    entered();
    return new Promise((_resolve, reject) => {
      rejectHash = reject;
    });
  });
  const controller = new AbortController();
  const reason = Error('cancel range digest');
  let writes = 0;
  const job = (uploadParameterEntries as unknown as TestUpload)(entries(682), {
    signal: controller.signal,
    createTensor: () => ({}),
    readRange: () => bytes.slice().buffer,
    writeRange: () => writes++
  });
  await started;
  controller.abort(reason);
  await assert.rejects(job, (error) => error === reason);
  assert.equal(writes, 0);
  rejectHash(Error('late digest failure'));
  await nextTurn();
});

test('abort during a submitted write waits for the write to settle', async () => {
  let entered!: () => void, finish!: () => void;
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  const controller = new AbortController();
  const reason = Error('cancel pending write');
  let ended = false;
  const job = (uploadParameterEntries as unknown as TestUpload)(entries(682), {
    signal: controller.signal,
    createTensor: () => ({}),
    readRange: () => bytes.slice().buffer,
    writeRange: () => {
      entered();
      return new Promise<void>((resolve) => {
        finish = resolve;
      });
    }
  }).finally(() => {
    ended = true;
  });
  await started;
  controller.abort(reason);
  await nextTurn();
  assert.equal(ended, false);
  finish();
  await assert.rejects(job, (error) => error === reason);
});
