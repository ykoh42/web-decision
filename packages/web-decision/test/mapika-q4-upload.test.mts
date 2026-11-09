import test from 'node:test';
import assert from 'node:assert/strict';
import {uploadMapikaQ4Weights} from '../src/mapika-q4/upload.js';

type UploadArgs = Parameters<typeof uploadMapikaQ4Weights>[0];
type TestArgs = Omit<UploadArgs, 'labels' | 'signal' | 'read' | 'lifecycle'> & {
  labels: {ids: number[]};
  signal?: AbortSignal | undefined;
  read: UploadArgs['read'];
  lifecycle: {disposed: boolean; loss: unknown};
};
type TensorRecord = UploadArgs['index']['records'][number];
type Shard = UploadArgs['index']['shards'][number];
type FakeRuntime = UploadArgs['runtime'] & {
  lib: {
    webGPUContext: {
      copyRawBytesToBuffer(data: Uint8Array, pointer: unknown, offset: number, count: number): unknown;
    };
  };
};

interface Fixture {
  args: TestArgs;
  all: Uint8Array;
  chunks: Uint8Array[];
  tensors: Uint8Array[];
  events: string[];
  dtypes: string[];
}

function fixture(): Fixture {
  const labels = {ids: Array.from({length: 255}, (_, i) => i + 1000)};
  const records: TensorRecord[] = [];
  let offset = 0;
  for (let i = 0; i <= 133; i++) {
    const nbytes = i === 133 ? 1020 : 4;
    records.push({index: i, shape: [nbytes / 4], dtype: i % 2 ? 'uint32' : 'int32', offset, nbytes});
    offset += nbytes;
  }
  const all = new Uint8Array(offset),
    view = new DataView(all.buffer);
  for (let i = 0; i < 133; i++) view.setInt32(i * 4, i, true);
  for (const [i, id] of labels.ids.entries()) view.setInt32(133 * 4 + i * 4, id, true);
  const shards: Shard[] = [400, 400, offset - 800].map((nbytes, i) => ({path: String(i), nbytes}));
  const chunks: Uint8Array[] = [];
  let pos = 0;
  for (const shard of shards) {
    chunks.push(all.slice(pos, pos + shard.nbytes));
    pos += shard.nbytes;
  }
  const tensors: Uint8Array[] = [],
    weights: unknown[] = [],
    events: string[] = [],
    dtypes: string[] = [];
  const runtime: FakeRuntime = {
    empty(shape: number[], dtype: string) {
      const bytes = new Uint8Array(shape[0]! * 4);
      tensors.push(bytes);
      dtypes.push(dtype);
      return {byteOffset: 0, getDataPtr: () => bytes};
    },
    lib: {
      webGPUContext: {
        copyRawBytesToBuffer(data, ptr, targetOffset, nbytes) {
          assert.equal(data.byteLength, nbytes);
          assert.equal(targetOffset % 4, 0);
          (ptr as Uint8Array).set(data, targetOffset);
          events.push('copy');
        }
      }
    }
  };
  const device: UploadArgs['device'] = {
    async sync() {
      events.push('sync');
    }
  };
  const read: UploadArgs['read'] = async (path: string) => chunks[Number(path)]!.slice().buffer;
  return {
    args: {
      index: {records, shards, totalBytes: offset},
      read,
      runtime,
      device,
      weights,
      labels,
      maxPrefetchBytes: 0,
      lifecycle: {loss: null, disposed: false},
      uncaptured: []
    } as unknown as TestArgs,
    all,
    chunks,
    tensors,
    events,
    dtypes
  };
}

test('streams uint32 tensors and verifies the split label buffer at index 133', async () => {
  const {args, all, tensors, events, dtypes} = fixture();
  await uploadMapikaQ4Weights(args as unknown as UploadArgs);
  const actual = new Uint8Array(all.length);
  let offset = 0;
  for (const bytes of tensors) {
    actual.set(bytes, offset);
    offset += bytes.length;
  }
  assert.deepEqual(actual, all);
  assert.ok(dtypes.includes('uint32'));
  assert.equal(args.weights.length, 134);
  assert.equal(events.filter((event) => event === 'sync').length, 3);
});

test('rejects mismatched labels and a lost device during upload', async () => {
  const changed = fixture();
  changed.chunks[1]![132]! ^= 1;
  await assert.rejects(uploadMapikaQ4Weights(changed.args as unknown as UploadArgs), {
    code: 'INVALID_ASSET',
    message: /labels/i
  });
  const lost = fixture();
  lost.args.device.sync = async () => {
    lost.args.lifecycle.loss = {reason: 'test'};
  };
  await assert.rejects(uploadMapikaQ4Weights(lost.args as unknown as UploadArgs), {code: 'MODEL_LOAD_FAILED'});
});

test('preserves cancellation and does not read the remaining shard', async () => {
  const value = fixture(),
    controller = new AbortController(),
    reason = Error('cancel');
  const paths: string[] = [],
    read = value.args.read;
  value.args.read = async (path) => {
    paths.push(path);
    if (path === '1') controller.abort(reason);
    return read(path);
  };
  value.args.signal = controller.signal;
  await assert.rejects(uploadMapikaQ4Weights(value.args as unknown as UploadArgs), (error) => error === reason);
  assert.deepEqual(paths, ['0', '1']);
});
