import test from 'node:test';
import assert from 'node:assert/strict';
import {uploadMapikaWeights} from '../src/mapika/upload.js';

function fixture() {
  const labels = {ids: Array.from({length: 255}, (_, i) => i + 1000)};
  const records = [];
  let offset = 0;
  for (let i = 0; i <= 320; i++) {
    const nbytes = i === 320 ? 1020 : 4;
    records.push({index: i, shape: [nbytes / 4], dtype: 'int32', offset, nbytes});
    offset += nbytes;
  }
  const all = new Uint8Array(offset),
    view = new DataView(all.buffer);
  for (let i = 0; i < 320; i++) view.setInt32(i * 4, i, true);
  for (const [i, id] of labels.ids.entries()) view.setInt32(320 * 4 + i * 4, id, true);
  const shards = [400, 888, offset - 1288].map((nbytes, i) => ({path: String(i), nbytes}));
  const chunks = [];
  let pos = 0;
  for (const shard of shards) {
    chunks.push(all.slice(pos, pos + shard.nbytes));
    pos += shard.nbytes;
  }
  const tensors = [],
    weights = [],
    events = [];
  const runtime = {
    empty(shape) {
      const bytes = new Uint8Array(shape[0] * 4);
      const tensor = {byteOffset: 0, getDataPtr: () => bytes};
      tensors.push(bytes);
      return tensor;
    },
    lib: {
      webGPUContext: {
        copyRawBytesToBuffer(data, ptr, targetOffset, nbytes) {
          assert.equal(data.byteLength, nbytes);
          assert.equal(targetOffset % 4, 0);
          assert.equal(nbytes % 4, 0);
          ptr.set(data, targetOffset);
          events.push('copy');
        }
      }
    }
  };
  const device = {
    async sync() {
      events.push('sync');
    }
  };
  const read = async (path) => chunks[Number(path)].slice().buffer;
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
    },
    all,
    chunks,
    tensors,
    events
  };
}

test('reassembles split verified shards and matches compiled labels before copying', async () => {
  const {args, all, tensors, events} = fixture();
  await uploadMapikaWeights(args);
  const actual = new Uint8Array(all.length);
  let offset = 0;
  for (const bytes of tensors) {
    actual.set(bytes, offset);
    offset += bytes.length;
  }
  assert.deepEqual(actual, all);
  assert.equal(args.weights.length, 321);
  assert.equal(events.filter((event) => event === 'sync').length, 3);
});

test('rejects a tokenizer label mismatch and a lost GPU during upload', async () => {
  const changed = fixture();
  changed.chunks[1][884] ^= 1;
  await assert.rejects(uploadMapikaWeights(changed.args), {code: 'INVALID_ASSET', message: /labels/i});
  const lost = fixture();
  lost.args.device.sync = async () => {
    lost.args.lifecycle.loss = {reason: 'test'};
  };
  await assert.rejects(uploadMapikaWeights(lost.args), {code: 'MODEL_LOAD_FAILED'});
});

test('cancellation prevents later shards from being read', async () => {
  const value = fixture(),
    controller = new AbortController(),
    reason = Error('cancel');
  const paths = [],
    read = value.args.read;
  value.args.read = async (path) => {
    paths.push(path);
    if (path === '1') controller.abort(reason);
    return read(path);
  };
  value.args.signal = controller.signal;
  await assert.rejects(uploadMapikaWeights(value.args), (error) => error === reason);
  assert.deepEqual(paths, ['0', '1']);
});
