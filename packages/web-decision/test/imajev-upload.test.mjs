import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {setImmediate as nextTurn} from 'node:timers/promises';
import test from 'node:test';
import {uploadImajevRows} from '../src/imajev/upload.mjs';

const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const bytes = Uint8Array.from({length: 20}, (_, i) => i + 1);
const chunk = (offset) => ({offset, bytes: 4, sha256: digest(bytes.subarray(offset, offset + 4))});
const row = (offset) => ({
  path: 'shared.bin',
  offset,
  bytes: 8,
  shape: [2],
  dtype: 'float32',
  chunks: [chunk(offset), chunk(offset + 4)],
  rangeSha256: digest(bytes.subarray(offset, offset + 8))
});
const rotaryBytes = Uint8Array.of(0, 0, 128, 63);
const rotary = {
  shape: [1],
  dtype: 'float32',
  bytesHex: Buffer.from(rotaryBytes).toString('hex'),
  sha256: digest(rotaryBytes)
};

function fixture() {
  const events = [],
    tensors = [];
  const owner = {
    phase: 'prepared',
    lost: false,
    blocked: false,
    uncaptured: [],
    retained: false,
    released: false,
    hold() {
      events.push('hold');
      return {
        release: () => {
          owner.released = true;
          events.push('release');
        }
      };
    },
    retain() {
      owner.retained = true;
      events.push('retain');
    }
  };
  const args = {
    owner,
    rows: [row(4), row(12), rotary],
    async readRange(path, offset, length) {
      events.push(`read:${path}:${offset}:${length}`);
      return bytes.slice(offset, offset + length);
    },
    makeHasher() {
      const hash = createHash('sha256');
      return {update: (part) => hash.update(part), digest: () => hash.digest('hex')};
    },
    allocate(_row, size) {
      const tensor = new Uint8Array(size);
      tensors.push(tensor);
      return {tensor, capacityBytes: size};
    },
    async writeAndSettle(tensor, offset, value) {
      tensor.set(value, offset);
      events.push(`write:${offset}`);
    }
  };
  return {owner, args, events, tensors};
}

test('uploads distinct ranges of a shared file and the rotary constant in graph order', async () => {
  const f = fixture();
  const result = await uploadImajevRows(f.args);
  assert.equal(f.owner.phase, 'prepared');
  assert.equal(f.owner.released, true);
  assert.equal(f.owner.retained, false);
  assert.deepEqual(result, [bytes.slice(4, 12), bytes.slice(12, 20), rotaryBytes]);
  assert.deepEqual(
    f.events.filter((event) => event.startsWith('read:')),
    ['read:shared.bin:4:4', 'read:shared.bin:8:4', 'read:shared.bin:12:4', 'read:shared.bin:16:4']
  );
});

test('corruption or undersized allocation retains the owner and does not release it', async () => {
  for (const mode of ['corrupt', 'capacity']) {
    const f = fixture();
    if (mode === 'corrupt') f.args.readRange = async () => new Uint8Array(4);
    else f.args.allocate = () => ({tensor: {}, capacityBytes: 0});
    await assert.rejects(uploadImajevRows(f.args));
    assert.equal(f.owner.phase, 'failed');
    assert.equal(f.owner.retained, true);
    assert.equal(f.owner.released, false);
  }
});

test('abort during a submitted write waits for settlement before retaining resources', async () => {
  const f = fixture(),
    controller = new AbortController();
  let resolveWrite, started;
  const pendingWrite = new Promise((resolve) => {
    resolveWrite = resolve;
  });
  const entered = new Promise((resolve) => {
    started = resolve;
  });
  f.args.signal = controller.signal;
  f.args.writeAndSettle = async () => {
    started();
    await pendingWrite;
  };
  const result = uploadImajevRows(f.args);
  await entered;
  controller.abort();
  assert.equal(f.owner.retained, false);
  assert.equal(f.owner.released, false);
  resolveWrite();
  await assert.rejects(result, {name: 'AbortError'});
  assert.equal(f.owner.retained, true);
  assert.equal(f.owner.released, false);
});

test('abort during chunk hash retains owner and observes late failure without submitting a write', async (t) => {
  const f = fixture();
  let entered, rejectHash;
  const started = new Promise((resolve) => {
    entered = resolve;
  });
  t.mock.method(crypto.subtle, 'digest', () => {
    entered();
    return new Promise((_resolve, reject) => {
      rejectHash = reject;
    });
  });
  const controller = new AbortController();
  const reason = Error('cancel chunk hash');
  f.args.signal = controller.signal;
  const result = uploadImajevRows(f.args);
  await started;
  controller.abort(reason);
  await assert.rejects(result, (error) => error === reason);
  assert.equal(f.owner.phase, 'failed');
  assert.equal(f.owner.retained, true);
  assert.equal(f.owner.released, false);
  assert.equal(
    f.events.some((event) => event.startsWith('write:')),
    false
  );
  rejectHash(Error('late chunk hash failure'));
  await nextTurn();
});

test('snapshots the binding list before asynchronous reads', async () => {
  const f = fixture();
  let resume, started;
  const wait = new Promise((resolve) => {
    resume = resolve;
  });
  const entered = new Promise((resolve) => {
    started = resolve;
  });
  const read = f.args.readRange;
  f.args.readRange = async (...args) => {
    started();
    await wait;
    return read(...args);
  };
  const result = uploadImajevRows(f.args);
  await entered;
  f.args.rows[1].offset = 0;
  resume();
  assert.deepEqual(await result, [bytes.slice(4, 12), bytes.slice(12, 20), rotaryBytes]);
});
