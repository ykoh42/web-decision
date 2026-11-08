import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {selectFakeClefRows} from '../src/clef/rows.mjs';
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
function fixture() {
  const layout = [3, 2, 4, 3],
    columns = 64,
    files = new Map(),
    banks = [];
  for (let bank = 0; bank < 4; bank++) {
    const rows = layout[bank];
    const make = (kind, width, dtype) => {
      const bytes = Buffer.alloc(rows * width * (dtype === 'uint32' ? 4 : 2));
      if (dtype === 'uint32')
        for (let i = 0; i < bytes.length / 4; i++) bytes.writeUInt32LE((bank * 1000 + i) >>> 0, i * 4);
      else for (let i = 0; i < bytes.length / 2; i++) bytes.writeUInt16LE(0x3800 + bank, i * 2);
      const path = `${bank}-${kind}`;
      files.set(path, bytes);
      const chunks = [];
      for (let offset = 0; offset < bytes.length; offset += 13) {
        const part = bytes.subarray(offset, offset + 13);
        chunks.push({offset, length: part.length, sha256: hash(part)});
      }
      return {path, shape: [rows, width], dtype, bytes: bytes.length, chunks};
    };
    banks.push({rows, packed: make('packed', 8, 'uint32'), scales: make('scales', 2, 'float16')});
  }
  return {layout, columns, banks, files};
}
const options = (f, readRange) => ({
  fake: true,
  layout: f.layout,
  columns: f.columns,
  readRange: readRange ?? ((entry, part) => f.files.get(entry.path).subarray(part.offset, part.offset + part.length))
});

test('abort during selected-row hashing returns promptly and handles a late rejection', async (t) => {
  const f = fixture(),
    original = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  let started, rejectHash;
  const entered = new Promise((resolve) => {
    started = resolve;
  });
  const pending = new Promise((_, reject) => {
    rejectHash = reject;
  });
  Object.defineProperty(globalThis, 'crypto', {
    configurable: true,
    value: {
      subtle: {
        digest() {
          started();
          return pending;
        }
      }
    }
  });
  t.after(() => (original ? Object.defineProperty(globalThis, 'crypto', original) : delete globalThis.crypto));
  const controller = new AbortController(),
    reason = Error('selected hash cancelled');
  const operation = selectFakeClefRows([0], f.banks, {...options(f), signal: controller.signal});
  const outcome = operation.then(
    () => 'resolved',
    (error) => error
  );
  await entered;
  controller.abort(reason);
  const observed = await Promise.race([outcome, new Promise((resolve) => setTimeout(() => resolve('pending'), 50))]);
  rejectHash(Error('late selected hash failure'));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(observed, reason);
});

test('abort after a selected-row digest settles takes precedence over hash mismatch', async (t) => {
  const f = fixture(),
    original = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  let started, resolveHash;
  const entered = new Promise((resolve) => {
    started = resolve;
  });
  const pending = new Promise((resolve) => {
    resolveHash = resolve;
  });
  Object.defineProperty(globalThis, 'crypto', {
    configurable: true,
    value: {
      subtle: {
        digest() {
          started();
          return pending;
        }
      }
    }
  });
  t.after(() => (original ? Object.defineProperty(globalThis, 'crypto', original) : delete globalThis.crypto));
  const controller = new AbortController(),
    reason = Error('selected hash cancelled');
  const operation = selectFakeClefRows([0], f.banks, {...options(f), signal: controller.signal});
  await entered;
  resolveHash(new Uint8Array(32).buffer);
  queueMicrotask(() => queueMicrotask(() => controller.abort(reason)));
  await assert.rejects(operation, (error) => error === reason);
});

test('selects stable independent rows across bank boundaries and duplicates', async () => {
  const f = fixture(),
    ids = [11, 0, 3, 5, 9, 11];
  const result = await selectFakeClefRows(ids, f.banks, options(f));
  assert.deepEqual(result.unique, [11, 0, 3, 5, 9]);
  assert.deepEqual([...result.local], [0, 1, 2, 3, 4, 0]);
  assert.deepEqual([...result.rowBanks], [3, 0, 1, 2, 3]);
  assert.equal(result.packed[0], 3016);
  assert.equal(result.scales[0], 0x3803);
});

test('only intersecting ranges are read; selected corruption and nonfinite scales are rejected', async () => {
  const f = fixture(),
    reads = [];
  const read = (entry, part) => {
    reads.push([entry.path, part.offset]);
    return f.files.get(entry.path).subarray(part.offset, part.offset + part.length);
  };
  const untouchedPath = f.banks[3].packed.path;
  f.files.get(untouchedPath)[0] ^= 1;
  const result = await selectFakeClefRows([0, 0], f.banks, options(f, read));
  assert.deepEqual([...result.local], [0, 0]);
  assert.deepEqual(reads, [
    ['0-packed', 0],
    ['0-packed', 13],
    ['0-packed', 26],
    ['0-scales', 0]
  ]);
  assert.equal(result.packed.length, 8);
  // A range intersecting the chosen row is authenticated in full, including
  // bytes outside that row.
  f.files.get('0-packed')[35] ^= 1;
  await assert.rejects(selectFakeClefRows([0], f.banks, options(f)), /hash mismatch/);
  f.files.get('0-packed')[35] ^= 1;
  const scale = f.banks[0].scales,
    bytes = f.files.get(scale.path);
  bytes.writeUInt16LE(0x7e00, 0);
  for (const part of scale.chunks) part.sha256 = hash(bytes.subarray(part.offset, part.offset + part.length));
  await assert.rejects(selectFakeClefRows([0], f.banks, options(f)), /Nonfinite/);
});

test('row-spanning chunks and the final partial chunk cover each selected row exactly', async () => {
  const f = fixture(),
    reads = [];
  const result = await selectFakeClefRows(
    [2, 3, 11],
    f.banks,
    options(f, (entry, part) => {
      reads.push(`${entry.path}:${part.offset}`);
      return f.files.get(entry.path).subarray(part.offset, part.offset + part.length);
    })
  );
  assert.deepEqual([...result.rowBanks], [0, 1, 3]);
  assert.equal(result.packed[0], 16);
  assert.equal(result.packed[8], 1000);
  assert.equal(result.packed[16], 3016);
  assert.ok(reads.includes('0-packed:91'));
  assert.ok(!reads.some((value) => value.startsWith('2-')));
});

test('snapshots bank metadata before callbacks and rejects cancellation', async () => {
  const f = fixture();
  let calls = 0;
  const result = await selectFakeClefRows(
    [11, 0],
    f.banks,
    options(f, (entry, part) => {
      if (++calls === 1) {
        f.banks[3] = f.banks[0];
        f.banks[1].packed.chunks.length = 0;
      }
      return f.files.get(entry.path).subarray(part.offset, part.offset + part.length);
    })
  );
  assert.deepEqual(result.unique, [11, 0]);
  assert.deepEqual([...result.rowBanks], [3, 0]);
  const g = fixture(),
    controller = new AbortController();
  await assert.rejects(
    selectFakeClefRows([0], g.banks, {
      ...options(g, (entry, part) => {
        controller.abort();
        return g.files.get(entry.path).subarray(part.offset, part.offset + part.length);
      }),
      signal: controller.signal
    }),
    {name: 'AbortError'}
  );
});

test('real 65535/65536 bank boundary selects only its four authenticated ranges', async () => {
  const {selectClefRows} = await import('../src/clef/rows.mjs');
  const limit = 4 * 1024 * 1024,
    zeros = Buffer.alloc(limit),
    digest = new Map();
  const lengthHash = (length) => {
    if (!digest.has(length)) digest.set(length, hash(zeros.subarray(0, length)));
    return digest.get(length);
  };
  const layout = [65536, 65536, 65536, 51712];
  const stream = (bank, rows, width, dtype) => {
    const bytes = rows * width * (dtype === 'uint32' ? 4 : 2),
      chunks = [];
    for (let offset = 0; offset < bytes; offset += limit) {
      const length = Math.min(limit, bytes - offset);
      chunks.push({offset, length, sha256: lengthHash(length)});
    }
    return {path: `${bank}-${dtype}`, shape: [rows, width], dtype, bytes, chunks};
  };
  const banks = layout.map((rows, bank) => ({
    rows,
    packed: stream(bank, rows, 512, 'uint32'),
    scales: stream(bank, rows, 128, 'float16')
  }));
  const reads = [];
  const result = await selectClefRows([65535, 65536, 65535], banks, {
    readRange(entry, part) {
      reads.push([entry.path, part.offset]);
      return zeros.subarray(0, part.length);
    }
  });
  assert.deepEqual(result.unique, [65535, 65536]);
  assert.deepEqual([...result.local], [0, 1, 0]);
  assert.deepEqual([...result.rowBanks], [0, 1]);
  assert.equal(reads.length, 4);
  assert.deepEqual(
    reads.map(([path]) => path),
    ['0-uint32', '0-float16', '1-uint32', '1-float16']
  );
});

test('range callbacks cannot replace the trusted descriptor or later chunk metadata', async () => {
  for (const asynchronous of [false, true]) {
    const f = fixture();
    const read = async (entry, part) => {
      const original = f.files.get(entry.path).subarray(part.offset, part.offset + part.length);
      const altered = Buffer.from(original);
      altered[0] ^= 0xff;
      if (asynchronous) await Promise.resolve();
      part.sha256 = hash(altered);
      part.offset = 0;
      part.length = altered.length;
      entry.chunks[0].sha256 = part.sha256;
      return altered;
    };
    await assert.rejects(selectFakeClefRows([0], f.banks, options(f, read)), /hash mismatch/);
  }
  const f = fixture();
  const result = await selectFakeClefRows(
    [0],
    f.banks,
    options(f, (entry, part) => {
      const bytes = f.files.get(entry.path).subarray(part.offset, part.offset + part.length);
      part.offset = 99;
      part.length = 1;
      entry.chunks.length = 0;
      return bytes;
    })
  );
  assert.equal(result.packed[7], 7);
  assert.equal(result.scales[0], 0x3800);
});
