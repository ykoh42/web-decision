import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {setImmediate as nextTurn} from 'node:timers/promises';
import {uploadParameters} from '../src/nari/upload.mjs';

const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const hash = async (bytes) => digest(bytes);
const ROTARY = ['body.rotary_emb.inv_freq', 'body.rotary_emb.original_inv_freq'];
function fixture(fileCount = 728) {
  const bytes = new Uint8Array([0x12, 0x34]),
    letterBytes = new Uint8Array(52 * 2560 * 4);
  const ordered = [],
    inventory = [],
    data = new Map();
  for (let i = 0; i < fileCount; i++) {
    const path = `f-${i}.pending`,
      file = {path, bytes: 2, sha256: digest(bytes)};
    ordered.push({
      target: `body.w${i}`,
      graphArgument: `p${i}`,
      shape: [1],
      dtype: 'float16',
      source: 'planned-asset',
      file
    });
    inventory.push({...file, chunks: [{offset: 0, bytes: 2, sha256: digest(bytes)}]});
    data.set(path, bytes);
  }
  for (let i = 0; i < 2; i++)
    ordered.push({
      target: ROTARY[i],
      graphArgument: `p${fileCount + i}`,
      shape: [32],
      dtype: 'float32',
      source: 'pinned-config-rotary'
    });
  const letterFile = {path: 'letter.pending', bytes: letterBytes.length, sha256: digest(letterBytes)};
  inventory.push({...letterFile, chunks: [{offset: 0, bytes: letterBytes.length, sha256: digest(letterBytes)}]});
  data.set(letterFile.path, letterBytes);
  const owner = {
    phase: 'prepared',
    lost: false,
    retained: 0,
    held: false,
    hold() {
      if (this.held) throw Error('already held');
      this.held = true;
      return {
        release: () => {
          this.held = false;
        }
      };
    },
    retain() {
      this.retained++;
    },
    dispose() {
      if (this.held || this.retained) throw Error('owner held or retained');
    }
  };
  const writes = [],
    allocations = [],
    read = [];
  const args = {
    fake: true,
    expectedBindings: fileCount + 2,
    owner,
    plan: {ordered, letterFile},
    inventory,
    rotary: Object.fromEntries(ROTARY.map((k) => [k, Float32Array.from({length: 32}, (_, i) => 1 / (i + 1))])),
    readRange: async (path, offset, count) => {
      read.push(path);
      return data.get(path).slice(offset, offset + count);
    },
    sha256: hash,
    makeHasher: () => {
      const h = createHash('sha256');
      return {update: (b) => h.update(b), digest: async () => h.digest('hex')};
    },
    allocate: (entry, size) => {
      const tensor = {target: entry.target ?? 'letters', shape: entry.shape, dtype: entry.dtype};
      allocations.push(tensor);
      return {tensor, capacityBytes: size};
    },
    writeAndSettle: async (tensor, offset, bytes) => {
      writes.push({target: tensor.target, offset, bytes: Uint8Array.from(bytes)});
    }
  };
  return {args, owner, writes, allocations, read, data};
}

test('fake 728 files, two rotary and separate original letter rows retain graph order and aligned padding', async () => {
  const f = fixture(),
    out = await uploadParameters(f.args);
  assert.equal(f.owner.phase, 'prepared');
  assert.equal(f.owner.retained, 0);
  assert.equal(out.signed, false);
  assert.equal(f.owner.held, false);
  assert.equal(out.parameters.length, 730);
  assert.equal(f.allocations.length, 731);
  assert.equal(f.read.length, 729);
  assert.deepEqual(
    out.parameters.map((x) => x.graphArgument),
    Array.from({length: 730}, (_, i) => `p${i}`)
  );
  assert.equal(out.parameters[728].source, 'pinned-config-rotary');
  assert.equal(out.parameters[729].target, ROTARY[1]);
  assert.equal(out.letterRows.target, 'letters');
  assert.equal(out.letterSha256, f.args.plan.letterFile.sha256);
  assert.deepEqual([...f.writes[0].bytes], [0x12, 0x34, 0, 0]);
  assert.equal(f.writes[728].bytes.length, 128);
  assert.deepEqual(f.writes[728].bytes, f.writes[729].bytes);
  assert.equal(f.writes[730].bytes.length, 52 * 2560 * 4);
});

test('mixed upload binds 604 files and rejects full-Q4 binding count', async () => {
  const mixed = fixture(604);
  const result = await uploadParameters(mixed.args);
  assert.equal(result.parameters.length, 606);
  assert.equal(mixed.read.length, 605);
  assert.equal(result.parameters[604].source, 'pinned-config-rotary');
  const mismatched = fixture(604);
  mismatched.args.expectedBindings = 730;
  await assert.rejects(uploadParameters(mismatched.args));
  assert.equal(mismatched.allocations.length, 0);
  assert.equal(mismatched.owner.retained, 1);
});

test('invalid plan, shape, inventory and rotary fail closed and retain owner', async () => {
  for (const mutation of [
    (f) => f.args.plan.ordered.pop(),
    (f) => (f.args.plan.ordered[1].graphArgument = 'p0'),
    (f) => (f.args.plan.ordered[0].file.bytes = 4),
    (f) => (f.args.inventory[0].chunks[0].offset = 1),
    (f) => (f.args.inventory[0].chunks[0].bytes = 3),
    (f) => (f.args.inventory[1].path = f.args.inventory[0].path),
    (f) => (f.args.plan.ordered[728].target = ROTARY[1]),
    (f) => (f.args.rotary[ROTARY[1]][0] = 2),
    (f) => (f.args.plan.letterFile.bytes = 4)
  ]) {
    const f = fixture();
    mutation(f);
    await assert.rejects(uploadParameters(f.args));
    assert.equal(f.owner.phase, 'failed');
    assert.equal(f.owner.retained, 1);
    assert.equal(f.allocations.length, 0);
  }
});

test('sparse tensor dimensions are rejected before allocation', async () => {
  const f = fixture();
  f.args.plan.ordered[0].shape = Array(2);
  f.args.plan.ordered[0].shape[1] = 1;
  await assert.rejects(uploadParameters(f.args));
  assert.equal(f.allocations.length, 0);
  assert.equal(f.owner.retained, 1);
});

test('range mismatch and full hash mismatch after writes both retain ownership', async () => {
  for (const mode of ['range', 'whole']) {
    const f = fixture();
    if (mode === 'range') f.data.set('f-0.pending', new Uint8Array([9, 9]));
    else {
      f.args.plan.ordered[0].file.sha256 = '0'.repeat(64);
      f.args.inventory[0].sha256 = '0'.repeat(64);
    }
    await assert.rejects(uploadParameters(f.args), mode === 'range' ? /range hash/ : /whole file hash/);
    assert.equal(f.owner.phase, 'failed');
    assert.equal(f.owner.retained, 1);
    assert.equal(f.writes.length, mode === 'range' ? 0 : 1);
  }
});

test('multiple contiguous ranges preserve offsets and only pad final odd FP16 tail', async () => {
  const f = fixture(),
    bytes = Uint8Array.of(1, 2, 3, 4, 5, 6),
    file = f.args.plan.ordered[0].file;
  file.bytes = 6;
  file.sha256 = digest(bytes);
  f.args.plan.ordered[0].shape = [3];
  f.data.set(file.path, bytes);
  f.args.inventory[0] = {
    ...file,
    chunks: [
      {offset: 0, bytes: 4, sha256: digest(bytes.slice(0, 4))},
      {offset: 4, bytes: 2, sha256: digest(bytes.slice(4))}
    ]
  };
  await uploadParameters(f.args);
  assert.equal(f.writes[0].offset, 0);
  assert.deepEqual([...f.writes[0].bytes], [1, 2, 3, 4]);
  assert.equal(f.writes[1].offset, 4);
  assert.deepEqual([...f.writes[1].bytes], [5, 6, 0, 0]);
});

test('aligned ranges reuse owned verified bytes while an unaligned tail is padded', async () => {
  const f = fixture();
  const bytes = Uint8Array.of(1, 2, 3, 4, 5, 6);
  const file = f.args.plan.ordered[0].file;
  file.bytes = bytes.length;
  file.sha256 = digest(bytes);
  f.args.plan.ordered[0].shape = [3];
  f.data.set(file.path, bytes);
  f.args.inventory[0] = {
    ...file,
    chunks: [
      {offset: 0, bytes: 4, sha256: digest(bytes.subarray(0, 4))},
      {offset: 4, bytes: 2, sha256: digest(bytes.subarray(4))}
    ]
  };
  const hashed = [],
    written = [];
  let sourceView,
    guards = 0;
  f.args.readRange = async (path, offset, count) => {
    const view = f.data.get(path).subarray(offset, offset + count);
    if (path === file.path && offset === 0) sourceView = view;
    return view;
  };
  f.args.sha256 = async (data) => {
    if (hashed.length < 2) hashed.push(data);
    return digest(data);
  };
  f.args.beforeUpload = async () => {
    if (++guards === 2) sourceView.fill(9);
  };
  f.args.writeAndSettle = async (_tensor, offset, data) => {
    if (written.length < 2) written.push({offset, data, snapshot: Uint8Array.from(data)});
  };
  await uploadParameters(f.args);
  assert.notEqual(hashed[0].buffer, sourceView.buffer);
  assert.equal(written[0].data, hashed[0]);
  assert.deepEqual([...written[0].snapshot], [1, 2, 3, 4]);
  assert.equal(written[1].offset, 4);
  assert.notEqual(written[1].data, hashed[1]);
  assert.deepEqual([...written[1].snapshot], [5, 6, 0, 0]);
});

test('cancel during an aligned verified write waits and keeps the owned bytes and lease', async () => {
  const f = fixture();
  const source = Uint8Array.of(1, 2, 3, 4);
  const file = f.args.plan.ordered[0].file;
  file.bytes = source.length;
  file.sha256 = digest(source);
  f.args.plan.ordered[0].shape = [2];
  f.args.inventory[0] = {...file, chunks: [{offset: 0, bytes: 4, sha256: digest(source)}]};
  const controller = new AbortController();
  const reason = Error('cancel aligned write');
  f.args.signal = controller.signal;
  let hashed,
    written,
    settle,
    started,
    reads = 0,
    writes = 0;
  const begun = new Promise((resolve) => {
    started = resolve;
  });
  f.args.readRange = async () => {
    reads++;
    return source;
  };
  f.args.sha256 = async (bytes) => {
    hashed = bytes;
    return digest(bytes);
  };
  f.args.writeAndSettle = (_tensor, offset, bytes) => {
    writes++;
    assert.equal(offset, 0);
    written = bytes;
    started();
    return new Promise((resolve) => {
      settle = resolve;
    });
  };
  let ended = false;
  const job = uploadParameters(f.args).finally(() => {
    ended = true;
  });
  await begun;
  assert.equal(written, hashed);
  assert.notEqual(written.buffer, source.buffer);
  assert.deepEqual([...written], [1, 2, 3, 4]);
  source.fill(9);
  controller.abort(reason);
  await nextTurn();
  assert.equal(ended, false);
  assert.deepEqual([...written], [1, 2, 3, 4]);
  assert.equal(f.owner.phase, 'uploading');
  assert.equal(f.owner.retained, 0);
  assert.equal(f.owner.held, true);
  assert.throws(() => f.owner.dispose(), /held/);
  settle();
  await assert.rejects(job, (error) => error === reason);
  assert.equal(reads, 1);
  assert.equal(writes, 1);
  assert.equal(f.owner.phase, 'failed');
  assert.equal(f.owner.retained, 1);
  assert.equal(f.owner.held, true);
  assert.throws(() => f.owner.dispose(), /held or retained/);
});

test('cancel while submitted fake GPU write is unresolved waits for settlement then retains owner', async () => {
  const f = fixture(),
    controller = new AbortController();
  f.args.signal = controller.signal;
  let settle, started;
  const begun = new Promise((r) => (started = r));
  f.args.writeAndSettle = () => {
    started();
    return new Promise((r) => (settle = r));
  };
  let ended = false;
  const job = uploadParameters(f.args).finally(() => {
    ended = true;
  });
  await begun;
  controller.abort();
  await Promise.resolve();
  assert.equal(ended, false);
  assert.equal(f.owner.phase, 'uploading');
  assert.equal(f.owner.retained, 0);
  assert.throws(() => f.owner.dispose(), /held/);
  settle();
  await assert.rejects(job, {name: 'AbortError'});
  assert.equal(f.owner.phase, 'failed');
  assert.equal(f.owner.retained, 1);
  assert.equal(f.owner.held, true);
  assert.throws(() => f.owner.dispose(), /held or retained/);
});

test('cancelling range and whole file hashes retains the owner and observes late failures', async () => {
  for (const stage of ['range', 'whole']) {
    const f = fixture();
    const controller = new AbortController();
    const reason = Error(`cancel ${stage} hash`);
    let rejectHash, entered;
    const started = new Promise((resolve) => {
      entered = resolve;
    });
    const delayed = () => {
      entered();
      return new Promise((_resolve, reject) => {
        rejectHash = reject;
      });
    };
    f.args.signal = controller.signal;
    if (stage === 'range') f.args.sha256 = delayed;
    else f.args.makeHasher = () => ({update() {}, digest: delayed});
    const job = uploadParameters(f.args);
    await started;
    controller.abort(reason);
    await assert.rejects(job, (error) => error === reason);
    assert.equal(f.owner.phase, 'failed');
    assert.equal(f.owner.retained, 1);
    assert.equal(f.owner.held, true);
    assert.equal(f.writes.length, stage === 'range' ? 0 : 1);
    rejectHash(Error('late hash failure'));
    await nextTurn();
  }
});

test('device loss, insufficient allocation and rejected submission retain ownership', async () => {
  for (const mode of ['loss', 'capacity', 'rejected']) {
    const f = fixture();
    if (mode === 'loss')
      f.args.writeAndSettle = async () => {
        f.owner.lost = true;
      };
    if (mode === 'capacity') f.args.allocate = () => ({tensor: {}, capacityBytes: 2});
    if (mode === 'rejected')
      f.args.writeAndSettle = async () => {
        throw Error('sync failed');
      };
    await assert.rejects(uploadParameters(f.args));
    assert.equal(f.owner.phase, 'failed');
    assert.equal(f.owner.retained, 1);
  }
});

test('cancellation during lease release cannot report success', async () => {
  const f = fixture(),
    controller = new AbortController();
  f.args.signal = controller.signal;
  f.owner.hold = () => ({
    release() {
      controller.abort();
    }
  });
  await assert.rejects(uploadParameters(f.args), {name: 'AbortError'});
  assert.equal(f.owner.retained, 1);
  assert.equal(f.owner.phase, 'failed');
});

test('asynchronous lease release must settle before success and cancellation is observed', async () => {
  for (const cancel of [false, true]) {
    const f = fixture(),
      controller = new AbortController();
    f.args.signal = controller.signal;
    let settle, started;
    const begun = new Promise((r) => (started = r));
    f.owner.hold = () => {
      f.owner.held = true;
      return {
        release() {
          started();
          return new Promise(
            (r) =>
              (settle = () => {
                f.owner.held = false;
                r();
              })
          );
        }
      };
    };
    let ended = false;
    const job = uploadParameters(f.args).finally(() => {
      ended = true;
    });
    await begun;
    await Promise.resolve();
    assert.equal(ended, false);
    assert.throws(() => f.owner.dispose(), /held/);
    if (cancel) controller.abort();
    settle();
    if (cancel) {
      await assert.rejects(job, {name: 'AbortError'});
      assert.equal(f.owner.retained, 1);
    } else {
      await job;
      assert.equal(f.owner.phase, 'prepared');
    }
  }
});

test('rejected asynchronous lease release retains owner', async () => {
  const f = fixture();
  f.owner.hold = () => ({
    release: async () => {
      throw Error('release failed');
    }
  });
  await assert.rejects(uploadParameters(f.args), /release failed/);
  assert.equal(f.owner.phase, 'failed');
  assert.equal(f.owner.retained, 1);
});

test('resource guard failure stops the next upload and retains the owner', async () => {
  const f = fixture();
  let checks = 0;
  f.args.beforeUpload = async () => {
    if (++checks === 3) throw Error('resource guard failed');
  };
  await assert.rejects(uploadParameters(f.args), /resource guard failed/);
  assert.equal(f.writes.length, 1);
  assert.equal(f.owner.phase, 'failed');
  assert.equal(f.owner.retained, 1);
});
