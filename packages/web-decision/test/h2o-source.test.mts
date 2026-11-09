import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash, randomBytes, webcrypto} from 'node:crypto';
import {createIncrementalSha256} from '../src/h2o/hasher.mjs';
import {openFakeH2oSource} from '../src/h2o/source.mjs';
import {readVerifiedRange} from '../src/strands/range-reader.mjs';
import {Hash} from '../src/h2o/sha256.mjs';
import {parseStrictJson} from '../src/h2o/strict-json.mjs';

const digest = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');
const encode = (text: string) => new TextEncoder().encode(text);
const desc = (path: string, bytes: Uint8Array) => ({path, bytes: bytes.length, sha256: digest(bytes)});
const pieces = (path: string, data: Uint8Array, cuts: [number, number][]) => ({
  ...desc(path, data),
  chunks: cuts.map(([offset, length]) => ({offset, length, sha256: digest(data.slice(offset, offset + length))}))
});

test('strict JSON rejects duplicate decoded keys while allowing keys in different objects', () => {
  assert.deepEqual(parseStrictJson(encode('{"a": 1, "nested": {"a": 2}}')), {a: 1, nested: {a: 2}});
  assert.throws(() => parseStrictJson(encode('{"nested": {"key": 1, "\\u006bey": 2}}')), /Duplicate JSON object key/);
  assert.deepEqual(parseStrictJson(encode('{"text": "escaped \\\" quote", "items": [{"x": 1}]}')), {
    text: 'escaped " quote',
    items: [{x: 1}]
  });
});

test('cached fast-sha256 streams synthetic bytes matching Node and WebCrypto', async () => {
  for (const n of [0, 1, 55, 56, 63, 64, 65, 511, 100003]) {
    const data = new Uint8Array(randomBytes(n)),
      h = createIncrementalSha256(Hash);
    for (let offset = 0; offset < data.length; offset += 73) h.update(data.subarray(offset, offset + 73));
    const output = h.digest();
    assert.equal(output, digest(data));
    assert.equal(output, Buffer.from(await webcrypto.subtle.digest('SHA-256', data)).toString('hex'));
    assert.throws(() => h.update(Uint8Array.of(1)));
    assert.throws(() => h.digest());
  }
  const h = createIncrementalSha256(Hash);
  assert.throws(() => h.update([1, 2]), RangeError);
});

test('ES module exposes the incremental hasher', () => {
  const h = createIncrementalSha256(Hash);
  h.update(encode('abc'));
  assert.equal(h.digest(), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
});

function fixture() {
  const blobs = new Map([
    ['h2o-stage.log', encode('stage')],
    ['manifest.pending.json', encode('manifest')],
    ['model.pending.wasm', encode('wasm')],
    ['tokenizer/tokenizer.json', encode('tokenizer')],
    ['tokenizer/tokenizer_config.json', encode('config')],
    ['tokenizer_contract.json', encode('contract')],
    ['a.pending', Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8)],
    ['b.pending', Uint8Array.of(9, 10, 11, 12)]
  ]);
  const field = (name: string) => desc(name, blobs.get(name)!);
  const stage = field('h2o-stage.log'),
    manifest = field('manifest.pending.json'),
    token = field('tokenizer/tokenizer.json');
  const policy: {
    stage: ReturnType<typeof desc>;
    manifest: ReturnType<typeof desc>;
    manifestSha256: string;
    wasm: ReturnType<typeof desc>;
    tokenizer: ReturnType<typeof desc>;
    tokenizerConfig: ReturnType<typeof desc>;
    tokenizerContract: ReturnType<typeof desc>;
    weightCount: number;
    preparePlan: (...args: unknown[]) => Promise<unknown>;
  } = {
    stage,
    manifest,
    manifestSha256: manifest.sha256,
    wasm: field('model.pending.wasm'),
    tokenizer: token,
    tokenizerConfig: field('tokenizer/tokenizer_config.json'),
    tokenizerContract: field('tokenizer_contract.json'),
    weightCount: 2,
    async preparePlan() {
      return {
        parameters: [{...field('a.pending'), kind: 'file'}, {...field('b.pending'), kind: 'file'}, {kind: 'rotary'}]
      };
    }
  };
  const first = pieces('a.pending', blobs.get('a.pending')!, [
      [0, 4],
      [4, 4]
    ]),
    second = pieces('b.pending', blobs.get('b.pending')!, [[0, 4]]),
    tokenizer = pieces(token.path, blobs.get(token.path)!, [
      [0, 4],
      [4, 5]
    ]);
  const receipt = 'd'.repeat(64);
  const data = {
    format: 'h2o-range-inventory-v1',
    manifestSha256: manifest.sha256,
    auditReceiptSha256: receipt,
    files: [first, second, tokenizer]
  };
  return {blobs, policy, data, receipt};
}

function fakeFetch(f: ReturnType<typeof fixture>, mode = 'valid'): typeof fetch {
  return async (url, options) => {
    const name = new URL(url instanceof Request ? url.url : url).pathname.slice('/assets/'.length),
      data = f.blobs.get(name);
    assert.ok(data, `unexpected path ${name}`);
    const range = /^bytes=(\d+)-(\d+)$/.exec(new Headers(options?.headers).get('Range') ?? '');
    assert.ok(range);
    const start = Number(range[1]),
      end = Number(range[2]);
    const bytes = data.slice(start, end + 1);
    if (mode === 'tamper') bytes[0] = bytes[0]! ^ 1;
    return new Response(bytes, {
      status: mode === 'full' ? 200 : 206,
      headers: {
        'Content-Range': mode === 'wrong-range' ? 'bytes 0-0/1' : `bytes ${start}-${end}/${data.length}`,
        'Content-Length': String(bytes.length)
      }
    });
  };
}

async function openFixture(f: ReturnType<typeof fixture>, data = f.data, extra: Record<string, unknown> = {}) {
  const raw = encode(JSON.stringify(data));
  return openFakeH2oSource({
    fake: true,
    base: 'https://example.invalid/assets/',
    inventoryBytes: raw,
    inventorySha256: digest(raw),
    auditReceiptSha256: f.receipt,
    policy: f.policy as unknown as Parameters<typeof openFakeH2oSource>[0]['policy'],
    reader: readVerifiedRange,
    makeHasher: () => createIncrementalSha256(Hash),
    ...extra
  });
}

test('source plan cancellation rejects promptly and observes a late failure', async () => {
  const f = fixture(),
    original = globalThis.fetch,
    controller = new AbortController(),
    reason = new Error('synthetic cancellation'),
    fetched = [];
  let startPlan!: () => void, rejectPlan!: (reason?: unknown) => void;
  const started = new Promise<void>((resolve) => {
      startPlan = resolve;
    }),
    pending = new Promise<never>((_resolve, reject) => {
      rejectPlan = reject;
    });
  f.policy.preparePlan = (...args: unknown[]) => {
    const signal = args[3];
    assert.ok(signal instanceof AbortSignal);
    startPlan();
    return pending;
  };
  const fetch = fakeFetch(f);
  globalThis.fetch = (...args) => {
    fetched.push(new URL(args[0] instanceof Request ? args[0].url : args[0]).pathname);
    return fetch(...args);
  };
  try {
    const task = openFixture(f, f.data, {signal: controller.signal});
    await started;
    controller.abort(reason);
    await assert.rejects(task, (error) => error === reason);
    rejectPlan(new Error('late synthetic rejection'));
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(fetched.length, 2);
  } finally {
    globalThis.fetch = original;
  }
});

test('source inventory digest cancellation rejects promptly and observes a late rejection', async (t) => {
  const f = fixture(),
    controller = new AbortController(),
    reason = new Error('synthetic cancellation');
  let startDigest!: () => void, rejectDigest!: (reason?: unknown) => void;
  const started = new Promise<void>((resolve) => {
      startDigest = resolve;
    }),
    pending = new Promise<never>((_resolve, reject) => {
      rejectDigest = reject;
    });
  t.mock.method(crypto.subtle, 'digest', () => {
    startDigest();
    return pending;
  });
  let reads = 0;
  const task = openFixture(f, f.data, {
    signal: controller.signal,
    reader: () => {
      reads++;
      throw new Error('unexpected read');
    }
  });
  await started;
  controller.abort(reason);
  await assert.rejects(task, (error) => error === reason);
  rejectDigest(new Error('late synthetic digest failure'));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(reads, 0);
});

test('source cancellation beats an uncooperative reader and ignores late invalid bytes', async () => {
  const f = fixture(),
    controller = new AbortController(),
    reason = new Error('synthetic cancellation');
  let startRead!: () => void, resolveRead!: (value: {bytes: ArrayBuffer}) => void;
  const started = new Promise<void>((resolve) => {
      startRead = resolve;
    }),
    pending = new Promise<{bytes: ArrayBuffer}>((resolve) => {
      resolveRead = resolve;
    });
  const task = openFixture(f, f.data, {
    signal: controller.signal,
    reader: () => {
      startRead();
      return pending;
    }
  });
  await started;
  controller.abort(reason);
  await assert.rejects(task, (error) => error === reason);
  resolveRead({bytes: new ArrayBuffer(0)});
  await new Promise((resolve) => setImmediate(resolve));
});

test('per-range cancellation wins over late digest failure and keeps the source usable', async (t) => {
  const f = fixture(),
    controller = new AbortController(),
    reason = new Error('synthetic cancellation');
  const reader = async (_base: unknown, spec: {path: string; offset: number; length: number}) => {
    const data = f.blobs.get(spec.path)!.slice(spec.offset, spec.offset + spec.length);
    return {bytes: data.buffer};
  };
  const source = await openFixture(f, f.data, {reader});
  let startDigest!: () => void, rejectDigest!: (reason?: unknown) => void;
  const started = new Promise<void>((resolve) => {
      startDigest = resolve;
    }),
    pending = new Promise<never>((_resolve, reject) => {
      rejectDigest = reject;
    });
  t.mock.method(crypto.subtle, 'digest', () => {
    startDigest();
    return pending;
  });
  const spec = {path: 'a.pending', totalBytes: 8, ...source.inventory['a.pending']![0]!};
  const task = source.readRange('https://example.invalid/assets/', spec, {signal: controller.signal});
  await started;
  controller.abort(reason);
  await assert.rejects(task, (error) => error === reason);
  rejectDigest(new Error('late synthetic digest failure'));
  await new Promise((resolve) => setImmediate(resolve));
  t.mock.restoreAll();
  const result = await source.readRange('https://example.invalid/assets/', spec);
  assert.deepEqual([...new Uint8Array(result.bytes)], [1, 2, 3, 4]);
  source.close();
});

test('source uses actual range reader with synthetic HTTP and pins every weight range', async () => {
  const f = fixture(),
    original = globalThis.fetch;
  globalThis.fetch = fakeFetch(f);
  try {
    const source = await openFixture(f);
    assert.equal(new TextDecoder().decode(source.tokenizer), 'tokenizer');
    const spec = {path: 'a.pending', totalBytes: 8, ...source.inventory['a.pending']![0]!};
    const reply = await source.readRange('https://example.invalid/assets/', spec);
    assert.deepEqual([...new Uint8Array(reply.bytes)], [1, 2, 3, 4]);
    const full = createIncrementalSha256(Hash);
    for (const part of source.inventory['a.pending']!) {
      const result = await source.readRange('https://example.invalid/assets/', {
        path: 'a.pending',
        totalBytes: 8,
        ...part
      });
      full.update(new Uint8Array(result.bytes));
    }
    assert.equal(full.digest(), digest(f.blobs.get('a.pending')!));
    await assert.rejects(source.readRange('https://example.invalid/assets/', {...spec, path: 'b.pending'}), {
      code: 'INVALID_ASSET'
    });
    await assert.rejects(source.readRange('https://elsewhere.invalid/assets/', spec), {code: 'INVALID_ASSET'});
    await assert.rejects(
      source.readRange('https://example.invalid/assets/', spec, {cache: true} as unknown as NonNullable<
        Parameters<typeof source.readRange>[2]
      >),
      {
        code: 'INVALID_ASSET'
      }
    );
    await assert.rejects(
      source.readRange('https://example.invalid/assets/', spec, {offline: true} as unknown as NonNullable<
        Parameters<typeof source.readRange>[2]
      >),
      {
        code: 'INVALID_ASSET'
      }
    );
    await assert.rejects(
      source.readRange('https://example.invalid/assets/', spec, {cache: 0} as unknown as NonNullable<
        Parameters<typeof source.readRange>[2]
      >),
      {
        code: 'INVALID_ASSET'
      }
    );
    await assert.rejects(
      source.readRange('https://example.invalid/assets/', spec, {maxPrefetchBytes: 1} as unknown as NonNullable<
        Parameters<typeof source.readRange>[2]
      >),
      {
        code: 'INVALID_ASSET'
      }
    );
    await assert.rejects(source.readRange('https://example.invalid/assets/', {...spec, extra: true} as typeof spec), {
      code: 'INVALID_ASSET'
    });
    source.close();
    await assert.rejects(source.readRange('https://example.invalid/assets/', spec), {code: 'INVALID_ASSET'});
  } finally {
    globalThis.fetch = original;
  }
});

test('caller abort cancels a synthetic in-flight range request', async () => {
  const f = fixture(),
    original = globalThis.fetch;
  globalThis.fetch = fakeFetch(f);
  try {
    const source = await openFixture(f),
      controller = new AbortController();
    const spec = {path: 'a.pending', totalBytes: 8, ...source.inventory['a.pending']![0]!};
    let started!: () => void;
    const ready = new Promise<void>((resolve) => {
      started = resolve;
    });
    globalThis.fetch = (_url, options) =>
      new Promise((_resolve, reject) => {
        options!.signal!.addEventListener('abort', () => reject(options!.signal!.reason), {once: true});
        started();
      });
    const task = source.readRange('https://example.invalid/assets/', spec, {signal: controller.signal});
    await ready;
    controller.abort();
    await assert.rejects(task, {name: 'AbortError'});
    source.close();
  } finally {
    globalThis.fetch = original;
  }
});

test('reader rejects synthetic 200, bad Content-Range and corrupted bytes', async () => {
  for (const mode of ['full', 'wrong-range', 'tamper']) {
    const f = fixture(),
      original = globalThis.fetch;
    globalThis.fetch = fakeFetch(f, mode);
    try {
      await assert.rejects(openFixture(f), {code: 'INVALID_ASSET'});
    } finally {
      globalThis.fetch = original;
    }
  }
});

test('inventory mismatches and caller cancellation fail without a network request', async () => {
  const f = fixture(),
    original = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => {
    calls++;
    throw new Error('unexpected fetch');
  };
  try {
    await assert.rejects(openFixture(f, {...f.data, auditReceiptSha256: 'a'.repeat(64)}), {code: 'INVALID_ASSET'});
    const raw = encode(JSON.stringify(f.data));
    await assert.rejects(
      openFakeH2oSource({
        fake: true,
        base: 'https://example.invalid/assets/',
        inventoryBytes: raw,
        inventorySha256: '0'.repeat(64),
        auditReceiptSha256: f.receipt,
        policy: f.policy as unknown as Parameters<typeof openFakeH2oSource>[0]['policy'],
        reader: readVerifiedRange,
        makeHasher: () => createIncrementalSha256(Hash)
      }),
      {code: 'INVALID_ASSET'}
    );
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(openFixture(f, f.data, {signal: controller.signal}), {name: 'AbortError'});
    assert.equal(calls, 0);
  } finally {
    globalThis.fetch = original;
  }
});

test('tokenizer full-file hash mismatch is rejected even with correct chunk hashes', async () => {
  const f = fixture(),
    original = globalThis.fetch;
  f.policy.tokenizer.sha256 = '0'.repeat(64);
  f.data.files.find((item) => item.path === f.policy.tokenizer.path)!.sha256 = '0'.repeat(64);
  globalThis.fetch = fakeFetch(f);
  try {
    await assert.rejects(openFixture(f), {code: 'INVALID_ASSET'});
  } finally {
    globalThis.fetch = original;
  }
});
