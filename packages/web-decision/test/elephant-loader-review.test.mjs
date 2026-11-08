import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {access, readFile} from 'node:fs/promises';
import {setImmediate as nextTurn} from 'node:timers/promises';
import test from 'node:test';
import {
  ELEPHANT_LIMITS,
  ELEPHANT_MODEL,
  loadElephantCandidate,
  loadFakeElephantCandidate,
  validateElephantAdapter,
  validateElephantOptions
} from '../src/elephant/loader.mjs';
import {SLOT_LABELS} from '../src/elephant/protocol.mjs';
import {indexElephantFiles, openElephantSource} from '../src/elephant/source.mjs';
import {createElephantTokenizer} from '../src/elephant/tokenizer.mjs';
import {uploadParameterEntries} from '../src/parameter-upload.mjs';

const root = new URL('../../../models/elephant-integration-20261030/', import.meta.url);
const paths = {
  'inventory.json': 'connected-meta-gpu-stage-20261101-v1/range-inventory.pending.json',
  'model.wasm': 'dlight-meta-gpu-stage-20261101-v1/model.patched.pending.wasm',
  'tokenizer.json': 'source/tokenizer.json',
  'tokenizer_config.json': 'source/tokenizer_config.json',
  'weights/0000.bin': 'converted-q4-20261101-v1/0000.bin',
  'weights/0683.bin': 'converted-q4-20261101-v1/0683.bin'
};

test('the local Elephant inventory validates all 684 entries and authenticates the first and last range', async (t) => {
  if (process.env.WEB_DECISION_SKIP_LOCAL_MODEL_ASSETS === '1') return t.skip('Local model assets disabled.');
  try {
    await Promise.all(Object.values(paths).map((path) => access(new URL(path, root))));
  } catch {
    t.skip('Local Elephant model assets are not available');
    return;
  }
  const inventoryBytes = await readFile(new URL(paths['inventory.json'], root));
  assert.equal(
    createHash('sha256').update(inventoryBytes).digest('hex'),
    'd68385516ef60ec2feafd5d7996534356ed7531de8491c45c2fd00073831366b'
  );
  const inventory = JSON.parse(inventoryBytes);
  const entries = indexElephantFiles(inventory);
  const reachedCreation = new Error('all entries validated');
  await assert.rejects(
    uploadParameterEntries(entries, {
      expectedCount: 684,
      createTensor() {
        throw reachedCreation;
      },
      readRange() {},
      writeRange() {}
    }),
    (error) => error === reachedCreation
  );

  t.mock.method(globalThis, 'fetch', async (url, options = {}) => {
    const name = new URL(url).pathname.slice(1);
    assert.ok(Object.hasOwn(paths, name), `Unexpected request: ${name}`);
    const data = await readFile(new URL(paths[name], root));
    if (!options.headers?.Range) return new Response(data);
    const match = /^bytes=(\d+)-(\d+)$/.exec(options.headers.Range);
    assert.ok(match);
    const start = Number(match[1]);
    const end = Number(match[2]);
    return new Response(data.subarray(start, end + 1), {
      status: 206,
      headers: {'Content-Range': `bytes ${start}-${end}/${data.length}`}
    });
  });
  const source = await openElephantSource(new URL('https://assets.example/'));
  assert.equal(source.entries.length, 684);
  for (const index of [0, 683]) {
    const entry = source.entries[index];
    const part = entry.ranges[0];
    const bytes = await source.readRange(entry, 0, part);
    assert.equal(bytes.byteLength, part.bytes);
    assert.equal(createHash('sha256').update(new Uint8Array(bytes)).digest('hex'), part.sha256);
  }
  const entry = source.entries[0];
  await assert.rejects(source.readRange(entry, 0, {...entry.ranges[0]}), {code: 'INVALID_ASSET'});
});

test('a corrupted last parameter is rejected before it is uploaded', async () => {
  const data = Uint8Array.of(1, 2, 3, 4);
  const sha256 = createHash('sha256').update(data).digest('hex');
  const inventory = {
    state: 'local-ranges-pending-review',
    parameters: Array.from({length: 684}, (_, index) => ({
      position: index + 5,
      target: `weight-${index}`,
      shape: [1],
      dtype: 'float32',
      kind: 'file',
      path: `${String(index).padStart(4, '0')}.bin`,
      fileOffset: 0,
      bytes: 4,
      ranges: [{offset: 0, bytes: 4, sha256}]
    }))
  };
  const written = [];
  await assert.rejects(
    uploadParameterEntries(indexElephantFiles(inventory), {
      expectedCount: 684,
      createTensor: (entry) => entry.position,
      readRange: (entry) => (entry.position === 688 ? new Uint8Array(4).buffer : data.slice().buffer),
      writeRange: (tensor) => written.push(tensor)
    }),
    /Range hash mismatch/
  );
  assert.equal(written.length, 683);
  assert.equal(written.at(-1), 687);
});

function loadFixture() {
  const events = [];
  const device = {lost: new Promise(() => {}), destroy: () => events.push('destroy')};
  const adapter = {
    features: new Set(['shader-f16']),
    limits: {...ELEPHANT_LIMITS},
    requestDevice: async () => device
  };
  const owner = {
    lifecycle: {loss: null, disposed: false},
    uncaptured: [],
    disposalGuard: {blocked: false, retain: () => events.push('retain')}
  };
  return {
    events,
    device,
    adapter,
    options: {model: ELEPHANT_MODEL, assetsUrl: 'https://assets.example/'},
    deps: {
      fake: true,
      gpu: {requestAdapter: async () => adapter},
      openSource: async () => ({wasm: new Uint8Array(), tokenizer: {}, tokenizerConfig: {}}),
      prepareTokenizer: () => ({}),
      instantiate: async () => ({dispose: () => events.push('dispose')}),
      initialize: () => owner,
      upload: async () => [],
      runner: () => ({}),
      session: () => ({})
    }
  };
}

test('a device returned after load cancellation is destroyed', async () => {
  const f = loadFixture();
  const controller = new AbortController();
  let resolveDevice;
  f.adapter.requestDevice = () => new Promise((resolve) => (resolveDevice = resolve));
  const pending = loadFakeElephantCandidate({...f.options, signal: controller.signal}, f.deps);
  while (!resolveDevice) await nextTurn();
  controller.abort();
  await assert.rejects(pending, {name: 'AbortError'});
  resolveDevice(f.device);
  await nextTurn();
  assert.deepEqual(f.events, ['destroy']);
});

test('device loss during upload retains the initialized runtime', async () => {
  const f = loadFixture();
  let lose, entered;
  f.device.lost = new Promise((resolve) => (lose = resolve));
  const started = new Promise((resolve) => (entered = resolve));
  f.deps.upload = async (_owner, _source, signal) =>
    new Promise((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(signal.reason), {once: true});
      entered();
    });
  const pending = loadFakeElephantCandidate(f.options, f.deps);
  await started;
  lose({reason: 'unknown'});
  await assert.rejects(pending, {code: 'MODEL_LOAD_FAILED'});
  assert.deepEqual(f.events, ['retain']);
});

test('invalid load options are rejected without evaluating user getters', () => {
  const {options} = loadFixture();
  let accessed = false;
  const getter = {
    ...options,
    get cache() {
      accessed = true;
      return false;
    }
  };
  const cases = [
    null,
    [],
    Object.create({custom: true}),
    getter,
    {...options, unknown: true},
    {...options, model: 'wrong'},
    {...options, assetsUrl: ''},
    {...options, assetsUrl: 'file:///model'},
    {...options, assetsUrl: 'https://user:password@example.test/'},
    {...options, assetsUrl: 'https://example.test/?query'},
    {...options, cache: true},
    {...options, offline: true},
    {...options, maxPrefetchBytes: 1},
    {...options, onProgress() {}},
    {...options, debertaReadback: true}
  ];
  for (const value of cases) assert.throws(() => validateElephantOptions(value), {code: 'INVALID_INPUT'});
  assert.equal(accessed, false);
  assert.equal(validateElephantOptions(options).base.href, 'https://assets.example/');
  assert.throws(() => loadElephantCandidate(options), /runtime bindings/);
  assert.throws(() => loadFakeElephantCandidate(options, {}), /Fake dependencies/);
});

test('unsupported adapters fail before loading model assets', async () => {
  for (const change of ['missing', 'f16', 'fallback', 'limits']) {
    const f = loadFixture();
    let opened = false;
    f.deps.openSource = () => {
      opened = true;
    };
    if (change === 'missing') f.deps.gpu = null;
    if (change === 'f16') f.adapter.features.clear();
    if (change === 'fallback') f.adapter.info = {isFallbackAdapter: true};
    if (change === 'limits') f.adapter.limits.maxStorageBufferBindingSize--;
    await assert.rejects(loadFakeElephantCandidate(f.options, f.deps), {code: 'UNSUPPORTED_ENVIRONMENT'});
    assert.equal(opened, false);
  }
  assert.throws(() => validateElephantAdapter(null), {code: 'UNSUPPORTED_ENVIRONMENT'});
});

test('malformed inventory metadata cannot bind parameters to the graph', () => {
  const makeInventory = () => ({
    state: 'local-ranges-pending-review',
    parameters: Array.from({length: 684}, (_, index) => ({
      position: index + 5,
      target: `weight-${index}`,
      shape: [1],
      dtype: 'float32',
      kind: 'file',
      path: `${String(index).padStart(4, '0')}.bin`,
      fileOffset: 0,
      bytes: 4,
      ranges: []
    }))
  });
  for (const mutate of [
    (value) => (value.state = 'unverified'),
    (value) => value.parameters.pop(),
    (value) => (value.parameters[1].position = 5),
    (value) => (value.parameters[1].target = value.parameters[0].target),
    (value) => (value.parameters[1].kind = 'rotary'),
    (value) => (value.parameters[1].path = '../weight'),
    (value) => (value.parameters[1].fileOffset = 4),
    (value) => (value.parameters[1].shape = [0]),
    (value) => (value.parameters[1].bytes = -1),
    (value) => (value.parameters[1].ranges = null)
  ]) {
    const inventory = makeInventory();
    mutate(inventory);
    assert.throws(() => indexElephantFiles(inventory), {code: 'INVALID_ASSET'});
  }
});

test('the tokenizer rejects invalid metadata and incompatible answer slots', async () => {
  const data = {normalizer: {type: 'NFC'}, added_tokens: [{content: '<reserved>'}]};
  const config = {tokenizer_class: 'Qwen3_5Tokenizer'};
  class SyntheticTokenizer {
    encode(label) {
      return {ids: [SLOT_LABELS.indexOf(label)]};
    }
    decode(ids) {
      return SLOT_LABELS[ids[0]];
    }
  }
  for (const [Tokenizer, tokens, settings] of [
    [null, data, config],
    [SyntheticTokenizer, data, {tokenizer_class: 'other'}],
    [SyntheticTokenizer, {...data, normalizer: {type: 'NFD'}}, config],
    [SyntheticTokenizer, {...data, added_tokens: [{content: ''}]}, config],
    [SyntheticTokenizer, {...data, added_tokens: [{content: 'same'}, {content: 'same'}]}, config],
    [
      class {
        encode() {
          return {ids: [0, 1]};
        }
      },
      data,
      config
    ],
    [SyntheticTokenizer, data, config]
  ]) {
    await assert.rejects(createElephantTokenizer(Tokenizer, tokens, settings), {code: 'INVALID_ASSET'});
  }
});

test('load failures release an unused device and retain an initialized owner', async () => {
  const beforeRuntime = loadFixture();
  beforeRuntime.deps.instantiate = async () => {
    throw new Error('bad wasm');
  };
  await assert.rejects(loadFakeElephantCandidate(beforeRuntime.options, beforeRuntime.deps), {
    code: 'MODEL_LOAD_FAILED'
  });
  assert.deepEqual(beforeRuntime.events, ['destroy']);
  const duringUpload = loadFixture();
  duringUpload.deps.upload = async () => {
    throw new Error('bad range');
  };
  await assert.rejects(loadFakeElephantCandidate(duringUpload.options, duringUpload.deps), {code: 'MODEL_LOAD_FAILED'});
  assert.deepEqual(duringUpload.events, ['retain']);
});

test('source rejects HTTP errors, truncated metadata and an exact-size corrupt inventory', async (t) => {
  const base = new URL('https://assets.example/');
  const fetchMock = t.mock.method(globalThis, 'fetch');
  fetchMock.mock.mockImplementation(async () => new Response('missing', {status: 404}));
  await assert.rejects(openElephantSource(base), {code: 'ASSET_LOAD_FAILED'});
  fetchMock.mock.mockImplementation(async () => new Response(new Uint8Array(2)));
  await assert.rejects(openElephantSource(base), {code: 'ASSET_LOAD_FAILED'});
  fetchMock.mock.mockImplementation((url) =>
    new URL(url).pathname === '/inventory.json'
      ? Promise.resolve(new Response(new Uint8Array(431398)))
      : new Promise(() => {})
  );
  await assert.rejects(openElephantSource(base), {code: 'INVALID_ASSET'});
});

test('source cancellation interrupts a pending fetch without replacing its reason', async (t) => {
  const controller = new AbortController();
  const reason = new Error('caller stopped loading');
  t.mock.method(globalThis, 'fetch', () => new Promise(() => {}));
  const pending = openElephantSource(new URL('https://assets.example/'), {signal: controller.signal});
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
});

test('successful loader transfers ordered weights into a session after the pre-device guard', async () => {
  const f = loadFixture();
  const events = f.events;
  f.deps.beforeDevice = () => events.push('guard');
  f.adapter.requestDevice = async () => {
    events.push('device');
    return f.device;
  };
  f.deps.upload = async () => ['first', 'last'];
  f.deps.runner = (owner) => owner.weights;
  f.deps.session = ({runner}) => runner;
  assert.deepEqual(await loadFakeElephantCandidate(f.options, f.deps), ['first', 'last']);
  assert.deepEqual(events, ['guard', 'device']);
});

test('pre-device failure stops loading before requesting a device', async () => {
  const f = loadFixture();
  let requested = false;
  f.deps.beforeDevice = () => {
    throw new Error('resource reserve');
  };
  f.adapter.requestDevice = async () => {
    requested = true;
    return f.device;
  };
  await assert.rejects(loadFakeElephantCandidate(f.options, f.deps), {code: 'MODEL_LOAD_FAILED'});
  assert.equal(requested, false);
});

test('device loss while instantiation is pending disposes a late runtime', async () => {
  const f = loadFixture();
  let lose, resolveRuntime;
  f.device.lost = new Promise((resolve) => (lose = resolve));
  f.deps.instantiate = () => new Promise((resolve) => (resolveRuntime = resolve));
  const pending = loadFakeElephantCandidate(f.options, f.deps);
  while (!resolveRuntime) await nextTurn();
  lose({reason: 'unknown'});
  await assert.rejects(pending, {code: 'MODEL_LOAD_FAILED'});
  assert.deepEqual(f.events, ['destroy']);
  resolveRuntime({dispose: () => f.events.push('late-dispose')});
  await nextTurn();
  assert.deepEqual(f.events, ['destroy', 'late-dispose']);
});

test('partial initialization does not dispose a runtime that may own GPU work', async () => {
  const f = loadFixture();
  f.deps.initialize = () => {
    throw new Error('partial GPU initialization');
  };
  await assert.rejects(loadFakeElephantCandidate(f.options, f.deps), {code: 'MODEL_LOAD_FAILED'});
  assert.deepEqual(f.events, []);
});
