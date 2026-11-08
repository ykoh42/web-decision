import assert from 'node:assert/strict';
import {getEventListeners} from 'node:events';
import {readFile} from 'node:fs/promises';
import {setImmediate as nextTurn} from 'node:timers/promises';
import test from 'node:test';
import {openNariSource} from '../src/nari/source.mjs';
import {NARI_FULL, NARI_MIXED} from '../src/nari/variants.mjs';

const metadata = await readFile(new URL('./fixtures/nari/metadata.json', import.meta.url));
const mixedMetadata = await readFile(new URL('./fixtures/nari/mixed-metadata.json', import.meta.url));
const copy = (bytes) => Uint8Array.from(bytes).buffer;
const base = 'https://example.test/nari/';

test('pinned source accepts only inventory ranges and rejects corruption', async () => {
  let calls = 0;
  const reader = async (_base, entry) => {
    calls++;
    if (entry.path === 'nari-metadata.json') return {bytes: copy(metadata)};
    return {bytes: new ArrayBuffer(entry.length)};
  };
  const source = await openNariSource(base, {reader});
  assert.equal(source.plan.ordered.length, 730);
  assert.equal(source.inventory.length, 729);
  assert.equal(Object.isFrozen(source.inventory[0].chunks), true);
  await assert.rejects(source.readRange('config.json', 0, 5), {code: 'INVALID_ASSET'});
  assert.equal(calls, 1);
  await assert.rejects(source.readAsset('config.json'), {code: 'INVALID_ASSET'});
  source.close();
  await assert.rejects(source.readAsset('config.json'), {code: 'INVALID_ASSET'});
});

test('rejects altered metadata before exposing any ranges', async () => {
  const altered = Uint8Array.from(metadata);
  altered[0] ^= 1;
  await assert.rejects(openNariSource(base, {reader: async () => ({bytes: altered.buffer})}), {code: 'INVALID_ASSET'});
});

test('variants accept only their own pinned metadata', async () => {
  const source = await openNariSource(base, {variant: NARI_MIXED, reader: async () => ({bytes: copy(mixedMetadata)})});
  assert.equal(source.plan.ordered.length, 606);
  assert.equal(source.inventory.length, 605);
  source.close();
  for (const [variant, bytes] of [
    [NARI_FULL, mixedMetadata],
    [NARI_MIXED, metadata]
  ])
    await assert.rejects(openNariSource(base, {variant, reader: async () => ({bytes: copy(bytes)})}), {
      code: 'INVALID_ASSET'
    });
  await assert.rejects(
    openNariSource(base, {variant: {...NARI_MIXED}, reader: async () => ({bytes: copy(mixedMetadata)})}),
    {code: 'INVALID_ASSET'}
  );
});

test('closing the source aborts an outstanding range read', async () => {
  let entered;
  const started = new Promise((resolve) => {
    entered = resolve;
  });
  const reader = async (_base, entry, options) => {
    if (entry.path === 'nari-metadata.json') return {bytes: copy(metadata)};
    return new Promise((_resolve, reject) => {
      options.signal.addEventListener('abort', () => reject(options.signal.reason), {once: true});
      entered();
    });
  };
  const source = await openNariSource(base, {reader});
  const pending = source.readAsset('config.json');
  await started;
  source.close();
  await assert.rejects(pending, {name: 'AbortError'});
});

function deferred() {
  let resolve, reject;
  const promise = new Promise((success, failure) => {
    resolve = success;
    reject = failure;
  });
  return {promise, resolve, reject};
}

test('metadata reader and hash cancellation preserve the caller reason', async (t) => {
  for (const stage of ['reader', 'hash']) {
    const delayed = deferred();
    const entered = deferred();
    if (stage === 'hash')
      t.mock.method(crypto.subtle, 'digest', () => {
        entered.resolve();
        return delayed.promise;
      });
    const controller = new AbortController();
    const reason = Error(`cancel metadata ${stage}`);
    const pending = openNariSource(base, {
      signal: controller.signal,
      reader: () => {
        if (stage === 'reader') {
          entered.resolve();
          return delayed.promise;
        }
        return {bytes: copy(metadata)};
      }
    });
    await entered.promise;
    controller.abort(reason);
    await assert.rejects(pending, (error) => error === reason);
    delayed.reject(Error('late metadata failure'));
    await nextTurn();
    t.mock.restoreAll();
  }
});

test('range reader and hash cancellation preserve the caller reason', async (t) => {
  const config = JSON.parse(metadata).assets.find((entry) => entry.path === 'config.json');
  const chunk = config.chunks[0];
  for (const stage of ['reader', 'hash']) {
    const delayed = deferred();
    const entered = deferred();
    const source = await openNariSource(base, {
      reader: (_base, entry) => {
        if (entry.path === 'nari-metadata.json') return {bytes: copy(metadata)};
        if (stage === 'reader') {
          entered.resolve();
          return delayed.promise;
        }
        return {bytes: new ArrayBuffer(entry.length)};
      }
    });
    if (stage === 'hash')
      t.mock.method(crypto.subtle, 'digest', () => {
        entered.resolve();
        return delayed.promise;
      });
    const controller = new AbortController();
    const reason = Error(`cancel range ${stage}`);
    const pending = source.readRange(config.path, chunk.offset, chunk.bytes, {signal: controller.signal});
    await entered.promise;
    controller.abort(reason);
    await assert.rejects(pending, (error) => error === reason);
    delayed.reject(Error('late range failure'));
    await nextTurn();
    source.close();
    t.mock.restoreAll();
  }
});

test('full asset hash observes both operation cancellation and source close', async (t) => {
  const config = JSON.parse(metadata).assets.find((entry) => entry.path === 'config.json');
  for (const kind of ['operation', 'close']) {
    const delayed = deferred();
    const entered = deferred();
    const source = await openNariSource(base, {
      reader: (_base, entry) => ({
        bytes: entry.path === 'nari-metadata.json' ? copy(metadata) : new ArrayBuffer(entry.length)
      })
    });
    let hashes = 0;
    t.mock.method(crypto.subtle, 'digest', () => {
      if (++hashes === 1) return Promise.resolve(Uint8Array.from(Buffer.from(config.chunks[0].sha256, 'hex')).buffer);
      entered.resolve();
      return delayed.promise;
    });
    const controller = new AbortController();
    const reason = Error('cancel full hash');
    const pending = source.readAsset(config.path, {signal: controller.signal});
    await entered.promise;
    if (kind === 'operation') controller.abort(reason);
    else source.close();
    await assert.rejects(pending, (error) => (kind === 'operation' ? error === reason : error.name === 'AbortError'));
    delayed.reject(Error('late full hash failure'));
    await nextTurn();
    source.close();
    t.mock.restoreAll();
  }
});

test('operation abort detaches the full-hash source listener before digest settles', async (t) => {
  const config = JSON.parse(metadata).assets.find((entry) => entry.path === 'config.json');
  const NativeController = AbortController;
  const controllers = [];
  t.mock.method(
    globalThis,
    'AbortController',
    class extends NativeController {
      constructor() {
        super();
        controllers.push(this);
      }
    }
  );
  const source = await openNariSource(base, {
    reader: (_base, entry) => ({
      bytes: entry.path === 'nari-metadata.json' ? copy(metadata) : new ArrayBuffer(entry.length)
    })
  });
  const sourceSignal = controllers[0].signal;
  const entered = deferred();
  const delayed = deferred();
  let hashes = 0;
  t.mock.method(crypto.subtle, 'digest', () => {
    if (++hashes === 1) return Promise.resolve(Uint8Array.from(Buffer.from(config.chunks[0].sha256, 'hex')).buffer);
    entered.resolve();
    return delayed.promise;
  });
  const controller = new NativeController();
  const reason = Error('cancel operation');
  const pending = source.readAsset(config.path, {signal: controller.signal});
  await entered.promise;
  assert.equal(getEventListeners(sourceSignal, 'abort').length, 1);
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
  assert.equal(getEventListeners(sourceSignal, 'abort').length, 0);
  assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
  delayed.reject(Error('late digest failure'));
  await nextTurn();
  source.close();
});

test('rejects non-HTTP and unsafe base URLs before reading', async () => {
  for (const value of ['file:///tmp/', 'https://user:pass@example.test/', 'https://example.test/?query=1'])
    await assert.rejects(
      openNariSource(value, {
        reader: () => {
          throw Error('unexpected read');
        }
      }),
      {code: 'INVALID_ASSET'}
    );
});
