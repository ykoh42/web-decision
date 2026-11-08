import assert from 'node:assert/strict';
import {getEventListeners} from 'node:events';
import {setImmediate as nextTurn} from 'node:timers/promises';
import test from 'node:test';
import {elephantFiles, openElephantSource} from '../src/elephant/source.mjs';
import {ELEPHANT_MODEL} from '../src/elephant/contract.mjs';
import {mapika4Files, openMapika4Source} from '../src/mapika4b/source.mjs';
import {MAPIKA4_MODEL} from '../src/mapika4b/contract.mjs';
import {openFlashSource} from '../src/pplx/source.mjs';
import {FLASH_GRAPH_SHA256, FLASH_MANIFEST_SHA256} from '../src/pplx/contract.mjs';

const cases = [
  {
    name: 'Perplexity',
    open: openFlashSource,
    first: 'manifest.pending.json',
    files: {
      'manifest.pending.json': {bytes: 527849, sha256: FLASH_MANIFEST_SHA256},
      'model.pending.wasm': {bytes: 4952155, sha256: FLASH_GRAPH_SHA256},
      'tokenizer.json': {bytes: 19989492, sha256: '6f32ce20dc35f57a7f9ad1eac03525bd7d30f9df8cea6507e958279cc3657706'},
      'tokenizer_config.json': {bytes: 1268, sha256: '188cf0b9db5cfd6e92f1b63c646f5975be6f8383c6c9a2fa5dc2b48a8b229402'}
    }
  },
  {name: 'Mapika 4B', open: openMapika4Source, first: 'inventory.json', files: mapika4Files(MAPIKA4_MODEL)},
  {name: 'Elephant', open: openElephantSource, first: 'inventory.json', files: elephantFiles(ELEPHANT_MODEL)}
];

function deferred() {
  let resolve, reject;
  const promise = new Promise((success, failure) => {
    resolve = success;
    reject = failure;
  });
  return {promise, resolve, reject};
}

for (const item of cases) {
  test(`${item.name} cancels sibling fetches and body on initial asset failure`, async (t) => {
    const entered = deferred();
    const bodyRead = deferred();
    const pending = [];
    const signals = [];
    let cancelled = 0;
    const body = new ReadableStream({
      pull() {
        entered.resolve();
        return bodyRead.promise;
      },
      cancel() {
        cancelled++;
      }
    });
    t.mock.method(globalThis, 'fetch', async (url, {signal}) => {
      const name = new URL(url).pathname.split('/').at(-1);
      signals.push(signal);
      if (name === item.first) {
        await entered.promise;
        return new Response(null, {status: 500});
      }
      if (name === 'tokenizer_config.json') return new Response(body, {status: 200});
      const wait = deferred();
      pending.push(wait);
      return wait.promise;
    });
    const caller = new AbortController();
    const job = item.open(new URL('https://example.test/assets/'), {signal: caller.signal});
    await assert.rejects(job, (error) => error.code === 'ASSET_LOAD_FAILED' && /500/.test(error.cause?.message));
    assert.equal(signals.length, 4);
    assert.ok(signals.every((signal) => signal.aborted));
    assert.equal(cancelled, 1);
    assert.equal(caller.signal.aborted, false);
    assert.equal(getEventListeners(caller.signal, 'abort').length, 0);
    for (const wait of pending) wait.reject(Error('late fetch failure'));
    bodyRead.reject(Error('late body failure'));
    await nextTurn();
  });

  test(`${item.name} preserves caller abort and cancels late responses`, async (t) => {
    const pending = [];
    const signals = [];
    t.mock.method(globalThis, 'fetch', (_url, {signal}) => {
      signals.push(signal);
      const wait = deferred();
      pending.push(wait);
      return wait.promise;
    });
    const caller = new AbortController();
    const reason = Error('caller cancellation');
    const job = item.open(new URL('https://example.test/assets/'), {signal: caller.signal});
    while (pending.length < 4) await nextTurn();
    caller.abort(reason);
    await assert.rejects(job, (error) => error === reason);
    assert.ok(signals.every((signal) => signal.aborted));
    assert.equal(getEventListeners(caller.signal, 'abort').length, 0);
    let cancelled = 0;
    pending[0].resolve(new Response(new ReadableStream({cancel: () => cancelled++})));
    for (const wait of pending.slice(1)) wait.reject(Error('late fetch failure'));
    await nextTurn();
    assert.equal(cancelled, 1);
  });

  test(`${item.name} checks caller cancellation after the batch before parsing`, async (t) => {
    const byLength = new Map(Object.values(item.files).map((file) => [file.bytes, file.sha256]));
    t.mock.method(crypto.subtle, 'digest', (_algorithm, bytes) => {
      const sha = byLength.get(bytes.byteLength);
      assert.ok(sha);
      // These synthetic zero bytes are deliberately not authenticated assets.
      return Promise.resolve(Uint8Array.from(Buffer.from(sha, 'hex')).buffer);
    });
    t.mock.method(globalThis, 'fetch', (url) => {
      const name = new URL(url).pathname.split('/').at(-1);
      let remaining = item.files[name].bytes;
      return new Response(
        new ReadableStream({
          pull(controller) {
            if (!remaining) return controller.close();
            const length = Math.min(remaining, 64 * 1024);
            remaining -= length;
            controller.enqueue(new Uint8Array(length));
          }
        })
      );
    });
    const caller = new AbortController();
    const reason = Error('cancel before parsing');
    const check = caller.signal.throwIfAborted.bind(caller.signal);
    let checks = 0;
    caller.signal.throwIfAborted = () => {
      if (++checks === 3) caller.abort(reason);
      check();
    };
    await assert.rejects(
      item.open(new URL('https://example.test/assets/'), {signal: caller.signal}),
      (error) => error === reason
    );
    assert.equal(checks, 3);
    assert.equal(getEventListeners(caller.signal, 'abort').length, 0);
  });
}
