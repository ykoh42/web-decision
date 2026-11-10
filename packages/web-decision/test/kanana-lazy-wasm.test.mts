import assert from 'node:assert/strict';
import test from 'node:test';
import {createLazyKananaWasmLoader} from '../src/kanana/lazy-wasm.js';

test('defers Kanana WASM loading and does not let one request abort another', async () => {
  const requests: Array<{
    signal: AbortSignal;
    resolve(bytes: ArrayBuffer): void;
  }> = [];
  const loadWasm = createLazyKananaWasmLoader((signal) => {
    if (!signal) throw new Error('the request signal was not forwarded');
    return new Promise<ArrayBuffer>((resolve, reject) => {
      const abort = () => reject(signal.reason);
      signal.addEventListener('abort', abort, {once: true});
      requests.push({
        signal,
        resolve(bytes) {
          signal.removeEventListener('abort', abort);
          resolve(bytes);
        }
      });
    });
  });

  assert.equal(requests.length, 0);
  const firstController = new AbortController();
  const secondController = new AbortController();
  const first = loadWasm(firstController.signal);
  const second = loadWasm(secondController.signal);
  assert.deepEqual(
    requests.map(({signal}) => signal),
    [firstController.signal, secondController.signal]
  );

  firstController.abort(new Error('first image request cancelled'));
  await assert.rejects(first, /first image request cancelled/);

  const bytes = new ArrayBuffer(4);
  requests[1]!.resolve(bytes);
  assert.equal(await second, bytes);
  assert.equal(await loadWasm(), bytes);
  assert.equal(requests.length, 2);
});

test('a cancelled Kanana WASM read is retried on the next image request', async () => {
  let reads = 0;
  const loadWasm = createLazyKananaWasmLoader(async (signal) => {
    reads++;
    if (reads === 1) {
      if (!signal) throw new Error('the first vision load must carry its request signal');
      await new Promise<never>((_resolve, reject) => {
        signal.addEventListener('abort', () => reject(signal.reason), {once: true});
      });
    }
    return new ArrayBuffer(2);
  });
  const controller = new AbortController();
  const cancelled = loadWasm(controller.signal);
  controller.abort(new Error('cancel during vision WASM fetch'));
  await assert.rejects(cancelled, /cancel during vision WASM fetch/);
  assert.equal(reads, 1);

  const bytes = await loadWasm();
  assert.equal(bytes.byteLength, 2);
  assert.equal(reads, 2);
});
