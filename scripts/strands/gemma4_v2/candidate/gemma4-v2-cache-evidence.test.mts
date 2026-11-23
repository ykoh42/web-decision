import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {test} from 'node:test';
import {verifyGemmaV2Cache} from './gemma4-v2-cache-evidence.mts';

function installCache(t: import('node:test').TestContext, expected: Uint8Array, stored: Uint8Array) {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'caches');
  const digest = createHash('sha256').update(expected).digest('hex');
  const key = new Request(`https://example.test/.web-decision-cache/sha256/?digest=${digest}`);
  const marker = new Request('https://example.test/.web-decision-cache/complete/');
  Object.defineProperty(globalThis, 'caches', {
    configurable: true,
    value: {
      async open() {
        return {
          async keys() {
            return [key, marker];
          },
          async match(request: Request) {
            return request.url === key.url ? new Response(Uint8Array.from(stored)) : undefined;
          }
        };
      }
    }
  });
  t.after(() =>
    original ? Object.defineProperty(globalThis, 'caches', original) : Reflect.deleteProperty(globalThis, 'caches')
  );
}

test('rehashes the actual cache bytes and ignores non-digest markers', async (t) => {
  const bytes = new Uint8Array([1, 2, 3]);
  installCache(t, bytes, bytes);
  assert.deepEqual(await verifyGemmaV2Cache(), {verifiedBytes: 3, verifiedEntries: 1});
});

test('rejects a same-size corrupted cache entry', async (t) => {
  installCache(t, new Uint8Array([1, 2, 3]), new Uint8Array([1, 2, 4]));
  await assert.rejects(verifyGemmaV2Cache(), /Corrupted cached entry/);
});
