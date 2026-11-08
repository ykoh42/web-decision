import assert from 'node:assert/strict';
import test from 'node:test';
import {readManifestResponse} from '../src/assets.js';
import {readPinnedManifest} from '../src/pinned-manifest.js';

const primitiveCause = 'stream failure';

function failedResponse(): Pick<Response, 'body'> {
  return {
    body: new ReadableStream<Uint8Array<ArrayBuffer>>({
      start(controller) {
        controller.error(primitiveCause);
      }
    })
  };
}

async function thrownValue(operation: () => Promise<unknown>): Promise<unknown> {
  try {
    await operation();
  } catch (cause: unknown) {
    return cause;
  }
  return undefined;
}

test('manifest readers preserve AbortError detection for primitive causes via the prototype chain', async () => {
  const originalName = Object.getOwnPropertyDescriptor(String.prototype, 'name');
  Object.defineProperty(String.prototype, 'name', {configurable: true, value: 'AbortError'});
  try {
    assert.equal(await thrownValue(() => readManifestResponse(failedResponse())), primitiveCause);
    assert.equal(
      await thrownValue(() => readPinnedManifest(failedResponse(), undefined, 'unused', 'test')),
      primitiveCause
    );
  } finally {
    if (originalName) Object.defineProperty(String.prototype, 'name', originalName);
    else Reflect.deleteProperty(String.prototype, 'name');
  }
});
