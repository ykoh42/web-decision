import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {test} from 'node:test';
import {createGemma4V2RunnerFactory} from '../src/strands/gemma4-v2-runner-factory.mjs';

function fixtures() {
  const base = new URL('https://example.test/verified/');
  const contents = new Map<string, Uint8Array>();
  const asset = (name: string, bytes: Uint8Array) => {
    contents.set(new URL(name, base).href, bytes);
    return {base, name, metadata: {bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex')}};
  };
  return {
    contents,
    assets: {
      wasm: asset('graph.wasm', new Uint8Array([0, 97, 115, 109])),
      localFrequency: asset('local.bin', new Uint8Array(512)),
      globalFrequency: asset('global.bin', new Uint8Array(1024))
    }
  };
}

test('verifies each graph asset before requesting a GPU device', async (t) => {
  const {contents, assets} = fixtures();
  const requests: string[] = [];
  t.mock.method(globalThis, 'fetch', async (input: string) => {
    requests.push(String(input));
    const bytes = contents.get(String(input));
    assert.ok(bytes);
    return new Response(Uint8Array.from(bytes));
  });
  const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  let descriptor: GPUDeviceDescriptor | undefined;
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      gpu: {
        async requestAdapter() {
          return {
            limits: {
              maxBufferSize: 1000,
              maxStorageBufferBindingSize: 900,
              maxStorageBuffersPerShaderStage: 8,
              maxComputeWorkgroupStorageSize: 16384,
              maxComputeInvocationsPerWorkgroup: 256
            },
            async requestDevice(value: GPUDeviceDescriptor) {
              descriptor = value;
              throw new Error('device stopped');
            }
          };
        }
      }
    }
  });
  t.after(() =>
    original
      ? Object.defineProperty(globalThis, 'navigator', original)
      : Reflect.deleteProperty(globalThis, 'navigator')
  );
  const factory = createGemma4V2RunnerFactory({
    assets,
    async createImageInserter() {
      throw new Error('unreachable');
    }
  });
  await assert.rejects(factory({source: {} as never, readout: new Uint8Array(), offline: false}), /device stopped/);
  assert.equal(requests.length, 3);
  assert.equal(descriptor?.requiredLimits?.maxStorageBufferBindingSize, 900);
});

test('rejects changed graph bytes before attempting GPU allocation', async (t) => {
  const {contents, assets} = fixtures();
  contents.set(assets.wasm.base.href + assets.wasm.name, new Uint8Array([1, 97, 115, 109]));
  t.mock.method(
    globalThis,
    'fetch',
    async (input: string) => new Response(Uint8Array.from(contents.get(String(input))!))
  );
  const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  let requested = false;
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      gpu: {
        async requestAdapter() {
          requested = true;
          return null;
        }
      }
    }
  });
  t.after(() =>
    original
      ? Object.defineProperty(globalThis, 'navigator', original)
      : Reflect.deleteProperty(globalThis, 'navigator')
  );
  const factory = createGemma4V2RunnerFactory({
    assets,
    async createImageInserter() {
      throw new Error('unreachable');
    }
  });
  await assert.rejects(
    factory({source: {} as never, readout: new Uint8Array(), offline: false}),
    /hash|digest|integrity|checksum/i
  );
  assert.equal(requested, false);
});

test('does not fetch graph assets or request a device in offline mode when cache is empty', async (t) => {
  const {assets} = fixtures();
  const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  let requested = false;
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      gpu: {
        async requestAdapter() {
          requested = true;
          return null;
        }
      }
    }
  });
  t.after(() =>
    original
      ? Object.defineProperty(globalThis, 'navigator', original)
      : Reflect.deleteProperty(globalThis, 'navigator')
  );
  let fetched = false;
  t.mock.method(globalThis, 'fetch', async () => {
    fetched = true;
    throw new Error('unexpected network');
  });
  const cache = {
    async match() {
      return undefined;
    },
    async put() {
      throw new Error('unexpected cache write');
    }
  };
  const factory = createGemma4V2RunnerFactory({
    assets,
    async createImageInserter() {
      throw new Error('unreachable');
    }
  });
  await assert.rejects(factory({source: {} as never, readout: new Uint8Array(), cache, offline: true}), /offline/i);
  assert.equal(requested, false);
  assert.equal(fetched, false);
});

test('rejects a same-size corrupted cached graph before offline GPU allocation', async (t) => {
  const {assets, contents} = fixtures();
  const entries = new Map<string, Response>();
  for (const asset of Object.values(assets)) {
    const key = new URL('/.web-decision-cache/sha256/', asset.base);
    key.searchParams.set('digest', asset.metadata.sha256);
    const bytes = Uint8Array.from(contents.get(new URL(asset.name, asset.base).href)!);
    if (asset === assets.wasm) bytes[0] = bytes[0]! ^ 1;
    entries.set(key.href, new Response(bytes));
  }
  const cache = {
    async match(key: string) {
      return entries.get(key)?.clone();
    },
    async put() {
      throw new Error('unexpected cache write');
    }
  };
  const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  let requested = false;
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      gpu: {
        async requestAdapter() {
          requested = true;
          return null;
        }
      }
    }
  });
  t.after(() =>
    original
      ? Object.defineProperty(globalThis, 'navigator', original)
      : Reflect.deleteProperty(globalThis, 'navigator')
  );
  let fetched = false;
  t.mock.method(globalThis, 'fetch', async () => {
    fetched = true;
    throw new Error('unexpected network');
  });
  const factory = createGemma4V2RunnerFactory({
    assets,
    async createImageInserter() {
      throw new Error('unreachable');
    }
  });
  await assert.rejects(factory({source: {} as never, readout: new Uint8Array(), cache, offline: true}), /offline/i);
  assert.equal(requested, false);
  assert.equal(fetched, false);
});
