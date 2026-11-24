import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test from 'node:test';
import type {AssetCache} from '../src/assets.js';
import {createGemma4V2OfflinePreparation, pinnedGemma4V2OfflineAssets} from '../src/strands/gemma4-v2-offline.mjs';
import {GEMMA4_V2_REPOSITORY, GEMMA4_V2_REVISION} from '../src/strands/gemma4-v2-source.mjs';

class MemoryCache implements AssetCache {
  readonly entries = new Map<string, Response>();
  async match(key: string) {
    return this.entries.get(key)?.clone();
  }
  async put(key: string, response: Response) {
    this.entries.set(key, response.clone());
  }
}

test('pinned checkpoint assets use the immutable revision directory with a trailing slash', () => {
  const {artifacts, ...files} = pinnedGemma4V2OfflineAssets([]);
  assert.deepEqual(artifacts, []);
  const expected = `https://huggingface.co/${GEMMA4_V2_REPOSITORY}/resolve/${GEMMA4_V2_REVISION}/`;
  for (const asset of Object.values(files)) {
    assert.equal(asset.base.href, expected);
    assert.ok(asset.base.pathname.endsWith('/'));
    assert.equal(asset.base.search, '');
    assert.equal(asset.base.hash, '');
  }
});

function verifiable<T extends {hasCompleteOfflineWeights(): Promise<boolean>}>(source: T) {
  return {...source, verifyOfflineWeights: source.hasCompleteOfflineWeights.bind(source)};
}

function sampleAssets() {
  const base = new URL('https://example.test/pinned/');
  const contents = new Map<string, Uint8Array>();
  const asset = (name: string) => {
    const bytes = new TextEncoder().encode(name);
    contents.set(new URL(name, base).href, bytes);
    return {base, name, metadata: {bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex')}};
  };
  return {
    contents,
    assets: {
      tokenizer: asset('tokenizer.json'),
      tokenizerConfig: asset('tokenizer_config.json'),
      readout: asset('readout.safetensors'),
      modelConfig: asset('config.json'),
      decisionConfig: asset('decision_config.json'),
      processorConfig: asset('processor_config.json'),
      artifacts: [asset('model.wasm'), asset('vision.wasm')]
    }
  };
}

test('prepares every pinned static asset and weights before reporting offline ready', async (t) => {
  const {assets, contents} = sampleAssets();
  const cache = new MemoryCache();
  let requests = 0;
  t.mock.method(globalThis, 'fetch', async (input: string) => {
    requests++;
    const bytes = contents.get(String(input));
    assert.ok(bytes);
    return new Response(Uint8Array.from(bytes));
  });
  let weightsComplete = false;
  const source = {
    async hasCompleteOfflineWeights() {
      return weightsComplete;
    },
    async prepareWeightsOffline(options?: {onProgress?: (bytes: number, total: number) => void}) {
      weightsComplete = true;
      options?.onProgress?.(3, 3);
    }
  };
  let imagesComplete = false;
  const images = {
    async isOfflineReady() {
      return imagesComplete;
    },
    async prepareOffline() {
      imagesComplete = true;
    }
  };
  const preparation = createGemma4V2OfflinePreparation(verifiable(source), cache, assets, images);
  assert.equal(await preparation.isOfflineReady(), false);
  const progress: number[] = [];
  await preparation.prepareOffline({onProgress: (event) => progress.push(event.loadedBytes)});
  assert.deepEqual(progress, [3]);
  assert.equal(requests, 8);
  assert.equal(await preparation.isOfflineReady(), true);
  imagesComplete = false;
  assert.equal(await preparation.isOfflineReady(), false);
  imagesComplete = true;
  cache.entries.delete([...cache.entries.keys()][0]!);
  assert.equal(await preparation.isOfflineReady(), false);
  assert.equal(requests, 8);
  preparation.dispose();
  await assert.rejects(preparation.prepareOffline(), /closed/i);
  await assert.rejects(preparation.isOfflineReady(), /closed/i);
});

test('rejects omitted graph assets and fails when static asset integrity differs', async (t) => {
  const {assets, contents} = sampleAssets();
  const cache = new MemoryCache();
  let preparedWeights = false;
  const source = {
    async hasCompleteOfflineWeights() {
      return false;
    },
    async prepareWeightsOffline() {
      preparedWeights = true;
    }
  };
  assert.throws(
    () =>
      createGemma4V2OfflinePreparation(
        verifiable(source),
        cache,
        {...assets, artifacts: []},
        {
          async isOfflineReady() {
            return false;
          },
          async prepareOffline() {}
        }
      ),
    /incomplete/i
  );
  t.mock.method(globalThis, 'fetch', async (input: string) => {
    const bytes = contents.get(String(input));
    assert.ok(bytes);
    return new Response(Uint8Array.from([...bytes, 1]));
  });
  let imagesComplete = false;
  const images = {
    async isOfflineReady() {
      return imagesComplete;
    },
    async prepareOffline() {
      imagesComplete = true;
    }
  };
  const preparation = createGemma4V2OfflinePreparation(verifiable(source), cache, assets, images);
  await assert.rejects(preparation.prepareOffline());
  assert.equal(preparedWeights, false);
  assert.equal(await preparation.isOfflineReady(), false);
  preparation.dispose();
});

test('cancelled preparation can be retried without marking the model ready', async (t) => {
  const {assets, contents} = sampleAssets();
  const cache = new MemoryCache();
  let complete = false;
  const source = {
    async hasCompleteOfflineWeights() {
      return complete;
    },
    async prepareWeightsOffline() {
      complete = true;
    }
  };
  let started!: () => void;
  const waiting = new Promise<void>((resolve) => {
    started = resolve;
  });
  let hold = true;
  t.mock.method(globalThis, 'fetch', async (input: string, init?: RequestInit) => {
    const bytes = contents.get(String(input));
    assert.ok(bytes);
    if (!hold) return new Response(Uint8Array.from(bytes));
    started();
    const signal = init?.signal;
    assert.ok(signal);
    return new Promise<Response>((_resolve, reject) => {
      if (signal.aborted) reject(signal.reason);
      else signal.addEventListener('abort', () => reject(signal.reason), {once: true});
    });
  });
  let imagesComplete = false;
  const images = {
    async isOfflineReady() {
      return imagesComplete;
    },
    async prepareOffline() {
      imagesComplete = true;
    }
  };
  const preparation = createGemma4V2OfflinePreparation(verifiable(source), cache, assets, images);
  const controller = new AbortController();
  const first = preparation.prepareOffline({signal: controller.signal});
  await waiting;
  controller.abort(new Error('cancelled'));
  await assert.rejects(first, /cancelled/i);
  assert.equal(await preparation.isOfflineReady(), false);
  hold = false;
  await preparation.prepareOffline();
  assert.equal(await preparation.isOfflineReady(), true);
  preparation.dispose();
});

test('readiness and preparation reject when disposed during a pending weight check', async (t) => {
  const {assets, contents} = sampleAssets();
  const cache = new MemoryCache();
  t.mock.method(globalThis, 'fetch', async (input: string) => {
    const bytes = contents.get(String(input));
    assert.ok(bytes);
    return new Response(Uint8Array.from(bytes));
  });
  let finishCheck!: (ready: boolean) => void;
  let started!: () => void;
  const pending = new Promise<void>((resolve) => {
    started = resolve;
  });
  const source = {
    hasCompleteOfflineWeights() {
      started();
      return new Promise<boolean>((resolve) => {
        finishCheck = resolve;
      });
    },
    async prepareWeightsOffline() {
      assert.fail('should not prepare after disposal');
    }
  };
  const images = {
    async isOfflineReady() {
      return true;
    },
    async prepareOffline() {}
  };
  const preparation = createGemma4V2OfflinePreparation(verifiable(source), cache, assets, images);
  const preparing = preparation.prepareOffline();
  await pending;
  const readiness = preparation.isOfflineReady();
  const preparationRejected = assert.rejects(preparing, /closed/i);
  const readinessRejected = assert.rejects(readiness, /closed/i);
  preparation.dispose();
  await Promise.all([preparationRejected, readinessRejected]);
  finishCheck(true);
  await assert.rejects(preparation.isOfflineReady(), /closed/i);
});

test('one caller can cancel shared preparation without cancelling another caller', async (t) => {
  const {assets, contents} = sampleAssets();
  const cache = new MemoryCache();
  t.mock.method(globalThis, 'fetch', async (input: string) => {
    const bytes = contents.get(String(input));
    assert.ok(bytes);
    return new Response(Uint8Array.from(bytes));
  });
  let complete = false;
  let starts = 0;
  let started!: () => void;
  let finish!: () => void;
  const waiting = new Promise<void>((resolve) => {
    started = resolve;
  });
  const pending = new Promise<void>((resolve) => {
    finish = resolve;
  });
  const source = {
    async hasCompleteOfflineWeights() {
      return complete;
    },
    async prepareWeightsOffline(options?: {signal?: AbortSignal | null}) {
      starts++;
      started();
      await pending;
      options?.signal?.throwIfAborted();
      complete = true;
    }
  };
  const images = {
    async isOfflineReady() {
      return true;
    },
    async prepareOffline() {}
  };
  const preparation = createGemma4V2OfflinePreparation(verifiable(source), cache, assets, images);
  const controller = new AbortController();
  const first = preparation.prepareOffline({signal: controller.signal});
  const second = preparation.prepareOffline();
  await waiting;
  controller.abort(new Error('first caller cancelled'));
  await assert.rejects(first, /first caller cancelled/i);
  finish();
  await second;
  assert.equal(starts, 1);
  assert.equal(await preparation.isOfflineReady(), true);
  preparation.dispose();
});

test('a failed static asset cache write keeps earlier files and permits retry', async (t) => {
  const {assets, contents} = sampleAssets();
  const cache = new MemoryCache();
  const originalPut = cache.put.bind(cache);
  let failing = true;
  cache.put = async (key, response) => {
    if (failing && key.includes(assets.readout.metadata.sha256)) throw new Error('cache write failed');
    return originalPut(key, response);
  };
  const requests = new Map<string, number>();
  t.mock.method(globalThis, 'fetch', async (input: string) => {
    const url = String(input);
    requests.set(url, (requests.get(url) ?? 0) + 1);
    const bytes = contents.get(url);
    assert.ok(bytes);
    return new Response(Uint8Array.from(bytes));
  });
  let weightChecks = 0;
  const source = {
    async hasCompleteOfflineWeights() {
      weightChecks++;
      return true;
    },
    async prepareWeightsOffline() {
      assert.fail('weights are already complete');
    }
  };
  const images = {
    async isOfflineReady() {
      return true;
    },
    async prepareOffline() {
      assert.fail('images are already complete');
    }
  };
  const preparation = createGemma4V2OfflinePreparation(verifiable(source), cache, assets, images);
  await assert.rejects(preparation.prepareOffline(), /could not be cached/i);
  assert.equal(weightChecks, 0);
  assert.equal(await preparation.isOfflineReady(), false);
  const retained = [...cache.entries.keys()];
  assert.equal(retained.length, 2);
  failing = false;
  await preparation.prepareOffline();
  assert.equal(await preparation.isOfflineReady(), true);
  for (const key of retained) assert.ok(cache.entries.has(key));
  assert.equal(requests.get(new URL(assets.tokenizer.name, assets.tokenizer.base).href), 1);
  assert.equal(requests.get(new URL(assets.tokenizerConfig.name, assets.tokenizerConfig.base).href), 1);
  assert.equal(requests.get(new URL(assets.readout.name, assets.readout.base).href), 2);
  preparation.dispose();
});

test('offline preparation never fetches missing static assets', async (t) => {
  const {assets} = sampleAssets();
  const cache = new MemoryCache();
  let requests = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    throw new Error('offline preparation must not fetch');
  });
  const source = {
    async hasCompleteOfflineWeights() {
      return false;
    },
    async verifyOfflineWeights() {
      return false;
    },
    async prepareWeightsOffline() {
      throw new Error('offline preparation must not fetch weights');
    }
  };
  const images = {
    async isOfflineReady() {
      return false;
    },
    async prepareOffline() {
      throw new Error('offline preparation must not fetch codecs');
    }
  };
  const preparation = createGemma4V2OfflinePreparation(source, cache, assets, images, true);
  await assert.rejects(preparation.prepareOffline(), /offline/i);
  assert.equal(requests, 0);
  preparation.dispose();
});

test('cancelling explicit weight revalidation preserves the reason and allows retry', async (t) => {
  const {assets, contents} = sampleAssets();
  const cache = new MemoryCache();
  t.mock.method(
    globalThis,
    'fetch',
    async (input: string) => new Response(Uint8Array.from(contents.get(String(input))!))
  );
  let hold = true;
  let entered!: () => void;
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  const source = {
    async hasCompleteOfflineWeights() {
      return true;
    },
    async verifyOfflineWeights(signal?: AbortSignal | null) {
      if (!hold) return true;
      entered();
      await new Promise<void>((_resolve, reject) =>
        signal!.addEventListener('abort', () => reject(signal!.reason), {once: true})
      );
      return false;
    },
    async prepareWeightsOffline() {
      assert.fail('intact weights must not be fetched');
    }
  };
  const images = {
    async isOfflineReady() {
      return true;
    },
    async prepareOffline() {
      assert.fail('codecs already cached');
    }
  };
  const preparation = createGemma4V2OfflinePreparation(source, cache, assets, images);
  const controller = new AbortController();
  const reason = new Error('stop weight revalidation');
  const pending = preparation.prepareOffline({signal: controller.signal});
  await started;
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
  hold = false;
  await preparation.prepareOffline();
  assert.equal(await preparation.isOfflineReady(), true);
  preparation.dispose();
});

test('offline preparation refuses missing codecs or invalid weights without contacting the network', async (t) => {
  const {assets, contents} = sampleAssets();
  const cache = new MemoryCache();
  let requests = 0;
  t.mock.method(globalThis, 'fetch', async (input: string) => {
    requests++;
    return new Response(Uint8Array.from(contents.get(String(input))!));
  });
  let weightsReady = true;
  let imagesReady = true;
  const source = {
    async hasCompleteOfflineWeights() {
      return true;
    },
    async verifyOfflineWeights() {
      return weightsReady;
    },
    async prepareWeightsOffline() {
      assert.fail('offline cannot repair weights');
    }
  };
  const images = {
    async isOfflineReady() {
      return imagesReady;
    },
    async prepareOffline() {
      assert.fail('offline cannot fetch codecs');
    }
  };
  const online = createGemma4V2OfflinePreparation(source, cache, assets, images);
  await online.prepareOffline();
  const before = requests;
  const offline = createGemma4V2OfflinePreparation(source, cache, assets, images, true);
  imagesReady = false;
  await assert.rejects(offline.prepareOffline(), /image decoder.*offline/i);
  assert.equal(requests, before);
  imagesReady = true;
  weightsReady = false;
  await assert.rejects(offline.prepareOffline(), /weights.*offline/i);
  assert.equal(requests, before);
  offline.dispose();
  online.dispose();
});
