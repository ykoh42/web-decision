import assert from 'node:assert/strict';
import {test} from 'node:test';
import {loadGemma4V2Model, GEMMA4_V2_MODEL, type Gemma4V2RunnerFactory} from '../src/strands/gemma4-v2-loader.mjs';
import type {Gemma4V2OwnedRunner} from '../src/strands/gemma4-v2-session.mjs';
import type {openPinnedGemma4V2Source} from '../src/strands/gemma4-v2-source.mjs';
import type {readVerifiedAsset} from '../src/assets.js';
import {GEMMA4_V2_SCORING_BYTES, GEMMA4_V2_SCORING_SHA256} from '../src/strands/gemma4-v2-scoring.mjs';
import {readFileSync} from 'node:fs';

type Source = Awaited<ReturnType<typeof openPinnedGemma4V2Source>>;
const json = (value: unknown) => new TextEncoder().encode(JSON.stringify(value)).buffer;
function readout() {
  const header = json({weight: {dtype: 'BF16', shape: [255, 1536], data_offsets: [0, 255 * 1536 * 2]}});
  const buffer = new ArrayBuffer(8 + header.byteLength + 255 * 1536 * 2);
  new DataView(buffer).setBigUint64(0, BigInt(header.byteLength), true);
  new Uint8Array(buffer, 8, header.byteLength).set(new Uint8Array(header));
  return buffer;
}
function assets() {
  return {
    'config.json': json({text_config: {num_hidden_layers: 35, hidden_size: 1536}}),
    'decision_config.json': json({
      prompt: 'strands-chat',
      attention_mode: 'prefix_bidir',
      pooling: 'last',
      max_length_validated: 32768,
      temperature: 1.5456,
      final_logit_softcapping: 30
    }),
    'processor_config.json': json({image_processor: {patch_size: 16, pooling_kernel_size: 3, max_soft_tokens: 280}}),
    'tokenizer.json': json({
      added_tokens: [
        {special: true, content: '<bos>'},
        {special: false, content: 'word'}
      ]
    }),
    'tokenizer_config.json': json({}),
    'readout.safetensors': readout()
  } as Record<string, ArrayBuffer>;
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return {promise, resolve};
}

test('loads pinned assets, connects source and runner, and disposes both once', async () => {
  const fixtures = assets();
  const seen: string[] = [];
  let sourceDisposed = 0;
  let runnerDisposed = 0;
  let releases = 0;
  let tokenEntries: unknown;
  const source = {
    dispose() {
      sourceDisposed++;
    }
  } as unknown as Source;
  const runner: Gemma4V2OwnedRunner = {
    async run(input) {
      assert.equal(input.images[0], 'decoded');
      return [0, 0];
    },
    releaseCachedMemory() {
      releases++;
    },
    dispose() {
      runnerDisposed++;
    }
  };
  const createRunner: Gemma4V2RunnerFactory = async (input) => {
    assert.equal(input.source, source);
    assert.equal(input.readout.length, 255 * 1536 * 2);
    return runner;
  };
  const readAsset = (async (_base, manifest, name) => {
    seen.push(name);
    assert.equal(typeof manifest.files[name]!.sha256, 'string');
    return fixtures[name]!;
  }) as typeof readVerifiedAsset;
  const model = await loadGemma4V2Model(
    {},
    {
      createRunner,
      readAsset,
      openSource: async () => source,
      prepareImage: async () => ({tokens: 2, value: 'decoded'}),
      createTokenizer(model) {
        tokenEntries = (model as {added_tokens: unknown}).added_tokens;
        return {
          encode(text) {
            return {ids: Array.from(text, () => 8)};
          }
        };
      }
    }
  );
  assert.equal(model.modelId, GEMMA4_V2_MODEL);
  assert.equal(await model.isOfflineReady(), false);
  await assert.rejects(model.prepareOffline(), /cache and pinned graph assets/i);
  assert.equal(seen.length, 6);
  assert.deepEqual(tokenEntries, [{special: false, content: 'word'}]);
  const result = await model.systemOne({state: 'hello', questions: {q: {instructions: 'yes?'}}, images: ['abc']});
  assert.equal(result.answers.q?.type, 'noul');
  await model.releaseCachedMemory();
  assert.equal(releases, 1);
  await model.dispose();
  await model.dispose();
  await assert.rejects(model.isOfflineReady(), /closed/i);
  await assert.rejects(model.prepareOffline(), /closed/i);
  assert.equal(runnerDisposed, 1);
  assert.equal(sourceDisposed, 1);
});

test('rejects invalid pinned configuration before opening the source', async () => {
  const fixtures = assets();
  fixtures['decision_config.json'] = json({prompt: 'unexpected'});
  let opened = false;
  await assert.rejects(
    loadGemma4V2Model(
      {},
      {
        createRunner: async () => {
          throw new Error('runner must not open');
        },
        openSource: async () => {
          opened = true;
          throw new Error('source must not open');
        },
        readAsset: (async (_base, _manifest, name) => fixtures[name]!) as typeof readVerifiedAsset
      }
    ),
    /configuration/i
  );
  assert.equal(opened, false);
});

test('releases late runner and source when load is cancelled', async () => {
  const fixtures = assets();
  const waiting = deferred<Gemma4V2OwnedRunner>();
  const entered = deferred<void>();
  const controller = new AbortController();
  let sourceDisposed = 0;
  let runnerDisposed = 0;
  const source = {
    dispose() {
      sourceDisposed++;
    }
  } as unknown as Source;
  const loading = loadGemma4V2Model(
    {signal: controller.signal},
    {
      createRunner: async () => {
        entered.resolve();
        return waiting.promise;
      },
      openSource: async () => source,
      readAsset: (async (_base, _manifest, name) => fixtures[name]!) as typeof readVerifiedAsset,
      createTokenizer() {
        return {
          encode() {
            return {ids: []};
          }
        };
      }
    }
  );
  await entered.promise;
  controller.abort(new Error('cancelled'));
  await assert.rejects(loading, /cancelled/i);
  assert.equal(sourceDisposed, 1);
  waiting.resolve({
    async run() {
      return [0, 0];
    },
    releaseCachedMemory() {},
    dispose() {
      runnerDisposed++;
    }
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(runnerDisposed, 1);
});

test('loader connects the offline coordinator to graph pins, image preparation, and source', async (t) => {
  const {existsSync} = await import('node:fs');
  const {readFile} = await import('node:fs/promises');
  const {resolve} = await import('node:path');
  const {createHash} = await import('node:crypto');
  const checkpoint = resolve(process.env.WEB_DECISION_STRANDS_V2_CHECKPOINT ?? '../strands-e2b-v2');
  if (!existsSync(resolve(checkpoint, 'tokenizer.json')))
    return t.skip('Pinned real tokenizer fixture is not available');
  const graphBytes = new Uint8Array([0, 97, 115, 109]);
  const graph = {
    base: new URL('https://example.test/'),
    name: 'graph.wasm',
    metadata: {
      bytes: graphBytes.length,
      sha256: createHash('sha256').update(graphBytes).digest('hex')
    }
  };
  const scoringBytes = readFileSync(new URL('../assets/strands-gemma4-v2/scoring.bin', import.meta.url));
  const scoring = {
    base: new URL('https://example.test/'),
    name: 'scoring.bin',
    metadata: {bytes: GEMMA4_V2_SCORING_BYTES, sha256: GEMMA4_V2_SCORING_SHA256}
  };
  const cacheEntries = new Map<string, Response>();
  const cache = {
    async match(key: string) {
      return cacheEntries.get(key)?.clone();
    },
    async put(key: string, response: Response) {
      cacheEntries.set(key, response.clone());
    }
  };
  let weights = false;
  let images = false;
  let sourceClosed = false;
  let holdReady = false;
  const entered = deferred<void>();
  const gate = deferred<void>();
  const source = {
    async hasCompleteOfflineWeights() {
      if (holdReady) {
        entered.resolve();
        await gate.promise;
      }
      return weights;
    },
    async verifyOfflineWeights() {
      return weights;
    },
    async prepareWeightsOffline(options?: {onProgress?: (bytes: number, total: number) => void}) {
      weights = true;
      options?.onProgress?.(3, 3);
    },
    dispose() {
      sourceClosed = true;
    }
  } as unknown as Source;
  const imagePreparer = {
    async prepareImage() {
      return {
        tokens: 2,
        value: {pixels: new Float32Array(), positions: new Int32Array(), tokens: 2, width: 1, height: 1}
      };
    },
    async prepareOffline() {
      images = true;
    },
    async isOfflineReady() {
      return images;
    }
  };
  t.mock.method(globalThis, 'fetch', async (input: string) => {
    const url = new URL(String(input));
    if (url.href === new URL(graph.name, graph.base).href) return new Response(graphBytes);
    if (url.href === new URL(scoring.name, scoring.base).href) return new Response(scoringBytes);
    return new Response(
      Uint8Array.from(await readFile(resolve(checkpoint, decodeURIComponent(url.pathname.split('/').at(-1)!))))
    );
  });
  const fixtures = assets();
  fixtures[scoring.name] = scoringBytes.buffer.slice(
    scoringBytes.byteOffset,
    scoringBytes.byteOffset + scoringBytes.byteLength
  );
  const model = await loadGemma4V2Model(
    {cache},
    {
      createRunner: async () => ({
        async run() {
          return [0, 0];
        },
        releaseCachedMemory() {},
        dispose() {}
      }),
      graphAssets: [graph],
      scoringAsset: scoring,
      imagePreparer,
      readAsset: (async (_base, _manifest, name) => fixtures[name]!) as typeof readVerifiedAsset,
      openSource: async () => source,
      createTokenizer: () => ({encode: (text) => ({ids: Array.from(text, () => 1)})})
    }
  );
  assert.equal(await model.isOfflineReady(), false);
  const progress: number[] = [];
  await model.prepareOffline({onProgress: ({loadedBytes}) => progress.push(loadedBytes)});
  assert.equal(await model.isOfflineReady(), true);
  const scoringKey = [...cacheEntries.keys()].find((key) => key.includes(GEMMA4_V2_SCORING_SHA256));
  assert.ok(scoringKey);
  cacheEntries.delete(scoringKey);
  assert.equal(await model.isOfflineReady(), false);
  await model.prepareOffline();
  assert.equal(await model.isOfflineReady(), true);
  assert.deepEqual(progress, [3]);
  assert.equal(images, true);
  assert.equal(weights, true);
  holdReady = true;
  const pending = model.isOfflineReady();
  await entered.promise;
  const disposal = model.dispose();
  await assert.rejects(pending, /closed/i);
  await disposal;
  gate.resolve();
  assert.equal(sourceClosed, true);
  await assert.rejects(model.isOfflineReady(), /closed/i);
});

test('rejects distinct inference and offline image preparers', async () => {
  const imagePreparer = {
    async prepareImage() {
      return {
        tokens: 1,
        value: {pixels: new Float32Array(), positions: new Int32Array(), tokens: 1, width: 1, height: 1}
      };
    },
    async prepareOffline() {},
    async isOfflineReady() {
      return true;
    }
  };
  await assert.rejects(
    loadGemma4V2Model(
      {},
      {
        createRunner: async () => {
          throw new Error('must not load');
        },
        imagePreparer,
        prepareImage: imagePreparer.prepareImage
      }
    ),
    /one Gemma 4 v2 image preparer/
  );
});

test('the Gemma adapter opens the model from its curated ID alone', async () => {
  const {createDecisionModelLoader} = await import('../src/curated/load-decision-model.ts');
  const {createGemma4V2CuratedAdapter} = await import('../src/curated/adapters/strands-gemma4-v2.mjs');
  const fixtures = assets();
  let sourceClosed = false;
  let runnerClosed = false;
  let called = 0;
  const source = {
    dispose() {
      sourceClosed = true;
    }
  } as unknown as Source;
  const adapter = createGemma4V2CuratedAdapter(
    (options) =>
      loadGemma4V2Model(options, {
        readAsset: (async (_base, _manifest, name) => fixtures[name]!) as typeof readVerifiedAsset,
        openSource: async () => source,
        createTokenizer: () => ({encode: (text) => ({ids: Array.from(text, () => 9)})}),
        prepareImage: async () => ({tokens: 2, value: 'decoded'}),
        createRunner: async () => ({
          async run(input) {
            called++;
            assert.ok(input.stateIds.length);
            return [0, 0];
          },
          releaseCachedMemory() {},
          dispose() {
            runnerClosed = true;
          }
        })
      }),
    async () => null
  );
  const description = {modelId: GEMMA4_V2_MODEL};
  const loadFixtureModel = createDecisionModelLoader({
    isModelId: (value): value is typeof GEMMA4_V2_MODEL => value === GEMMA4_V2_MODEL,
    describe: () => description,
    loaderDescription: () => description,
    loadAdapter: () => async () => adapter,
    approvedLoadOptions: {[GEMMA4_V2_MODEL]: ['maxLength']}
  });
  const model = await loadFixtureModel({modelId: GEMMA4_V2_MODEL, maxLength: 65536});
  assert.equal(model.modelId, GEMMA4_V2_MODEL);
  assert.equal(model.description, description);
  const result = await model.systemOne({state: 'hello', questions: {q: {instructions: 'yes?'}}});
  assert.equal(result.answers.q?.type, 'noul');
  assert.equal(called, 1);
  await assert.rejects(loadFixtureModel({modelId: GEMMA4_V2_MODEL, assetsUrl: 'https://example.test'}), /unsupported/i);
  await model.dispose();
  assert.equal(sourceClosed, true);
  assert.equal(runnerClosed, true);
});

test('curated adapter rejects invalid load-time maxLength before opening the cache', async () => {
  const {createGemma4V2CuratedAdapter} = await import('../src/curated/adapters/strands-gemma4-v2.mjs');
  let opened = false;
  const adapter = createGemma4V2CuratedAdapter(
    async () => {
      throw new Error('must not load');
    },
    async () => {
      opened = true;
      return null;
    }
  );
  for (const maxLength of [0, -1, 1.5, NaN, Infinity]) {
    await assert.rejects(adapter.loadModel({maxLength}, {modelId: GEMMA4_V2_MODEL}), /invalid/i);
  }
  assert.equal(opened, false);
});

test('verified scoring asset reaches the session and rejects the wrong pin', async () => {
  const data = readFileSync(new URL('../assets/strands-gemma4-v2/scoring.bin', import.meta.url));
  const fixtures = assets();
  fixtures['scoring.bin'] = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  const scoringAsset = {
    base: new URL('https://example.test/'),
    name: 'scoring.bin',
    metadata: {bytes: GEMMA4_V2_SCORING_BYTES, sha256: GEMMA4_V2_SCORING_SHA256}
  };
  const source = {dispose() {}} as unknown as Source;
  let opened = 0;
  const dependencies = {
    scoringAsset,
    readAsset: (async (_base, _manifest, name, options) => {
      if (name === 'scoring.bin') assert.equal(options?.onBytes, undefined);
      return fixtures[name]!;
    }) as typeof readVerifiedAsset,
    openSource: async () => {
      opened++;
      return source;
    },
    createTokenizer: () => ({encode: (text: string) => ({ids: Array.from(text, () => 1)})}),
    prepareImage: async () => ({tokens: 1, value: null}),
    createRunner: async () => ({
      async run() {
        return [215, 216];
      },
      releaseCachedMemory() {},
      dispose() {}
    })
  };
  const model = await loadGemma4V2Model({}, dependencies);
  const result = await model.systemOne({
    state: 'state',
    questions: {q: {type: 'choice', instructions: 'Choose', criteria: {A: 'a', B: 'b'}}}
  });
  assert.equal(result.answers.q?.type, 'choice');
  if (result.answers.q?.type === 'choice') assert.equal(result.answers.q.choice, 'B');
  await model.dispose();
  await assert.rejects(
    loadGemma4V2Model(
      {},
      {...dependencies, scoringAsset: {...scoringAsset, metadata: {...scoringAsset.metadata, sha256: '0'.repeat(64)}}}
    ),
    /scoring asset pin/i
  );
  for (const metadata of [null, undefined]) {
    await assert.rejects(
      loadGemma4V2Model(
        {},
        {...dependencies, scoringAsset: {...scoringAsset, metadata} as unknown as typeof scoringAsset}
      ),
      {code: 'INVALID_ASSET'}
    );
  }
  assert.equal(opened, 1);
});
