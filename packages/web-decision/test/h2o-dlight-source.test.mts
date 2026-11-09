import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {openH2oAssetSource} from '../src/h2o/source-entry.mjs';
import {
  preparePublicCandidatePlan,
  verifyWasm,
  H2O_BASELINE_MODEL_ID,
  H2O_DLIGHT_MODEL_ID,
  WASM_SHA256,
  DLIGHT_WASM_SHA256
} from '../src/h2o/plan.mjs';
import {loadFakeH2oCandidate, createH2oBrowserFactories, PRELIMINARY_LIMITS} from '../src/h2o/loader.mjs';

const root = new URL('../../../', import.meta.url),
  model = new URL('models/h2o-integration-20261030/', root);
const pub = new URL('experiments/h2o-public-evidence-preparation-20261031/h2o-public-metadata.pending.json', root);
const candidate = new URL('experiments/h2o-dlight-private-preparation-20261031/outputs/local-20261031-v1/', root);
const files: Record<string, URL> = {
  'evidence/inventory.json': new URL('h2o-inventory-31d7b5f89e2c41a6/inventory.pending.json', model),
  'h2o-public-metadata.json': pub,
  'h2o-dlight-metadata.json': new URL('h2o-dlight-metadata.json', candidate),
  'manifest.pending.json': new URL('h2o-run-c476df0e9ff43007/candidate/manifest.pending.json', model),
  'model.pending.wasm': new URL('full-split-webgpu-20261030-v2/model.pending.wasm', model),
  'model.dlight.pending.wasm': new URL('model.dlight.pending.wasm', candidate),
  'tokenizer/tokenizer.json': new URL('checkpoint/tokenizer/tokenizer.json', model),
  'tokenizer/tokenizer_config.json': new URL('checkpoint/tokenizer/tokenizer_config.json', model),
  'tokenizer_contract.json': new URL('checkpoint/tokenizer_contract.json', model),
  'tensor-0415-retained.pending': new URL('h2o-run-c476df0e9ff43007/candidate/tensor-0415-retained.pending', model)
};
const artifactInputs = [
  pub,
  files['manifest.pending.json']!,
  files['h2o-dlight-metadata.json']!,
  files['model.pending.wasm']!,
  files['model.dlight.pending.wasm']!
];
const artifactSkip = artifactInputs.every(existsSync) ? false : 'Local H2O graph and metadata evidence not prepared';
const sourceSkip = Object.values(files).every(existsSync) ? false : 'Local H2O source evidence not prepared';
const sha = (x: Uint8Array) => createHash('sha256').update(x).digest('hex');
const base = 'https://example.invalid/assets/';
async function withAssets(fn: (context: {seen: string[]; setMutate: (value: string | null) => void}) => Promise<void>) {
  const original = globalThis.fetch,
    seen: string[] = [];
  let mutate: string | null = null;
  globalThis.fetch = async (url, options) => {
    const name = new URL(url instanceof Request ? url.url : url).pathname.slice('/assets/'.length),
      path = files[name];
    assert.ok(path, `unexpected asset ${name}`);
    seen.push(name);
    const data = await readFile(path),
      m = /^bytes=(\d+)-(\d+)$/.exec(new Headers(options?.headers).get('Range') ?? '');
    assert.ok(m);
    const start = Number(m[1]),
      end = Number(m[2]),
      bytes = Uint8Array.from(data.subarray(start, end + 1));
    if (mutate === name) bytes[0] = bytes[0]! ^ 1;
    return new Response(bytes, {
      status: 206,
      headers: {'Content-Range': `bytes ${start}-${end}/${data.length}`, 'Content-Length': String(bytes.length)}
    });
  };
  try {
    return await fn({
      seen,
      setMutate: (value: string | null) => {
        mutate = value;
      }
    });
  } finally {
    globalThis.fetch = original;
  }
}

test(
  'DLight provenance preserves baseline ABI and pins a separate owned WASM snapshot',
  {skip: artifactSkip},
  async () => {
    const [metadata, manifest, proof, baseline, graph] = await Promise.all(
      [
        pub,
        files['manifest.pending.json']!,
        files['h2o-dlight-metadata.json']!,
        files['model.pending.wasm']!,
        files['model.dlight.pending.wasm']!
      ].map((path) => readFile(path))
    );
    const a = await preparePublicCandidatePlan(metadata, manifest);
    const b = await preparePublicCandidatePlan(metadata, manifest, H2O_DLIGHT_MODEL_ID, proof);
    assert.equal(a.wasmSha256, WASM_SHA256);
    assert.equal(b.wasmSha256, DLIGHT_WASM_SHA256);
    assert.deepEqual(b.parameters, a.parameters);
    assert.equal(b.parameters.length, 682);
    assert.deepEqual(b.labels, a.labels);
    assert.deepEqual(b.labelIds, a.labelIds);
    const owned = await verifyWasm(graph, H2O_DLIGHT_MODEL_ID);
    assert.notEqual(owned, graph);
    assert.equal(sha(owned), DLIGHT_WASM_SHA256);
    await assert.rejects(verifyWasm(baseline, H2O_DLIGHT_MODEL_ID), {code: 'INVALID_ASSET'});
    await assert.rejects(verifyWasm(graph, H2O_BASELINE_MODEL_ID), {code: 'INVALID_ASSET'});
    await assert.rejects(preparePublicCandidatePlan(metadata, manifest, H2O_DLIGHT_MODEL_ID, metadata), {
      code: 'INVALID_ASSET'
    });
    const altered = Uint8Array.from(proof!);
    altered[10] = altered[10]! ^ 1;
    await assert.rejects(preparePublicCandidatePlan(metadata, manifest, H2O_DLIGHT_MODEL_ID, altered), {
      code: 'INVALID_ASSET'
    });
    const input = Uint8Array.from(graph!),
      pending = verifyWasm(input, H2O_DLIGHT_MODEL_ID);
    input[0] = input[0]! ^ 1;
    assert.equal(sha(await pending), DLIGHT_WASM_SHA256);
  }
);

test('source picks exact variant paths and rejects modified candidate files', {skip: sourceSkip}, async () =>
  withAssets(async ({seen, setMutate}) => {
    const a = await openH2oAssetSource({assetsUrl: base, model: H2O_BASELINE_MODEL_ID});
    assert.equal(sha(a.wasm), WASM_SHA256);
    assert.equal(a.candidateMetadata, undefined);
    a.close();
    assert.ok(seen.includes('model.pending.wasm'));
    assert.equal(seen.includes('model.dlight.pending.wasm'), false);
    seen.length = 0;
    const b = await openH2oAssetSource({assetsUrl: base, model: H2O_DLIGHT_MODEL_ID});
    assert.equal(sha(b.wasm), DLIGHT_WASM_SHA256);
    assert.equal(b.candidateMetadata!.length, 178971);
    assert.ok(seen.includes('model.dlight.pending.wasm'));
    assert.ok(seen.includes('h2o-dlight-metadata.json'));
    assert.equal(seen.includes('model.pending.wasm'), false);
    const path = 'tensor-0415-retained.pending',
      spec = {path, totalBytes: 64, ...b.inventory[path]![0]!};
    assert.equal((await b.readRange(base, spec)).bytes.byteLength, 64);
    setMutate(path);
    await assert.rejects(b.readRange(base, spec), {code: 'INVALID_ASSET'});
    setMutate(null);
    b.close();
    for (const path of ['h2o-dlight-metadata.json', 'model.dlight.pending.wasm']) {
      setMutate(path);
      await assert.rejects(openH2oAssetSource({assetsUrl: base, model: H2O_DLIGHT_MODEL_ID}), {code: 'INVALID_ASSET'});
    }
  })
);

test('verified DLight byte snapshot reaches instantiate despite later source mutation', {skip: sourceSkip}, async () =>
  withAssets(async () => {
    const source = await openH2oAssetSource({assetsUrl: base, model: H2O_DLIGHT_MODEL_ID});
    let instantiated, instanceArgs;
    const factories = createH2oBrowserFactories(
      {
        Tokenizer: class {},
        instantiate: async (bytes: Uint8Array) => {
          instantiated = sha(bytes);
          instanceArgs = bytes;
          return {dispose() {}};
        },
        createPolyfillWASI: () => ({})
      } as unknown as Parameters<typeof createH2oBrowserFactories>[0],
      H2O_DLIGHT_MODEL_ID
    );
    const originalVerify = factories.verifyWasm;
    let verified;
    const device = {lost: new Promise(() => {}), destroy() {}};
    const deps = {
      ...factories,
      fake: true,
      expectedModel: H2O_DLIGHT_MODEL_ID,
      gpu: {
        async requestAdapter() {
          return {
            features: new Set(['shader-f16']),
            limits: PRELIMINARY_LIMITS,
            async requestDevice() {
              return device;
            }
          };
        }
      },
      async openSource() {
        return source;
      },
      async verifyWasm(bytes: Uint8Array) {
        verified = await originalVerify(bytes, undefined);
        return verified;
      },
      async prepareTokenizer() {
        source.wasm.fill(0);
        return {tokenizer: {}, addedTokens: []};
      },
      initialize() {
        return {disposalGuard: {retain() {}}};
      },
      async upload() {
        return [];
      },
      attach() {
        return {};
      },
      async createSession() {
        return {ok: true};
      },
      makeHasher() {}
    };
    const result = await loadFakeH2oCandidate(
      {model: H2O_DLIGHT_MODEL_ID, assetsUrl: base},
      deps as unknown as Parameters<typeof loadFakeH2oCandidate>[1]
    );
    assert.equal((result as unknown as {ok: boolean}).ok, true);
    assert.equal(instantiated, DLIGHT_WASM_SHA256);
    assert.equal(instanceArgs, verified);
    assert.notEqual(instanceArgs, source.wasm);
  })
);
