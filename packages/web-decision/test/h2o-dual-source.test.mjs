import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {existsSync} from 'node:fs';
import {openH2oAssetSource} from '../src/h2o/source-entry.mjs';
import {validateOptions} from '../src/h2o/loader.mjs';
import {
  preparePublicCandidatePlan,
  verifyWasm,
  H2O_DUAL_MODEL_ID,
  H2O_DLIGHT_MODEL_ID,
  DUAL_WASM_SHA256
} from '../src/h2o/plan.mjs';

const root = new URL('../../../', import.meta.url);
const model = new URL('models/h2o-integration-20261030/', root);
const prepared = new URL('experiments/h2o-q4-dual-private-preparation-20261101/local-20261101-v1/', root);
const files = {
  'evidence/inventory.json': new URL('h2o-inventory-31d7b5f89e2c41a6/inventory.pending.json', model),
  'h2o-public-metadata.json': new URL(
    'experiments/h2o-public-evidence-preparation-20261031/h2o-public-metadata.pending.json',
    root
  ),
  'h2o-dual-metadata.json': new URL('h2o-dual-metadata.json', prepared),
  'manifest.pending.json': new URL('h2o-run-c476df0e9ff43007/candidate/manifest.pending.json', model),
  'model.dual.pending.wasm': new URL('model.dual.pending.wasm', prepared),
  'tokenizer/tokenizer.json': new URL('checkpoint/tokenizer/tokenizer.json', model),
  'tokenizer/tokenizer_config.json': new URL('checkpoint/tokenizer/tokenizer_config.json', model),
  'tokenizer_contract.json': new URL('checkpoint/tokenizer_contract.json', model)
};
const skip = Object.values(files).every(existsSync) ? false : 'Local H2O dual assets not prepared';
const base = 'https://example.invalid/assets/';

test('dual loader accepts only its matching model and supported options', () => {
  assert.equal(
    validateOptions({model: H2O_DUAL_MODEL_ID, assetsUrl: base}, H2O_DUAL_MODEL_ID).model,
    H2O_DUAL_MODEL_ID
  );
  assert.throws(() => validateOptions({model: H2O_DUAL_MODEL_ID, assetsUrl: base}, H2O_DLIGHT_MODEL_ID), {
    code: 'INVALID_INPUT'
  });
  assert.throws(() => validateOptions({model: H2O_DUAL_MODEL_ID, assetsUrl: base, cache: true}, H2O_DUAL_MODEL_ID), {
    code: 'INVALID_INPUT'
  });
});

test('dual graph binds the baseline order and rejects another variant or altered bytes', {skip}, async () => {
  const [metadata, manifest, candidate, wasm] = await Promise.all(
    [
      files['h2o-public-metadata.json'],
      files['manifest.pending.json'],
      files['h2o-dual-metadata.json'],
      files['model.dual.pending.wasm']
    ].map((path) => readFile(path))
  );
  const original = await preparePublicCandidatePlan(metadata, manifest);
  const dual = await preparePublicCandidatePlan(metadata, manifest, H2O_DUAL_MODEL_ID, candidate);
  assert.equal(dual.wasmSha256, DUAL_WASM_SHA256);
  assert.deepEqual(dual.parameters, original.parameters);
  assert.deepEqual(dual.labelIds, original.labelIds);
  const verified = await verifyWasm(wasm, H2O_DUAL_MODEL_ID);
  assert.ok(Buffer.from(verified).equals(wasm));
  assert.notEqual(verified, wasm);
  await assert.rejects(verifyWasm(wasm, H2O_DLIGHT_MODEL_ID), {code: 'INVALID_ASSET'});
  await assert.rejects(preparePublicCandidatePlan(metadata, manifest, H2O_DLIGHT_MODEL_ID, candidate), {
    code: 'INVALID_ASSET'
  });
  const changed = Uint8Array.from(candidate);
  changed[0] ^= 1;
  await assert.rejects(preparePublicCandidatePlan(metadata, manifest, H2O_DUAL_MODEL_ID, changed), {
    code: 'INVALID_ASSET'
  });
});

test('dual source fetches its own graph and rejects corrupted files', {skip}, async () => {
  const originalFetch = globalThis.fetch;
  const seen = [];
  let corrupt;
  globalThis.fetch = async (url, options) => {
    const name = new URL(url).pathname.slice('/assets/'.length),
      path = files[name];
    assert.ok(path, `unexpected asset ${name}`);
    seen.push(name);
    const data = await readFile(path),
      range = /^bytes=(\d+)-(\d+)$/.exec(options.headers.Range);
    assert.ok(range);
    const start = Number(range[1]),
      end = Number(range[2]);
    const bytes = Uint8Array.from(data.subarray(start, end + 1));
    if (name === corrupt) bytes[0] ^= 1;
    return new Response(bytes, {
      status: 206,
      headers: {'Content-Range': `bytes ${start}-${end}/${data.length}`, 'Content-Length': String(bytes.length)}
    });
  };
  try {
    const source = await openH2oAssetSource({assetsUrl: base, model: H2O_DUAL_MODEL_ID});
    assert.equal(source.wasm.byteLength, 4916749);
    assert.equal(source.candidateMetadata.byteLength, 179056);
    assert.ok(seen.includes('model.dual.pending.wasm'));
    assert.ok(seen.includes('h2o-dual-metadata.json'));
    source.close();
    for (const path of ['model.dual.pending.wasm', 'h2o-dual-metadata.json']) {
      corrupt = path;
      await assert.rejects(openH2oAssetSource({assetsUrl: base, model: H2O_DUAL_MODEL_ID}), {code: 'INVALID_ASSET'});
    }
  } finally {
    globalThis.fetch = originalFetch;
  }
});
