import assert from 'node:assert/strict';
import test from 'node:test';
import type {AssetSourceOptions, ManifestSourceSpec} from '../src/asset-source-types.js';
import type {XDecisionLoadOptions} from '../src/index.js';
import {XDECISION_MODEL} from '../src/xdecision/contract.mjs';
import {loadXDecisionModel} from '../src/xdecision/loader.mjs';

test('forwards a caller-supplied runtime spec to the injected source opener', async () => {
  const customSpec: ManifestSourceSpec = {
    manifest: {files: {}},
    maxAssetBytes: 1,
    totalBytes: 1
  };
  let receivedSpec: ManifestSourceSpec | undefined;
  const runtime = {
    ort: {},
    spec: customSpec,
    openSource: async (options: AssetSourceOptions & {readonly spec?: ManifestSourceSpec}) => {
      receivedSpec = options.spec;
      throw new Error('stop after inspecting the forwarded source spec');
    }
  };
  const options: XDecisionLoadOptions = {
    model: XDECISION_MODEL,
    assetsUrl: 'https://example.test/xdecision/'
  };

  await assert.rejects(loadXDecisionModel(options, runtime), {code: 'MODEL_LOAD_FAILED'});
  assert.equal(receivedSpec, customSpec);
});
