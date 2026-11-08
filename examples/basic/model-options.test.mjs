import test from 'node:test';
import assert from 'node:assert/strict';
import {describeModel, isDecisionModel} from '../../packages/web-decision/src/model-registry.js';
import {limitsPlugin} from './limits-plugin.mjs';
import {choices, findChoice, loadOptions as createLoadOptions} from './model-options.mjs';

const loadOptions = (key, url, options = {}) =>
  createLoadOptions(key, url, {baseUrl: 'https://example.test/app/', ...options});

test('example token limits match the model descriptors', () => {
  const plugin = limitsPlugin(describeModel);
  const id = plugin.resolveId('virtual:web-decision-token-limits');
  const module = plugin.load(id);
  const limits = JSON.parse(module.slice('export default '.length, -1));
  for (const {model} of choices) {
    assert.deepEqual(
      limits[model],
      describeModel(model).features.textChoice.limits.filter((limit) => limit.resource === 'tokens'),
      model
    );
  }
});

test('every example choice selects a registered model', () => {
  assert.equal(choices.length, 23);
  assert.equal(new Set(choices.map((choice) => choice.key)).size, choices.length);
  for (const choice of choices) assert.equal(isDecisionModel(choice.model), true, choice.key);
  assert.equal(findChoice('k32').model, 'kev-0.6b-q4f16');
  assert.equal(findChoice('k32').assets, '/kev-0.6b-q4f16-k32-experimental/');
  assert.equal(findChoice('d1-omni').model, 'experimental-d1-omni-600m-q8');
  assert.equal(findChoice('d1-omni').assets, '/d1-omni-600m-q8/');
});

test('asset URL validation uses the supplied browser base and preserves valid paths', () => {
  for (const url of ['/assets/', '../models/', 'https://assets.example.test/models/', '/assets/%3F%23']) {
    assert.equal(loadOptions('nari', ` ${url} `).assetsUrl, url);
  }
  for (const [input, expected] of [
    ['/assets/?', '/assets/'],
    ['../models/#', '../models/'],
    ['../models/?#', '../models/'],
    ['https://assets.example.test/models/?', 'https://assets.example.test/models/'],
    ['?', 'https://example.test/app/'],
    ['#', 'https://example.test/app/'],
    ['?#', 'https://example.test/app/']
  ])
    assert.equal(loadOptions('nari', input).assetsUrl, expected);
  for (const url of [
    'file:///tmp/assets/',
    'https://user:secret@example.test/',
    'https://example.test/?v=1',
    'https://example.test/#fragment',
    'http://['
  ]) {
    assert.throws(() => loadOptions('nari', url), /asset (directory )?URL/i);
  }
  assert.throws(() => createLoadOptions('nari', '/relative/', {baseUrl: undefined}), /valid HTTP/);
});

test('model-specific loader options preserve H2O and Strands restrictions', () => {
  const onProgress = () => {};
  const h2o = loadOptions('h2o', ' http://127.0.0.1:41782/h2o-assets/ ', {onProgress});
  assert.deepEqual(h2o, {
    model: 'experimental-h2o-lightning-4b-q4f16',
    assetsUrl: 'http://127.0.0.1:41782/h2o-assets/',
    cache: false
  });
  assert.equal(Object.hasOwn(h2o, 'onProgress'), false);
  const dlight = loadOptions('h2o-dlight', '/h2o/', {onProgress});
  assert.deepEqual(dlight, {model: 'experimental-h2o-lightning-4b-q4f16-dlight', assetsUrl: '/h2o/', cache: false});
  assert.equal(Object.hasOwn(dlight, 'onProgress'), false);
  const dual = loadOptions('h2o-dual', '/h2o/', {onProgress});
  assert.deepEqual(dual, {model: 'experimental-h2o-lightning-4b-q4f16-dual', assetsUrl: '/h2o/', cache: false});
  assert.equal(Object.hasOwn(dual, 'onProgress'), false);
  assert.throws(() => loadOptions('h2o-dlight', '/h2o/', {packedReadback: true}), /only for DeBERTa/);
  const strands = loadOptions('strands', '/strands-assets/', {onProgress});
  assert.equal(strands.model, 'experimental-strands-qwen-2b-fp16');
  assert.equal(strands.cache, false);
  assert.equal(strands.onProgress, onProgress);
  const opendecider = loadOptions('opendecider-nano', '/opendecider-nano-assets/', {onProgress});
  assert.deepEqual(opendecider, {
    model: 'experimental-opendecider-nano-fp16',
    assetsUrl: '/opendecider-nano-assets/',
    cache: false,
    onProgress
  });
  const nari = loadOptions('nari', '/nari-assets/', {onProgress});
  assert.deepEqual(nari, {model: 'experimental-nari-decision-4b-q4f16', assetsUrl: '/nari-assets/', cache: false});
  const nariMixed = loadOptions('nari-mixed', '/mixed-assets/', {onProgress});
  assert.deepEqual(nariMixed, {
    model: 'experimental-nari-decision-4b-mixed16-q4f16',
    assetsUrl: '/mixed-assets/',
    cache: false
  });
  const imajev = loadOptions('imajev', '/imajev-assets/', {onProgress});
  assert.deepEqual(imajev, {model: 'experimental-imajev-4b-candidate', assetsUrl: '/imajev-assets/', cache: false});
  const pplx = loadOptions('pplx', '/pplx-assets/', {onProgress});
  assert.deepEqual(pplx, {
    model: 'experimental-pplx-decider-flash-v1.2-q4f16-dlight',
    assetsUrl: '/pplx-assets/',
    cache: false
  });
  assert.deepEqual(loadOptions('mapika4b', '/mapika4b/', {onProgress}), {
    model: 'experimental-mapika-4b-q4f16',
    assetsUrl: '/mapika4b/',
    cache: false
  });
  assert.deepEqual(loadOptions('mapika4b-dlight', '/mapika4b-dlight/', {onProgress}), {
    model: 'experimental-mapika-4b-q4f16-dlight',
    assetsUrl: '/mapika4b-dlight/',
    cache: false
  });
  assert.deepEqual(loadOptions('mapika4b-dlight-all', '/mapika4b-all/', {onProgress}), {
    model: 'experimental-mapika-4b-q4f16-dlight-all',
    assetsUrl: '/mapika4b-all/',
    cache: false
  });
  assert.deepEqual(loadOptions('elephant', '/elephant/', {onProgress}), {
    model: 'experimental-elephant-9b-q4f16-dlight',
    assetsUrl: '/elephant/',
    cache: false
  });
  assert.deepEqual(loadOptions('elephant-512', '/elephant-512/', {onProgress}), {
    model: 'experimental-elephant-9b-q4f16-dlight-512',
    assetsUrl: '/elephant-512/',
    cache: false
  });
  assert.deepEqual(loadOptions('clef', ' /clef-assets/ ', {onProgress}), {
    model: 'experimental-clef-9b-q4f16-selected',
    assetsUrl: '/clef-assets/',
    cache: false
  });
  assert.throws(() => loadOptions('clef', '/clef/', {packedReadback: true}), /only for DeBERTa/);
  assert.equal(loadOptions('deberta', '/deberta/', {packedReadback: true}).debertaReadback, 'packed');
  assert.throws(() => loadOptions('h2o', '/h2o/', {packedReadback: true}), /only for DeBERTa/);
  assert.throws(() => loadOptions('h2o', '  '), /asset directory URL/);
  assert.throws(() => loadOptions('unknown', '/assets/'), /supported model/);
});
