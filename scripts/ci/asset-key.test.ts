import assert from 'node:assert/strict';
import {test} from 'node:test';
import {hasQuotedAssetKey} from './asset-key.ts';

test('matches exact quoted asset keys in the generated object', () => {
  assert.equal(hasQuotedAssetKey('{"model.wasm": "data:"}', 'model.wasm'), true);
  assert.equal(hasQuotedAssetKey("{'model.wasm': 'data:'}", 'model.wasm'), true);
});

test('does not treat a different asset, mismatched quotes, or a string value as the key', () => {
  for (const source of ['{"modelXwasm": 1}', '{"model.wasm\': 1}', '{"value": "model.wasm"}']) {
    assert.equal(hasQuotedAssetKey(source, 'model.wasm'), false);
  }
});
