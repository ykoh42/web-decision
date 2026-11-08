import assert from 'node:assert/strict';
import test from 'node:test';
import {indexMapika4Files, mapika4Files} from '../src/mapika4b/source.mjs';
import {MAPIKA4_DLIGHT_ALL_MODEL, MAPIKA4_DLIGHT_MODEL, MAPIKA4_MODEL} from '../src/mapika4b/contract.mjs';

function inventory() {
  const parameters = Array.from({length: 680}, (_, index) => ({
    position: index + 5,
    target: `tensor-${index}`,
    shape: ['1'],
    dtype: 'torch.float32',
    kind: 'file',
    path: `weights-${Math.min(index, 675)}`,
    fileOffset: index * 4,
    bytes: 4,
    ranges: [{offset: 0, bytes: 4, sha256: 'a'.repeat(64)}]
  }));
  parameters.push(
    ...[680, 681].map((index) => ({
      position: index + 5,
      target: `rotary-${index}`,
      shape: ['1'],
      dtype: 'torch.float32',
      kind: 'rotary',
      values: [0.5]
    }))
  );
  return {state: 'local-ranges-pending-review', parameters};
}

test('indexes file segments and preserves binding order', () => {
  const {entries, files} = indexMapika4Files(inventory());
  assert.equal(entries.length, 682);
  assert.equal(files.size, 676);
  assert.equal(files.get('weights-675'), 680 * 4);
  assert.equal(entries[0].bindingKind, 'q4-candidate');
  assert.equal(entries[681].bindingKind, 'derived-rotary');
});

test('rejects unsafe paths, invalid ranges and incomplete inventory', () => {
  for (const change of [
    (value) => value.parameters.pop(),
    (value) => (value.parameters[0].path = '../escape'),
    (value) => (value.parameters[0].fileOffset = -1),
    (value) => (value.parameters[0].bytes = 0),
    (value) => (value.parameters[0].position = 7),
    (value) => (value.parameters[0].shape = ['not-a-number']),
    (value) => (value.parameters[0].path = value.parameters[1].path)
  ]) {
    const fixture = inventory();
    change(fixture);
    assert.throws(() => indexMapika4Files(fixture), {code: 'INVALID_ASSET'});
  }
});

test('pins distinct graph bytes for each variant and rejects other models', () => {
  const baseline = mapika4Files(MAPIKA4_MODEL);
  const dlight = mapika4Files(MAPIKA4_DLIGHT_MODEL);
  assert.equal(baseline['model.wasm'].bytes, 4963401);
  assert.equal(dlight['model.wasm'].bytes, 4996669);
  assert.notEqual(baseline['model.wasm'].sha256, dlight['model.wasm'].sha256);
  const all = mapika4Files(MAPIKA4_DLIGHT_ALL_MODEL);
  assert.equal(all['model.wasm'].bytes, 5168538);
  assert.notEqual(all['model.wasm'].sha256, dlight['model.wasm'].sha256);
  assert.deepEqual(baseline['inventory.json'], dlight['inventory.json']);
  assert.throws(() => mapika4Files('other'), {code: 'INVALID_ASSET'});
});
