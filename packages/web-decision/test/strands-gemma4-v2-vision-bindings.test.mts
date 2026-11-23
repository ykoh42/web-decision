import assert from 'node:assert/strict';
import {test} from 'node:test';
import type {DLDevice, Instance} from '../src/runtime/tvm-types.js';
import {
  assertGemma4V2VisionGraphBindings,
  gemma4V2VisionFloat32,
  loadGemma4V2VisionWeights,
  loadPinnedGemma4V2VisionBindings
} from '../src/strands/gemma4-v2-vision-bindings.mts';

test('pinned vision manifest contains complete ordered original checkpoint bindings', async () => {
  const bindings = await loadPinnedGemma4V2VisionBindings();
  assert.equal(bindings.patch.length, 2);
  assert.deepEqual(
    bindings.patch.map((item) => item.shape),
    [
      [2, 10240, 768],
      [768, 768]
    ]
  );
  assert.equal(bindings.projection[0]?.name, 'model.embed_vision.embedding_projection.weight');
  assert.deepEqual(bindings.projection[0]?.shape, [1536, 768]);
  assert.equal(bindings.layers.length, 16);
  for (let i = 0; i < 16; i++) {
    const layer = bindings.layers[i]!;
    assert.equal(layer.layer, i);
    assert.equal(layer.parameters.length, 41);
    assertGemma4V2VisionGraphBindings(layer.parameters, layer.parameters);
    const wrongOrder = layer.parameters.slice();
    [wrongOrder[0], wrongOrder[1]] = [wrongOrder[1]!, wrongOrder[0]!];
    assert.throws(() => assertGemma4V2VisionGraphBindings(layer.parameters, wrongOrder), /graph arguments/);
  }
});

test('converts raw little-endian BF16 while preserving exact bits', () => {
  const bytes = new Uint8Array([0x80, 0x3f, 0x00, 0xbf, 0x80, 0x7f, 0x00, 0x00]);
  assert.deepEqual(Array.from(gemma4V2VisionFloat32(bytes)), [1, -0.5, Infinity, 0]);
  assert.throws(() => gemma4V2VisionFloat32(new Uint8Array(3)), /length/);
});

test('rejects a corrupt weight before creating a TVM tensor', async () => {
  let allocated = false;
  const runtime = {
    empty() {
      allocated = true;
      throw new Error('Unexpected allocation');
    }
  } as unknown as Instance;
  const source = {
    async readTensor() {
      return new Uint8Array(2);
    }
  };
  await assert.rejects(loadGemma4V2VisionWeights(runtime, {} as DLDevice, source), /digest differs/);
  assert.equal(allocated, false);
});
