import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {open, readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {createPolyfillWASI, instantiate} from '../../../packages/web-decision/src/runtime/tvm.generated.mjs';
import {
  createGemma4V1Decoder,
  type Gemma4V1Graph
} from '../../../packages/web-decision/src/strands/gemma4-v1-decoder.mjs';
import {createGemma4V1TvmStages} from '../../../packages/web-decision/src/strands/gemma4-v1-runtime.mjs';
import {
  createGemma4V1Source,
  type Gemma4V1Inventory
} from '../../../packages/web-decision/src/strands/gemma4-v1-source.mjs';

const [directory, inventoryPath, base, adapter, reference, mode] = process.argv.slice(2);
assert.ok(directory && inventoryPath && base && adapter && reference);
const hash = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');
const graph = JSON.parse(await readFile(join(directory, 'graph.json'), 'utf8')) as Gemma4V1Graph & {
  target: string;
  archiveApproved: boolean;
  graphSha256: string;
  inventorySha256: string;
};
assert.equal(graph.target, 'cpu-wasm');
assert.equal(graph.archiveApproved, false);
const wasm = await readFile(join(directory, 'graph.wasm'));
assert.equal(hash(wasm), graph.graphSha256);
const inventoryRaw = await readFile(inventoryPath);
assert.equal(hash(inventoryRaw), graph.inventorySha256);
const inventory = JSON.parse(inventoryRaw.toString()) as Gemma4V1Inventory;
const fixture = JSON.parse(await readFile(reference, 'utf8'));
assert.equal(fixture.modelSha, inventory.base.sha256);
assert.equal(fixture.adapterSha, inventory.adapter.sha256);
const files = {
  'model.safetensors': await open(join(base, inventory.base.file), 'r'),
  'lora/adapter_model.safetensors': await open(join(adapter, inventory.adapter.file), 'r')
};
try {
  const recent = new Map<string, ArrayBuffer>();
  let sourceReads = 0;
  const bindings = [...graph.final, ...graph.ple, ...graph.layers.flatMap(({project, tail}) => [...project, ...tail])];
  const source = createGemma4V1Source(inventory, bindings, {}, async (_base, spec, options) => {
    options?.signal?.throwIfAborted();
    const key = spec.path + ':' + spec.offset;
    let bytes = recent.get(key);
    if (!bytes) {
      const buffer = new Uint8Array(spec.length);
      const file = files[spec.path as keyof typeof files];
      assert.equal((await file.read(buffer, 0, buffer.length, spec.offset)).bytesRead, buffer.length);
      assert.equal(hash(buffer), spec.sha256);
      if (recent.size === 4) recent.delete(recent.keys().next().value!);
      bytes = buffer.buffer;
      recent.set(key, bytes);
      sourceReads++;
    }
    return {bytes, cachedByHash: false};
  });
  const runtime = await instantiate(new Uint8Array(wasm).buffer, createPolyfillWASI());
  const stages = createGemma4V1TvmStages(runtime, runtime.cpu(), source);
  const decoder = createGemma4V1Decoder(graph, stages, source);
  const signal = AbortSignal.timeout(360_000);
  try {
    const questions = mode === '--shared' ? [fixture.question, fixture.question] : [fixture.question];
    const outputs = await decoder.run(fixture.state, questions, signal);
    assert.equal(outputs.length, questions.length);
    for (const row of outputs.slice(1)) assert.deepEqual(row, outputs[0]);
    const [row] = outputs;
    const actual = Array.from(row!);
    const errors = actual.map((value, index) => Math.abs(value - fixture.logits[index]));
    assert.ok(actual.every(Number.isFinite));
    assert.equal(actual.indexOf(Math.max(...actual)), fixture.logits.indexOf(Math.max(...fixture.logits)));
    const maxAbsoluteLogitError = Math.max(...errors);
    assert.ok(maxAbsoluteLogitError <= 0.1, 'CPU graph ABI smoke tolerance');
    console.log(
      JSON.stringify({
        testedQuestions: questions.length,
        sharedPrefix: questions.length > 1,
        actual,
        expected: fixture.logits,
        maxAbsoluteLogitError,
        verifiedLocalRangeReads: sourceReads,
        graphSha256: graph.graphSha256,
        gpuExecuted: false,
        releaseApproved: false
      })
    );
  } finally {
    await decoder.dispose();
  }
} finally {
  await Promise.all(Object.values(files).map((file) => file.close()));
}
