import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {
  materializeQwen35VisionSourceWeight,
  loadQwen35VisionParameterTensors
} from '../src/strands/qwen35-vision-weights.mts';
import type {Qwen35PinnedSource} from '../src/strands/qwen35-source.mts';
import type {Qwen35VisionArtifact} from '../src/strands/qwen35-vision-runner.mts';
import type {Qwen35VisionWeightBinding} from '../src/strands/qwen35-vision-weights.mts';
import {verifyPinnedQwen35VisionArtifacts} from '../src/strands/qwen35-vision-contract.mts';
import {createQwen35VisionRuntimeRunner} from '../src/strands/qwen35-vision-runtime.mts';
import type {Qwen35VisionRuntime, Qwen35VisionTvmJs} from '../src/strands/qwen35-vision-runtime.mts';

function hash(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function tensorPayloadSource(payload: Uint8Array): Qwen35PinnedSource {
  const tensor = {
    name: 'model.visual.test.weight',
    shape: [2],
    dtype: 'BF16' as const,
    elementCount: 2,
    sourceDataByteOffset: 0,
    storageByteOffset: 0,
    storageWordOffset: 0,
    sourceByteCount: payload.byteLength,
    packedByteCount: payload.byteLength,
    paddingByteCount: 0,
    sourcePayloadSha256: hash(payload),
    packedTensorSha256: hash(payload)
  };
  return {
    manifest: {assets: {base: {tensors: [tensor]}}} as never,
    async readTensor(_asset, name) {
      assert.equal(name, tensor.name);
      return payload;
    },
    async readTensorRange() {
      throw new Error('unexpected range read');
    },
    async commit() {
      return false;
    }
  };
}

const sampleBinding: Qwen35VisionWeightBinding = {
  index: 0,
  targetName: 'p_visual_test_weight',
  sourceName: 'model.visual.test.weight',
  shape: [2],
  elementCount: 2,
  sourceByteCount: 4,
  sourcePayloadSha256: '99dc7d6da1d73a8240f9d5e7a5f5c1ad0b42dbfa85aab1b6f7ec4f5c90e2ad38',
  sourceDtype: 'BF16',
  targetDtype: 'float32',
  conversion: 'exact-bf16-to-float32'
};

test('widens verified source BF16 vision weights exactly to little-endian float32', async () => {
  const sourceBytes = new Uint8Array([0x80, 0x3f, 0x00, 0xc0]);
  const binding = {...sampleBinding, sourcePayloadSha256: hash(sourceBytes)};
  const output = await materializeQwen35VisionSourceWeight(tensorPayloadSource(sourceBytes), binding);
  assert.deepEqual(Array.from(new Uint32Array(output.buffer)), [0x3f800000, 0xc0000000]);
});

test('rejects a corrupted source BF16 payload before conversion', async () => {
  const sourceBytes = new Uint8Array([0x80, 0x3f, 0x00, 0xc0]);
  const binding = {...sampleBinding, sourcePayloadSha256: hash(new Uint8Array([0, 0, 0, 0]))};
  await assert.rejects(materializeQwen35VisionSourceWeight(tensorPayloadSource(sourceBytes), binding), {
    code: 'INVALID_ASSET'
  });
});

test('validates all pinned graph and source bindings before uploading any tensor', async () => {
  const assetRoot = new URL('../assets/qwen35/vision/', import.meta.url);
  const [fixtureBytes, bindingBytes] = await Promise.all([
    readFile(new URL('fixture.json', assetRoot)),
    readFile(new URL('weight-bindings.json', assetRoot))
  ]);
  const artifact = JSON.parse(fixtureBytes.toString('utf8')) as Qwen35VisionArtifact;
  const bindingFile = JSON.parse(bindingBytes.toString('utf8')) as {
    readonly bindings: readonly Qwen35VisionWeightBinding[];
  };
  const controller = new AbortController();
  let reads = 0;
  const source = {
    manifest: {assets: {base: {tensors: []}}},
    async readTensor() {
      reads += 1;
      throw new Error('must not read source after cancellation');
    }
  } as unknown as Qwen35PinnedSource;
  const loading = loadQwen35VisionParameterTensors({
    runtime: {
      empty() {
        throw new Error('must not upload after cancellation');
      }
    },
    device: {deviceType: 1, deviceId: 0, async sync() {}},
    source,
    artifact,
    bindings: bindingFile.bindings,
    signal: controller.signal
  });
  controller.abort();
  await assert.rejects(loading, {name: 'AbortError'});
  assert.equal(reads, 0);
});

test('rechecks pinned bytes before TVM instantiation and releases its scope on source cancellation', async () => {
  const assetRoot = new URL('../assets/qwen35/vision/', import.meta.url);
  const [fixtureBytes, wasmBytes, bindingBytes, wordManifestBytes] = await Promise.all([
    readFile(new URL('fixture.json', assetRoot)),
    readFile(new URL('qwen35-dynamic-vision.webgpu.wasm', assetRoot)),
    readFile(new URL('weight-bindings.json', assetRoot)),
    readFile(new URL('../assets/qwen35/pinned-word-assets-manifest.json', import.meta.url))
  ]);
  const artifact = JSON.parse(fixtureBytes.toString('utf8')) as Qwen35VisionArtifact;
  const bindingFile = JSON.parse(bindingBytes.toString('utf8')) as {
    readonly bindings: readonly Qwen35VisionWeightBinding[];
  };
  const firstBinding = bindingFile.bindings[0]!;
  const wordManifest = JSON.parse(wordManifestBytes.toString('utf8')) as {
    readonly assets: {readonly base: {readonly tensors: readonly Record<string, unknown>[]}};
  };
  const firstTensor = wordManifest.assets.base.tensors.find(({name}) => name === firstBinding.sourceName);
  assert.ok(firstTensor);
  const sourceFirstBinding = [...bindingFile.bindings].sort((left, right) => {
    const leftTensor = wordManifest.assets.base.tensors.find(({name}) => name === left.sourceName)!;
    const rightTensor = wordManifest.assets.base.tensors.find(({name}) => name === right.sourceName)!;
    return Number(leftTensor.sourceDataByteOffset) - Number(rightTensor.sourceDataByteOffset);
  })[0]!;
  const artifacts = {
    fixtureBytes: new Uint8Array(fixtureBytes),
    wasmBytes: new Uint8Array(wasmBytes),
    artifact,
    bindings: bindingFile.bindings
  };
  await verifyPinnedQwen35VisionArtifacts(artifacts);

  const controller = new AbortController();
  const calls = {
    instantiations: 0,
    beginScopes: 0,
    endScopes: 0,
    runtimeDisposals: 0,
    gpuDisposals: 0,
    reads: 0
  };
  const device = {async sync() {}};
  const runtime = {
    initWebGPU() {},
    beginScope() {
      calls.beginScopes += 1;
    },
    endScope() {
      calls.endScopes += 1;
    },
    dispose() {
      calls.runtimeDisposals += 1;
    },
    webgpu() {
      return device;
    },
    cpu() {
      return {deviceType: 1, deviceId: 0, async sync() {}};
    },
    empty() {
      throw new Error('must not upload after source cancellation');
    },
    createVirtualMachine() {
      throw new Error('must not create the graph runner before source weights load');
    }
  } as unknown as Qwen35VisionRuntime;
  const tvmjs: Qwen35VisionTvmJs = {
    createPolyfillWASI() {
      return {imports: {}, start() {}};
    },
    async instantiate(wasm) {
      calls.instantiations += 1;
      assert.equal(wasm.byteLength, wasmBytes.byteLength);
      return runtime;
    }
  };
  const source = {
    manifest: wordManifest,
    async readTensor(_asset: string, name: string) {
      calls.reads += 1;
      assert.equal(name, sourceFirstBinding.sourceName);
      controller.abort();
      return new Uint8Array(sourceFirstBinding.sourceByteCount);
    }
  } as unknown as Qwen35PinnedSource;
  await assert.rejects(
    createQwen35VisionRuntimeRunner({
      tvmjs,
      gpuDevice: {
        destroy() {
          calls.gpuDisposals += 1;
        }
      } as unknown as GPUDevice,
      source,
      artifacts,
      signal: controller.signal
    }),
    {name: 'AbortError'}
  );
  assert.deepEqual(calls, {
    instantiations: 1,
    beginScopes: 1,
    endScopes: 1,
    runtimeDisposals: 1,
    gpuDisposals: 1,
    reads: 1
  });

  const corruptedArtifacts = {...artifacts, wasmBytes: new Uint8Array(artifacts.wasmBytes)};
  corruptedArtifacts.wasmBytes[0] = corruptedArtifacts.wasmBytes[0]! ^ 0xff;
  await assert.rejects(verifyPinnedQwen35VisionArtifacts(corruptedArtifacts), {code: 'INVALID_ASSET'});
});
