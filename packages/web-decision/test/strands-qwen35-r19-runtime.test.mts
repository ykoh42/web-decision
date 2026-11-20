import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {
  instantiatePinnedQwen35R19Runtime,
  type Qwen35R19Device,
  type Qwen35R19Runtime,
  type Qwen35R19TvmJs
} from '../src/strands/qwen35-r19-runtime.mts';

const [fixtureBytes, wasmBytes] = await Promise.all([
  readFile(new URL('../assets/qwen35/r19/fixture.json', import.meta.url)),
  readFile(new URL('../assets/qwen35/r19/qwen35-text-pointer.webgpu.wasm', import.meta.url))
]);

function fakeTvm() {
  const calls = {instantiations: 0, begins: 0, ends: 0, disposals: 0, gpuDisposals: 0};
  const device: Qwen35R19Device = {deviceType: 15, deviceId: 0, async sync() {}};
  const runtime = {
    initWebGPU() {},
    beginScope() {
      calls.begins++;
    },
    endScope() {
      calls.ends++;
    },
    dispose() {
      calls.disposals++;
    },
    webgpu: () => device,
    cpu: () => ({deviceType: 1, deviceId: 0, async sync() {}}),
    empty() {
      throw new Error('Unexpected R19 verification tensor allocation.');
    },
    makeShapeTuple() {
      throw new Error('Unexpected R19 verification shape allocation.');
    },
    createVirtualMachine: () => ({getFunction: () => () => undefined})
  } as unknown as Qwen35R19Runtime;
  const tvmjs: Qwen35R19TvmJs = {
    createPolyfillWASI: () => ({imports: {}, start() {}}),
    async instantiate(wasm) {
      calls.instantiations++;
      assert.equal(wasm.byteLength, wasmBytes.byteLength);
      return runtime;
    }
  };
  const gpuDevice = {destroy: () => calls.gpuDisposals++} as unknown as GPUDevice;
  return {tvmjs, gpuDevice, calls};
}

test('R19 runtime accepts the packed graph fixture and exact WASM', async () => {
  const mocks = fakeTvm();
  const handle = await instantiatePinnedQwen35R19Runtime({
    tvmjs: mocks.tvmjs,
    fixtureBytes: new Uint8Array(fixtureBytes),
    wasmBytes: new Uint8Array(wasmBytes),
    gpuDevice: mocks.gpuDevice
  });
  assert.equal(mocks.calls.instantiations, 1);
  await handle.dispose();
  assert.equal(mocks.calls.begins, 1);
  assert.equal(mocks.calls.ends, 1);
  assert.equal(mocks.calls.disposals, 1);
  assert.equal(mocks.calls.gpuDisposals, 1);
});

test('R19 runtime rejects changed artifact bytes before TVM instantiation', async () => {
  const mocks = fakeTvm();
  const changedWasm = new Uint8Array(wasmBytes);
  changedWasm[0] = changedWasm[0]! ^ 1;
  await assert.rejects(
    instantiatePinnedQwen35R19Runtime({
      tvmjs: mocks.tvmjs,
      fixtureBytes: new Uint8Array(fixtureBytes),
      wasmBytes: changedWasm,
      gpuDevice: mocks.gpuDevice
    }),
    /R19 TVM\.js artifact bytes differ from the pinned fixture/u
  );
  assert.equal(mocks.calls.instantiations, 0);
  assert.equal(mocks.calls.gpuDisposals, 1);
});
