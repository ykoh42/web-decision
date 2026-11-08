import assert from 'node:assert/strict';
import test from 'node:test';
import {WebDecisionError} from '../src/kev.js';
import {
  installOpenDeciderNanoGpuCompatibility,
  rewriteOpenDeciderNanoShader
} from '../src/opendecider-nano/gpu-compat.mjs';

const contextPrototype = {
  createShader(info, code) {
    return {kind: 'sync', info, code};
  },
  async createShaderAsync(info, code) {
    return {kind: 'async', info, code};
  },
  deviceAllocDataSpace(nbytes) {
    return nbytes;
  }
};

function createContext() {
  return Object.assign(Object.create(contextPrototype), {id: Symbol('runtime-context')});
}

test('rewrites scalar i8 arrays and nested boolean stores', () => {
  const source = `
    @group(0) @binding(0) var<storage, read_write> flags: array<i8>;
    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      flags[gid.x] = bool((gid.x + select(1u, 2u, gid.x > 0u)) > 1u);
    }
  `;
  const result = rewriteOpenDeciderNanoShader(source);
  assert.equal(result.changed, true);
  assert.equal(result.arrayCount, 1);
  assert.equal(result.storeCount, 1);
  assert.match(result.code, /flags: array<i32>/);
  assert.match(result.code, /flags\[gid\.x\] = i32\(bool\(\(gid\.x \+ select\(1u, 2u, gid\.x > 0u\)\) > 1u\)\);/);
});

test('leaves unrelated integer and boolean shaders unchanged', () => {
  const source = `
    // An old comment may mention i8 and array<i8>.
    var<storage, read_write> values: array<i32>;
    var<workgroup> localFlags: array<bool, 4>;
    fn readLast(index: u32) -> i32 { return values[index - 1u]; }
  `;
  assert.deepEqual(rewriteOpenDeciderNanoShader(source), {
    code: source,
    changed: false,
    arrayCount: 0,
    storeCount: 0
  });
});

test('rejects i8 uses outside the pinned array layout', () => {
  assert.throws(
    () => rewriteOpenDeciderNanoShader('var<private> value: i8;'),
    (error) => error instanceof WebDecisionError && error.code === 'UNSUPPORTED_ENVIRONMENT'
  );
});

test('rewrites one runtime context and expands small TVM storage allocations only', async () => {
  const prototype = contextPrototype;
  const originalAlloc = prototype.deviceAllocDataSpace;
  const originalShader = prototype.createShader;
  const originalAsyncShader = prototype.createShaderAsync;
  const first = createContext();
  const second = createContext();
  const compatibility = installOpenDeciderNanoGpuCompatibility(first);

  assert.equal(first.deviceAllocDataSpace(4096), 8192);
  assert.equal(first.deviceAllocDataSpace(8192), 8192);
  assert.equal(first.deviceAllocDataSpace(2048), 2048);
  const shader = first.createShader({}, 'var<storage, read_write> flags: array<i8>;');
  assert.match(shader.code, /array<i32>/);
  const asyncShader = await first.createShaderAsync({}, 'var<storage, read_write> flags: array<i8>;');
  assert.match(asyncShader.code, /array<i32>/);
  assert.equal(second.deviceAllocDataSpace(4096), 4096);
  assert.match(second.createShader({}, 'var<storage> flags: array<i8>;').code, /array<i8>/);
  assert.equal(Object.getPrototypeOf(first).deviceAllocDataSpace, originalAlloc);
  assert.equal(Object.getPrototypeOf(first).createShader, originalShader);
  assert.equal(Object.getPrototypeOf(first).createShaderAsync, originalAsyncShader);
  assert.deepEqual(compatibility.report, {
    shaderModules: 2,
    i8ArrayDeclarations: 2,
    boolStores: 0,
    expandedStorageAllocations: 1
  });

  compatibility.restore();
  compatibility.restore();
  assert.equal(Object.hasOwn(first, 'deviceAllocDataSpace'), false);
  assert.equal(Object.hasOwn(first, 'createShader'), false);
  assert.equal(Object.hasOwn(first, 'createShaderAsync'), false);
  assert.equal(first.deviceAllocDataSpace(4096), 4096);
});
