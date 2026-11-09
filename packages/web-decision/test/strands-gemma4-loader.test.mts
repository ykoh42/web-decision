import assert from 'node:assert/strict';
import test from 'node:test';
import {loadStrandsGemma4FromSource} from '../src/strands/gemma4-e2b-loader.mjs';

function syntheticRuntime() {
  const events: string[] = [];
  let pointer = 0;
  const context = {
    pendingGPUToCPUCopy: null as Promise<void> | null,
    copyRawBytesToBuffer() {
      events.push('upload');
    },
    gpuBufferFromPtr() {
      return {size: 4};
    }
  };
  class Tensor {
    declare shape: number[];
    declare dtype: string;
    declare byteOffset: number;
    declare pointer: number;
    declare values: number[] | Float32Array;
    constructor(shape: number[], dtype: string) {
      this.shape = shape;
      this.dtype = dtype;
      this.byteOffset = 0;
      this.pointer = pointer++;
      this.values = [];
    }

    copyFrom(value: Tensor | ArrayLike<number>) {
      if (value instanceof Tensor) {
        this.values = value.values;
        context.pendingGPUToCPUCopy = Promise.resolve();
      } else {
        this.values = Array.from(value);
      }
      return this;
    }

    getDataPtr() {
      return this.pointer;
    }

    toArray() {
      return Array.from(this.values);
    }
  }
  const device = {
    async sync() {
      const pending = context.pendingGPUToCPUCopy;
      context.pendingGPUToCPUCopy = null;
      await pending;
    }
  };
  let loseDevice!: (info: {reason: string}) => void;
  const lost = new Promise<{reason: string}>((resolve) => {
    loseDevice = resolve;
  });
  const gpu = {
    lost,
    addEventListener() {},
    pushErrorScope() {},
    async popErrorScope() {
      return null;
    },
    destroy() {
      events.push('gpu.destroy');
      loseDevice({reason: 'destroyed'});
    }
  };
  const runtime = {
    lib: {webGPUContext: context},
    initWebGPU() {},
    beginScope() {
      events.push('runtime.begin');
    },
    endScope() {
      events.push('runtime.end');
    },
    webgpu() {
      return device;
    },
    cpu() {
      return {deviceType: 'cpu'};
    },
    empty(shape: number[], dtype: string) {
      return new Tensor([...shape], dtype);
    },
    createVirtualMachine() {
      return {
        getFunction(name: string) {
          assert.equal(name, 'main');
          return (ids: unknown, answerIndex: unknown, optionIndices: unknown, ...weights: unknown[]) => {
            assert.ok(ids instanceof Tensor);
            assert.ok(answerIndex instanceof Tensor);
            assert.ok(optionIndices instanceof Tensor);
            assert.equal(weights.length, 1248);
            const output = new Tensor([1, optionIndices.values.length], 'float32');
            output.values = new Float32Array(optionIndices.values.length);
            output.values[output.values.length - 1] = 2;
            return output;
          };
        }
      };
    },
    getGlobalFunc(name: string) {
      assert.equal(name, 'vm.builtin.memory_manager.clear');
      return () => events.push('memory.clear');
    },
    dispose() {
      events.push('runtime.dispose');
      gpu.destroy();
    }
  };
  return {events, gpu, runtime};
}

function syntheticSource() {
  const expectedParameters = Array.from({length: 1248}, (_, index) => ({index, shape: [1], dtype: 'float16'}));
  const records = expectedParameters.map(({index, shape, dtype}) => ({
    name: `param_${index}`,
    shape,
    dtype,
    nbytes: 2,
    byteOffset: index * 2
  }));
  const bytes = new Uint8Array(records.length * 2);
  return {
    assets: {
      graphWasm: Uint8Array.of(1),
      tokenizer: new TextEncoder().encode('{}'),
      tokenizerConfig: new TextEncoder().encode('{}')
    },
    cacheBase: new URL('https://assets.example/cache/'),
    plan: {
      expectedParameters,
      shards: [
        {path: 'params_shard_0.bin', bytes: bytes.byteLength, sha256: 'a'.repeat(64), records, expectedParameters}
      ]
    },
    reportProgress() {},
    bytes
  };
}

test('Gemma 4 requires a non-empty assetsUrl before requesting a GPU adapter', async () => {
  for (const assetsUrl of [undefined, '', ' ', null, 42]) {
    let adapterRequests = 0;
    await assert.rejects(
      loadStrandsGemma4FromSource({
        assetsUrl,
        createTokenizer() {},
        createRuntime() {},
        gpuApi: {
          async requestAdapter() {
            adapterRequests++;
          }
        }
      } as unknown as Parameters<typeof loadStrandsGemma4FromSource>[0]),
      {code: 'INVALID_INPUT'}
    );
    assert.equal(adapterRequests, 0);
  }
});

test('Gemma 4 loader gives the session an owned runner that can decide and dispose', async () => {
  const source = syntheticSource();
  const {events, gpu, runtime} = syntheticRuntime();
  const adapter = {
    features: new Set(['shader-f16']),
    limits: {
      maxBufferSize: 100663296,
      maxStorageBufferBindingSize: 100663296,
      maxStorageBuffersPerShaderStage: 8
    },
    async requestDevice() {
      return gpu;
    }
  };
  const tokenizer = {
    encode(text: string, {add_special_tokens}: {add_special_tokens?: boolean}) {
      const ids = Array.from(text, (character) => character.codePointAt(0)!);
      return {ids: add_special_tokens ? [2, ...ids] : ids};
    },
    decode(ids: number[]) {
      return ids.map((id) => String.fromCodePoint(id)).join('');
    }
  };
  const model = await loadStrandsGemma4FromSource(
    {
      assetsUrl: 'https://assets.example/model/',
      createTokenizer: () => tokenizer,
      createRuntime: () => runtime
    },
    {
      preparedAdapter: adapter,
      verifySource: async () => source,
      readFile: async () => source.bytes
    } as unknown as Parameters<typeof loadStrandsGemma4FromSource>[1]
  );

  assert.equal(events.filter((event) => event === 'upload').length, 1248);
  assert.equal(typeof model.releaseCachedMemory, 'function');
  const result = await model.decide({
    state: 'A small state.',
    questions: [{instructions: 'Choose one.', options: ['red', 'blue']}]
  });
  assert.equal(result.answers[0]!.selected, 'blue');
  await model.releaseCachedMemory();
  await model.dispose();
  assert.ok(events.includes('memory.clear'));
  assert.ok(events.includes('runtime.dispose'));
  assert.ok(events.includes('gpu.destroy'));
  await assert.rejects(
    model.decide({state: 'closed', questions: [{instructions: 'Choose one.', options: ['yes', 'no']}]}),
    {code: 'SESSION_CLOSED'}
  );
});
