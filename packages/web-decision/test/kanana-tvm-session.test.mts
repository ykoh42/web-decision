import assert from 'node:assert/strict';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import test from 'node:test';

const outputDirectory = process.env.KANANA_TVM_SESSION_DIST;
if (!outputDirectory) throw new Error('Compile the Kanana TVM session before running its focused tests.');
const {createKananaTvmInferenceSession, createKananaTvmVisionInferenceSession} = await import(
  pathToFileURL(join(outputDirectory, 'tvm-session.js')).href
);
const {KANANA_EMBEDDING_SIZE, KANANA_IMAGE_TOKEN_ID} = await import(
  pathToFileURL(join(outputDirectory, 'contract.mjs')).href
);

function weightRecord(name = 'blocks.0.weight') {
  return {name, shape: [2], dtype: 'float16', length: 4};
}

type FakeTensor = {
  shape: number[];
  dtype: string;
  targetDevice: unknown;
  value?: any;
  bytes?: Uint8Array;
  copyFrom(value: any): FakeTensor;
  copyFromRawBytes(bytes: Uint8Array): FakeTensor;
  toRawBytes(): Uint8Array;
};
type FakeSetupOptions = {
  parameterNames?: string[];
  arity?: number;
  outputValues?: number[];
  inferenceError?: unknown;
  syncError?: unknown;
  inferenceDelay?: number;
};

function fakeSetup(options: FakeSetupOptions = {}) {
  const parameterNames = options.parameterNames ?? ['input_ids', 'attention_mask', 'blocks_0_weight'];
  const arity = options.arity ?? parameterNames.length;
  const outputValues = options.outputValues ?? [0.25, -2, 4.5];
  const inferenceError = options.inferenceError;
  const syncError = options.syncError;
  const inferenceDelay = options.inferenceDelay ?? 0;
  const calls: {
    functions: string[];
    mainArgs: any[];
    inferenceInputs: number[][][];
    scalar: Array<[number, string]>;
    tensors: FakeTensor[];
    sync: number;
    scopeBegins: number;
    scopeEnds: number;
    scopeDepth: number;
    activeInferences: number;
    maxConcurrentInferences: number;
    disposed: number;
  } = {
    functions: [],
    mainArgs: [],
    inferenceInputs: [],
    scalar: [],
    tensors: [],
    sync: 0,
    scopeBegins: 0,
    scopeEnds: 0,
    scopeDepth: 0,
    activeInferences: 0,
    maxConcurrentInferences: 0,
    disposed: 0
  };
  const device = {
    async sync() {
      calls.sync++;
      if (syncError) throw syncError;
    }
  };
  const outputBytes = new Uint8Array(new Float32Array(outputValues).buffer).slice();
  const output = {
    shape: [1, outputValues.length],
    dtype: 'float32',
    toRawBytes: () => outputBytes.slice()
  };
  const vm = {
    getFunction(name: string) {
      calls.functions.push(name);
      if (name === 'main')
        return async (...args: any[]) => {
          calls.mainArgs = args;
          calls.inferenceInputs.push(args.slice(0, 2).map(({value}: {value: number[]}) => [...value]));
          calls.activeInferences++;
          calls.maxConcurrentInferences = Math.max(calls.maxConcurrentInferences, calls.activeInferences);
          try {
            if (inferenceDelay) await new Promise((resolve) => setTimeout(resolve, inferenceDelay));
            if (inferenceError) throw inferenceError;
            return output;
          } finally {
            calls.activeInferences--;
          }
        };
      if (name === 'get_function_arity') return () => arity;
      if (name === 'get_function_param_name')
        return (_functionName: string, index: {value: number}) => parameterNames[index.value];
      throw new Error(`Unexpected TVM function: ${name}`);
    }
  };
  const runtime = {
    cpu: () => ({kind: 'cpu'}),
    webgpu: () => device,
    scalar(value: number, dtype: string) {
      calls.scalar.push([value, dtype]);
      return {value, dtype};
    },
    empty(shape: number[], dtype: string, targetDevice: unknown) {
      const tensor: FakeTensor = {
        shape: [...shape],
        dtype,
        targetDevice,
        copyFrom(value: any) {
          this.value = value;
          return this;
        },
        copyFromRawBytes(bytes: Uint8Array) {
          this.bytes = bytes.slice();
          return this;
        },
        toRawBytes() {
          return this.bytes?.slice() ?? this.value?.toRawBytes?.() ?? new Uint8Array();
        }
      };
      calls.tensors.push(tensor);
      return tensor;
    },
    initWebGPU() {},
    beginScope() {
      calls.scopeBegins++;
      calls.scopeDepth++;
    },
    endScope() {
      calls.scopeEnds++;
      calls.scopeDepth--;
    },
    dispose() {},
    createVirtualMachine() {
      return vm;
    }
  };
  return {
    calls,
    setup: {
      runtime,
      vm,
      dispose() {
        calls.disposed++;
      }
    }
  };
}

test('rejects a TVM main with a mismatched arity or parameter name and closes its scope', async () => {
  const wrongArity = fakeSetup({arity: 4});
  await assert.rejects(
    createKananaTvmInferenceSession({
      setup: wrongArity.setup,
      weightSource: {weights: [{...weightRecord(), bytes: new Uint8Array([1, 2, 3, 4])}]}
    }),
    {code: 'parameter_mismatch'}
  );
  assert.equal(wrongArity.calls.disposed, 1);

  const wrongName = fakeSetup({parameterNames: ['tokens', 'attention_mask', 'blocks_0_weight']});
  await assert.rejects(
    createKananaTvmInferenceSession({
      setup: wrongName.setup,
      weightSource: {weights: [{...weightRecord(), bytes: new Uint8Array([1, 2, 3, 4])}]}
    }),
    {code: 'parameter_mismatch'}
  );
  assert.equal(wrongName.calls.disposed, 1);
});

test('runs short int32 inputs with manifest weights and returns hidden-state values', async () => {
  const {calls, setup} = fakeSetup({
    parameterNames: ['input_ids', 'attention_mask', 'blocks_0_weight', 'blocks_1_weight']
  });
  const sourceBytes: Record<string, Uint8Array> = {
    'blocks.0.weight': new Uint8Array([1, 2, 3, 4]),
    'blocks.1.weight': new Uint8Array([5, 6, 7, 8])
  };
  const records = [weightRecord('blocks.1.weight'), weightRecord('blocks.0.weight')];
  const reads: Array<[string, number]> = [];
  const session = await createKananaTvmInferenceSession({
    setup,
    weightSource: {
      manifest: {weights: records},
      readWeight: async (requested: {name: string}, index: number) => {
        reads.push([requested.name, index]);
        return sourceBytes[requested.name];
      }
    }
  });

  const result = await session.infer(new Int32Array([7, 8]), new Int32Array([1, 1]));

  assert.deepEqual(reads, [
    ['blocks.0.weight', 1],
    ['blocks.1.weight', 0]
  ]);
  assert.deepEqual(calls.scalar, [
    [0, 'int'],
    [1, 'int'],
    [2, 'int'],
    [3, 'int']
  ]);
  assert.deepEqual(
    calls.mainArgs.slice(0, 2).map(({shape, dtype, value}) => [shape, dtype, [...value]]),
    [
      [[1, 2], 'int32', [7, 8]],
      [[1, 2], 'int32', [1, 1]]
    ]
  );
  assert.deepEqual(calls.mainArgs[2].bytes, sourceBytes['blocks.0.weight']);
  assert.deepEqual(calls.mainArgs[3].bytes, sourceBytes['blocks.1.weight']);
  assert.deepEqual(result.shape, [1, 3]);
  assert.equal(result.dtype, 'float32');
  assert.deepEqual([...result.values], [0.25, -2, 4.5]);
  assert.equal(calls.sync, 3);
  assert.equal(calls.scopeBegins, 1);
  assert.equal(calls.scopeEnds, 1);
  assert.equal(calls.scopeDepth, 0);

  await session.dispose();
  assert.equal(calls.disposed, 1);
});

test('runs multimodal text graphs with image embeddings, ordered positions, and four-axis positions', async () => {
  const parameterNames = ['input_ids', 'attention_mask', 'image_embeddings', 'image_positions', 'position_ids'];
  const {calls, setup} = fakeSetup({parameterNames, arity: parameterNames.length});
  const session = await createKananaTvmInferenceSession({setup, weightSource: {weights: []}});
  const ids = [1, KANANA_IMAGE_TOKEN_ID, KANANA_IMAGE_TOKEN_ID, 2];
  const embeddings = Float32Array.from({length: 2 * KANANA_EMBEDDING_SIZE}, (_, index) => index / 10);
  const positionIds = Int32Array.from({length: 4 * ids.length}, (_, index) => index % ids.length);

  await session.infer(ids, [1, 1, 1, 1], {
    imageEmbeddings: {shape: [2, KANANA_EMBEDDING_SIZE], values: embeddings},
    imagePositions: [1, 2],
    positionIds
  });

  assert.deepEqual(
    calls.mainArgs.slice(0, 5).map(({shape, dtype}) => [shape, dtype]),
    [
      [[1, 4], 'int32'],
      [[1, 4], 'int32'],
      [[2, KANANA_EMBEDDING_SIZE], 'float32'],
      [[2], 'int32'],
      [[4, 1, 4], 'int32']
    ]
  );
  assert.deepEqual([...calls.mainArgs[2].value], [...embeddings]);
  assert.deepEqual([...calls.mainArgs[3].value], [1, 2]);
  assert.deepEqual([...calls.mainArgs[4].value], [...positionIds]);
  await session.dispose();
});

test('uses empty image tensors and linear positions for text-only prompts on a multimodal graph', async () => {
  const parameterNames = ['input_ids', 'attention_mask', 'image_embeddings', 'image_positions', 'position_ids'];
  const {calls, setup} = fakeSetup({parameterNames, arity: parameterNames.length});
  const session = await createKananaTvmInferenceSession({setup, weightSource: {weights: []}});

  await session.infer([7, 8], [1, 1]);

  assert.deepEqual(calls.mainArgs[2].shape, [0, KANANA_EMBEDDING_SIZE]);
  assert.deepEqual(calls.mainArgs[3].shape, [0]);
  assert.deepEqual(calls.mainArgs[4].shape, [4, 1, 2]);
  assert.deepEqual([...calls.mainArgs[4].value], [0, 1, 0, 1, 0, 1, 0, 1]);
  await session.dispose();
});

test('rejects multimodal inputs that do not match the image tokens or embedding dimension', async () => {
  const parameterNames = ['input_ids', 'attention_mask', 'image_embeddings', 'image_positions', 'position_ids'];
  const {setup, calls} = fakeSetup({parameterNames, arity: parameterNames.length});
  const session = await createKananaTvmInferenceSession({setup, weightSource: {weights: []}});
  const valid = {
    imageEmbeddings: {shape: [1, KANANA_EMBEDDING_SIZE], values: new Float32Array(KANANA_EMBEDDING_SIZE)},
    imagePositions: [1],
    positionIds: new Int32Array(8)
  };

  await assert.rejects(session.infer([1, KANANA_IMAGE_TOKEN_ID], [1, 1], {...valid, imagePositions: [0]}), {
    code: 'invalid_input'
  });
  await assert.rejects(
    session.infer([1, KANANA_IMAGE_TOKEN_ID], [1, 1], {
      ...valid,
      imageEmbeddings: {shape: [1, KANANA_EMBEDDING_SIZE - 1], values: new Float32Array(KANANA_EMBEDDING_SIZE - 1)}
    }),
    {code: 'invalid_input'}
  );
  await assert.rejects(
    session.infer([1, KANANA_IMAGE_TOKEN_ID], [1, 1], {
      ...valid,
      imageEmbeddings: {
        shape: [1, KANANA_EMBEDDING_SIZE],
        values: new Float64Array(KANANA_EMBEDDING_SIZE).fill(3.5e38)
      }
    }),
    {code: 'invalid_input'}
  );
  const outOfRangePositions = new Float64Array(valid.positionIds);
  outOfRangePositions[0] = 0x80000000;
  await assert.rejects(
    session.infer([1, KANANA_IMAGE_TOKEN_ID], [1, 1], {...valid, positionIds: outOfRangePositions}),
    {code: 'invalid_input'}
  );
  assert.equal(calls.mainArgs.length, 0);
  await session.dispose();
});

test('runs a dynamic vision graph with model-derived patch and interpolation inputs', async () => {
  const parameterNames = ['pixels', 'positions', 'indices', 'weights', 'merged_tokens'];
  const outputValues = Array.from({length: 2_560}, () => 0.25);
  const {calls, setup} = fakeSetup({parameterNames, arity: parameterNames.length, outputValues});
  const session = await createKananaTvmVisionInferenceSession({
    setup,
    weightSource: {weights: []},
    patchValues: 3,
    numPositionEmbeddings: 9,
    spatialMergeSize: 2,
    maxPatchCount: 4
  });
  const pixels = Float32Array.from({length: 12}, (_, index) => index / 10);

  const result = await session.infer(pixels, [1, 2, 2]);

  assert.deepEqual(
    calls.mainArgs.slice(0, 5).map(({shape, dtype}) => [shape, dtype]),
    [
      [[4, 3], 'float32'],
      [[4, 2], 'int32'],
      [[4, 4], 'int32'],
      [[4, 4], 'float32'],
      [[1], 'int32']
    ]
  );
  assert.deepEqual(result.shape, [1, 2_560]);
  assert.equal(result.values.length, 2_560);
  assert.equal(calls.scopeBegins, 1);
  assert.equal(calls.scopeEnds, 1);
  await session.dispose();
});

test('rejects vision inputs beyond the compiled graph range or with the wrong pixel count', async () => {
  const parameterNames = ['pixels', 'positions', 'indices', 'weights', 'merged_tokens'];
  const {calls, setup} = fakeSetup({parameterNames, arity: parameterNames.length});
  const session = await createKananaTvmVisionInferenceSession({
    setup,
    weightSource: {weights: []},
    patchValues: 3,
    numPositionEmbeddings: 9,
    spatialMergeSize: 2,
    maxPatchCount: 4
  });

  await assert.rejects(session.infer(new Float32Array(3), [1, 2, 2]), {code: 'invalid_input'});
  await assert.rejects(session.infer(new Float64Array(12).fill(Number.MAX_VALUE), [1, 2, 2]), {code: 'invalid_input'});
  await assert.rejects(session.infer(new Float32Array(5 * 3), [1, 2, 10]), {code: 'input_too_large'});
  await assert.rejects(session.infer(new Float32Array(3), [1, 1_073_741_824, 2]), {code: 'input_too_large'});
  assert.equal(calls.mainArgs.length, 0);
  await session.dispose();
});

test('matches a Safetensors source name to its explicit TVM parameter name', async () => {
  const parameterName = 'p_language_model_layers_0_self_attn_q_proj_weight';
  const {calls, setup} = fakeSetup({parameterNames: ['input_ids', 'attention_mask', parameterName]});
  const bytes = new Uint8Array([1, 2, 3, 4]);
  const session = await createKananaTvmInferenceSession({
    setup,
    weightSource: {
      manifest: {
        weights: [
          {
            ...weightRecord('model.language_model.layers.0.self_attn.q_proj.weight'),
            parameterName
          }
        ]
      },
      readWeight: async () => bytes
    }
  });

  await session.infer([7], [1]);

  assert.deepEqual(calls.mainArgs[2].bytes, bytes);
  await session.dispose();
  assert.equal(calls.disposed, 1);
});

test('rejects empty or over-limit input without truncating or invoking the model', async () => {
  const {calls, setup} = fakeSetup({parameterNames: ['input_ids', 'attention_mask'], arity: 2});
  const session = await createKananaTvmInferenceSession({setup, weightSource: {weights: []}});

  await assert.rejects(session.infer([], []), {code: 'invalid_input'});
  let oversizedInputReads = 0;
  const oversizedIds = new Proxy([], {
    get(target, property, receiver) {
      if (property === 'length') return 1_000_000_000;
      if (property === '0') oversizedInputReads++;
      return Reflect.get(target, property, receiver);
    }
  });
  await assert.rejects(session.infer(oversizedIds, [1]), {code: 'input_too_long'});
  assert.equal(oversizedInputReads, 0);

  const longIds = new Int32Array(20_481).fill(1);
  const longMask = new Int32Array(longIds.length).fill(1);
  await assert.rejects(session.infer(longIds, longMask), {code: 'input_too_long'});
  assert.equal(calls.mainArgs.length, 0);
  await session.dispose();
});

test('ends each temporary scope after multiple sequential inferences', async () => {
  const {calls, setup} = fakeSetup({parameterNames: ['input_ids', 'attention_mask'], arity: 2});
  const session = await createKananaTvmInferenceSession({setup, weightSource: {weights: []}});

  await session.infer([1], [1]);
  await session.infer([2, 3], [1, 1]);

  assert.equal(calls.scopeBegins, 2);
  assert.equal(calls.scopeEnds, 2);
  assert.equal(calls.scopeDepth, 0);
  await session.dispose();
});

test('propagates inference failures and keeps the owning scope disposable', async () => {
  const inferenceError = Object.assign(new Error('TVM inference failed'), {code: 'runtime_failure'});
  const {calls, setup} = fakeSetup({
    parameterNames: ['input_ids', 'attention_mask'],
    arity: 2,
    inferenceError
  });
  const session = await createKananaTvmInferenceSession({setup, weightSource: {weights: []}});

  await assert.rejects(session.infer([1], [1]), (error) => {
    assert.equal((error as Error).message, 'Kanana TVM graph execution failed: Error: TVM inference failed');
    assert.equal((error as Error).cause, inferenceError);
    assert.equal((error as Error & {code?: string}).code, 'runtime_failure');
    return true;
  });
  assert.equal(calls.disposed, 0);
  assert.equal(calls.scopeBegins, 1);
  assert.equal(calls.scopeEnds, 1);
  assert.equal(calls.scopeDepth, 0);
  await session.dispose();
  assert.equal(calls.disposed, 1);
});

test('reports the WebGPU synchronization stage and preserves its cause', async () => {
  const syncError = new Error('GPU synchronization failed');
  const {calls, setup} = fakeSetup({
    parameterNames: ['input_ids', 'attention_mask'],
    arity: 2,
    syncError
  });
  const session = await createKananaTvmInferenceSession({setup, weightSource: {weights: []}});

  await assert.rejects(session.infer([1], [1]), (error) => {
    assert.equal(
      (error as Error).message,
      'Kanana TVM WebGPU execution synchronization failed: Error: GPU synchronization failed'
    );
    assert.equal((error as Error).cause, syncError);
    return true;
  });
  assert.equal(calls.scopeDepth, 0);
  await session.dispose();
});

test('serializes overlapping inferences and waits for accepted calls before disposal', async () => {
  const {calls, setup} = fakeSetup({
    parameterNames: ['input_ids', 'attention_mask'],
    arity: 2,
    inferenceDelay: 5
  });
  const session = await createKananaTvmInferenceSession({setup, weightSource: {weights: []}});
  const first = session.infer([1], [1]);
  const second = session.infer([2], [1]);
  const closing = session.dispose();

  const results = await Promise.all([first, second, closing]);

  assert.deepEqual([...results[0].values], [0.25, -2, 4.5]);
  assert.deepEqual([...results[1].values], [0.25, -2, 4.5]);
  assert.deepEqual(calls.inferenceInputs, [
    [[1], [1]],
    [[2], [1]]
  ]);
  assert.equal(calls.maxConcurrentInferences, 1);
  assert.equal(calls.scopeBegins, 2);
  assert.equal(calls.scopeEnds, 2);
  assert.equal(calls.scopeDepth, 0);
  assert.equal(calls.disposed, 1);
});

test('snapshots queued input arrays when infer is called', async () => {
  const {calls, setup} = fakeSetup({
    parameterNames: ['input_ids', 'attention_mask'],
    arity: 2,
    inferenceDelay: 10
  });
  const session = await createKananaTvmInferenceSession({setup, weightSource: {weights: []}});
  const first = session.infer([1], [1]);
  const ids = Int32Array.from([2]);
  const mask = Int32Array.from([1]);
  const second = session.infer(ids, mask);
  ids[0] = 3;
  mask[0] = 0;

  await Promise.all([first, second]);

  assert.deepEqual(calls.inferenceInputs, [
    [[1], [1]],
    [[2], [1]]
  ]);
  await session.dispose();
});

test('disposes the setup scope once and rejects inference after disposal', async () => {
  const {calls, setup} = fakeSetup();
  const session = await createKananaTvmInferenceSession({
    setup,
    weightSource: {
      weights: [{...weightRecord(), bytes: new Uint8Array([1, 2, 3, 4])}]
    }
  });

  await session.dispose();
  await session.dispose();

  assert.equal(calls.disposed, 1);
  await assert.rejects(session.infer([1], [1]), {code: 'session_disposed'});
});
