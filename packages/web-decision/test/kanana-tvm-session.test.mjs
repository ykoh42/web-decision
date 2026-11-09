import assert from 'node:assert/strict';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import test from 'node:test';

const outputDirectory = process.env.KANANA_TVM_SESSION_DIST;
if (!outputDirectory) throw new Error('Compile the Kanana TVM session before running its focused tests.');
const {createKananaTvmInferenceSession} = await import(pathToFileURL(join(outputDirectory, 'tvm-session.js')).href);

function weightRecord(name = 'blocks.0.weight') {
  return {name, shape: [2], dtype: 'float16', length: 4};
}

function fakeSetup({
  parameterNames = ['input_ids', 'attention_mask', 'blocks_0_weight'],
  arity = parameterNames.length,
  outputValues = [0.25, -2, 4.5],
  inferenceError,
  inferenceDelay = 0
} = {}) {
  const calls = {
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
    }
  };
  const outputBytes = new Uint8Array(new Float32Array(outputValues).buffer).slice();
  const output = {
    shape: [1, outputValues.length],
    dtype: 'float32',
    toRawBytes: () => outputBytes.slice()
  };
  const vm = {
    getFunction(name) {
      calls.functions.push(name);
      if (name === 'main')
        return async (...args) => {
          calls.mainArgs = args;
          calls.inferenceInputs.push(args.slice(0, 2).map(({value}) => [...value]));
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
      if (name === 'get_function_param_name') return (_functionName, index) => parameterNames[index.value];
      throw new Error(`Unexpected TVM function: ${name}`);
    }
  };
  const runtime = {
    cpu: () => ({kind: 'cpu'}),
    webgpu: () => device,
    scalar(value, dtype) {
      calls.scalar.push([value, dtype]);
      return {value, dtype};
    },
    empty(shape, dtype, targetDevice) {
      const tensor = {
        shape: [...shape],
        dtype,
        targetDevice,
        copyFrom(value) {
          this.value = value;
          return this;
        },
        copyFromRawBytes(bytes) {
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
  const sourceBytes = {
    'blocks.0.weight': new Uint8Array([1, 2, 3, 4]),
    'blocks.1.weight': new Uint8Array([5, 6, 7, 8])
  };
  const records = [weightRecord('blocks.1.weight'), weightRecord('blocks.0.weight')];
  const reads = [];
  const session = await createKananaTvmInferenceSession({
    setup,
    weightSource: {
      manifest: {weights: records},
      readWeight: async (requested, index) => {
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

test('rejects empty or over-limit input without truncating or invoking the model', async () => {
  const {calls, setup} = fakeSetup({parameterNames: ['input_ids', 'attention_mask'], arity: 2});
  const session = await createKananaTvmInferenceSession({setup, weightSource: {weights: []}});

  await assert.rejects(session.infer([], []), {code: 'invalid_input'});
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
  const inferenceError = new Error('TVM inference failed');
  const {calls, setup} = fakeSetup({
    parameterNames: ['input_ids', 'attention_mask'],
    arity: 2,
    inferenceError
  });
  const session = await createKananaTvmInferenceSession({setup, weightSource: {weights: []}});

  await assert.rejects(session.infer([1], [1]), inferenceError);
  assert.equal(calls.disposed, 0);
  assert.equal(calls.scopeBegins, 1);
  assert.equal(calls.scopeEnds, 1);
  assert.equal(calls.scopeDepth, 0);
  await session.dispose();
  assert.equal(calls.disposed, 1);
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
