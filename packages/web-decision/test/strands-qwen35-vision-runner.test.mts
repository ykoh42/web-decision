import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {prepareQwen35GraphInput} from '../src/strands/qwen35-graph-input.mts';
import {createQwen35Session} from '../src/strands/qwen35-session.mts';
import {
  createQwen35VisionFeatureRunner,
  isPinnedQwen35VisionRopeInitializationEvidence,
  type Qwen35VisionArtifact,
  type Qwen35VisionArtifactArgument
} from '../src/strands/qwen35-vision-runner.mts';
import type {DLDevice, Instance, Tensor} from '../src/runtime/tvm-types.js';

const utf8 = new TextEncoder();
const fixtureDirectory = new URL('./fixtures/qwen35-image-input/', import.meta.url);
const goldens = JSON.parse(
  await readFile(new URL('./fixtures/qwen35-text-vision-goldens.json', import.meta.url), 'utf8')
) as {
  readonly cases: readonly {
    readonly name: string;
    readonly files: readonly string[];
    readonly gridThw: readonly (readonly [number, number, number])[];
  }[];
};
const fourImageCase = goldens.cases.find(({name}) => name === 'four-image-boundary')!;
const require = createRequire(import.meta.url);
const magickWasm = await readFile(require.resolve('@imagemagick/magick-wasm/magick.wasm'));

const inputMetadata = [
  ['pixel_values', 'float32', ['T.int64(4) * s34', 'T.int64(1536)'], [256, 1536]],
  ['grid_thw', 'int32', ['T.int64(1)', 'T.int64(3)'], [1, 3]],
  ['interp_indices', 'int32', ['T.int64(4) * s34', 'T.int64(4)'], [256, 4]],
  ['interp_weights', 'float32', ['T.int64(4) * s34', 'T.int64(4)'], [256, 4]],
  ['position_ids_by_token', 'int32', ['s34', 'T.int64(8)'], [64, 8]],
  ['cu_seqlens', 'int32', ['T.int64(2)'], [2]]
] as const;

function artifact(mainArgumentOrder: readonly Qwen35VisionArtifactArgument[] = makeArguments()): Qwen35VisionArtifact {
  return {
    format: 'qwen35-dynamic-per-image-vision-tvmjs-fixture-v1',
    visionRopeInitialization: {
      restoredBuffers: ['inv_freq', 'original_inv_freq'],
      sourceModelParameterDtype: 'bfloat16',
      bufferDtype: 'float32',
      matchesPinnedInitialization: true,
      bufferSha256: {
        inv_freq: '9cc63b858a37a16bf14e559754d6fee5f50ae74aabfce29fec9097b70902849c',
        original_inv_freq: '9cc63b858a37a16bf14e559754d6fee5f50ae74aabfce29fec9097b70902849c'
      }
    },
    shapeContract: {
      oneImagePerInvocation: true,
      patchRows: {formula: '4*S', minimum: 256, maximum: 65_536}
    },
    mainArgumentOrder
  };
}

test('requires pinned FP32 RoPE buffer evidence for the BF16 source model', () => {
  const evidence = artifact().visionRopeInitialization;
  assert.equal(isPinnedQwen35VisionRopeInitializationEvidence(evidence), true);
  assert.equal(isPinnedQwen35VisionRopeInitializationEvidence(undefined), false);
  assert.equal(isPinnedQwen35VisionRopeInitializationEvidence({...evidence, bufferDtype: 'bfloat16'}), false);
  assert.equal(
    isPinnedQwen35VisionRopeInitializationEvidence({
      ...evidence,
      bufferSha256: {...evidence.bufferSha256, inv_freq: '0'.repeat(64)}
    }),
    false
  );
});

function makeArguments(): Qwen35VisionArtifactArgument[] {
  return [
    ...inputMetadata.map(([name, dtype, relaxShape, shape]) => ({
      name,
      sourceKind: 'USER_INPUT' as const,
      dtype,
      relaxShape,
      shape
    })),
    {
      name: 'p_visual_test_weight',
      sourceKind: 'PARAMETER',
      dtype: 'float32',
      relaxShape: ['T.int64(1)'],
      shape: [1]
    }
  ];
}

function tokenizerFixture() {
  const specialTokens = [
    ['<|image_pad|>', 248056],
    ['<|vision_end|>', 248054],
    ['<|endoftext|>', 248044],
    ['<bos>', 248043]
  ] as const;
  const byteDecoder = Object.fromEntries(
    Array.from({length: 256}, (_, byte) => [String.fromCharCode(byte + 0x100), byte])
  );
  const added = new Set(specialTokens.map(([, id]) => id));
  return {
    decoder: {byte_decoder: byteDecoder},
    get_added_tokens_decoder: () => added,
    encode(text: string, {add_special_tokens}: {readonly add_special_tokens: boolean}) {
      const ids: number[] = [];
      const tokens: string[] = [];
      if (add_special_tokens) {
        ids.push(248043);
        tokens.push('<bos>');
      }
      let cursor = 0;
      while (cursor < text.length) {
        const special = specialTokens.find(([token]) => text.startsWith(token, cursor));
        if (special) {
          tokens.push(special[0]);
          ids.push(special[1]);
          cursor += special[0].length;
          continue;
        }
        let end = cursor + 1;
        while (end < text.length && !specialTokens.some(([token]) => text.startsWith(token, end))) end += 1;
        for (const byte of utf8.encode(text.slice(cursor, end).normalize('NFC'))) {
          tokens.push(String.fromCharCode(byte + 0x100));
          ids.push(byte + 1);
        }
        cursor = end;
      }
      return {ids, tokens};
    }
  };
}

async function imagePayloads(imageCount: number) {
  const fixtureImages = await Promise.all(
    fourImageCase!.files.map(async (file) => {
      const bytes = await readFile(new URL(file, fixtureDirectory));
      return `data:image/png;base64,${bytes.toString('base64')}`;
    })
  );
  return Array.from({length: imageCount}, (_, index) => fixtureImages[index % fixtureImages.length]!);
}

async function graphInput(imageCount = 4) {
  const images = await imagePayloads(imageCount);
  return prepareQwen35GraphInput(
    tokenizerFixture(),
    {
      state: 'Use these images in their listed order.',
      images,
      questions: {
        choice: {type: 'choice', instructions: 'Choose one.', criteria: {first: null, second: null}}
      }
    },
    {magickWasm, maxBatch: 1, maxImages: imageCount}
  );
}

function bytesOf(values: Int32Array | Float32Array): Uint8Array {
  return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
}

class MockTensor implements Tensor {
  readonly shape: number[];
  readonly dtype: string;
  readonly device: DLDevice;
  private bytes = new Uint8Array();

  constructor(
    shape: readonly number[],
    dtype: string,
    bytes?: Uint8Array,
    device: DLDevice = {deviceType: 1, deviceId: 0, sync: async () => {}}
  ) {
    this.shape = [...shape];
    this.dtype = dtype;
    this.device = device;
    if (bytes) this.bytes = bytes.slice();
  }

  copyFrom(
    data:
      | Tensor
      | number[]
      | Float32Array
      | Float64Array
      | Int32Array
      | Int8Array
      | Uint8Array
      | Uint32Array
      | Uint8ClampedArray
  ): this {
    if (Array.isArray(data)) this.bytes = bytesOf(Float32Array.from(data)).slice();
    else if (ArrayBuffer.isView(data))
      this.bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
    else this.bytes = (data as unknown as MockTensor).bytes.slice();
    return this;
  }

  copyFromRawBytes(bytes: Uint8Array): this {
    this.bytes = bytes.slice();
    return this;
  }

  toRawBytes(): Uint8Array {
    return this.bytes.slice();
  }

  toArray(): Uint8Array {
    return this.bytes.slice();
  }
}

type CapturedTensor = Readonly<{shape: readonly number[]; dtype: string; bytes: Uint8Array}>;

function capture(tensor: unknown): CapturedTensor {
  const value = tensor as MockTensor;
  return {shape: value.shape, dtype: value.dtype, bytes: value.toRawBytes()};
}

function createRuntime(
  parameter: Tensor,
  {
    sync = async () => {},
    mainFailure,
    endScopeFailure,
    outputValue
  }: Readonly<{sync?: () => Promise<void>; mainFailure?: Error; endScopeFailure?: Error; outputValue?: number}> = {}
) {
  const calls: CapturedTensor[][] = [];
  const gpuDevice = {deviceType: 15, deviceId: 0, sync};
  const runtime: Pick<Instance, 'beginScope' | 'endScope' | 'createVirtualMachine' | 'empty' | 'webgpu' | 'cpu'> = {
    beginScope() {},
    endScope() {
      if (endScopeFailure) throw endScopeFailure;
    },
    webgpu: () => gpuDevice,
    cpu: () => ({deviceType: 1, deviceId: 0, sync: async () => {}}),
    empty(shape, dtype, device) {
      const normalized = Array.isArray(shape) ? shape : [shape];
      return new MockTensor(normalized, dtype, undefined, device);
    },
    createVirtualMachine(device) {
      assert.equal(device, gpuDevice);
      return {
        getFunction(name) {
          assert.equal(name, 'main');
          return (...args) => {
            calls.push(args.map(capture));
            if (mainFailure) throw mainFailure;
            const positions = args[4] as MockTensor;
            const rows = positions!.shape[0]!;
            const value = outputValue ?? calls.length;
            const output = new Float32Array(rows * 2048).fill(value);
            return new MockTensor([rows, 2048], 'float32', bytesOf(output));
          };
        }
      };
    }
  };
  return {runtime, calls, parameters: new Map([['p_visual_test_weight', parameter]])};
}

test('retains finite full precision features only with the FP32 vision fixture', async () => {
  const input = await graphInput(1);
  const legacy = artifact();
  const fp32 = {...legacy, activationPrecision: 'float32'} as const;
  const run = (graph: Qwen35VisionArtifact, outputValue: number) => {
    const mock = createRuntime(new MockTensor([1], 'float32'), {outputValue});
    return createQwen35VisionFeatureRunner({
      artifact: graph,
      runtime: mock.runtime,
      parameterTensors: mock.parameters
    }).run(input);
  };
  const fractional = Math.fround(0.1);
  assert.equal((await run(fp32, fractional))[0]![0], fractional);
  await assert.rejects(run(legacy, fractional), /preserve source BF16/);
  await assert.rejects(run(fp32, Infinity), /finite float32/);
});

test('runs the dynamic single-image TVM ABI in source order for four images', async () => {
  const input = await graphInput();
  const parameter = new MockTensor([1], 'float32', new Uint8Array(4));
  const mock = createRuntime(parameter);
  const runner = createQwen35VisionFeatureRunner({
    artifact: artifact(),
    runtime: mock.runtime,
    parameterTensors: mock.parameters
  });

  const outputs = await runner.run(input);

  assert.equal(input.imageCount, 4);
  assert.equal(outputs.length, input.imageCount);
  assert.equal(mock.calls.length, input.imageCount);
  let rowOffset = 0;
  for (let index = 0; index < input.imageCount; index += 1) {
    const geometry = input.visionGeometry[index]!;
    const args = mock.calls[index]!;
    assert.equal(args.length, 7);
    assert.deepEqual(
      args.slice(0, 6).map(({shape, dtype}) => ({shape, dtype})),
      [
        {shape: [geometry.patchRows, 1536], dtype: 'float32'},
        {shape: [1, 3], dtype: 'int32'},
        {shape: [geometry.patchRows, 4], dtype: 'int32'},
        {shape: [geometry.patchRows, 4], dtype: 'float32'},
        {shape: [geometry.featureRows, 8], dtype: 'int32'},
        {shape: [2], dtype: 'int32'}
      ]
    );
    assert.deepEqual(
      args[0]!.bytes,
      bytesOf(input.imageInput!.pixelValues.subarray(rowOffset * 1536, (rowOffset + geometry.patchRows) * 1536))
    );
    assert.deepEqual(args[1]!.bytes, bytesOf(geometry.gridThwInput));
    assert.deepEqual(args[2]!.bytes, bytesOf(geometry.interpIndices));
    assert.deepEqual(args[3]!.bytes, bytesOf(geometry.interpWeights));
    assert.deepEqual(args[4]!.bytes, bytesOf(geometry.positionIdsByToken));
    assert.deepEqual(args[5]!.bytes, bytesOf(geometry.cuSeqlens));
    assert.deepEqual(outputs[index], new Float32Array(geometry.featureRows * 2048).fill(index + 1));
    rowOffset += geometry.patchRows;
  }
});

test('honors a configured upstream image limit above the default of four', async () => {
  const input = await graphInput(5);
  const mock = createRuntime(new MockTensor([1], 'float32', new Uint8Array(4)));
  const runner = createQwen35VisionFeatureRunner({
    artifact: artifact(),
    runtime: mock.runtime,
    parameterTensors: mock.parameters
  });

  const outputs = await runner.run(input);

  assert.equal(input.imageCount, 5);
  assert.equal(outputs.length, 5);
  assert.equal(mock.calls.length, 5);
  for (let index = 0; index < outputs.length; index += 1) {
    assert.deepEqual(outputs[index], new Float32Array(input.visionGeometry[index]!.featureRows * 2048).fill(index + 1));
  }
});

test('Qwen systemOne carries five configured images through packing and runner in source order', async () => {
  const images = await imagePayloads(5);
  let receivedInput: Awaited<ReturnType<typeof graphInput>> | undefined;
  const session = createQwen35Session({
    tokenizer: tokenizerFixture(),
    modelName: 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610',
    graphOptions: {magickWasm, maxBatch: 1, maxImages: 5},
    runner: {
      async run(chunk, input) {
        receivedInput = input;
        assert.equal(input.imageCount, 5);
        return chunk.questions.map(() => [0.2, 0.8]);
      },
      async releaseCachedMemory() {},
      async dispose() {}
    }
  });

  try {
    const response = await session.systemOne({
      state: 'Inspect these images in their listed order.',
      images,
      questions: {truth: {type: 'noul', instructions: 'Does the claim hold?'}}
    });

    assert.deepEqual(receivedInput?.request.images, images);
    assert.deepEqual(receivedInput?.imageInput?.imageGridThw, [...fourImageCase!.gridThw, fourImageCase!.gridThw[0]]);
    assert.deepEqual(
      receivedInput?.visionGeometry.map(({gridThw}) => gridThw),
      [...fourImageCase!.gridThw, fourImageCase!.gridThw[0]]
    );
    assert.equal(response.answers.truth?.type, 'noul');
    if (response.answers.truth?.type !== 'noul') assert.fail('Expected the source noul response variant.');
    assert.equal(response.answers.truth.noul, 0.8);
  } finally {
    await session.dispose();
  }
});

test('rejects an artifact whose input shapes do not share the dynamic patch symbol', () => {
  const invalidArguments = makeArguments().map((argument) =>
    argument.name === 'position_ids_by_token' ? {...argument, relaxShape: ['T.int64(64)', 'T.int64(8)']} : argument
  );
  assert.throws(
    () =>
      createQwen35VisionFeatureRunner({
        artifact: artifact(invalidArguments),
        runtime: createRuntime(new MockTensor([1], 'float32')).runtime,
        parameterTensors: new Map([['p_visual_test_weight', new MockTensor([1], 'float32')]])
      }),
    /dynamic \[P=4\*S\] single-image inputs/
  );
});

test('rejects vision weights whose shape or dtype differs from the TVM artifact', () => {
  const argumentsWithWrongShape = makeArguments().map((argument) =>
    argument.sourceKind === 'PARAMETER' ? {...argument, shape: [2]} : argument
  );
  assert.throws(
    () =>
      createQwen35VisionFeatureRunner({
        artifact: artifact(argumentsWithWrongShape),
        runtime: createRuntime(new MockTensor([1], 'float32')).runtime,
        parameterTensors: new Map([['p_visual_test_weight', new MockTensor([1], 'float32')]])
      }),
    /loaded vision parameter p_visual_test_weight does not match the dynamic vision graph shape/
  );

  const argumentsWithWrongDtype = makeArguments().map((argument) =>
    argument.sourceKind === 'PARAMETER' ? {...argument, dtype: 'bfloat16'} : argument
  );
  assert.throws(
    () =>
      createQwen35VisionFeatureRunner({
        artifact: artifact(argumentsWithWrongDtype),
        runtime: createRuntime(new MockTensor([1], 'float32')).runtime,
        parameterTensors: new Map([['p_visual_test_weight', new MockTensor([1], 'float32')]])
      }),
    /invalid shape or dtype metadata/
  );
});

test('stops before the next image when cancellation arrives during GPU readback', async () => {
  const input = await graphInput();
  let signalSyncStarted: () => void = () => {};
  let finishSync: () => void = () => {};
  const syncStarted = new Promise<void>((resolve) => (signalSyncStarted = resolve));
  const syncGate = new Promise<void>((resolve) => (finishSync = resolve));
  const mock = createRuntime(new MockTensor([1], 'float32'), {
    async sync() {
      signalSyncStarted();
      await syncGate;
    }
  });
  const runner = createQwen35VisionFeatureRunner({
    artifact: artifact(),
    runtime: mock.runtime,
    parameterTensors: mock.parameters
  });
  const controller = new AbortController();
  const pending = runner.run(input, controller.signal);

  await syncStarted;
  controller.abort(new Error('stop after the active image'));
  finishSync();

  await assert.rejects(pending, /stop after the active image/);
  assert.equal(mock.calls.length, 1);
});

test('preserves the TVM failure when scope cleanup also fails', async () => {
  const input = await graphInput();
  const mainFailure = new Error('TVM main failed');
  const mock = createRuntime(new MockTensor([1], 'float32'), {
    mainFailure,
    endScopeFailure: new Error('scope cleanup failed')
  });
  const runner = createQwen35VisionFeatureRunner({
    artifact: artifact(),
    runtime: mock.runtime,
    parameterTensors: mock.parameters
  });

  await assert.rejects(runner.run(input), (error) => error === mainFailure);
  assert.equal(mock.calls.length, 1);
});

test('reports a scope cleanup failure after successful inference', async () => {
  const input = await graphInput();
  const cleanupFailure = new Error('scope cleanup failed');
  const mock = createRuntime(new MockTensor([1], 'float32'), {endScopeFailure: cleanupFailure});
  const runner = createQwen35VisionFeatureRunner({
    artifact: artifact(),
    runtime: mock.runtime,
    parameterTensors: mock.parameters
  });

  await assert.rejects(runner.run(input), (error) => error === cleanupFailure);
  assert.equal(mock.calls.length, 1);
});
