declare const GPUBufferUsage: Readonly<{
  STORAGE: number;
  COPY_DST: number;
  COPY_SRC: number;
  UNIFORM: number;
  MAP_READ: number;
}>;
declare const GPUMapMode: Readonly<{READ: number}>;

const shaderSource =
  '// Function: attention_grid_kernel\n//----------------------------------------\n// Function: attention_grid_kernel\n//----------------------------------------\nenable f16;\n\n@group(0) @binding(0) var<storage, read> key_ptr : array<f16>;\n@group(0) @binding(1) var<storage, read> length_i32_ptr : array<i32>;\n@group(0) @binding(2) var<storage, read_write> output_ptr : array<f32>;\n@group(0) @binding(3) var<storage, read> query_ptr : array<f16>;\n@group(0) @binding(4) var<storage, read> segments_ptr : array<i32>;\n@group(0) @binding(5) var<storage, read> valid_ptr : array<i32>;\n@group(0) @binding(6) var<storage, read> value_ptr : array<f32>;\n\nstruct PODArgs {\n  sequence_length: i32,\n  packGridDimX: u32\n}\n@group(0) @binding(7) var<uniform> podArgs : PODArgs;\n\nvar<workgroup> state : array<f32, 384>;\nvar<workgroup> partial : array<f32, 128>;\nvar<workgroup> scores : array<f32, 64>;\n@compute @workgroup_size(128, 1, 1)\nfn attention_grid_kernel(\n  @builtin(workgroup_id) blockIdx : vec3<u32>,\n  @builtin(num_workgroups) gridDim : vec3<u32>,\n  @builtin(local_invocation_id) threadIdx : vec3<u32>\n) {\n  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }\n  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);\n  var query_index : array<i32, 1>;\n  var head : array<i32, 1>;\n  var query_segment : array<i32, 1>;\n  var tile_count : array<i32, 1>;\n  var key_base : array<i32, 1>;\n  var key_slot : array<i32, 1>;\n  var feature_lane : array<i32, 1>;\n  var partial_sum : array<f32, 1>;\n  var channel : array<i32, 1>;\n  var feature_lane_1 : array<i32, 1>;\n  var offset : array<i32, 1>;\n  var key_slot_1 : array<i32, 1>;\n  var feature_lane_2 : array<i32, 1>;\n  var in_range : array<bool, 1>;\n  var safe_key_index : array<i32, 1>;\n  var visible : array<bool, 1>;\n  var bias : array<f32, 1>;\n  var tile_maximum : array<f32, 1>;\n  var previous_maximum : array<f32, 1>;\n  var tile_denominator : array<f32, 1>;\n  var key_base_1 : array<i32, 1>;\n  var key_slot_2 : array<i32, 1>;\n  var feature_lane_3 : array<i32, 1>;\n  var partial_sum_1 : array<f32, 1>;\n  var channel_1 : array<i32, 1>;\n  var feature_lane_4 : array<i32, 1>;\n  var offset_1 : array<i32, 1>;\n  var key_slot_3 : array<i32, 1>;\n  var feature_lane_5 : array<i32, 1>;\n  var raw_key_index : array<i32, 1>;\n  var in_range_1 : array<bool, 1>;\n  var safe_key_index_1 : array<i32, 1>;\n  var visible_1 : array<bool, 1>;\n  var bias_1 : array<f32, 1>;\n  var accumulator : array<f32, 1>;\n  var maximum : array<f32, 1>;\n  var denominator : array<f32, 1>;\n  var raw_key_index_1 : array<i32, 1>;\n  var probability : array<f32, 1>;\n  var probability_f16 : array<f16, 1>;\n  var probability_f32 : array<f32, 1>;\n  query_index[0i] = v__1;\n  head[0i] = i32(blockIdx.y);\n  query_segment[0i] = segments_ptr[query_index[0i]];\n  let cse_v2 : i32 = (i32(threadIdx.x) * 3i);\n  state[(i32(threadIdx.x) * 3i)] = -3.402823e+38f;\n  let cse_v6 : i32 = ((i32(threadIdx.x) * 3i) + 1i);\n  state[((i32(threadIdx.x) * 3i) + 1i)] = 0.000000e+00f;\n  let cse_v7 : i32 = ((i32(threadIdx.x) * 3i) + 2i);\n  state[((i32(threadIdx.x) * 3i) + 2i)] = 0.000000e+00f;\n  workgroupBarrier();\n  tile_count[0i] = ((length_i32_ptr[0i] + 63i)>>6u);\n  let cse_v3 : i32 = (i32(threadIdx.x)>>1u);\n  let cse_v4 : i32 = (i32(threadIdx.x) & 1i);\n  for (var tile_loop : i32 = 0i; tile_loop < tile_count[0i]; tile_loop++) {\n    key_base[0i] = (tile_loop * 64i);\n    key_slot[0i] = (i32(threadIdx.x)>>1u);\n    feature_lane[0i] = (i32(threadIdx.x) & 1i);\n    key_slot[0i] = (key_base[0i] + key_slot[0i]);\n    key_slot[0i] = min(key_slot[0i], (length_i32_ptr[0i] - 1i));\n    partial_sum[0i] = 0.000000e+00f;\n    for (var feature_group : i32 = 0i; feature_group < 64i; feature_group++) {\n      channel[0i] = ((feature_group * 2i) + feature_lane[0i]);\n      partial_sum[0i] = fma(f32(query_ptr[((((head[0i] * podArgs.sequence_length) * 128i) + (query_index[0i] * 128i)) + channel[0i])]), f32(key_ptr[((((head[0i] * 128i) + channel[0i]) * podArgs.sequence_length) + key_slot[0i])]), partial_sum[0i]);\n    }\n    partial[i32(threadIdx.x)] = partial_sum[0i];\n    workgroupBarrier();\n    feature_lane_1[0i] = (i32(threadIdx.x) & 1i);\n    offset[0i] = 1i;\n    if (feature_lane_1[0i] < offset[0i]) {\n      partial[i32(threadIdx.x)] = (partial[i32(threadIdx.x)] + partial[(i32(threadIdx.x) + offset[0i])]);\n    }\n    workgroupBarrier();\n    key_slot_1[0i] = (i32(threadIdx.x)>>1u);\n    feature_lane_2[0i] = (i32(threadIdx.x) & 1i);\n    key_base[0i] = (key_base[0i] + key_slot_1[0i]);\n    in_range[0i] = bool((key_base[0i] < length_i32_ptr[0i]));\n    safe_key_index[0i] = min(key_base[0i], (length_i32_ptr[0i] - 1i));\n    visible[0i] = bool((((bool(in_range[0i]) & (key_base[0i] <= query_index[0i])) & (valid_ptr[safe_key_index[0i]] != 0i)) & ((segments_ptr[safe_key_index[0i]] == query_segment[0i]) | (segments_ptr[safe_key_index[0i]] == 0i))));\n    var condval : f32;\n    if (bool(visible[0i])) {\n      condval = 0.000000e+00f;\n} else {\n      condval = -6.550400e+04f;\n}\n    bias[0i] = condval;\n    bias[0i] = fma(partial[i32(threadIdx.x)], 8.838835e-02f, bias[0i]);\n    if (feature_lane_2[0i] == 0i) {\n      var condval_1 : f32;\n      if (bool(in_range[0i])) {\n        condval_1 = bias[0i];\n} else {\n        condval_1 = -3.402823e+38f;\n}\n      scores[key_slot_1[0i]] = condval_1;\n    }\n    workgroupBarrier();\n    tile_maximum[0i] = -3.402823e+38f;\n    for (var slot : i32 = 0i; slot < 64i; slot++) {\n      tile_maximum[0i] = max(tile_maximum[0i], scores[slot]);\n    }\n    previous_maximum[0i] = state[(i32(threadIdx.x) * 3i)];\n    tile_maximum[0i] = max(previous_maximum[0i], tile_maximum[0i]);\n    tile_denominator[0i] = 0.000000e+00f;\n    for (var slot_1 : i32 = 0i; slot_1 < 64i; slot_1++) {\n      tile_denominator[0i] = (tile_denominator[0i] + exp((scores[slot_1] - tile_maximum[0i])));\n    }\n    previous_maximum[0i] = exp((previous_maximum[0i] - tile_maximum[0i]));\n    workgroupBarrier();\n    state[(i32(threadIdx.x) * 3i)] = tile_maximum[0i];\n    workgroupBarrier();\n    state[((i32(threadIdx.x) * 3i) + 1i)] = fma(state[((i32(threadIdx.x) * 3i) + 1i)], previous_maximum[0i], tile_denominator[0i]);\n    workgroupBarrier();\n  }\n  let cse_v5 : i32 = i32(threadIdx.x);\n  for (var tile_loop_1 : i32 = 0i; tile_loop_1 < tile_count[0i]; tile_loop_1++) {\n    key_base_1[0i] = (tile_loop_1 * 64i);\n    key_slot_2[0i] = (i32(threadIdx.x)>>1u);\n    feature_lane_3[0i] = (i32(threadIdx.x) & 1i);\n    key_slot_2[0i] = (key_base_1[0i] + key_slot_2[0i]);\n    key_slot_2[0i] = min(key_slot_2[0i], (length_i32_ptr[0i] - 1i));\n    partial_sum_1[0i] = 0.000000e+00f;\n    for (var feature_group_1 : i32 = 0i; feature_group_1 < 64i; feature_group_1++) {\n      channel_1[0i] = ((feature_group_1 * 2i) + feature_lane_3[0i]);\n      partial_sum_1[0i] = fma(f32(query_ptr[((((head[0i] * podArgs.sequence_length) * 128i) + (query_index[0i] * 128i)) + channel_1[0i])]), f32(key_ptr[((((head[0i] * 128i) + channel_1[0i]) * podArgs.sequence_length) + key_slot_2[0i])]), partial_sum_1[0i]);\n    }\n    partial[i32(threadIdx.x)] = partial_sum_1[0i];\n    workgroupBarrier();\n    feature_lane_4[0i] = (i32(threadIdx.x) & 1i);\n    offset_1[0i] = 1i;\n    if (feature_lane_4[0i] < offset_1[0i]) {\n      partial[i32(threadIdx.x)] = (partial[i32(threadIdx.x)] + partial[(i32(threadIdx.x) + offset_1[0i])]);\n    }\n    workgroupBarrier();\n    key_slot_3[0i] = (i32(threadIdx.x)>>1u);\n    feature_lane_5[0i] = (i32(threadIdx.x) & 1i);\n    raw_key_index[0i] = (key_base_1[0i] + key_slot_3[0i]);\n    in_range_1[0i] = bool((raw_key_index[0i] < length_i32_ptr[0i]));\n    safe_key_index_1[0i] = min(raw_key_index[0i], (length_i32_ptr[0i] - 1i));\n    visible_1[0i] = bool((((bool(in_range_1[0i]) & (raw_key_index[0i] <= query_index[0i])) & (valid_ptr[safe_key_index_1[0i]] != 0i)) & ((segments_ptr[safe_key_index_1[0i]] == query_segment[0i]) | (segments_ptr[safe_key_index_1[0i]] == 0i))));\n    var condval_2 : f32;\n    if (bool(visible_1[0i])) {\n      condval_2 = 0.000000e+00f;\n} else {\n      condval_2 = -6.550400e+04f;\n}\n    bias_1[0i] = condval_2;\n    bias_1[0i] = fma(partial[i32(threadIdx.x)], 8.838835e-02f, bias_1[0i]);\n    if (feature_lane_5[0i] == 0i) {\n      var condval_3 : f32;\n      if (bool(in_range_1[0i])) {\n        condval_3 = bias_1[0i];\n} else {\n        condval_3 = -3.402823e+38f;\n}\n      scores[key_slot_3[0i]] = condval_3;\n    }\n    workgroupBarrier();\n    accumulator[0i] = state[((i32(threadIdx.x) * 3i) + 2i)];\n    maximum[0i] = state[(i32(threadIdx.x) * 3i)];\n    denominator[0i] = state[((i32(threadIdx.x) * 3i) + 1i)];\n    for (var slot_2 : i32 = 0i; slot_2 < 64i; slot_2++) {\n      raw_key_index_1[0i] = (key_base_1[0i] + slot_2);\n      if (raw_key_index_1[0i] < length_i32_ptr[0i]) {\n        probability[0i] = (exp((scores[slot_2] - maximum[0i])) / denominator[0i]);\n        probability_f16[0i] = f16(probability[0i]);\n        probability_f32[0i] = f32(probability_f16[0i]);\n        accumulator[0i] = fma(probability_f32[0i], value_ptr[((((head[0i] * podArgs.sequence_length) * 128i) + (raw_key_index_1[0i] * 128i)) + i32(threadIdx.x))], accumulator[0i]);\n      }\n    }\n    workgroupBarrier();\n    state[((i32(threadIdx.x) * 3i) + 2i)] = accumulator[0i];\n    workgroupBarrier();\n  }\n  output_ptr[((((head[0i] * podArgs.sequence_length) * 128i) + (query_index[0i] * 128i)) + i32(threadIdx.x))] = state[((i32(threadIdx.x) * 3i) + 2i)];\n}\n\n\n';
type AdapterInformation = Readonly<{vendor: string; architecture: string}>;
type AttentionInputs = Readonly<{
  query: Uint16Array;
  key: Uint16Array;
  value: Float32Array;
  valid: Int32Array;
  segments: Int32Array;
}>;
type DeviceBuffers = Readonly<{
  query: GPUBuffer;
  key: GPUBuffer;
  value: GPUBuffer;
  valid: GPUBuffer;
  segments: GPUBuffer;
  sequenceLength: GPUBuffer;
  output: GPUBuffer;
  arguments: GPUBuffer;
}>;
type CaseDiagnostics = Readonly<{
  sequenceLength: number;
  allInvalid: boolean;
  adapter: AdapterInformation;
  maxAbs: number;
  rms: number;
  elements: number;
}>;
type SoftmaxState = Readonly<{maximum: number; denominator: number}>;

const HEADS = 16;
const KV_HEADS = 8;
const DIMENSION = 128;
const ATTENTION_SCALE = Math.fround(0.088388346135616302);
const TILE_WIDTH = 64;

function halfToFloat(half: number): number {
  const sign = half & 0x8000 ? -1 : 1;
  const exponent = (half >> 10) & 31;
  const fraction = half & 1023;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

function floatToHalf(value: number): number {
  const float = new Float32Array([value]);
  const bits = new Uint32Array(float.buffer)[0]!;
  const sign = (bits >>> 16) & 0x8000;
  let exponent = ((bits >>> 23) & 255) - 127 + 15;
  let mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | ((mantissa + 0x1000 + ((mantissa >> 13) & 1)) >> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;
  mantissa += 0x1000 + ((mantissa >> 13) & 1);
  if (mantissa & 0x800000) {
    mantissa = 0;
    exponent += 1;
  }
  if (exponent >= 31) return sign | 0x7c00;
  return sign | (exponent << 10) | (mantissa >> 13);
}

function randomValues(length: number, seed: number, scale: number): Float32Array {
  let state = seed >>> 0;
  const values = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    values[index] = (((state >>> 0) / 4294967296) * 2 - 1) * scale;
  }
  return values;
}

function fma32(left: number, right: number, addend: number): number {
  return Math.fround(left * right + addend);
}

function makeBuffer(device: GPUDevice, data: ArrayBufferView, usage: GPUBufferUsageFlags): GPUBuffer {
  const bytes = data.byteLength;
  const buffer = device.createBuffer({size: Math.max(4, (bytes + 3) & ~3), usage, mappedAtCreation: true});
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}

function createQuery(sequenceLength: number, seed: number): Uint16Array {
  const values = randomValues(HEADS * sequenceLength * DIMENSION, seed, 0.12);
  return Uint16Array.from(values, floatToHalf);
}

function createKeyAndValue(sequenceLength: number, seed: number): Pick<AttentionInputs, 'key' | 'value'> {
  const keyValues = randomValues(KV_HEADS * DIMENSION * sequenceLength, seed + 31, 0.12);
  const valueValues = randomValues(KV_HEADS * sequenceLength * DIMENSION, seed + 67, 0.5);
  const key = new Uint16Array(HEADS * DIMENSION * sequenceLength);
  const value = new Float32Array(HEADS * sequenceLength * DIMENSION);

  for (let head = 0; head < HEADS; head += 1) {
    const kvHead = Math.floor(head / 2);
    copyKeyHead(key, keyValues, head, kvHead, sequenceLength);
    copyValueHead(value, valueValues, head, kvHead, sequenceLength);
  }
  return {key, value};
}

function copyKeyHead(
  key: Uint16Array,
  keyValues: Float32Array,
  head: number,
  kvHead: number,
  sequenceLength: number
): void {
  for (let feature = 0; feature < DIMENSION; feature += 1) {
    for (let position = 0; position < sequenceLength; position += 1) {
      const source = (kvHead * DIMENSION + feature) * sequenceLength + position;
      const target = (head * DIMENSION + feature) * sequenceLength + position;
      key[target] = floatToHalf(keyValues[source]!);
    }
  }
}

function copyValueHead(
  value: Float32Array,
  valueValues: Float32Array,
  head: number,
  kvHead: number,
  sequenceLength: number
): void {
  for (let position = 0; position < sequenceLength; position += 1) {
    for (let feature = 0; feature < DIMENSION; feature += 1) {
      value[(head * sequenceLength + position) * DIMENSION + feature] =
        valueValues[(kvHead * sequenceLength + position) * DIMENSION + feature]!;
    }
  }
}

function createMask(sequenceLength: number, allInvalid: boolean): Pick<AttentionInputs, 'valid' | 'segments'> {
  const valid = new Int32Array(sequenceLength);
  const segments = new Int32Array(sequenceLength);
  for (let position = 0; position < sequenceLength; position += 1) {
    const firstBoundary = Math.ceil(sequenceLength / 3);
    const secondBoundary = Math.ceil((2 * sequenceLength) / 3);
    segments[position] = position < firstBoundary ? 0 : position < secondBoundary ? 1 : 2;
    valid[position] =
      allInvalid || position % 11 === 5 || (sequenceLength > 4 && position === sequenceLength - 1) ? 0 : 1;
  }
  return {valid, segments};
}

function createInputs(sequenceLength: number, allInvalid: boolean, seed: number): AttentionInputs {
  return {
    query: createQuery(sequenceLength, seed),
    ...createKeyAndValue(sequenceLength, seed),
    ...createMask(sequenceLength, allInvalid)
  };
}

function createDeviceBuffers(device: GPUDevice, inputs: AttentionInputs, sequenceLength: number): DeviceBuffers {
  const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const outputBytes = HEADS * sequenceLength * DIMENSION * Float32Array.BYTES_PER_ELEMENT;
  const pod = new ArrayBuffer(8);
  new DataView(pod).setInt32(0, sequenceLength, true);
  new DataView(pod).setUint32(4, sequenceLength, true);
  const argumentsBuffer = device.createBuffer({
    size: pod.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true
  });
  new Uint8Array(argumentsBuffer.getMappedRange()).set(new Uint8Array(pod));
  argumentsBuffer.unmap();

  return {
    query: makeBuffer(device, inputs.query, storageUsage),
    key: makeBuffer(device, inputs.key, storageUsage),
    value: makeBuffer(device, inputs.value, storageUsage),
    valid: makeBuffer(device, inputs.valid, storageUsage),
    segments: makeBuffer(device, inputs.segments, storageUsage),
    sequenceLength: makeBuffer(device, new Int32Array([sequenceLength]), storageUsage),
    output: device.createBuffer({size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC}),
    arguments: argumentsBuffer
  };
}

function destroyDeviceBuffers(buffers: DeviceBuffers): void {
  for (const buffer of Object.values(buffers)) buffer.destroy();
}

async function dispatchKernel(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  buffers: DeviceBuffers,
  sequenceLength: number
): Promise<Float32Array> {
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      {binding: 0, resource: {buffer: buffers.key}},
      {binding: 1, resource: {buffer: buffers.sequenceLength}},
      {binding: 2, resource: {buffer: buffers.output}},
      {binding: 3, resource: {buffer: buffers.query}},
      {binding: 4, resource: {buffer: buffers.segments}},
      {binding: 5, resource: {buffer: buffers.valid}},
      {binding: 6, resource: {buffer: buffers.value}},
      {binding: 7, resource: {buffer: buffers.arguments}}
    ]
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(sequenceLength, HEADS, 1);
  pass.end();
  const outputBytes = HEADS * sequenceLength * DIMENSION * Float32Array.BYTES_PER_ELEMENT;
  const readback = device.createBuffer({size: outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  try {
    encoder.copyBufferToBuffer(buffers.output, 0, readback, 0, outputBytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    return new Float32Array(readback.getMappedRange().slice(0));
  } finally {
    readback.destroy();
  }
}

function calculateScores(
  inputs: AttentionInputs,
  sequenceLength: number,
  head: number,
  queryIndex: number
): Float32Array {
  const scores = new Float32Array(sequenceLength);
  for (let keyIndex = 0; keyIndex < sequenceLength; keyIndex += 1) {
    let even = 0;
    let odd = 0;
    for (let group = 0; group < DIMENSION / 2; group += 1) {
      const evenFeature = group * 2;
      const oddFeature = evenFeature + 1;
      const queryBase = (head * sequenceLength + queryIndex) * DIMENSION;
      even = fma32(
        halfToFloat(inputs.query[queryBase + evenFeature]!),
        halfToFloat(inputs.key[(head * DIMENSION + evenFeature) * sequenceLength + keyIndex]!),
        even
      );
      odd = fma32(
        halfToFloat(inputs.query[queryBase + oddFeature]!),
        halfToFloat(inputs.key[(head * DIMENSION + oddFeature) * sequenceLength + keyIndex]!),
        odd
      );
    }
    const dot = Math.fround(even + odd);
    const sameSegment = inputs.segments[keyIndex] === inputs.segments[queryIndex] || inputs.segments[keyIndex] === 0;
    const visible = keyIndex <= queryIndex && inputs.valid[keyIndex] !== 0 && sameSegment;
    scores[keyIndex] = fma32(dot, ATTENTION_SCALE, visible ? 0 : -65504);
  }
  return scores;
}

function calculateSoftmax(scores: Float32Array, sequenceLength: number): SoftmaxState {
  const tiles = Math.ceil(sequenceLength / TILE_WIDTH);
  let maximum = Math.fround(-3.402823466e38);
  let denominator = 0;
  for (let tile = 0; tile < tiles; tile += 1) {
    const base = tile * TILE_WIDTH;
    let tileMaximum = Math.fround(-3.402823466e38);
    for (let slot = 0; slot < TILE_WIDTH; slot += 1) {
      const keyIndex = base + slot;
      const score = keyIndex < sequenceLength ? scores[keyIndex]! : Math.fround(-3.402823466e38);
      tileMaximum = Math.max(tileMaximum, score);
    }
    const nextMaximum = Math.max(maximum, tileMaximum);
    let tileDenominator = 0;
    for (let slot = 0; slot < TILE_WIDTH; slot += 1) {
      const keyIndex = base + slot;
      const score = keyIndex < sequenceLength ? scores[keyIndex]! : Math.fround(-3.402823466e38);
      tileDenominator = Math.fround(tileDenominator + Math.fround(Math.exp(Math.fround(score - nextMaximum))));
    }
    const previousScale = Math.fround(Math.exp(Math.fround(maximum - nextMaximum)));
    denominator = fma32(denominator, previousScale, tileDenominator);
    maximum = nextMaximum;
  }
  return {maximum, denominator};
}

function calculateOutputForQuery(
  inputs: AttentionInputs,
  scores: Float32Array,
  sequenceLength: number,
  head: number,
  queryIndex: number,
  softmax: SoftmaxState
): Float32Array {
  const output = new Float32Array(DIMENSION);
  const tiles = Math.ceil(sequenceLength / TILE_WIDTH);
  for (let feature = 0; feature < DIMENSION; feature += 1) {
    let accumulator = 0;
    for (let tile = 0; tile < tiles; tile += 1) {
      for (let slot = 0; slot < TILE_WIDTH; slot += 1) {
        const keyIndex = tile * TILE_WIDTH + slot;
        if (keyIndex < sequenceLength) {
          const probability = Math.fround(
            Math.fround(Math.exp(Math.fround(scores[keyIndex]! - softmax.maximum))) / softmax.denominator
          );
          const roundedProbability = halfToFloat(floatToHalf(probability));
          accumulator = fma32(
            roundedProbability,
            inputs.value[(head * sequenceLength + keyIndex) * DIMENSION + feature]!,
            accumulator
          );
        }
      }
    }
    output[feature] = accumulator;
  }
  return output;
}

function calculateExpected(inputs: AttentionInputs, sequenceLength: number): Float32Array {
  const expected = new Float32Array(HEADS * sequenceLength * DIMENSION);
  for (let head = 0; head < HEADS; head += 1) {
    for (let queryIndex = 0; queryIndex < sequenceLength; queryIndex += 1) {
      const scores = calculateScores(inputs, sequenceLength, head, queryIndex);
      const softmax = calculateSoftmax(scores, sequenceLength);
      const output = calculateOutputForQuery(inputs, scores, sequenceLength, head, queryIndex, softmax);
      const outputBase = (head * sequenceLength + queryIndex) * DIMENSION;
      for (let feature = 0; feature < DIMENSION; feature += 1) {
        expected[outputBase + feature] = output[feature]!;
      }
    }
  }
  return expected;
}

function compareOutputs(actual: Float32Array, expected: Float32Array): Readonly<{maxAbs: number; rms: number}> {
  let maxAbs = 0;
  let sumSquared = 0;
  for (let index = 0; index < actual.length; index += 1) {
    const delta = Math.abs(actual[index]! - expected[index]!);
    maxAbs = Math.max(maxAbs, delta);
    sumSquared += delta * delta;
  }
  return {maxAbs, rms: Math.sqrt(sumSquared / actual.length)};
}

async function runCase(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  adapter: AdapterInformation,
  sequenceLength: number,
  allInvalid: boolean,
  seed: number
): Promise<CaseDiagnostics> {
  const inputs = createInputs(sequenceLength, allInvalid, seed);
  const buffers = createDeviceBuffers(device, inputs, sequenceLength);
  try {
    const actual = await dispatchKernel(device, pipeline, buffers, sequenceLength);
    const expected = calculateExpected(inputs, sequenceLength);
    const errors = compareOutputs(actual, expected);
    return {...errors, sequenceLength, allInvalid, adapter, elements: actual.length};
  } finally {
    destroyDeviceBuffers(buffers);
  }
}

async function runSharedKernelCases() {
  const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
  if (!adapter) throw new Error('no adapter');
  if (!adapter.features.has('shader-f16')) throw new Error('shader-f16 unavailable');
  const information = {vendor: adapter.info?.vendor ?? '', architecture: adapter.info?.architecture ?? ''};
  const device = await adapter.requestDevice({requiredFeatures: ['shader-f16']});
  try {
    const shader = device.createShaderModule({code: shaderSource});
    const compilation = await shader.getCompilationInfo();
    const errors = compilation.messages.filter(({type}) => type === 'error');
    if (errors.length) throw new Error(errors.map(({message}) => message).join('\n'));
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {module: shader, entryPoint: 'attention_grid_kernel'}
    });
    const inputs = [
      {sequenceLength: 1, allInvalid: false},
      {sequenceLength: 7, allInvalid: false},
      {sequenceLength: 65, allInvalid: false},
      {sequenceLength: 129, allInvalid: false},
      {sequenceLength: 7, allInvalid: true}
    ] as const;
    const cases: CaseDiagnostics[] = [];
    for (const {sequenceLength, allInvalid} of inputs) {
      const seed = 0x91ab27 + sequenceLength + (allInvalid ? 999 : 0);
      cases.push(await runCase(device, pipeline, information, sequenceLength, allInvalid, seed));
    }
    return {
      adapter: information,
      feature: 'shader-f16',
      wgslBytes: new TextEncoder().encode(shaderSource).length,
      cases
    };
  } finally {
    device.destroy();
  }
}

Object.assign(window, {runSharedKernelCases});
