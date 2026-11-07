// Function: main_kernel
//----------------------------------------
// Function: main_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read> bias_ptr : array<f16>;
@group(0) @binding(1) var<storage, read> k_ptr : array<f16>;
@group(0) @binding(2) var<storage, read_write> out_ptr : array<f16>;
@group(0) @binding(3) var<storage, read> q_ptr : array<f16>;
@group(0) @binding(4) var<storage, read> v_ptr : array<f16>;

struct PODArgs {
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

var<workgroup> partial : array<f32, 256>;
var<workgroup> scores : array<f32, 256>;
@compute @workgroup_size(128, 2, 1)
fn main_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var row : array<i32, 1>;
  var query_row : array<i32, 1>;
  var state : array<f32, 3>;
  var dot : array<f32, 1>;
  var qk : array<f32, 1>;
  var tile_max : array<f32, 1>;
  var next_max : array<f32, 1>;
  var alpha : array<f32, 1>;
  var weight_sum : array<f32, 1>;
  var value_sum : array<f32, 1>;
  var key_in_tile : array<i32, 1>;
  var reduce_lane : array<i32, 1>;
  var key : array<i32, 1>;
  var dim : array<i32, 1>;
  var weight : array<f32, 1>;
  row[0i] = ((v__1 * 2i) + i32(threadIdx.y));
  query_row[0i] = min(row[0i], (podArgs.seq_len - 1i));
  state[0i] = -3.402823e+38f;
  state[1i] = 0.000000e+00f;
  state[2i] = 0.000000e+00f;
  let cse_v4 : i32 = ((i32(blockIdx.y) * podArgs.seq_len) * 128i);
  for (var tile : i32 = 0i; tile < ((podArgs.seq_len + 127i)>>7u); tile++) {
    key_in_tile[0i] = i32(threadIdx.x);
    reduce_lane[0i] = 0i;
    let cse_v2 : i32 = (tile * 128i);
    key[0i] = ((tile * 128i) + key_in_tile[0i]);
    dot[0i] = 0.000000e+00f;
    for (var chunk : i32 = 0i; chunk < 128i; chunk++) {
      dim[0i] = ((reduce_lane[0i] * 128i) + chunk);
      if (key[0i] < podArgs.seq_len) {
        dot[0i] = fma(f32(q_ptr[(((query_row[0i] * 128i) + ((i32(blockIdx.y) * podArgs.seq_len) * 128i)) + dim[0i])]), f32(k_ptr[(((key[0i] * 128i) + ((i32(blockIdx.y) * podArgs.seq_len) * 128i)) + dim[0i])]), dot[0i]);
      }
    }
    let cse_v3 : i32 = (i32(threadIdx.y) * 128i);
    partial[(((i32(threadIdx.y) * 128i) + key_in_tile[0i]) + reduce_lane[0i])] = dot[0i];
    workgroupBarrier();
    qk[0i] = 0.000000e+00f;
    let cse_v5 : i32 = ((i32(threadIdx.y) * 128i) + i32(threadIdx.x));
    qk[0i] = (qk[0i] + partial[((i32(threadIdx.y) * 128i) + i32(threadIdx.x))]);
    qk[0i] = f32(f16(qk[0i]));
    qk[0i] = f32((f16(qk[0i]) * 8.837891e-02h));
    key[0i] = ((tile * 128i) + i32(threadIdx.x));
    if (key[0i] < podArgs.seq_len) {
      qk[0i] = f32((f16(qk[0i]) + bias_ptr[((query_row[0i] * podArgs.seq_len) + key[0i])]));
      scores[((i32(threadIdx.y) * 128i) + i32(threadIdx.x))] = qk[0i];
    } else {
      scores[((i32(threadIdx.y) * 128i) + i32(threadIdx.x))] = -3.402823e+38f;
    }
    workgroupBarrier();
    tile_max[0i] = -3.402823e+38f;
    for (var key_lane : i32 = 0i; key_lane < 128i; key_lane++) {
      key[0i] = ((tile * 128i) + key_lane);
      if (key[0i] < podArgs.seq_len) {
        tile_max[0i] = max(tile_max[0i], scores[((i32(threadIdx.y) * 128i) + key_lane)]);
      }
    }
    next_max[0i] = max(state[0i], tile_max[0i]);
    if (state[1i] == 0.000000e+00f) {
      alpha[0i] = 0.000000e+00f;
    } else {
      alpha[0i] = exp((state[0i] - next_max[0i]));
    }
    weight_sum[0i] = 0.000000e+00f;
    value_sum[0i] = 0.000000e+00f;
    for (var key_lane_1 : i32 = 0i; key_lane_1 < 128i; key_lane_1++) {
      key[0i] = ((tile * 128i) + key_lane_1);
      if (key[0i] < podArgs.seq_len) {
        weight[0i] = exp((scores[((i32(threadIdx.y) * 128i) + key_lane_1)] - next_max[0i]));
        weight_sum[0i] = (weight_sum[0i] + weight[0i]);
        value_sum[0i] = fma(weight[0i], f32(v_ptr[(((key[0i] * 128i) + ((i32(blockIdx.y) * podArgs.seq_len) * 128i)) + i32(threadIdx.x))]), value_sum[0i]);
      }
    }
    state[2i] = fma(state[2i], alpha[0i], value_sum[0i]);
    state[1i] = fma(state[1i], alpha[0i], weight_sum[0i]);
    state[0i] = next_max[0i];
  }
  if (row[0i] < podArgs.seq_len) {
    out_ptr[(((row[0i] * 128i) + ((i32(blockIdx.y) * podArgs.seq_len) * 128i)) + i32(threadIdx.x))] = f16((state[2i] / state[1i]));
  }
}


