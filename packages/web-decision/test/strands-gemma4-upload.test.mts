import assert from 'node:assert/strict';
import test from 'node:test';
import {uploadGemma4Shard} from '../src/strands/gemma4-upload.mjs';

type UploadOptions = Parameters<typeof uploadGemma4Shard>[0];
function harness(overrides: Partial<UploadOptions> = {}) {
  const allocations: {shape: number[]; byteOffset: number; getDataPtr(): number}[] = [],
    writes: {pointer: unknown; offset: number; bytes: number[]}[] = [],
    syncs: boolean[] = [];
  const context = {
    copyRawBytesToBuffer(data: Uint8Array, pointer: unknown, offset: number, length: number) {
      assert.equal(offset % 4, 0);
      assert.equal(length % 4, 0);
      assert.equal(data.length, length);
      writes.push({pointer, offset, bytes: [...data]});
    }
  };
  const options: UploadOptions = {
    bytes: Uint8Array.from([0xaa, 0xbb, 0xcc, 0xdd, 1, 2, 3, 4, 5, 6, 7, 8]),
    records: [
      {name: 'param_2', shape: [], dtype: 'bfloat16', byteOffset: 0, nbytes: 2},
      {name: 'param_0', shape: [1], dtype: 'bfloat16', byteOffset: 2, nbytes: 2},
      {name: 'param_1', shape: [3], dtype: 'bfloat16', byteOffset: 4, nbytes: 6}
    ],
    expectedParameters: [
      {index: 0, shape: [1], dtype: 'bfloat16'},
      {index: 1, shape: [3], dtype: 'bfloat16'},
      {index: 2, shape: [], dtype: 'bfloat16'}
    ],
    parameterCount: 3,
    allocate: async ({shape, nbytes}) => {
      const pointer = allocations.length;
      const tensor = {shape, byteOffset: 0, getDataPtr: () => pointer};
      allocations.push(tensor);
      return {tensor, capacityBytes: Math.ceil(nbytes / 4) * 4};
    },
    context,
    sync: async () => {
      syncs.push(true);
    },
    maxChunkBytes: 4,
    ...overrides
  };
  return {options, allocations, writes, syncs};
}

test('copies adjacent BF16 tails separately and preserves graph indices', async () => {
  const {options, allocations, writes, syncs} = harness();
  const result = await uploadGemma4Shard(options);
  assert.equal(result.get(2), allocations[0]);
  assert.equal(result.get(0), allocations[1]);
  assert.equal(result.get(1), allocations[2]);
  assert.deepEqual(writes, [
    {pointer: 0, offset: 0, bytes: [0xaa, 0xbb, 0, 0]},
    {pointer: 1, offset: 0, bytes: [0xcc, 0xdd, 0, 0]},
    {pointer: 2, offset: 0, bytes: [1, 2, 3, 4]},
    {pointer: 2, offset: 4, bytes: [5, 6, 0, 0]}
  ]);
  assert.equal(syncs.length, 1);
});

test('copies an unaligned source offset for float32 and bounds chunk sizes', async () => {
  const {options, writes} = harness({
    records: [{name: 'param_1', shape: [2], dtype: 'float32', byteOffset: 2, nbytes: 8}],
    expectedParameters: [{index: 1, shape: [2], dtype: 'float32'}]
  });
  await uploadGemma4Shard(options);
  assert.deepEqual(
    writes.map((write) => write.bytes),
    [
      [0xcc, 0xdd, 1, 2],
      [3, 4, 5, 6]
    ]
  );
});

test('rejects invalid or incomplete metadata before allocating', async () => {
  const expected = (
    index: number,
    shape: number[] = [],
    dtype: UploadOptions['expectedParameters'][number]['dtype'] = 'bfloat16'
  ) => ({index, shape, dtype});
  const cases: Partial<UploadOptions>[] = [
    {expectedParameters: [expected(0), expected(1)]},
    {expectedParameters: [expected(0), expected(1), expected(1)]},
    {
      records: [
        {name: 'param_0', shape: [], dtype: 'bfloat16', byteOffset: 0, nbytes: 2},
        {name: 'param_0', shape: [], dtype: 'bfloat16', byteOffset: 2, nbytes: 2}
      ],
      expectedParameters: [expected(0), expected(1)]
    },
    {
      records: [{name: 'param_0', shape: [], dtype: 'bfloat16', byteOffset: 0, nbytes: 2}],
      expectedParameters: [expected(3)]
    },
    {
      records: [{name: 'param_0', shape: [2], dtype: 'float32', byteOffset: 10, nbytes: 8}],
      expectedParameters: [expected(0, [2], 'float32')]
    },
    {
      records: [{name: 'param_0', shape: [1], dtype: 'float16', byteOffset: 0, nbytes: 2}],
      expectedParameters: [expected(0, [1])]
    },
    {
      records: [{name: 'param_0', shape: [1], dtype: 'bfloat16', byteOffset: 0, nbytes: 4}],
      expectedParameters: [expected(0, [1])]
    },
    {
      records: [{name: 'param_0', shape: [0], dtype: 'bfloat16', byteOffset: 0, nbytes: 0}],
      expectedParameters: [expected(0, [1])]
    },
    {
      records: [{name: 'param_0', shape: [Number.MAX_SAFE_INTEGER], dtype: 'float32', byteOffset: 0, nbytes: 4}],
      expectedParameters: [expected(0, [1])]
    },
    {
      records: [{name: 'param_0', shape: [], dtype: 'bfloat16', byteOffset: -1, nbytes: 2}],
      expectedParameters: [expected(0)]
    },
    {
      records: [
        {name: 'param_0', shape: [2], dtype: 'bfloat16', byteOffset: 0, nbytes: 4},
        {name: 'param_1', shape: [1], dtype: 'bfloat16', byteOffset: 2, nbytes: 2}
      ],
      expectedParameters: [expected(0, [2]), expected(1, [1])]
    },
    {
      records: [{name: 'param_0', shape: [1], dtype: 'float32', byteOffset: 0, nbytes: 4}],
      expectedParameters: [expected(0, [2])]
    },
    {
      records: [{name: 'param_0', shape: [2], dtype: 'bfloat16', byteOffset: 0, nbytes: 4}],
      expectedParameters: [expected(0, [1], 'float32')]
    },
    {maxChunkBytes: 3}
  ];
  for (const override of cases) {
    const {options, allocations, writes} = harness(override);
    await assert.rejects(uploadGemma4Shard(options));
    assert.equal(allocations.length, 0);
    assert.equal(writes.length, 0);
  }
});

test('rejects invalid destination capacity, offset, shape and alias', async () => {
  for (const allocate of [
    async () => ({tensor: {shape: [], byteOffset: 2, getDataPtr: () => 0}, capacityBytes: 4}),
    async () => ({tensor: {shape: [], byteOffset: 0, getDataPtr: () => 0}, capacityBytes: 2}),
    async () => ({tensor: {shape: [1], byteOffset: 0, getDataPtr: () => 0}, capacityBytes: 4})
  ]) {
    const {options, writes} = harness({allocate});
    await assert.rejects(uploadGemma4Shard(options));
    assert.equal(writes.length, 0);
  }
  const {options, writes} = harness({
    allocate: async ({shape}) => ({tensor: {shape, byteOffset: 0, getDataPtr: () => 9}, capacityBytes: 8})
  });
  await assert.rejects(uploadGemma4Shard(options));
  assert.equal(writes.length, 1);
});

test('does not return weights when writes or sync fail', async () => {
  const {options, syncs} = harness({
    context: {
      copyRawBytesToBuffer() {
        throw Error('write failed');
      }
    }
  });
  await assert.rejects(uploadGemma4Shard(options), /write failed/);
  assert.equal(syncs.length, 1);
  let attempts = 0;
  const failedSync = harness({
    sync: async () => {
      if (++attempts === 1) throw Error('sync failed');
    }
  });
  await assert.rejects(uploadGemma4Shard(failedSync.options), /sync failed/);
  assert.equal(attempts, 2);
});

test('aborts before allocation and drains a submitted write before rejecting', async () => {
  const controller = new AbortController();
  controller.abort();
  const before = harness({signal: controller.signal});
  await assert.rejects(uploadGemma4Shard(before.options), {name: 'AbortError'});
  assert.equal(before.allocations.length, 0);
  const inFlight = new AbortController();
  let drained = 0;
  const writes: {pointer: unknown; offset: number; length: number; bytes: number[]}[] = [];
  const during = harness({
    signal: inFlight.signal,
    context: {
      copyRawBytesToBuffer(data: Uint8Array, pointer: unknown, offset: number, length: number) {
        writes.push({pointer, offset, length, bytes: [...data]});
        inFlight.abort();
      }
    },
    sync: async () => {
      drained++;
    }
  });
  await assert.rejects(uploadGemma4Shard(during.options), {name: 'AbortError'});
  assert.equal(drained, 1);
  assert.equal(writes.length, 1);
  const afterAllocation = new AbortController();
  const pending = harness({
    signal: afterAllocation.signal,
    allocate: async ({shape}) => {
      afterAllocation.abort();
      return {tensor: {shape, byteOffset: 0, getDataPtr: () => 0}, capacityBytes: 4};
    }
  });
  await assert.rejects(uploadGemma4Shard(pending.options), {name: 'AbortError'});
  assert.equal(pending.writes.length, 0);
});

test('preserves both upload and drain failures', async () => {
  const writeError = Error('write failed');
  const drainError = Error('drain failed');
  let attempts = 0;
  const {options} = harness({
    context: {
      copyRawBytesToBuffer() {
        throw writeError;
      }
    },
    sync: async () => {
      attempts++;
      throw drainError;
    }
  });
  await assert.rejects(uploadGemma4Shard(options), (error) => {
    assert.ok(error instanceof AggregateError);
    assert.deepEqual(error.errors, [writeError, drainError]);
    assert.equal(error.cause, drainError);
    return true;
  });
  assert.equal(attempts, 1);
});

test('drains previous writes if allocation or check fails', async () => {
  for (const kind of ['allocate', 'check']) {
    const failure = Error(`${kind} failed`);
    let allocations = 0,
      checks = 0,
      syncs = 0;
    const {options, writes} = harness({
      allocate: async ({shape}) => {
        if (++allocations === 2 && kind === 'allocate') throw failure;
        const pointer = allocations;
        return {tensor: {shape, byteOffset: 0, getDataPtr: () => pointer}, capacityBytes: 8};
      },
      check: () => {
        if (++checks === 3 && kind === 'check') throw failure;
      },
      sync: async () => {
        syncs++;
      }
    });
    await assert.rejects(uploadGemma4Shard(options), (error) => error === failure);
    assert.equal(writes.length, 1);
    assert.equal(syncs, 1);
  }
});
