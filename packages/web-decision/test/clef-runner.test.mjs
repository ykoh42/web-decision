import test from 'node:test';
import assert from 'node:assert/strict';
import {ClefTvmRunner} from '../src/clef/runner.mjs';

const packed = {
  ids: [10, 11, 65537, 12],
  maxInputTokens: 512,
  questions: [
    {
      question_id: 'q0',
      question_type: 1,
      question_span: [0, 1],
      option_spans: [
        [1, 2],
        [2, 3]
      ],
      option_ids: ['0', '1'],
      options: ['a', 'b']
    }
  ]
};
const defer = () => {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return {promise, resolve};
};
function fixture({select, readback = Promise.resolve(), sync, lifecycle = {}, fn} = {}) {
  const events = [],
    ctx = {pendingGPUToCPUCopy: null, gpuBufferFromPtr: () => ({size: 2 ** 20}), copyRawBytesToBuffer() {}};
  const runtime = {
    lib: {webGPUContext: ctx},
    cpu: () => 'cpu',
    beginScope() {
      events.push('begin');
    },
    endScope() {
      events.push('end');
    },
    dispose() {
      events.push('dispose');
    },
    empty(shape, dtype, where) {
      return {
        shape,
        dtype,
        byteOffset: 0,
        getDataPtr: () => 0,
        copyFrom() {
          if (where === 'cpu') ctx.pendingGPUToCPUCopy = readback;
          return this;
        },
        toArray: () => Float32Array.of(1, 2)
      };
    }
  };
  const device = {
    async sync() {
      events.push('sync');
      if (sync) await sync();
      if (ctx.pendingGPUToCPUCopy) await ctx.pendingGPUToCPUCopy;
      ctx.pendingGPUToCPUCopy = null;
    }
  };
  const selector =
    select ||
    (async (ids) => {
      const unique = [...new Set(ids)];
      return {
        unique,
        local: Int32Array.from(ids, (id) => unique.indexOf(id)),
        rowBanks: Int32Array.from(unique, (id) => Math.floor(id / 65536)),
        packed: new Uint32Array(unique.length * 512),
        scales: new Uint16Array(unique.length * 128)
      };
    });
  const runner = new ClefTvmRunner({
    fake: true,
    selectRows: selector,
    runtime,
    device,
    gpu: {
      pushErrorScope() {},
      async popErrorScope() {
        return null;
      }
    },
    lifecycle,
    disposalGuard: {},
    uncaptured: [],
    teardownObserver: {async beforeDispose() {}, async afterDispose() {}},
    teardownTimeoutMs: 100,
    fn:
      fn ||
      ((...args) => {
        events.push(`graph:${args.length}`);
        return {dtype: 'float32', shape: [2]};
      }),
    weights: Array.from({length: 797}, () => ({})),
    source: {inputBanks: [], outputBanks: [], readRange() {}}
  });
  return {runner, events, ctx, lifecycle};
}

test('one joint graph receives 17 inputs and 797 weights and reads FP32 rows', async () => {
  const {runner, events} = fixture();
  const result = await runner.run(packed);
  assert.deepEqual(
    result.map((row) => Array.from(row)),
    [[1, 2]]
  );
  assert.deepEqual(events, ['begin', 'graph:814', 'sync', 'end']);
  await runner.dispose();
});

test('incorrect selected bank or local ID is rejected before GPU dispatch', async () => {
  for (const mode of ['bank', 'local']) {
    const {runner, events} = fixture({
      select: async (ids) => {
        const unique = [...new Set(ids)];
        const value = {
          unique,
          local: Int32Array.from(ids, (id) => unique.indexOf(id)),
          rowBanks: Int32Array.from(unique, (id) => Math.floor(id / 65536)),
          packed: new Uint32Array(unique.length * 512),
          scales: new Uint16Array(unique.length * 128)
        };
        if (mode === 'bank') value.rowBanks[0] = 3;
        else value.local[0] = 100;
        return value;
      }
    });
    await assert.rejects(runner.run(packed), /Invalid Clef local ID mapping/);
    assert.deepEqual(events, []);
    await runner.dispose();
  }
});

test('device loss stops before range scans and loss during readback forbids reuse', async () => {
  let selected = 0;
  const lost = fixture({
    lifecycle: {loss: true},
    select: async () => {
      selected++;
    }
  });
  await assert.rejects(lost.runner.run(packed), {code: 'INFERENCE_ERROR'});
  assert.equal(selected, 0);
  const waiting = defer(),
    entered = defer(),
    current = fixture({readback: waiting.promise, sync: () => entered.resolve()});
  const run = current.runner.run(packed);
  await entered.promise;
  current.lifecycle.loss = true;
  waiting.resolve();
  await assert.rejects(run, {code: 'INFERENCE_ERROR'});
  await assert.rejects(current.runner.run(packed), {code: 'INFERENCE_ERROR'});
  await assert.rejects(current.runner.dispose(), {code: 'INFERENCE_ERROR'});
});

test('an unresolved readback does not release the scope or permit runner reuse', async () => {
  const waiting = defer(),
    entered = defer(),
    current = fixture({readback: waiting.promise, sync: () => entered.resolve()});
  const run = current.runner.run(packed);
  await entered.promise;
  await assert.rejects(current.runner.run(packed), {code: 'INVALID_INPUT'});
  await assert.rejects(current.runner.dispose(), {code: 'INVALID_INPUT'});
  assert.deepEqual(current.events, ['begin', 'graph:814', 'sync']);
  waiting.resolve();
  await run;
  await current.runner.dispose();
});

test('unknown pending readback quarantines the runtime and prevents cleanup or reuse', async () => {
  const current = fixture();
  current.ctx.pendingGPUToCPUCopy = Promise.resolve();
  await assert.rejects(current.runner.run(packed), {code: 'INFERENCE_ERROR'});
  assert.equal(current.runner.poisoned, true);
  await assert.rejects(current.runner.run(packed), {code: 'INFERENCE_ERROR'});
  await assert.rejects(current.runner.dispose(), {code: 'INFERENCE_ERROR'});
  assert.equal(current.events.includes('dispose'), false);
});

test('device loss during a range read stops the scan before another range', async () => {
  let reads = 0;
  const current = fixture({
    select: async (ids, banks, {readRange}) => {
      await readRange({}, {}, undefined);
      await readRange({}, {}, undefined);
    }
  });
  current.runner.source.readRange = async () => {
    reads++;
    current.lifecycle.loss = true;
    return new Uint8Array(1);
  };
  await assert.rejects(current.runner.run(packed), {code: 'INFERENCE_ERROR'});
  assert.equal(reads, 1);
  assert.deepEqual(current.events, []);
});

test('input and output tables are selected independently with their own IDs', async () => {
  const calls = [];
  const current = fixture({
    select: async (ids, banks) => {
      calls.push({ids: Array.from(ids), banks});
      const unique = [...new Set(ids)];
      return {
        unique,
        local: Int32Array.from(ids, (id) => unique.indexOf(id)),
        rowBanks: Int32Array.from(unique, (id) => Math.floor(id / 65536)),
        packed: new Uint32Array(unique.length * 512),
        scales: new Uint16Array(unique.length * 128)
      };
    }
  });
  await current.runner.run(packed);
  assert.deepEqual(
    calls.map((call) => call.ids),
    [
      [10, 11, 65537, 12],
      [11, 65537]
    ]
  );
  assert.equal(calls[0].banks, current.runner.source.inputBanks);
  assert.equal(calls[1].banks, current.runner.source.outputBanks);
  assert.notEqual(calls[0].banks, calls[1].banks);
  await current.runner.dispose();
});
