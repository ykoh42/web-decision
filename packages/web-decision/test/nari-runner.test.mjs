import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {NariToqRunner, snapshotToqInput} from '../src/nari/runner.mjs';

const BINDINGS = JSON.parse(readFileSync(new URL('./fixtures/nari/metadata.json', import.meta.url))).ordered;

function fixture() {
  const raw = readFileSync(new URL('./fixtures/nari/short.json', import.meta.url));
  assert.equal(
    createHash('sha256').update(raw).digest('hex'),
    'caca40d05d3af0dbcc4767e588ba5d612196dbae36096d1840c216a327d7a38e'
  );
  return JSON.parse(raw);
}
function joint() {
  const raw = readFileSync(new URL('./fixtures/nari/joint.json', import.meta.url));
  assert.equal(
    createHash('sha256').update(raw).digest('hex'),
    'edd0094bbd94eba95d7f0741f687e4cf21f0e2fa96d795d70a6a32d44c345a4d'
  );
  const p = JSON.parse(raw),
    options = p.counts.map((n) => Array.from({length: n}, (_, i) => String(i)));
  return {
    inputIds: p.inputIds,
    optPos: p.optPos,
    optQ: p.optQ,
    optType: p.optType,
    slotPos: p.slots,
    nLetters: [26, 0],
    options
  };
}
function tensor(shape, dtype, device) {
  return {shape: [...shape], dtype, device, isView: false};
}

test('Q1 snapshot rejects malformed positions and owns its input arrays', () => {
  const input = fixture();
  const snapshot = snapshotToqInput(input);
  input.inputIds[0] = -1;
  assert.notEqual(snapshot.inputIds[0], -1);
  for (const change of [
    (value) => {
      value.inputIds = Array(362).fill(1);
    },
    (value) => {
      value.optPos[1] = value.optPos[0];
    },
    (value) => {
      value.slotPos[0] = value.optPos[1];
    },
    (value) => {
      value.optQ[1] = 1;
    },
    (value) => {
      value.optType[0] = 0;
    },
    (value) => {
      value.options[0][1] = value.options[0][0];
    }
  ]) {
    const bad = fixture();
    change(bad);
    assert.throws(() => snapshotToqInput(bad), {code: 'INVALID_INPUT'});
  }
});

test('TOQ host enforces independent token, option and question limits', () => {
  function packed(counts) {
    let position = 0;
    const options = [],
      optPos = [],
      optQ = [],
      optType = [],
      slotPos = [],
      nLetters = [];
    for (const [question, count] of counts.entries()) {
      options.push(Array.from({length: count}, (_, i) => String(i)));
      for (let i = 0; i < count; i++) {
        optPos.push(position++);
        optQ.push(question);
        optType.push(1);
      }
      nLetters.push(count <= 26 ? count : 0);
      slotPos.push(count <= 26 ? position++ : -1);
    }
    return {inputIds: Array(361).fill(7), optPos, optQ, optType, slotPos, nLetters, options};
  }
  const atLimit = packed(Array(8).fill(8));
  assert.equal(snapshotToqInput(atLimit).optPos.length, 64);
  assert.throws(() => snapshotToqInput(packed([65])), {code: 'INVALID_INPUT'});
  assert.throws(() => snapshotToqInput(packed(Array(9).fill(2))), {code: 'INVALID_INPUT'});
  const tooShort = packed([2]);
  tooShort.inputIds.length = 5;
  assert.throws(() => snapshotToqInput(tooShort), {code: 'INVALID_INPUT'});
});

test('Q1 runner sends the exact input ABI and reads both output shapes', async () => {
  const gpuDevice = {deviceType: 15, deviceId: 0},
    cpu = {deviceType: 1, deviceId: 0};
  const ctx = {pendingGPUToCPUCopy: null, copyRawBytesToBuffer() {}};
  const runtime = {
    lib: {webGPUContext: ctx},
    beginScope() {},
    endScope() {},
    cpu: () => cpu,
    empty(shape, dtype, device) {
      const result = tensor(shape, dtype, device);
      result.byteOffset = 0;
      result.getDataPtr = () => 1;
      result.copyFrom = (value) => {
        if (device === cpu) {
          ctx.pendingGPUToCPUCopy = Promise.resolve().then(() => {
            result.data = Float32Array.from(value.data);
          });
        } else result.data = ArrayBuffer.isView(value) ? value.slice() : value;
        return result;
      };
      result.toArray = () => result.data;
      return result;
    }
  };
  const device = {
    ...gpuDevice,
    async sync() {
      const pending = ctx.pendingGPUToCPUCopy;
      ctx.pendingGPUToCPUCopy = null;
      if (pending) await pending;
    }
  };
  const gpu = {
    pushErrorScope() {},
    async popErrorScope() {
      return null;
    }
  };
  const params = BINDINGS.map((value) => ({
    target: value.target,
    graphArgument: value.graphArgument,
    tensor: tensor(value.shape, value.dtype, device)
  }));
  const letterRows = tensor([2, 26, 2560], 'float32', device);
  let calls = 0;
  const fn = (...args) => {
    calls++;
    assert.equal(args.length, 739);
    const n = calls === 1 ? 79 : 361,
      count = calls === 1 ? 2 : 53,
      questions = calls === 1 ? 1 : 2;
    assert.deepEqual(
      args.slice(0, 9).map((value) => [value.dtype, value.shape]),
      [
        ['int32', [1, n]],
        ['int32', [1, n]],
        ['float16', [1, 1, n, n]],
        ['float16', [1, n]],
        ['int32', [1, count]],
        ['int32', [1, count]],
        ['int32', [1, count]],
        ['int32', [1, questions]],
        ['float32', [2, 26, 2560]]
      ]
    );
    assert.deepEqual(Array.from(args[5].data), Array(count).fill(1));
    assert.deepEqual(Array.from(args[7].data), calls === 1 ? [78] : [214, -1]);
    const head = {...tensor([1, count], 'float32', device), data: Float32Array.from({length: count}, (_, i) => i + 1)};
    const letters = {
      ...tensor([1, questions, 26], 'float32', device),
      data: Float32Array.from({length: questions * 26}, (_, i) => i)
    };
    return {size: () => 2, get: (i) => [head, letters][i]};
  };
  const runner = new NariToqRunner({
    bindings: BINDINGS,
    runtime,
    device,
    gpu,
    fn,
    parameters: params,
    letterRows,
    lifecycle: {loss: null, disposed: false},
    uncaptured: [],
    disposalGuard: {retain() {}, blocked: false},
    teardownObserver: {async beforeDispose() {}, async afterDispose() {}},
    teardownTimeoutMs: 100
  });
  const result = await runner.run(fixture());
  const resultLong = await runner.run(joint());
  assert.equal(calls, 2);
  assert.equal(resultLong.headScores.length, 53);
  assert.equal(resultLong.letterScores.length, 52);
  assert.deepEqual(Array.from(result.headScores), [1, 2]);
  assert.deepEqual(
    Array.from(result.letterScores),
    Array.from({length: 26}, (_, i) => i)
  );
});

test('teardown waits for device loss observation and retains on an unexpected loss', async () => {
  const device = {deviceType: 15, deviceId: 0, async sync() {}};
  const gpu = {
    pushErrorScope() {},
    async popErrorScope() {
      return null;
    }
  };
  const parameters = BINDINGS.map((value) => ({
    target: value.target,
    graphArgument: value.graphArgument,
    tensor: tensor(value.shape, value.dtype, device)
  }));
  let disposed = 0,
    retained = 0;
  const owner = {
    runtime: {
      endScope() {},
      dispose() {
        disposed++;
      }
    },
    device,
    gpu,
    fn() {},
    bindings: BINDINGS,
    parameters,
    letterRows: tensor([2, 26, 2560], 'float32', device),
    lifecycle: {loss: null, disposed: false},
    uncaptured: [],
    disposalGuard: {
      retain() {
        retained++;
      },
      blocked: false
    },
    teardownTimeoutMs: 50,
    teardownObserver: {
      async beforeDispose() {},
      async afterDispose() {
        throw Error('unexpected loss');
      }
    }
  };
  const runner = new NariToqRunner(owner);
  await assert.rejects(runner.dispose(), {code: 'INFERENCE_ERROR'});
  await assert.rejects(runner.dispose(), {code: 'INFERENCE_ERROR'});
  assert.equal(disposed, 1);
  assert.equal(retained, 1);
});
