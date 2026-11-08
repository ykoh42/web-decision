import test from 'node:test';
import assert from 'node:assert/strict';
import {ExperimentalDebertaSession} from '../src/deberta/model.mjs';
import {runWithPackedReadback} from '../src/deberta/packed-readback.mjs';
function makeSession({shortHeadBuffer = false, onPop = () => null, onSync = () => {}, onEnd = () => {}} = {}) {
  const events = [],
    buffers = [undefined],
    life = {disposed: false, loss: null},
    uncaptured = [];
  let scopes = 1,
    headCount = 0;
  const context = {
    bufferTable: buffers,
    getDeviceAPI(name) {
      assert.equal(name, 'deviceCopyWithinGPU');
      return (src, sourceOffset, dst, destOffset, nbytes) => {
        events.push({src, sourceOffset, dst, destOffset, nbytes});
        buffers[dst].bytes.set(buffers[src].bytes.subarray(sourceOffset, sourceOffset + nbytes), destOffset);
      };
    }
  };
  function gpuTensor(shape, dtype, bytes) {
    const ptr = buffers.length,
      buffer = {size: bytes, bytes: new Uint8Array(bytes)};
    buffers.push(buffer);
    return {
      shape,
      dtype,
      byteOffset: 0,
      getDataPtr: () => ptr,
      copyFrom() {
        return this;
      }
    };
  }
  const runtime = {
    lib: {webGPUContext: context},
    beginScope() {
      scopes++;
    },
    endScope() {
      scopes--;
      return onEnd();
    },
    dispose() {
      life.disposed = true;
    },
    cpu() {
      return 'cpu';
    },
    empty(shape, dtype, device) {
      if (device !== 'cpu') return gpuTensor(shape, dtype, shape[1] * (dtype === 'int32' ? 4 : 2));
      return {
        copyFrom(gpu) {
          this.raw = buffers[gpu.getDataPtr()].bytes.slice();
          return this;
        },
        toRawBytes() {
          return this.raw;
        }
      };
    }
  };
  const heads = [
      [0x3c00, 0x4000, 0x4200],
      [0x4400, 0x4500, 0x4600]
    ],
    head = () => {
      const t = gpuTensor([1, 3], 'float16', shortHeadBuffer ? 6 : 8),
        view = new DataView(buffers[t.getDataPtr()].bytes.buffer);
      for (const [i, value] of heads[headCount++].entries()) view.setUint16(i * 2, value, true);
      return t;
    };
  const gpu = {
      pushErrorScope() {},
      async popErrorScope() {
        return onPop(life, uncaptured);
      }
    },
    device = {
      async sync() {
        events.push('sync');
        return onSync();
      }
    },
    tokenizer = {encode: (s) => ({ids: [s.length + 10]})};
  const session = new ExperimentalDebertaSession({
    runtime,
    device,
    gpu,
    encoder: () => ({}),
    head,
    encoderWeights: [],
    headWeights: [],
    tokenizer,
    uncaptured,
    lifecycle: life
  });
  session.run = runWithPackedReadback;
  return {
    session,
    events,
    get scopes() {
      return scopes;
    }
  };
}
const request = {
  state: 'state',
  questions: [
    {instructions: 'first', options: ['a', 'b']},
    {instructions: 'second', options: ['c', 'd', 'e', 'f']}
  ]
};
test('packed output keeps scores in the correct multi-question order', async () => {
  const x = makeSession(),
    answer = await x.session.decide(request);
  assert.deepEqual(
    answer.answers.map((q) => q.probabilities.map((p) => p.logit)),
    [
      [1, 2],
      [3, 4, 5, 6]
    ]
  );
  assert.deepEqual(
    answer.answers.map((q) => q.selected),
    ['b', 'f']
  );
  assert.deepEqual(
    x.events.filter((e) => typeof e === 'object').map((e) => [e.sourceOffset, e.destOffset, e.nbytes]),
    [
      [0, 0, 8],
      [0, 8, 8]
    ]
  );
  assert.equal(x.events.filter((e) => e === 'sync').length, 1);
  assert.equal(x.scopes, 1);
  await x.session.dispose();
  assert.equal(x.scopes, 0);
});
test('packed readback rejects a head buffer smaller than its aligned copy', async () => {
  const x = makeSession({shortHeadBuffer: true});
  await assert.rejects(
    x.session.decide(request),
    (e) => e.code === 'INFERENCE_ERROR' && e.cause.message === 'Unexpected head tensor storage.'
  );
  assert.equal(x.events.filter((e) => typeof e === 'object').length, 0);
  assert.equal(x.scopes, 1);
  await x.session.dispose();
  assert.equal(x.scopes, 0);
});

test('packed readback rejects device loss during error scope cleanup', async () => {
  const x = makeSession({
    onPop: (life) => {
      life.loss = {reason: 'fake-loss'};
      return null;
    }
  });
  await assert.rejects(
    x.session.decide(request),
    (error) => error.code === 'INFERENCE_ERROR' && /lost/.test(error.cause?.message)
  );
  assert.equal(x.scopes, 1);
  await x.session.dispose();
});

for (const phase of ['onSync', 'onPop', 'onEnd']) {
  test(`packed readback rejects falsy ${phase} failures without losing their cause`, async () => {
    for (const reason of [undefined, null, false, 0, '']) {
      let failed = false;
      const x = makeSession({
        [phase]: () => {
          if (!failed) {
            failed = true;
            throw reason;
          }
        }
      });
      await assert.rejects(
        x.session.decide(request),
        (error) => error.code === 'INFERENCE_ERROR' && Object.hasOwn(error, 'cause') && Object.is(error.cause, reason)
      );
      await x.session.dispose();
    }
  });
}

test('packed readback preserves an earlier failure over later cleanup loss and validation error', async () => {
  const original = Error('original sync failure'),
    validation = Error('later validation failure');
  const x = makeSession({
    onSync: () => {
      throw original;
    },
    onPop: (life) => {
      life.loss = {reason: 'fake-loss'};
      return validation;
    }
  });
  await assert.rejects(
    x.session.decide(request),
    (error) => error.code === 'INFERENCE_ERROR' && error.cause === original
  );
  await x.session.dispose();
});

test('packed readback preserves a falsy original failure over later validation error', async () => {
  for (const reason of [undefined, null, false, 0, '']) {
    const x = makeSession({
      onSync: () => {
        throw reason;
      },
      onPop: (life) => {
        life.loss = {reason: 'fake-loss'};
        return Error('later validation failure');
      }
    });
    await assert.rejects(
      x.session.decide(request),
      (error) => error.code === 'INFERENCE_ERROR' && Object.hasOwn(error, 'cause') && Object.is(error.cause, reason)
    );
    await x.session.dispose();
  }
});

test('packed readback preserves an uncaptured error that appears with cleanup loss', async () => {
  const x = makeSession({
    onPop: (life, uncaptured) => {
      life.loss = {reason: 'fake-loss'};
      uncaptured.push('earlier uncaptured GPU error');
    }
  });
  await assert.rejects(
    x.session.decide(request),
    (error) =>
      error.code === 'INFERENCE_ERROR' && error.message === 'earlier uncaptured GPU error' && error.cause === undefined
  );
  await x.session.dispose();
});
