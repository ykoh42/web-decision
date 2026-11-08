import test from 'node:test';
import assert from 'node:assert/strict';
import {DecisionModel} from '../src/model.js';
import {ExperimentalDebertaSession} from '../src/deberta/model.mjs';
import {trackRuntime} from '../src/runtime-lifecycle.js';

const request = {state: 'state', questions: [{instructions: 'choose', options: ['a', 'b']}]};
const tokenizer = {encode: (text) => ({ids: Array.from(text, (c) => c.codePointAt(0) + 200)})};

function kevSession({pop, sync = () => {}, end = () => {}, trackDeviceLoss = false} = {}) {
  const state = {loss: null, disposed: false},
    events = [],
    uncaptured = [];
  const runtime = {
    beginScope() {
      events.push('begin');
    },
    endScope() {
      events.push('end');
      return end();
    },
    dispose() {
      events.push('dispose');
      state.disposed = true;
    },
    cpu() {
      return 'cpu';
    },
    getGlobalFunc() {
      return () => events.push('clear');
    },
    empty(shape) {
      return {
        copyFrom() {
          return this;
        },
        toRawBytes() {
          return new Uint8Array(shape[1] * 2);
        }
      };
    }
  };
  const device = {
    async sync() {
      events.push('sync');
      return sync();
    }
  };
  let resolveLost;
  const gpu = {
    lost: new Promise((resolve) => {
      resolveLost = resolve;
    }),
    pushErrorScope() {
      events.push('push');
    },
    async popErrorScope() {
      events.push('pop');
      return pop?.(lifecycle, uncaptured) ?? null;
    }
  };
  const lifecycle = trackDeviceLoss ? trackRuntime(runtime, gpu) : state;
  if (trackDeviceLoss) gpu.lost.then(() => runtime.dispose());
  const model = new DecisionModel(
    runtime,
    device,
    gpu,
    () => ({}),
    [],
    tokenizer,
    {maxStateTokens: 384, maxInputTokens: 1024},
    uncaptured,
    lifecycle
  );
  return {model, lifecycle, events, lose: resolveLost};
}

function debertaSession({pop, sync = () => {}, end = () => {}} = {}) {
  const lifecycle = {loss: null, disposed: false},
    events = [],
    uncaptured = [];
  const runtime = {
    beginScope() {
      events.push('begin');
    },
    endScope() {
      events.push('end');
      return end();
    },
    dispose() {
      events.push('dispose');
      lifecycle.disposed = true;
    },
    cpu() {
      return 'cpu';
    },
    getGlobalFunc() {
      return () => events.push('clear');
    },
    empty(shape) {
      return {
        copyFrom() {
          return this;
        },
        toRawBytes() {
          return new Uint8Array(shape[1] * 2);
        }
      };
    }
  };
  const device = {
    async sync() {
      events.push('sync');
      return sync();
    }
  };
  const gpu = {
    pushErrorScope() {
      events.push('push');
    },
    async popErrorScope() {
      events.push('pop');
      return pop?.(lifecycle, uncaptured) ?? null;
    }
  };
  const model = new ExperimentalDebertaSession({
    runtime,
    device,
    gpu,
    encoder: () => ({}),
    head: () => ({}),
    encoderWeights: [],
    headWeights: [],
    tokenizer,
    uncaptured,
    lifecycle
  });
  return {model, lifecycle, events};
}

test('Kev 0.6B decision rejects when the device is lost during GPU scope cleanup', async () => {
  const fixture = kevSession({
    pop: (lifecycle) => {
      lifecycle.loss = {reason: 'fake-loss'};
    }
  });
  await assert.rejects(
    fixture.model.decide(request),
    (error) => error.code === 'INFERENCE_ERROR' && /lost/.test(error.cause?.message)
  );
  await fixture.model.dispose();
  assert.deepEqual(fixture.events, ['begin', 'push', 'sync', 'pop', 'dispose']);
});

test('Kev 0.6B memory release rejects when runtime disposal occurs during GPU scope cleanup', async () => {
  const fixture = kevSession({
    pop: (lifecycle) => {
      lifecycle.disposed = true;
    }
  });
  await assert.rejects(
    fixture.model.releaseCachedMemory(),
    (error) => error.code === 'INFERENCE_ERROR' && /lost/.test(error.cause?.message)
  );
  await fixture.model.dispose();
  assert.deepEqual(fixture.events, ['begin', 'push', 'clear', 'sync', 'pop']);
});

test('Kev inference succeeds while the tracked device-loss signal remains active', async () => {
  const fixture = kevSession({trackDeviceLoss: true});
  const result = await fixture.model.decide(request);
  assert.equal(result.answers[0].selected, 'a');
  assert.equal(fixture.lifecycle.lossSignal.aborted, false);
  await fixture.model.dispose();
  assert.deepEqual(fixture.events, ['begin', 'push', 'sync', 'pop', 'end', 'end', 'dispose']);
});

for (const action of ['decide', 'releaseCachedMemory']) {
  test(`Kev ${action} rejects promptly when device loss interrupts a pending sync`, {timeout: 1000}, async () => {
    let startSync;
    const syncStarted = new Promise((resolve) => {
        startSync = resolve;
      }),
      fixture = kevSession({
        trackDeviceLoss: true,
        sync: () => {
          startSync();
          return new Promise(() => {});
        }
      });
    const operation = action === 'decide' ? fixture.model.decide(request) : fixture.model.releaseCachedMemory();
    await syncStarted;
    fixture.lose({reason: 'unknown', message: 'test device loss'});
    await assert.rejects(
      operation,
      (error) =>
        error.code === 'INFERENCE_ERROR' && error.cause instanceof Error && /device was lost/i.test(error.cause.message)
    );
    assert.equal(fixture.lifecycle.lossSignal.aborted, true);
    await fixture.model.dispose();
    assert.deepEqual(
      fixture.events,
      action === 'decide' ? ['begin', 'push', 'sync', 'dispose'] : ['begin', 'push', 'clear', 'sync', 'dispose']
    );
  });
}

test('Kev decision rejects promptly when device loss interrupts error-scope cleanup', {timeout: 1000}, async () => {
  let startPop;
  const popStarted = new Promise((resolve) => {
      startPop = resolve;
    }),
    fixture = kevSession({
      trackDeviceLoss: true,
      pop: () => {
        startPop();
        return new Promise(() => {});
      }
    });
  const decision = fixture.model.decide(request);
  await popStarted;
  fixture.lose({reason: 'unknown', message: 'test device loss'});
  await assert.rejects(
    decision,
    (error) =>
      error.code === 'INFERENCE_ERROR' && error.cause instanceof Error && /device was lost/i.test(error.cause.message)
  );
  await fixture.model.dispose();
  assert.deepEqual(fixture.events, ['begin', 'push', 'sync', 'pop', 'dispose']);
});

test('DeBERTa decision rejects when the device is lost during GPU scope cleanup', async () => {
  const fixture = debertaSession({
    pop: (lifecycle) => {
      lifecycle.loss = {reason: 'fake-loss'};
    }
  });
  await assert.rejects(
    fixture.model.decide(request),
    (error) => error.code === 'INFERENCE_ERROR' && /lost/.test(error.cause?.message)
  );
  await fixture.model.dispose();
  assert.deepEqual(fixture.events, ['begin', 'push', 'begin', 'sync', 'end', 'pop', 'end', 'dispose']);
});

test('DeBERTa memory release rejects when the device is lost during GPU scope cleanup', async () => {
  const fixture = debertaSession({
    pop: (lifecycle) => {
      lifecycle.loss = {reason: 'fake-loss'};
    }
  });
  await assert.rejects(
    fixture.model.releaseCachedMemory(),
    (error) => error.code === 'INFERENCE_ERROR' && /lost/.test(error.cause?.message)
  );
  await fixture.model.dispose();
  assert.deepEqual(fixture.events, ['begin', 'push', 'clear', 'sync', 'pop', 'end', 'dispose']);
});

for (const [family, make] of [
  ['Kev 0.6B', kevSession],
  ['DeBERTa', debertaSession]
]) {
  for (const action of ['decide', 'releaseCachedMemory']) {
    test(`${family} ${action} keeps an earlier operation error over cleanup loss and validation error`, async () => {
      const original = Error('original sync failure'),
        validation = Error('later validation failure');
      const fixture = make({
        sync: () => {
          throw original;
        },
        pop: (lifecycle) => {
          lifecycle.loss = {reason: 'fake-loss'};
          return validation;
        }
      });
      const result = action === 'decide' ? fixture.model.decide(request) : fixture.model.releaseCachedMemory();
      await assert.rejects(result, (error) => error.code === 'INFERENCE_ERROR' && error.cause === original);
      await fixture.model.dispose();
    });
    test(`${family} ${action} preserves a falsy original failure over a later validation error`, async () => {
      for (const reason of [undefined, null, false, 0, '']) {
        const fixture = make({
          sync: () => {
            throw reason;
          },
          pop: (lifecycle) => {
            lifecycle.loss = {reason: 'fake-loss'};
            return Error('later validation failure');
          }
        });
        const result = action === 'decide' ? fixture.model.decide(request) : fixture.model.releaseCachedMemory();
        await assert.rejects(
          result,
          (error) => error.code === 'INFERENCE_ERROR' && Object.hasOwn(error, 'cause') && Object.is(error.cause, reason)
        );
        await fixture.model.dispose();
      }
    });
    test(`${family} ${action} preserves an uncaptured error that appears with cleanup loss`, async () => {
      const fixture = make({
        pop: (lifecycle, uncaptured) => {
          lifecycle.loss = {reason: 'fake-loss'};
          uncaptured.push('earlier uncaptured GPU error');
        }
      });
      const result = action === 'decide' ? fixture.model.decide(request) : fixture.model.releaseCachedMemory();
      await assert.rejects(
        result,
        (error) =>
          error.code === 'INFERENCE_ERROR' &&
          error.message === 'earlier uncaptured GPU error' &&
          error.cause === undefined
      );
      await fixture.model.dispose();
    });
    for (const phase of ['sync', 'pop', 'end']) {
      test(`${family} ${action} rejects falsy ${phase} failures without losing their cause`, async () => {
        for (const reason of [undefined, null, false, 0, '']) {
          let failed = false;
          const failure = () => {
            if (!failed) {
              failed = true;
              throw reason;
            }
          };
          const fixture = make({[phase]: failure});
          const result = action === 'decide' ? fixture.model.decide(request) : fixture.model.releaseCachedMemory();
          await assert.rejects(
            result,
            (error) =>
              error.code === 'INFERENCE_ERROR' && Object.hasOwn(error, 'cause') && Object.is(error.cause, reason)
          );
          await fixture.model.dispose();
        }
      });
    }
  }
}

test('DeBERTa retains an earlier inference failure when closing the inner scope also fails', async () => {
  const original = Error('earlier inference failure'),
    cleanup = Error('later inner cleanup failure');
  let ended = false;
  const fixture = debertaSession({
    sync: () => {
      throw original;
    },
    end: () => {
      if (!ended) {
        ended = true;
        throw cleanup;
      }
    }
  });
  await assert.rejects(
    fixture.model.decide(request),
    (error) => error.code === 'INFERENCE_ERROR' && error.cause === original
  );
  await fixture.model.dispose();
});

test('DeBERTa retains a falsy inference failure when closing the inner scope also fails', async () => {
  for (const reason of [undefined, null, false, 0, '']) {
    let ended = false;
    const fixture = debertaSession({
      sync: () => {
        throw reason;
      },
      end: () => {
        if (!ended) {
          ended = true;
          throw Error('later inner cleanup failure');
        }
      }
    });
    await assert.rejects(
      fixture.model.decide(request),
      (error) => error.code === 'INFERENCE_ERROR' && Object.hasOwn(error, 'cause') && Object.is(error.cause, reason)
    );
    await fixture.model.dispose();
  }
});

for (const [family, make] of [
  ['Kev 0.6B', kevSession],
  ['DeBERTa', debertaSession]
]) {
  test(`${family} cancellation and dispose wait for a late falsy inference failure`, async () => {
    let start,
      resume,
      syncCount = 0;
    const started = new Promise((resolve) => {
      start = resolve;
    });
    const gate = new Promise((resolve) => {
      resume = resolve;
    });
    const fixture = make({
      sync: async () => {
        syncCount++;
        if (syncCount === 1) {
          start();
          await gate;
          throw undefined;
        }
      }
    });
    const controller = new AbortController(),
      reason = Error('cancel active');
    const first = fixture.model.decide(request, {signal: controller.signal});
    await started;
    const next = fixture.model.decide(request),
      closing = fixture.model.dispose();
    controller.abort(reason);
    await assert.rejects(first, (error) => error === reason);
    assert.equal(syncCount, 1);
    assert.equal(fixture.events.includes('dispose'), false);
    resume();
    assert.equal((await next).answers[0].selected, 'a');
    await closing;
    assert.equal(syncCount, 2);
    assert.equal(fixture.events.filter((event) => event === 'dispose').length, 1);
  });
}
