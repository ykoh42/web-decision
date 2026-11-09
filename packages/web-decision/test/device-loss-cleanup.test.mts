import test from 'node:test';
import assert from 'node:assert/strict';
import {DecisionModel} from '../src/model.js';
import {ExperimentalDebertaSession} from '../src/deberta/model.mjs';
import {trackRuntime} from '../src/runtime-lifecycle.js';
import type {DecisionResult} from '../src/curated/types.js';

const request = {state: 'state', questions: [{instructions: 'choose', options: ['a', 'b']}]};
const tokenizer = {encode: (text: string) => ({ids: Array.from(text, (c: string) => c.codePointAt(0)! + 200)})};

type TestLifecycle = {loss: unknown; disposed: boolean; lossSignal?: AbortSignal};
type FixtureOptions = {
  pop?: (lifecycle: TestLifecycle, uncaptured: string[]) => unknown;
  sync?: () => unknown;
  end?: () => unknown;
  trackDeviceLoss?: boolean;
};
type TestModel = {
  decide(request: unknown, options?: {signal?: AbortSignal}): Promise<DecisionResult>;
  releaseCachedMemory(): Promise<void>;
  dispose(): Promise<void>;
};
type ModelFixture = {model: TestModel; lifecycle: TestLifecycle; events: string[]};

function kevSession({pop, sync = () => {}, end = () => {}, trackDeviceLoss = false}: FixtureOptions = {}) {
  const state: TestLifecycle = {loss: null, disposed: false},
    events: string[] = [],
    uncaptured: string[] = [];
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
    empty(shape: number[]) {
      return {
        copyFrom() {
          return this;
        },
        toRawBytes() {
          return new Uint8Array(shape[1]! * 2);
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
  let resolveLost!: (value: unknown) => void;
  const gpu = {
    lost: new Promise<unknown>((resolve) => {
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
  const lifecycle: TestLifecycle = trackDeviceLoss ? trackRuntime(runtime, gpu) : state;
  if (trackDeviceLoss) gpu.lost.then(() => runtime.dispose());
  const model = new DecisionModel(
    runtime as unknown as ConstructorParameters<typeof DecisionModel>[0],
    device as unknown as ConstructorParameters<typeof DecisionModel>[1],
    gpu as unknown as ConstructorParameters<typeof DecisionModel>[2],
    () => ({}) as ReturnType<ConstructorParameters<typeof DecisionModel>[3]>,
    [] as ConstructorParameters<typeof DecisionModel>[4],
    tokenizer as unknown as ConstructorParameters<typeof DecisionModel>[5],
    {maxStateTokens: 384, maxInputTokens: 1024} as ConstructorParameters<typeof DecisionModel>[6],
    uncaptured,
    lifecycle as ConstructorParameters<typeof DecisionModel>[8]
  ) as unknown as TestModel;
  return {model, lifecycle, events, lose: resolveLost};
}

function debertaSession({pop, sync = () => {}, end = () => {}}: FixtureOptions = {}): ModelFixture {
  const lifecycle: TestLifecycle = {loss: null, disposed: false},
    events: string[] = [],
    uncaptured: string[] = [];
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
    empty(shape: number[]) {
      return {
        copyFrom() {
          return this;
        },
        toRawBytes() {
          return new Uint8Array(shape[1]! * 2);
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
  } as unknown as ConstructorParameters<typeof ExperimentalDebertaSession>[0]) as unknown as TestModel;
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
    (error: unknown) =>
      (error as {code?: unknown}).code === 'INFERENCE_ERROR' &&
      /lost/.test((error as {cause?: {message?: string} | null}).cause?.message as string)
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
    (error: unknown) =>
      (error as {code?: unknown}).code === 'INFERENCE_ERROR' &&
      /lost/.test((error as {cause?: {message?: string} | null}).cause?.message as string)
  );
  await fixture.model.dispose();
  assert.deepEqual(fixture.events, ['begin', 'push', 'clear', 'sync', 'pop']);
});

test('Kev inference succeeds while the tracked device-loss signal remains active', async () => {
  const fixture = kevSession({trackDeviceLoss: true});
  const result = await fixture.model.decide(request);
  assert.equal(result.answers[0]!.selected, 'a');
  assert.equal(fixture.lifecycle.lossSignal!.aborted, false);
  await fixture.model.dispose();
  assert.deepEqual(fixture.events, ['begin', 'push', 'sync', 'pop', 'end', 'end', 'dispose']);
});

for (const action of ['decide', 'releaseCachedMemory'] as const) {
  test(`Kev ${action} rejects promptly when device loss interrupts a pending sync`, {timeout: 1000}, async () => {
    let startSync!: () => void;
    const syncStarted = new Promise<void>((resolve) => {
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
      (error: unknown) =>
        (error as {code?: unknown}).code === 'INFERENCE_ERROR' &&
        (error as {cause?: unknown}).cause instanceof Error &&
        /device was lost/i.test((error as {cause: Error}).cause.message)
    );
    assert.equal(fixture.lifecycle.lossSignal!.aborted, true);
    await fixture.model.dispose();
    assert.deepEqual(
      fixture.events,
      action === 'decide' ? ['begin', 'push', 'sync', 'dispose'] : ['begin', 'push', 'clear', 'sync', 'dispose']
    );
  });
}

test('Kev decision rejects promptly when device loss interrupts error-scope cleanup', {timeout: 1000}, async () => {
  let startPop!: () => void;
  const popStarted = new Promise<void>((resolve) => {
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
    (error: unknown) =>
      (error as {code?: unknown}).code === 'INFERENCE_ERROR' &&
      (error as {cause?: unknown}).cause instanceof Error &&
      /device was lost/i.test((error as {cause: Error}).cause.message)
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
    (error: unknown) =>
      (error as {code?: unknown}).code === 'INFERENCE_ERROR' &&
      /lost/.test((error as {cause?: {message?: string} | null}).cause?.message as string)
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
    (error: unknown) =>
      (error as {code?: unknown}).code === 'INFERENCE_ERROR' &&
      /lost/.test((error as {cause?: {message?: string} | null}).cause?.message as string)
  );
  await fixture.model.dispose();
  assert.deepEqual(fixture.events, ['begin', 'push', 'clear', 'sync', 'pop', 'end', 'dispose']);
});

for (const [family, make] of [
  ['Kev 0.6B', kevSession],
  ['DeBERTa', debertaSession]
] as const satisfies readonly (readonly [string, (options?: FixtureOptions) => ModelFixture])[]) {
  for (const action of ['decide', 'releaseCachedMemory'] as const) {
    test(`${family} ${action} keeps an earlier operation error over cleanup loss and validation error`, async () => {
      const original = Error('original sync failure'),
        validation = Error('later validation failure');
      const fixture = make({
        sync: () => {
          throw original;
        },
        pop: (lifecycle: TestLifecycle) => {
          lifecycle.loss = {reason: 'fake-loss'};
          return validation;
        }
      });
      const result = action === 'decide' ? fixture.model.decide(request) : fixture.model.releaseCachedMemory();
      await assert.rejects(
        result,
        (error: unknown) =>
          (error as {code?: unknown}).code === 'INFERENCE_ERROR' && (error as {cause?: unknown}).cause === original
      );
      await fixture.model.dispose();
    });
    test(`${family} ${action} preserves a falsy original failure over a later validation error`, async () => {
      for (const reason of [undefined, null, false, 0, '']) {
        const fixture = make({
          sync: () => {
            throw reason;
          },
          pop: (lifecycle: TestLifecycle) => {
            lifecycle.loss = {reason: 'fake-loss'};
            return Error('later validation failure');
          }
        });
        const result = action === 'decide' ? fixture.model.decide(request) : fixture.model.releaseCachedMemory();
        await assert.rejects(
          result,
          (error: unknown) =>
            (error as {code?: unknown}).code === 'INFERENCE_ERROR' &&
            Object.hasOwn(error as object, 'cause') &&
            Object.is((error as {cause?: unknown}).cause, reason)
        );
        await fixture.model.dispose();
      }
    });
    test(`${family} ${action} preserves an uncaptured error that appears with cleanup loss`, async () => {
      const fixture = make({
        pop: (lifecycle: TestLifecycle, uncaptured: string[]) => {
          lifecycle.loss = {reason: 'fake-loss'};
          uncaptured.push('earlier uncaptured GPU error');
        }
      });
      const result = action === 'decide' ? fixture.model.decide(request) : fixture.model.releaseCachedMemory();
      await assert.rejects(
        result,
        (error) =>
          (error as {code?: unknown}).code === 'INFERENCE_ERROR' &&
          (error as {message?: unknown}).message === 'earlier uncaptured GPU error' &&
          (error as {cause?: unknown}).cause === undefined
      );
      await fixture.model.dispose();
    });
    for (const phase of ['sync', 'pop', 'end'] as const) {
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
            (error: unknown) =>
              (error as {code?: unknown}).code === 'INFERENCE_ERROR' &&
              Object.hasOwn(error as object, 'cause') &&
              Object.is((error as {cause?: unknown}).cause, reason)
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
    (error: unknown) =>
      (error as {code?: unknown}).code === 'INFERENCE_ERROR' && (error as {cause?: unknown}).cause === original
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
      (error: unknown) =>
        (error as {code?: unknown}).code === 'INFERENCE_ERROR' &&
        Object.hasOwn(error as object, 'cause') &&
        Object.is((error as {cause?: unknown}).cause, reason)
    );
    await fixture.model.dispose();
  }
});

for (const [family, make] of [
  ['Kev 0.6B', kevSession],
  ['DeBERTa', debertaSession]
] as const satisfies readonly (readonly [string, (options?: FixtureOptions) => ModelFixture])[]) {
  test(`${family} cancellation and dispose wait for a late falsy inference failure`, async () => {
    let start!: () => void,
      resume!: () => void,
      syncCount = 0;
    const started = new Promise<void>((resolve) => {
      start = resolve;
    });
    const gate = new Promise<void>((resolve) => {
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
    await assert.rejects(first, (error: unknown) => error === reason);
    assert.equal(syncCount, 1);
    assert.equal(fixture.events.includes('dispose'), false);
    resume();
    assert.equal((await next).answers[0]!.selected, 'a');
    await closing;
    assert.equal(syncCount, 2);
    assert.equal(fixture.events.filter((event: string) => event === 'dispose').length, 1);
  });
}
