import test from 'node:test';
import assert from 'node:assert/strict';
import {ExperimentalMapikaSession} from '../src/mapika/session.js';
import {trackRuntime} from '../src/runtime-lifecycle.js';

type TestSession = Omit<ExperimentalMapikaSession, 'runtime' | 'lifecycle'> & {
  runtime: NonNullable<ExperimentalMapikaSession['runtime']> & {lib: {webGPUContext: {copyRawBytesToBuffer(): void}}};
  lifecycle: NonNullable<ExperimentalMapikaSession['lifecycle']>;
};
type TestError = Error & {code: string; cause: Error};
function setup({
  clear = () => {},
  sync = async () => {},
  pop = async () => null
}: {
  clear?: () => void;
  sync?: () => Promise<void>;
  pop?: () => Promise<Error | null>;
} = {}) {
  const events: string[] = [],
    lifecycle: {loss: unknown; disposed: boolean} = {loss: null, disposed: false};
  const runtime = {
    beginScope() {
      events.push('begin');
    },
    endScope() {
      events.push('end');
    },
    getGlobalFunc() {
      return () => {
        events.push('clear');
        clear();
      };
    },
    dispose() {
      events.push('dispose');
      lifecycle.disposed = true;
    }
  };
  const device = {
    async sync() {
      events.push('sync');
      await sync();
    }
  };
  const gpu = {
    pushErrorScope(name: string) {
      events.push(`push:${name}`);
    },
    async popErrorScope() {
      events.push('pop');
      return pop();
    }
  };
  const session = new ExperimentalMapikaSession({
    runtime,
    device,
    gpu,
    lifecycle,
    uncaptured: [],
    weights: []
  } as unknown as ConstructorParameters<typeof ExperimentalMapikaSession>[0]) as TestSession;
  return {session, lifecycle, events};
}

function configureInference(session: TestSession) {
  session.runtime.empty = (shape: number[]) => ({
    byteOffset: 0,
    getDataPtr: () => 0,
    copyFrom() {
      return this;
    },
    toArray: () => new Float32Array(shape[1]! * 255)
  });
  session.runtime.cpu = () => 'cpu';
  session.runtime.lib = {webGPUContext: {copyRawBytesToBuffer() {}}};
  session.fn = () => ({});
  session.limits = {temperature: 1} as typeof session.limits;
  return {ids: [1, 2], groups: [{slot: 1, options: ['a', 'b']}]};
}

test('a failed inference waits for fallback synchronization before closing its scope', async () => {
  let finish!: () => void;
  let calls = 0,
    settled = false;
  const wait = new Promise<void>((resolve) => {
    finish = resolve;
  });
  const cause = new Error('initial sync failed');
  const {session, events} = setup({
    sync: async () => {
      if (++calls === 1) throw cause;
      await wait;
    }
  });
  const pending = session
    .run(configureInference(session))
    .then(
      () => ({ok: true as const}),
      (error: unknown) => ({ok: false as const, error})
    )
    .then((outcome) => {
      settled = true;
      return outcome;
    });
  try {
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(calls, 2);
    assert.equal(settled, false);
    assert.ok(!events.includes('end'));
  } finally {
    finish();
  }
  const outcome = await pending;
  assert.equal(outcome.ok, false);
  assert.equal((outcome.error as TestError).cause, cause);
  assert.ok(events.includes('end'));
  await session.dispose();
});

test('queued work and disposal wait for the fallback sync after a pre-sync failure', async () => {
  let finish!: () => void;
  let clears = 0,
    syncs = 0;
  const wait = new Promise<void>((resolve) => {
    finish = resolve;
  });
  const cause = new Error('clear failed after submitting work');
  const {session, events} = setup({
    clear: () => {
      if (++clears === 1) throw cause;
    },
    sync: async () => {
      if (++syncs === 1) await wait;
    }
  });
  const first = session.releaseCachedMemory().then(
    () => ({ok: true as const}),
    (error: unknown) => ({ok: false as const, error})
  );
  await new Promise((resolve) => setImmediate(resolve));
  const second = session.releaseCachedMemory(),
    disposal = session.dispose();
  try {
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(clears, 1);
    assert.ok(!events.includes('end'));
    assert.ok(!events.includes('dispose'));
  } finally {
    finish();
  }
  const outcome = await first;
  assert.equal(outcome.ok, false);
  assert.equal((outcome.error as TestError).cause, cause);
  await second;
  await disposal;
  assert.equal(clears, 2);
  assert.equal(syncs, 2);
  assert.ok(events.includes('dispose'));
});

test('preserves a falsy work failure if fallback synchronization also rejects', async () => {
  const {session, events} = setup({
    clear: () => {
      throw false;
    },
    sync: async () => {
      throw Error('drain failed');
    }
  });
  await assert.rejects(
    session.releaseCachedMemory(),
    (error) =>
      (error as TestError).code === 'INFERENCE_ERROR' &&
      Object.hasOwn(error as object, 'cause') &&
      (error as {cause: unknown}).cause === false
  );
  assert.equal(events.filter((event) => event === 'sync').length, 1);
  assert.ok(events.includes('end'));
  await session.dispose();
});

test('skips fallback synchronization if the device is already lost', async () => {
  let fixture!: ReturnType<typeof setup>;
  const cause = new Error('work failed');
  fixture = setup({
    clear: () => {
      fixture.lifecycle.loss = {reason: 'unknown'};
      throw cause;
    }
  });
  await assert.rejects(fixture.session.releaseCachedMemory(), (error) => (error as TestError).cause === cause);
  assert.ok(!fixture.events.includes('sync'));
  await fixture.session.dispose();
});

test('releases cached memory in queue order and disposes only after work settles', async () => {
  let finish!: () => void;
  const wait = new Promise<void>((resolve) => {
    finish = resolve;
  });
  const {session, events} = setup({sync: () => wait});
  const release = session.releaseCachedMemory();
  await new Promise((resolve) => setImmediate(resolve));
  const disposal = session.dispose();
  assert.ok(!events.includes('dispose'));
  finish();
  await release;
  await disposal;
  assert.deepEqual(events, [
    'begin',
    'push:out-of-memory',
    'push:validation',
    'clear',
    'sync',
    'pop',
    'pop',
    'end',
    'end',
    'dispose'
  ]);
  await assert.rejects(session.releaseCachedMemory(), {code: 'SESSION_CLOSED'});
});

test('reports GPU validation and device loss without poisoning disposal', async () => {
  const validation = Error('bad kernel'),
    first = setup({pop: async () => validation});
  await assert.rejects(
    first.session.releaseCachedMemory(),
    (error) => (error as TestError).code === 'INFERENCE_ERROR' && (error as TestError).cause === validation
  );
  assert.ok(first.events.includes('end'));
  await first.session.dispose();

  const lost = setup();
  lost.lifecycle.loss = {reason: 'unknown'};
  await assert.rejects(
    lost.session.releaseCachedMemory(),
    (error) => (error as TestError).code === 'INFERENCE_ERROR' && /lost/.test((error as TestError).cause.message)
  );
  await lost.session.dispose();
  assert.ok(lost.events.includes('dispose'));
});

test('reports device loss while GPU error scopes are being closed', async () => {
  let resolveLoss!: (info: {reason: string; message?: string}) => void;
  const lost = new Promise<{reason: string; message?: string}>((resolve) => {
    resolveLoss = resolve;
  });
  let pops = 0;
  const {session, events} = setup({
    pop: async () => {
      if (pops++ === 0) {
        resolveLoss({reason: 'unknown', message: 'device lost'});
        await lost;
      }
      return null;
    }
  });
  session.lifecycle = trackRuntime(session.runtime, {lost});

  const release = session.releaseCachedMemory();
  await lost;
  assert.equal((session.lifecycle.loss as {reason?: string} | null)?.reason, 'unknown');
  await assert.rejects(
    release,
    (error) => (error as TestError).code === 'INFERENCE_ERROR' && /lost/.test((error as TestError).cause?.message)
  );
  assert.deepEqual(events, ['begin', 'push:out-of-memory', 'push:validation', 'clear', 'sync', 'pop', 'pop', 'end']);
  await session.dispose();
});

test('does not return a Mapika answer if the device is lost while inference scopes close', async () => {
  const normal = setup();
  const expected = await normal.session.run(configureInference(normal.session));
  assert.equal(expected.answers[0]!.selected, 'a');
  await normal.session.dispose();

  let lost!: ReturnType<typeof setup>;
  lost = setup({
    pop: async () => {
      lost.lifecycle.loss = {reason: 'unknown'};
      return null;
    }
  });
  await assert.rejects(
    lost.session.run(configureInference(lost.session)),
    (error) => (error as TestError).code === 'INFERENCE_ERROR' && /lost/.test((error as TestError).cause?.message)
  );
  await lost.session.dispose();
});

test('does not close or dispose a runtime again after device loss disposes it during cleanup', async () => {
  let resolveLoss!: (info: {reason: string; message?: string}) => void;
  const lost = new Promise<{reason: string; message?: string}>((resolve) => {
    resolveLoss = resolve;
  });
  let pops = 0;
  const {session, events} = setup({
    pop: async () => {
      if (++pops === 2) {
        resolveLoss({reason: 'unknown'});
        await lost;
      }
      return null;
    }
  });
  session.lifecycle = trackRuntime(session.runtime, {lost});
  lost.then(() => session.runtime.dispose());

  await assert.rejects(
    session.releaseCachedMemory(),
    (error) => (error as TestError).code === 'INFERENCE_ERROR' && /lost/.test((error as TestError).cause?.message)
  );
  await session.dispose();
  assert.equal(events.filter((event) => event === 'dispose').length, 1);
  assert.ok(!events.includes('end'));
});

test('preserves an existing validation error if the device is also lost during cleanup', async () => {
  let fixture!: ReturnType<typeof setup>;
  const validation = Error('bad kernel');
  fixture = setup({
    pop: async () => {
      fixture.lifecycle.loss = {reason: 'unknown'};
      return validation;
    }
  });
  await assert.rejects(
    fixture.session.releaseCachedMemory(),
    (error) => (error as TestError).code === 'INFERENCE_ERROR' && (error as TestError).cause === validation
  );
  await fixture.session.dispose();
});

test('reports and preserves falsy failures from device work and cleanup', async () => {
  for (const cause of [undefined, null, false, 0, '']) {
    const work = setup({
      sync: async () => {
        throw cause;
      }
    });
    await assert.rejects(
      work.session.releaseCachedMemory(),
      (error) =>
        (error as TestError).code === 'INFERENCE_ERROR' &&
        Object.hasOwn(error as object, 'cause') &&
        Object.is((error as TestError).cause, cause)
    );
    await work.session.dispose();

    const pop = setup({
      pop: async () => {
        throw cause;
      }
    });
    await assert.rejects(
      pop.session.releaseCachedMemory(),
      (error) =>
        (error as TestError).code === 'INFERENCE_ERROR' &&
        Object.hasOwn(error as object, 'cause') &&
        Object.is((error as TestError).cause, cause)
    );
    await pop.session.dispose();

    const end = setup();
    const endScope = end.session.runtime.endScope;
    end.session.runtime.endScope = () => {
      throw cause;
    };
    await assert.rejects(
      end.session.releaseCachedMemory(),
      (error) =>
        (error as TestError).code === 'INFERENCE_ERROR' &&
        Object.hasOwn(error as object, 'cause') &&
        Object.is((error as TestError).cause, cause)
    );
    end.session.runtime.endScope = endScope;
    await end.session.dispose();
  }
});

test('keeps the first failure even if its cause is falsy', async () => {
  const fixture = setup({
    sync: async () => {
      throw null;
    },
    pop: async () => {
      throw Error('later');
    }
  });
  await assert.rejects(
    fixture.session.releaseCachedMemory(),
    (error) =>
      (error as TestError).code === 'INFERENCE_ERROR' &&
      Object.hasOwn(error as object, 'cause') &&
      (error as TestError).cause === null
  );
  await fixture.session.dispose();
});

test('keeps an uncaptured GPU error reported during cleanup ahead of device loss', async () => {
  let fixture!: ReturnType<typeof setup>;
  fixture = setup({
    pop: async () => {
      fixture.lifecycle.loss = {reason: 'unknown'};
      fixture.session.uncaptured.push('uncaptured GPU error');
      return null;
    }
  });
  await assert.rejects(fixture.session.releaseCachedMemory(), {
    code: 'INFERENCE_ERROR',
    message: 'uncaptured GPU error'
  });
  await fixture.session.dispose();
});
