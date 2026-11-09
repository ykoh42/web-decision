import assert from 'node:assert/strict';
import {test} from 'node:test';
import {processPrefetchedShards, validatePrefetchBytes} from '../src/shard-prefetch.js';

interface Deferred<Value> {
  readonly promise: Promise<Value>;
  resolve(value?: Value): void;
  reject(reason?: unknown): void;
}

function deferred<Value = unknown>(): Deferred<Value> {
  let fulfill!: (value: Value | PromiseLike<Value>) => void;
  let fail!: (reason?: unknown) => void;
  const promise = new Promise<Value>((resolve, reject) => {
    fulfill = resolve;
    fail = reject;
  });
  return {
    promise,
    resolve(value?: Value) {
      fulfill(value as Value);
    },
    reject(reason?: unknown) {
      fail(reason);
    }
  };
}

const tick = (): Promise<void> => new Promise((resolve) => setImmediate(resolve));

async function within<Value>(promise: PromiseLike<Value>, ms = 100): Promise<Value | 'still pending'> {
  let timer!: ReturnType<typeof setTimeout>;
  try {
    return await Promise.race([
      promise,
      new Promise<'still pending'>((resolve) => {
        timer = setTimeout(() => resolve('still pending'), ms);
      })
    ]);
  } finally {
    clearTimeout(timer);
  }
}

test('validates the explicit read-ahead byte budget', () => {
  assert.equal(validatePrefetchBytes(undefined), 0);
  assert.equal(validatePrefetchBytes(0), 0);
  assert.equal(validatePrefetchBytes(33554432), 33554432);
  for (const value of [-1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1, true, null, '33554432']) {
    assert.throws(() => validatePrefetchBytes(value), {code: 'INVALID_INPUT'});
  }
});

test('bounds read-ahead by the next shard size and preserves upload order', async () => {
  const records = [
    {name: 'a', nbytes: 80},
    {name: 'b', nbytes: 31},
    {name: 'c', nbytes: 40}
  ];
  const reads: string[] = [],
    uploads: Array<[string, unknown]> = [],
    gates = new Map<string, Deferred<unknown>>(),
    firstUpload = deferred();
  const task = processPrefetchedShards(
    records,
    async (shard) => {
      reads.push(shard.name);
      const gate = deferred();
      gates.set(shard.name, gate);
      return gate.promise;
    },
    async (shard, value) => {
      uploads.push([shard.name, value]);
      if (shard.name === 'a') await firstUpload.promise;
    },
    undefined,
    35
  );
  try {
    await tick();
    assert.deepEqual(reads, ['a', 'b']);
    gates.get('b')!.resolve('B');
    await tick();
    assert.deepEqual(uploads, []);
    gates.get('a')!.resolve('A');
    await tick();
    assert.deepEqual(uploads, [['a', 'A']]);
    assert.deepEqual(reads, ['a', 'b']);
    firstUpload.resolve();
    await tick();
    assert.deepEqual(uploads, [
      ['a', 'A'],
      ['b', 'B']
    ]);
    assert.deepEqual(reads, ['a', 'b', 'c']);
    gates.get('c')!.resolve('C');
    await task;
    assert.deepEqual(uploads, [
      ['a', 'A'],
      ['b', 'B'],
      ['c', 'C']
    ]);
  } finally {
    firstUpload.resolve();
    for (const gate of gates.values()) gate.resolve('cleanup');
  }
});

test('zero budget does not read a future shard during the current upload', async () => {
  const reads: number[] = [],
    firstUpload = deferred(),
    started = deferred();
  const task = processPrefetchedShards(
    [{nbytes: 10}, {nbytes: 10}],
    async (_, _signal, index) => {
      reads.push(index);
      return index;
    },
    async (_, value) => {
      if (value === 0) {
        started.resolve();
        await firstUpload.promise;
      }
    },
    undefined,
    0
  );
  try {
    await started.promise;
    assert.deepEqual(reads, [0]);
    firstUpload.resolve();
    await task;
    assert.deepEqual(reads, [0, 1]);
  } finally {
    firstUpload.resolve();
  }
});

test('a later shard failure aborts a stalled earlier read and returns its error', async () => {
  const first = deferred(),
    started = deferred(),
    failure = Error('later shard failed'),
    signals: AbortSignal[] = [];
  const task = processPrefetchedShards(
    [{nbytes: 10}, {nbytes: 10}],
    async (_, signal, index) => {
      signals.push(signal);
      if (index === 0) {
        started.resolve();
        return first.promise;
      }
      await started.promise;
      throw failure;
    },
    async () => {
      throw Error('Unexpected upload');
    },
    undefined,
    10
  );
  const observed = task.then(
    () => 'resolved',
    (error) => error
  );
  try {
    assert.equal(await within(observed), failure);
    assert.equal(signals.length, 2);
    assert(signals.every((signal) => signal.aborted));
  } finally {
    first.reject(Error('late first-read failure'));
    await observed;
  }
  await tick();
});

test('an undefined read-ahead failure retains its cause and waits for an active upload', async () => {
  const later = deferred(),
    started = deferred(),
    release = deferred();
  let completed = false;
  const task = processPrefetchedShards(
    [{nbytes: 1}, {nbytes: 1}],
    (_shard, _signal, index) => (index === 0 ? 'first' : later.promise),
    async () => {
      started.resolve();
      await release.promise;
      completed = true;
    },
    undefined,
    1
  );
  const observed = task.then(
    () => ({resolved: true}),
    (reason) => ({resolved: false, reason})
  );
  try {
    await started.promise;
    later.reject(undefined);
    assert.equal(await within(observed, 30), 'still pending');
    release.resolve();
    const result = await within(observed);
    assert.equal(completed, true);
    assert.deepEqual(result, {resolved: false, reason: undefined});
  } finally {
    release.resolve();
    await observed;
  }
});

test('an undefined later failure interrupts an uncooperative earlier read without changing the reason', async () => {
  const first = deferred(),
    started = deferred();
  const task = processPrefetchedShards(
    [{nbytes: 1}, {nbytes: 1}],
    async (_shard, _signal, index) => {
      if (index === 0) {
        started.resolve();
        return first.promise;
      }
      await started.promise;
      throw undefined;
    },
    async () => {
      throw Error('Unexpected upload');
    },
    undefined,
    1
  );
  const observed = task.then(
    () => ({resolved: true}),
    (reason) => ({resolved: false, reason})
  );
  try {
    assert.deepEqual(await within(observed), {resolved: false, reason: undefined});
  } finally {
    first.reject(Error('late fake read failure'));
    await observed;
  }
});

test('caller cancellation wins over an earlier read failure after the upload settles', async () => {
  const controller = new AbortController(),
    later = deferred(),
    started = deferred(),
    release = deferred();
  let completed = false;
  const task = processPrefetchedShards(
    [{nbytes: 1}, {nbytes: 1}],
    (_shard, _signal, index) => (index === 0 ? 'first' : later.promise),
    async () => {
      started.resolve();
      await release.promise;
      completed = true;
    },
    controller.signal,
    1
  );
  const observed = task.then(
    () => ({resolved: true}),
    (reason) => ({resolved: false, reason})
  );
  try {
    await started.promise;
    later.reject(undefined);
    await tick();
    controller.abort(null);
    assert.equal(await within(observed, 30), 'still pending');
    release.resolve();
    const result = await within(observed);
    assert.equal(completed, true);
    assert.deepEqual(result, {resolved: false, reason: null});
  } finally {
    release.resolve();
    await observed;
  }
});

test('a read failure observed before a later upload error remains the first failure', async () => {
  const later = deferred(),
    started = deferred(),
    release = deferred();
  const task = processPrefetchedShards(
    [{nbytes: 1}, {nbytes: 1}],
    (_shard, _signal, index) => (index === 0 ? 'first' : later.promise),
    async () => {
      started.resolve();
      await release.promise;
      throw Error('later upload failure');
    },
    undefined,
    1
  );
  const observed = task.then(
    () => ({resolved: true}),
    (reason) => ({resolved: false, reason})
  );
  try {
    await started.promise;
    later.reject(undefined);
    await tick();
    release.resolve();
    assert.deepEqual(await within(observed), {resolved: false, reason: undefined});
  } finally {
    release.resolve();
    await observed;
  }
});

test('the first callback promise rejection wins when read and upload fail in the same turn', async (t) => {
  for (const order of ['read-first', 'upload-first']) {
    await t.test(order, async () => {
      const readAhead = deferred(),
        upload = deferred(),
        started = deferred();
      const uploadFailure = Error('fake upload failure');
      const task = processPrefetchedShards(
        [{nbytes: 1}, {nbytes: 1}],
        (_shard, _signal, index) => (index === 0 ? 'first' : readAhead.promise),
        () => {
          started.resolve();
          return upload.promise;
        },
        undefined,
        1
      );
      const observed = task.then(
        () => ({resolved: true}),
        (reason) => ({resolved: false, reason})
      );
      try {
        await started.promise;
        if (order === 'read-first') {
          readAhead.reject(undefined);
          upload.reject(uploadFailure);
        } else {
          upload.reject(uploadFailure);
          readAhead.reject(undefined);
        }
        assert.deepEqual(await within(observed), {
          resolved: false,
          reason: order === 'read-first' ? undefined : uploadFailure
        });
      } finally {
        readAhead.reject(Error('cleanup'));
        upload.reject(Error('cleanup'));
        await observed;
      }
    });
  }
});

test('caller cancellation stops waiting for uncooperative reads and preserves the reason', async () => {
  const controller = new AbortController(),
    reason = Error('caller cancelled'),
    gates = [deferred(), deferred()],
    signals: AbortSignal[] = [];
  const task = processPrefetchedShards(
    [{nbytes: 10}, {nbytes: 10}],
    async (_, signal, index) => {
      signals.push(signal);
      return gates[index]!.promise;
    },
    async () => {
      throw Error('Unexpected upload');
    },
    controller.signal,
    10
  );
  const observed = task.then(
    () => 'resolved',
    (error) => error
  );
  try {
    await tick();
    assert.equal(signals.length, 2);
    controller.abort(reason);
    assert.equal(await within(observed), reason);
    assert(signals.every((signal) => signal.aborted));
  } finally {
    gates.forEach((gate) => gate.reject(Error('late read failure')));
    await observed;
  }
  await tick();
});

test('upload failure cancels the read-ahead and retains the upload error', async () => {
  const failure = Error('upload failed'),
    pending = deferred(),
    signals: AbortSignal[] = [];
  const task = processPrefetchedShards(
    [{nbytes: 10}, {nbytes: 10}],
    async (_, signal, index) => {
      signals.push(signal);
      return index === 0 ? 'first' : pending.promise;
    },
    async () => {
      throw failure;
    },
    undefined,
    10
  );
  const observed = task.then(
    () => 'resolved',
    (error) => error
  );
  try {
    assert.equal(await within(observed), failure);
    assert(signals.every((signal) => signal.aborted));
  } finally {
    pending.reject(Error('late read-ahead failure'));
    await observed;
  }
  await tick();
});

test('a falsy upload failure remains the first failure when cancelled read-ahead rejects later', async () => {
  for (const reason of [undefined, null, false, 0, '']) {
    const pending = deferred();
    const task = processPrefetchedShards(
      [{nbytes: 1}, {nbytes: 1}],
      (_shard, _signal, index) => (index === 0 ? 'first' : pending.promise),
      async () => {
        throw reason;
      },
      undefined,
      1
    );
    const observed = task.then(
      () => ({resolved: true}),
      (error) => ({resolved: false, error})
    );
    try {
      assert.deepEqual(await within(observed), {resolved: false, error: reason});
    } finally {
      pending.reject(Error('late fake read-ahead failure'));
      await observed;
    }
  }
  await tick();
});

test('cancellation during an upload aborts read-ahead but waits for the upload to settle', async () => {
  const controller = new AbortController(),
    reason = Error('cancel during upload'),
    pending = deferred(),
    started = deferred(),
    release = deferred(),
    signals: AbortSignal[] = [];
  const task = processPrefetchedShards(
    [{nbytes: 10}, {nbytes: 10}],
    async (_, signal, index) => {
      signals.push(signal);
      return index === 0 ? 'first' : pending.promise;
    },
    async () => {
      started.resolve();
      await release.promise;
    },
    controller.signal,
    10
  );
  const observed = task.then(
    () => 'resolved',
    (error) => error
  );
  try {
    await started.promise;
    controller.abort(reason);
    assert.equal(await within(observed, 30), 'still pending');
    assert(signals.every((signal) => signal.aborted));
    release.resolve();
    assert.equal(await within(observed), reason);
  } finally {
    release.resolve();
    pending.reject(Error('late read-ahead failure'));
    await observed;
  }
  await tick();
});
