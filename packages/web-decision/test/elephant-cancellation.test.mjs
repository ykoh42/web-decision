import test from 'node:test';
import assert from 'node:assert/strict';
import {ElephantTvmRunner} from '../src/elephant/runner.mjs';
import {createElephantSession} from '../src/elephant/session.mjs';

function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return {promise, resolve};
}

test('aborting an in-flight branch drains its readback and skips remaining branches before disposal', async () => {
  const events = [],
    started = deferred(),
    readback = deferred();
  const context = {pendingGPUToCPUCopy: null, copyRawBytesToBuffer() {}};
  const device = {
    async sync() {
      events.push('sync');
      started.resolve();
      const pending = context.pendingGPUToCPUCopy;
      if (pending) await pending;
      context.pendingGPUToCPUCopy = null;
    }
  };
  const runtime = {
    lib: {webGPUContext: context},
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
          if (where === 'cpu') context.pendingGPUToCPUCopy = readback.promise;
          return this;
        },
        toArray: () => new Float32Array(256)
      };
    }
  };
  const runner = new ElephantTvmRunner({
    runtime,
    device,
    gpu: {
      pushErrorScope() {},
      async popErrorScope() {
        return null;
      }
    },
    fn() {
      events.push('graph');
      return {dtype: 'float32', shape: [1, 256]};
    },
    weights: Array.from({length: 684}, () => ({})),
    lifecycle: {},
    disposalGuard: {},
    uncaptured: [],
    teardownObserver: {async beforeDispose() {}, async afterDispose() {}},
    teardownTimeoutMs: 100
  });
  const tokenizer = {
    encode: (text) => ({
      ids: [...text].filter((char) => char === '<' || char === 'Q').map((char) => char.charCodeAt(0))
    })
  };
  const session = createElephantSession({tokenizer, addedTokens: ['<reserved>'], runner});
  const controller = new AbortController();
  const decision = session.decide(
    {
      state: 'state',
      questions: [
        {instructions: 'first', options: ['a', 'b']},
        {instructions: 'second', options: ['c', 'd']}
      ]
    },
    {signal: controller.signal}
  );
  await started.promise;
  controller.abort();
  await assert.rejects(decision, {name: 'AbortError'});
  const disposal = session.dispose();
  await Promise.resolve();
  assert.deepEqual(events, ['begin', 'graph', 'sync']);
  readback.resolve();
  await disposal;
  assert.equal(events.filter((event) => event === 'graph').length, 1);
  assert.deepEqual(events, ['begin', 'graph', 'sync', 'end', 'end', 'dispose']);
});

const simpleTokenizer = {
  encode: (text) => ({ids: [...text].filter((char) => char === '<' || char === 'Q').map((char) => char.charCodeAt(0))})
};
const characterTokenizer = {encode: (text) => ({ids: Array.from(text, (character) => character.codePointAt(0))})};
const input = () => ({state: 'state', questions: [{instructions: 'choose', options: ['a', 'b']}]});

test('invalid score rows surface as inference errors and do not poison the session queue', async () => {
  const outputs = [null, [], [[1]], [[NaN, 0]], [[0, 2]]];
  const events = [];
  const session = createElephantSession({
    tokenizer: simpleTokenizer,
    addedTokens: ['<reserved>'],
    runner: {
      async run() {
        events.push('run');
        return outputs.shift();
      },
      async releaseCachedMemory() {
        events.push('release');
      },
      async dispose() {
        events.push('dispose');
      }
    }
  });
  for (let i = 0; i < 4; i++) await assert.rejects(session.decide(input()), {code: 'INFERENCE_ERROR'});
  const result = await session.decide(input());
  assert.equal(result.answers[0].selected, 'b');
  await session.releaseCachedMemory();
  await session.dispose();
  await assert.rejects(session.decide(input()), {code: 'SESSION_CLOSED'});
  await assert.rejects(session.releaseCachedMemory(), {code: 'SESSION_CLOSED'});
  assert.deepEqual(events, ['run', 'run', 'run', 'run', 'run', 'release', 'dispose']);
});

test('unsafe token boundaries and invalid requests fail before the runner is called', async () => {
  let calls = 0;
  const runner = {
    async run() {
      calls++;
      return [[0, 1]];
    },
    async releaseCachedMemory() {},
    async dispose() {}
  };
  const broken = createElephantSession({
    tokenizer: {encode: (text) => ({ids: [text.length]})},
    addedTokens: [],
    runner
  });
  await assert.rejects(
    broken.decide(input()),
    (error) => error.code === 'INVALID_INPUT' && /Unsafe token boundary/.test(error.cause.message)
  );
  await broken.dispose();
  const session = createElephantSession({tokenizer: simpleTokenizer, addedTokens: ['<reserved>'], runner});
  await assert.rejects(session.decide({...input(), state: '<reserved>'}), {code: 'INVALID_INPUT'});
  await assert.rejects(session.decide(input(), {temperature: 1}), {code: 'INVALID_INPUT'});
  assert.equal(calls, 0);
  await session.dispose();
});

test('the 255-option cap rejects before snapshotting an oversized array or tokenizing', async () => {
  let tokenizerCalls = 0,
    runnerCalls = 0,
    reads = 0;
  const session = createElephantSession({
    tokenizer: {
      encode() {
        tokenizerCalls++;
        return {ids: []};
      }
    },
    addedTokens: [],
    runner: {
      async run(packed) {
        runnerCalls++;
        return packed.branches.map((branch) => new Array(branch.options.length).fill(0));
      },
      async releaseCachedMemory() {},
      async dispose() {}
    }
  });
  const options = Array.from({length: 255}, (_, index) => `option ${index}`);
  const accepted = await session.decide({state: '', questions: [{instructions: 'Pick', options}]});
  assert.equal(accepted.answers[0].probabilities.length, 255);
  assert.equal(accepted.answers[0].selected, options[0]);

  const tokenizersBeforeReject = tokenizerCalls;
  const runsBeforeReject = runnerCalls;
  const tooMany = Array.from({length: 256}, (_, index) => `option ${index}`);
  Object.defineProperty(tooMany, '0', {
    get() {
      reads++;
      throw Error('option snapshot should not read an oversized array');
    }
  });
  await assert.rejects(session.decide({state: '', questions: [{instructions: 'Pick', options: tooMany}]}), {
    code: 'INVALID_INPUT'
  });
  assert.equal(reads, 0);
  assert.equal(tokenizerCalls, tokenizersBeforeReject);
  assert.equal(runnerCalls, runsBeforeReject);
  await session.dispose();
});

test('accepts more than 16,384 aggregate tokens when every branch fits its graph limit', async () => {
  let runnerCalls = 0,
    requestTokens;
  const questions = Array.from({length: 160}, () => ({instructions: 'Choose', options: ['a', 'b']}));
  const session = createElephantSession({
    tokenizer: characterTokenizer,
    addedTokens: [],
    maxBranchTokens: 512,
    runner: {
      async run(packed) {
        runnerCalls++;
        requestTokens = packed.tokenCount;
        assert.ok(packed.branches.every((branch) => branch.ids.length <= 512));
        return packed.branches.map(() => [0, 1]);
      },
      async releaseCachedMemory() {},
      async dispose() {}
    }
  });
  const result = await session.decide({state: 's'.repeat(32), questions});
  assert.ok(requestTokens > 16_384);
  assert.equal(result.answers.length, 160);
  assert.equal(runnerCalls, 1);
  await session.dispose();
});

test('accepts more than 16,384 small questions without a request-wide count cap', async () => {
  let runnerCalls = 0,
    requestTokens;
  const questions = Array.from({length: 16_385}, () => ({instructions: 'Pick', options: ['a', 'b']}));
  const session = createElephantSession({
    tokenizer: simpleTokenizer,
    addedTokens: [],
    runner: {
      async run(packed) {
        runnerCalls++;
        requestTokens = packed.tokenCount;
        assert.equal(packed.branches.length, 16_385);
        assert.ok(packed.branches.every((branch) => branch.ids.length <= 128));
        return packed.branches.map(() => [0, 1]);
      },
      async releaseCachedMemory() {},
      async dispose() {}
    }
  });
  const result = await session.decide({state: '', questions});
  assert.ok(requestTokens > 16_384);
  assert.equal(result.answers.length, 16_385);
  assert.equal(runnerCalls, 1);
  await session.dispose();
});

test('a cancelled queued decision never runs and prior work completes before disposal', async () => {
  const gate = deferred(),
    started = deferred(),
    events = [];
  const session = createElephantSession({
    tokenizer: simpleTokenizer,
    addedTokens: [],
    runner: {
      async run() {
        events.push('run');
        started.resolve();
        await gate.promise;
        return [[1, 0]];
      },
      async releaseCachedMemory() {
        events.push('release');
      },
      async dispose() {
        events.push('dispose');
      }
    }
  });
  const first = session.decide(input());
  await started.promise;
  const controller = new AbortController();
  const cancelled = session.decide(input(), {signal: controller.signal});
  controller.abort();
  await assert.rejects(cancelled, {name: 'AbortError'});
  const release = session.releaseCachedMemory();
  const disposal = session.dispose();
  assert.deepEqual(events, ['run']);
  gate.resolve();
  assert.equal((await first).answers[0].selected, 'a');
  await release;
  await disposal;
  assert.deepEqual(events, ['run', 'release', 'dispose']);
});

test('invalid graph inputs, output shapes, and readbacks fail as inference errors', async () => {
  for (const mode of ['input', 'shape', 'dtype', 'nan', 'length']) {
    let calls = 0;
    const context = {pendingGPUToCPUCopy: null, copyRawBytesToBuffer() {}};
    const device = {
      async sync() {
        const pending = context.pendingGPUToCPUCopy;
        if (pending) await pending;
        context.pendingGPUToCPUCopy = null;
      }
    };
    const runtime = {
      lib: {webGPUContext: context},
      cpu: () => 'cpu',
      beginScope() {},
      endScope() {},
      dispose() {},
      empty(shape, dtype, where) {
        return {
          shape,
          dtype,
          byteOffset: 0,
          getDataPtr: () => 0,
          copyFrom() {
            if (where === 'cpu') context.pendingGPUToCPUCopy = Promise.resolve();
            return this;
          },
          toArray() {
            const scores = new Float32Array(mode === 'length' ? 255 : 256);
            if (mode === 'nan') scores[0] = NaN;
            return scores;
          }
        };
      }
    };
    const runner = new ElephantTvmRunner({
      runtime,
      device,
      gpu: {
        pushErrorScope() {},
        async popErrorScope() {
          return null;
        }
      },
      fn() {
        calls++;
        return {dtype: mode === 'dtype' ? 'float16' : 'float32', shape: mode === 'shape' ? [1, 255] : [1, 256]};
      },
      weights: Array.from({length: 684}, () => ({})),
      lifecycle: {},
      disposalGuard: {},
      uncaptured: [],
      teardownObserver: {async beforeDispose() {}, async afterDispose() {}},
      teardownTimeoutMs: 100
    });
    const ids = mode === 'input' ? [1, 2, -1, 4] : [1, 2, 3, 4];
    await assert.rejects(
      runner.run({prefixIds: [1], branches: [{ids, options: ['a', 'b']}]}),
      {code: 'INFERENCE_ERROR'},
      mode
    );
    assert.equal(calls, mode === 'input' ? 0 : 1, mode);
    await runner.dispose();
  }
});
