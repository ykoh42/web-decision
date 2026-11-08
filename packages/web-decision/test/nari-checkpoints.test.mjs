import test from 'node:test';
import assert from 'node:assert/strict';
import {runCheckpointedStages} from '../src/nari/checkpoints.mjs';

function fixture(failingStage) {
  const events = [],
    scopes = [];
  let retained = 0;
  const gpu = {
    pushErrorScope(kind) {
      assert.equal(kind, 'validation');
      scopes.push(null);
      events.push('push');
    },
    async popErrorScope() {
      events.push('pop');
      return scopes.pop();
    }
  };
  const stages = Object.fromEntries(
    ['inputs', 'graph', 'head', 'letter'].map((name) => [
      name,
      {
        submit() {
          events.push('submit:' + name);
          if (name === failingStage) scopes[scopes.length - 1] = {name: 'GPUValidationError'};
          return name;
        },
        async sync() {
          events.push('sync:' + name);
        }
      }
    ])
  );
  return {
    gpu,
    stages,
    events,
    retain() {
      retained++;
    },
    retained: () => retained
  };
}

test('inputs, graph and both readbacks are individually synced and checked in order', async () => {
  const f = fixture();
  const output = await runCheckpointedStages(f);
  assert.deepEqual(output, {inputs: 'inputs', graph: 'graph', head: 'head', letter: 'letter'});
  assert.deepEqual(f.events, [
    'push',
    'submit:inputs',
    'sync:inputs',
    'pop',
    'push',
    'submit:graph',
    'sync:graph',
    'pop',
    'push',
    'submit:head',
    'sync:head',
    'pop',
    'push',
    'submit:letter',
    'sync:letter',
    'pop'
  ]);
  assert.equal(f.retained(), 0);
});

for (const stage of ['inputs', 'graph', 'head', 'letter'])
  test(`captured ${stage} validation error prevents later stages`, async () => {
    const f = fixture(stage);
    await assert.rejects(
      runCheckpointedStages(f),
      (error) => error.stage === stage && error.cause.name === 'GPUValidationError'
    );
    const index = ['inputs', 'graph', 'head', 'letter'].indexOf(stage);
    for (const next of ['inputs', 'graph', 'head', 'letter'].slice(index + 1))
      assert.equal(f.events.includes('submit:' + next), false);
    assert.equal(f.retained(), 0);
  });

test('a failed sync retains ownership and stops before the next readback', async () => {
  const f = fixture();
  f.stages.head.sync = async () => {
    f.events.push('sync:head');
    throw Error('unknown completion');
  };
  await assert.rejects(runCheckpointedStages(f), (error) => error.stage === 'head');
  assert.equal(f.retained(), 1);
  assert.equal(f.events.includes('submit:letter'), false);
});

test('failed validation scope closure retains ownership and stops', async () => {
  const f = fixture();
  f.gpu.popErrorScope = async () => {
    throw Error('scope state unknown');
  };
  await assert.rejects(runCheckpointedStages(f), (error) => error.stage === 'inputs');
  assert.equal(f.retained(), 1);
  assert.equal(f.events.includes('submit:head'), false);
});

test('submit failure retains ownership even before sync', async () => {
  const f = fixture();
  f.stages.inputs.submit = () => {
    throw Error('submission state unknown');
  };
  await assert.rejects(runCheckpointedStages(f), (error) => error.stage === 'inputs');
  assert.equal(f.retained(), 1);
  assert.equal(f.events.includes('sync:inputs'), false);
  assert.equal(f.events.includes('submit:head'), false);
});

test('falsy thrown values remain failures', async () => {
  for (const value of [undefined, null, false, 0, '']) {
    const f = fixture();
    f.stages.inputs.submit = () => {
      throw value;
    };
    await assert.rejects(runCheckpointedStages(f), (error) => error.stage === 'inputs' && error.cause === value);
    assert.equal(f.retained(), 1);
    assert.equal(f.events.includes('submit:head'), false);
  }
});
