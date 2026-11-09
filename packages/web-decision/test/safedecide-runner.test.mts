import test from 'node:test';
import assert from 'node:assert/strict';
import {createSafeDecideRunner} from '../src/safedecide/runner.mjs';

type TensorData = BigInt64Array | Float32Array;

interface TensorLike {
  dispose?(): void;
}

class Tensor {
  declare type: string;
  declare data: TensorData;
  declare dims: number[];
  declare disposed: boolean;

  constructor(type: string, data: TensorData, dims: number[]) {
    this.type = type;
    this.data = data;
    this.dims = dims;
    this.disposed = false;
  }

  dispose() {
    this.disposed = true;
  }
}

interface FakeSession {
  run(feeds: Record<string, TensorLike>, outputNames: string[]): Promise<unknown>;
  release(): Promise<void>;
}

interface RecordedCall {
  feeds: Record<string, Tensor>;
  outputNames: string[];
}

function fakeRuntime() {
  const calls: RecordedCall[] = [];
  const releases: boolean[] = [];
  return {
    calls,
    releases,
    ort: {Tensor},
    session: {
      async run(feeds: Record<string, TensorLike>, outputNames: string[]): Promise<unknown> {
        calls.push({feeds, outputNames} as unknown as RecordedCall);
        const [rows, width] = (feeds.input_ids as Tensor).dims;
        const logits = Float32Array.from({length: rows!}, (_, index) =>
          Number((feeds.input_ids as Tensor).data[index * width!])
        );
        return {logits: new Tensor('float32', logits, [rows!, 1])};
      },
      async release() {
        releases.push(true);
      }
    }
  };
}

test('SafeDecide batches bounded option pairs and right pads with the checkpoint pad id', async () => {
  const fake = fakeRuntime();
  const runner = createSafeDecideRunner(fake.ort, fake.session, 0, 2);
  const result = await runner.run([{inputIds: [4, 5]}, {inputIds: [6]}, {inputIds: [7, 8, 9]}]);

  assert.deepEqual(result, [4, 6, 7]);
  assert.deepEqual(
    fake.calls.map(({outputNames}) => outputNames),
    [['logits'], ['logits']]
  );
  assert.deepEqual(Array.from(fake.calls[0]!.feeds.input_ids!.data as ArrayLike<number | bigint>), [4n, 5n, 6n, 0n]);
  assert.deepEqual(Array.from(fake.calls[0]!.feeds.attention_mask!.data as ArrayLike<number | bigint>), [
    1n,
    1n,
    1n,
    0n
  ]);
  assert.deepEqual(Array.from(fake.calls[1]!.feeds.input_ids!.data as ArrayLike<number | bigint>), [7n, 8n, 9n]);
  for (const call of fake.calls) for (const tensor of Object.values(call.feeds)) assert.equal(tensor.disposed, true);
});

test('SafeDecide runs more than 512 pairs through eight-row chunks', async () => {
  const fake = fakeRuntime();
  const runner = createSafeDecideRunner(fake.ort, fake.session, 0);
  const rows = Array.from({length: 513}, (_, index) => ({inputIds: [index]}));
  const logits = await runner.run(rows);

  assert.deepEqual(
    logits,
    Array.from({length: 513}, (_, index) => index)
  );
  assert.equal(fake.calls.length, 65);
  assert.deepEqual(
    fake.calls.map(({feeds}) => feeds.input_ids!.dims[0]),
    [...Array(64).fill(8), 1]
  );
  await runner.dispose();
});

test('SafeDecide stops between chunks after abort and disposes the completed chunk tensors', async () => {
  const fake = fakeRuntime();
  const controller = new AbortController();
  const run = fake.session.run.bind(fake.session);
  fake.session.run = async (...args: Parameters<FakeSession['run']>) => {
    const result = await run(...args);
    controller.abort();
    return result;
  };
  const runner = createSafeDecideRunner(fake.ort, fake.session, 0, 2);
  await assert.rejects(runner.run([{inputIds: [1]}, {inputIds: [2]}, {inputIds: [3]}], controller.signal), {
    name: 'AbortError'
  });
  assert.equal(fake.calls.length, 1);
  for (const tensor of Object.values(fake.calls[0]!.feeds)) assert.equal(tensor.disposed, true);
  await runner.dispose();
});

test('SafeDecide releases the runtime once and rejects calls after disposal', async () => {
  const fake = fakeRuntime();
  const runner = createSafeDecideRunner(fake.ort, fake.session, 0, 2);
  await runner.dispose();
  await runner.dispose();
  assert.equal(fake.releases.length, 1);
  await assert.rejects(runner.run([{inputIds: [1]}]), {code: 'SESSION_CLOSED'});
});

test('SafeDecide rejects malformed ONNX output shapes and non-finite logits', async () => {
  const wrong = createSafeDecideRunner(
    {Tensor},
    {
      async run() {
        return {logits: new Tensor('float32', Float32Array.from([1, 2]), [1, 2])};
      },
      async release() {}
    },
    0
  );
  await assert.rejects(wrong.run([{inputIds: [1]}]), {code: 'INFERENCE_ERROR'});

  const nan = createSafeDecideRunner(
    {Tensor},
    {
      async run() {
        return {logits: new Tensor('float32', Float32Array.from([Number.NaN]), [1, 1])};
      },
      async release() {}
    },
    0
  );
  await assert.rejects(nan.run([{inputIds: [1]}]), {code: 'INFERENCE_ERROR'});
});
