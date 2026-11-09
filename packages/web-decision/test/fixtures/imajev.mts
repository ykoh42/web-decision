// Small synthetic tokenizer, metadata, score rows, and CPU-backed runtime.
// Used only by isolated source/installed tests; this is not a model loader.
import assert from 'node:assert/strict';

export function fixture<T>(
  createSession: (options: Record<string, unknown>) => T,
  {
    syncGate,
    syncStarted,
    rows: injectedRows
  }: {syncGate?: PromiseLike<unknown>; syncStarted?: () => void; rows?: Float32Array[]} = {}
) {
  const codes = Array.from({length: 256}, (_, i) => ({code: `C${String(i).padStart(3, '0')}`, token_id: 1000 + i}));
  const metadata = {version: 1, codes},
    calibration = {schema_version: '1.0', calibration_version: 'fake', temperatures: {}};
  const tokenizer = {
    encode(text: string) {
      if (text.startsWith('<|im_start|>user\n')) {
        const code = codes.find((item) => text.endsWith(item.code));
        return {ids: code ? [101, 102, code.token_id] : [101, 102]};
      }
      return {ids: [25]};
    }
  };
  const lifecycle: {loss: unknown; disposed: boolean} = {loss: null, disposed: false},
    events: string[] = [],
    scratch = new Float32Array(256);
  const rows = injectedRows ?? [new Float32Array(256), new Float32Array(256)];
  if (!injectedRows) {
    rows[0]!.set([1, 0, 3]);
    rows[1]!.set([2, 1, -2]);
  }
  let graphCalls = 0;
  class Tensor {
    declare shape: number[];
    declare dtype: string;
    declare device: string;
    declare byteOffset: number;
    declare value: unknown;
    constructor(shape: number[], dtype: string, device: string) {
      this.shape = shape;
      this.dtype = dtype;
      this.device = device;
      this.byteOffset = 0;
    }
    getDataPtr() {
      return this;
    }
    copyFrom(value: {scores?: unknown} | null) {
      this.value = value?.scores ?? value;
      if (this.device === 'cpu') runtime.lib.webGPUContext.pendingGPUToCPUCopy = Promise.resolve();
      return this;
    }
    toArray() {
      return this.value;
    }
  }
  const runtime = {
    beginScope() {
      events.push('begin');
    },
    endScope() {
      events.push('end');
    },
    dispose() {
      events.push('dispose');
      lifecycle.disposed = true;
    },
    empty(shape: number[], dtype: string, device: string) {
      return new Tensor(shape, dtype, device);
    },
    cpu() {
      return 'cpu';
    },
    getGlobalFunc() {
      return () => events.push('clear');
    },
    lib: {
      webGPUContext: {
        pendingGPUToCPUCopy: null as Promise<void> | null,
        copyRawBytesToBuffer(data: Uint8Array, target: {raw: Uint8Array}, offset: number, size: number) {
          assert.equal(offset, 0);
          assert.equal(size % 4, 0);
          target.raw = Uint8Array.from(data);
        }
      }
    }
  };
  const device = {
    async sync() {
      events.push('sync');
      const pending = runtime.lib.webGPUContext.pendingGPUToCPUCopy;
      runtime.lib.webGPUContext.pendingGPUToCPUCopy = null;
      syncStarted?.();
      await syncGate;
      await pending;
    }
  };
  const gpu = {
    pushErrorScope() {},
    async popErrorScope() {
      return null;
    }
  };
  const weights = ['fake'];
  function fn(
    ids: {value: number[]},
    positions: {value: number[]},
    mask: {shape: number[]},
    valid: {shape: number[]},
    last: {value: number[]},
    ...passedWeights: string[]
  ) {
    assert.deepEqual(Array.from(ids.value), [101, 102]);
    assert.deepEqual(Array.from(positions.value), [0, 1]);
    assert.deepEqual(mask.shape, [1, 1, 2, 2]);
    assert.deepEqual(valid.shape, [1, 2]);
    assert.deepEqual(Array.from(last.value), [1]);
    assert.deepEqual(passedWeights, weights);
    scratch.set(rows[graphCalls++ % rows.length]!);
    return {dtype: 'float32', shape: [1, 256], scores: scratch};
  }
  const model = createSession({
    tokenizer,
    manifest: metadata,
    calibration,
    maxInputTokens: 128,
    runtime,
    device,
    gpu,
    fn,
    weights,
    uncaptured: [],
    lifecycle,
    disposalGuard: {blocked: false, retain() {}}
  }) as Omit<T, 'runtime' | 'device' | 'lifecycle' | 'disposalGuard'> & {
    runtime: typeof runtime;
    device: typeof device;
    lifecycle: typeof lifecycle;
    disposalGuard: {retain(): void};
  };
  return {
    model,
    events,
    calls: () => graphCalls,
    request: {
      state: 'fake',
      questions: [
        {instructions: 'First?', options: ['red', 'blue']},
        {instructions: 'Second?', options: ['yes', 'no']}
      ]
    }
  };
}
