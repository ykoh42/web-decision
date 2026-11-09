import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test from 'node:test';
import {uploadFusedRecord} from '../src/strands/fused-upload.mjs';

interface TestRangeSpec {
  readonly path: string;
  readonly totalBytes: number;
  readonly offset: number;
  readonly length: number;
  readonly sha256: string;
}

interface TestRangeReadResult {
  readonly bytes: ArrayBuffer;
  readonly cachedByHash: boolean;
}

const BASE = new URL('https://example.test/strands/');
const POINTER = {allocation: 'gpu-buffer'};

function makeRecord(): {
  readonly sourcePath: string;
  readonly sourceOffset: number;
  readonly sourceFileBytes: number;
  readonly bytes: number;
  readonly chunks: readonly {readonly offset: number; readonly bytes: number; readonly sha256: string}[];
} {
  return {
    sourcePath: 'weights.bin',
    sourceOffset: 4,
    sourceFileBytes: 12,
    bytes: 8,
    chunks: [
      {offset: 0, bytes: 4, sha256: '1'.repeat(64)},
      {offset: 4, bytes: 4, sha256: '2'.repeat(64)}
    ]
  };
}

function result(bytes: Uint8Array): TestRangeReadResult {
  return {bytes: bytes.slice().buffer, cachedByHash: true};
}

test('reads and uploads chunks in order inside the awaited GPU write scope', async () => {
  const events: string[] = [];
  const record = makeRecord();
  const signal = new AbortController().signal;
  const written: Uint8Array[] = [];
  const readOffsets: number[] = [];
  const readRange = async (
    base: URL,
    spec: TestRangeSpec,
    options: {readonly signal?: AbortSignal | null | undefined; readonly offline?: boolean | undefined}
  ): Promise<TestRangeReadResult> => {
    events.push(`read:${spec.offset}`);
    readOffsets.push(spec.offset);
    assert.equal(base, BASE);
    assert.equal(spec.path, record.sourcePath);
    assert.equal(spec.totalBytes, record.sourceFileBytes);
    assert.equal(spec.length, 4);
    assert.equal(options.signal, signal);
    return result(spec.offset === 4 ? Uint8Array.of(1, 2, 3, 4) : Uint8Array.of(5, 6, 7, 8));
  };
  const context = {
    copyRawBytesToBuffer(data: Uint8Array, pointer: unknown, byteOffset: number, byteLength: number) {
      events.push(`copy:${byteOffset}:${byteLength}`);
      assert.equal(pointer, POINTER);
      written.push(data.slice());
    }
  };

  await uploadFusedRecord(
    BASE,
    record,
    context,
    POINTER,
    8,
    async () => {
      events.push('sync');
    },
    () => events.push('check'),
    async (work) => {
      events.push('scope.open');
      await work();
      events.push('scope.close');
    },
    {signal, readRange}
  );

  assert.deepEqual(events, [
    'check',
    'read:4',
    'check',
    'scope.open',
    'copy:8:4',
    'sync',
    'scope.close',
    'check',
    'check',
    'read:8',
    'check',
    'scope.open',
    'copy:12:4',
    'sync',
    'scope.close',
    'check'
  ]);
  assert.deepEqual(
    written.map((part) => [...part]),
    [
      [1, 2, 3, 4],
      [5, 6, 7, 8]
    ]
  );
  assert.deepEqual(readOffsets, [4, 8]);
});

test('rejects invalid metadata before checking an already-aborted signal', async () => {
  const controller = new AbortController();
  const reason = new Error('already cancelled');
  controller.abort(reason);
  const events: string[] = [];
  const invalidRecord = {...makeRecord(), sourceOffset: 2};

  await assert.rejects(
    uploadFusedRecord(
      BASE,
      invalidRecord,
      {
        copyRawBytesToBuffer() {
          events.push('copy');
        }
      },
      POINTER,
      0,
      async () => events.push('sync'),
      () => events.push('check'),
      undefined,
      {signal: controller.signal}
    ),
    {message: 'Invalid fused tensor range metadata'}
  );
  assert.deepEqual(events, []);
});

test('preserves cancellation after a read and skips GPU upload', async () => {
  const controller = new AbortController();
  const reason = new Error('cancel after range read');
  const events: string[] = [];

  await assert.rejects(
    uploadFusedRecord(
      BASE,
      makeRecord(),
      {
        copyRawBytesToBuffer() {
          events.push('copy');
        }
      },
      POINTER,
      0,
      async () => events.push('sync'),
      () => events.push('check'),
      async (work) => {
        events.push('scope');
        await work();
      },
      {
        signal: controller.signal,
        async readRange() {
          events.push('read');
          controller.abort(reason);
          return result(Uint8Array.of(1, 2, 3, 4));
        }
      }
    ),
    (error: unknown) => error === reason
  );
  assert.deepEqual(events, ['check', 'read']);
});

test('propagates a GPU sync failure through the write scope without starting the next chunk', async () => {
  const events: string[] = [];
  const failure = new Error('GPU synchronization failed');

  await assert.rejects(
    uploadFusedRecord(
      BASE,
      makeRecord(),
      {
        copyRawBytesToBuffer() {
          events.push('copy');
        }
      },
      POINTER,
      0,
      async () => {
        events.push('sync');
        throw failure;
      },
      () => events.push('check'),
      async (work) => {
        events.push('scope.open');
        try {
          await work();
        } catch (error) {
          events.push('scope.failure');
          throw error;
        } finally {
          events.push('scope.close');
        }
      },
      {
        async readRange(_base, spec) {
          events.push(`read:${spec.offset}`);
          return result(Uint8Array.of(1, 2, 3, 4));
        }
      }
    ),
    (error: unknown) => error === failure
  );
  assert.deepEqual(events, ['check', 'read:4', 'check', 'scope.open', 'copy', 'sync', 'scope.failure', 'scope.close']);
});

test('uses the verified range reader by default', async (t) => {
  const events: string[] = [];
  const data = Uint8Array.of(9, 8, 7, 6);
  const sha256 = createHash('sha256').update(data).digest('hex');
  const record = {
    sourcePath: 'default.bin',
    sourceOffset: 0,
    sourceFileBytes: 4,
    bytes: 4,
    chunks: [{offset: 0, bytes: 4, sha256}]
  };
  t.mock.method(globalThis, 'fetch', async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    events.push('fetch');
    assert.equal(String(input), new URL(record.sourcePath, BASE).href);
    assert.equal(new Headers(init?.headers).get('range'), 'bytes=0-3');
    return new Response(data, {
      status: 206,
      headers: {'content-range': 'bytes 0-3/4', 'content-length': '4'}
    });
  });
  await uploadFusedRecord(
    BASE,
    record,
    {
      copyRawBytesToBuffer(bytes) {
        events.push(`copy:${[...bytes].join(',')}`);
      }
    },
    POINTER,
    0,
    async () => events.push('sync'),
    () => events.push('check')
  );
  assert.deepEqual(events, ['check', 'fetch', 'check', 'copy:9,8,7,6', 'sync', 'check']);
});
