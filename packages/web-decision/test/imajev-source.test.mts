import assert from 'node:assert/strict';
import {getEventListeners} from 'node:events';
import {readFile} from 'node:fs/promises';
import {setImmediate as nextTurn} from 'node:timers/promises';
import test from 'node:test';
import {prepareImajevPlan} from '../src/imajev/plan.mjs';
import {openImajevSource} from '../src/imajev/source.mjs';

type ImajevMetadata = Parameters<typeof prepareImajevPlan>[0];
type ImajevBinding = ImajevMetadata['bindings'][number];
type ImajevFileBinding = Extract<ImajevBinding, {path: string}>;
type ImajevReader = NonNullable<NonNullable<Parameters<typeof openImajevSource>[1]>['reader']>;

const metadata = await readFile(new URL('./fixtures/imajev/metadata.json', import.meta.url));
const calibration = await readFile(new URL('./fixtures/imajev/calibration.json', import.meta.url));
const copy = (bytes: Uint8Array) => Uint8Array.from(bytes).buffer as ArrayBuffer;
const base = 'https://example.test/imajev/';

function reader(_base: URL, entry: Parameters<ImajevReader>[1]): ReturnType<ImajevReader> {
  if (entry.path === 'imajev-metadata.json') return Promise.resolve({bytes: copy(metadata)});
  if (entry.path === 'calibration.json') return Promise.resolve({bytes: copy(calibration)});
  return Promise.resolve({bytes: new ArrayBuffer(entry.length)});
}

test('pinned source reads an allowed asset and rejects unlisted or corrupt ranges', async () => {
  const source = await openImajevSource(base, {reader});
  assert.equal(source.plan.bindings.length, 1082);
  assert.equal(source.plan.maxInputTokens, 128);
  assert.ok(Object.isFrozen((source.plan.bindings[0]! as ImajevFileBinding).chunks));
  assert.deepEqual(await source.readAsset('calibration.json'), Uint8Array.from(calibration));
  await assert.rejects(source.readRange('calibration.json', 0, 1), {code: 'INVALID_ASSET'});
  const row = source.plan.bindings[0]! as ImajevFileBinding,
    chunk = row.chunks[0]!;
  await assert.rejects(source.readRange(row.path, chunk.offset, chunk.bytes), {code: 'INVALID_ASSET'});
  source.close();
  await assert.rejects(source.readAsset('calibration.json'), {code: 'INVALID_ASSET'});
});

test('rejects altered metadata and unsafe URLs before exposing any asset ranges', async () => {
  const altered = Uint8Array.from(metadata);
  altered[0]! ^= 1;
  await assert.rejects(openImajevSource(base, {reader: async () => ({bytes: altered.buffer})}), {
    code: 'INVALID_ASSET'
  });
  for (const url of ['file:///tmp/', 'https://user:pass@example.test/', 'https://example.test/?q=1'])
    await assert.rejects(
      openImajevSource(url, {
        reader: () => {
          throw Error('unexpected read');
        }
      }),
      {code: 'INVALID_ASSET'}
    );
});

test('metadata validation rejects duplicate chunks, changed sizes and rotary bindings', () => {
  const parse = (): ImajevMetadata => JSON.parse(metadata as unknown as string) as ImajevMetadata;
  const duplicate = parse();
  (duplicate.bindings[1]! as ImajevFileBinding).path = (duplicate.bindings[0]! as ImajevFileBinding).path;
  (duplicate.bindings[1]! as ImajevFileBinding).fileBytes = (duplicate.bindings[0]! as ImajevFileBinding).fileBytes;
  (duplicate.bindings[1]! as ImajevFileBinding).fileSha256 = (duplicate.bindings[0]! as ImajevFileBinding).fileSha256;
  (duplicate.bindings[1]! as ImajevFileBinding).offset = (duplicate.bindings[0]! as ImajevFileBinding).offset;
  (duplicate.bindings[1]! as ImajevFileBinding).chunks = (duplicate.bindings[0]! as ImajevFileBinding).chunks;
  assert.throws(() => prepareImajevPlan(duplicate));
  const wrongSize = parse();
  (wrongSize.bindings[0]! as ImajevFileBinding).bytes++;
  assert.throws(() => prepareImajevPlan(wrongSize));
  const wrongRotary = parse();
  wrongRotary.bindings[1080]!.target = 'wrong';
  assert.throws(() => prepareImajevPlan(wrongRotary));
});

test('closing the source aborts an outstanding range read', async () => {
  let entered!: () => void;
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  const source = await openImajevSource(base, {
    reader: async (_base, entry, options) => {
      if (entry.path === 'imajev-metadata.json') return {bytes: copy(metadata)};
      return new Promise((_resolve, reject) => {
        options.signal!.addEventListener('abort', () => reject(options.signal!.reason), {once: true});
        entered();
      });
    }
  });
  const pending = source.readAsset('calibration.json');
  await started;
  source.close();
  await assert.rejects(pending, {name: 'AbortError'});
});

function deferred<T = never>() {
  let resolve!: (value: T | PromiseLike<T>) => void, reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((success, failure) => {
    resolve = success;
    reject = failure;
  });
  return {promise, resolve, reject};
}

test('metadata reader and digest cancellation preserve the caller reason', async (t) => {
  for (const stage of ['reader', 'digest']) {
    const entered = deferred<void>();
    const delayed = deferred();
    if (stage === 'digest')
      t.mock.method(crypto.subtle, 'digest', () => {
        entered.resolve();
        return delayed.promise;
      });
    const controller = new AbortController();
    const reason = Error(`cancel metadata ${stage}`);
    const pending = openImajevSource(base, {
      signal: controller.signal,
      reader: (() => {
        if (stage === 'reader') {
          entered.resolve();
          return delayed.promise;
        }
        return {bytes: copy(metadata)};
      }) as ImajevReader
    });
    await entered.promise;
    controller.abort(reason);
    await assert.rejects(pending, (error) => error === reason);
    delayed.reject(Error('late metadata failure'));
    await nextTurn();
    t.mock.restoreAll();
  }
});

test('range reader and asset digests stop on operation abort or source close', async (t) => {
  for (const stage of ['reader', 'range-digest', 'asset-digest', 'asset-digest-close']) {
    const entered = deferred<void>();
    const delayed = deferred();
    const original = crypto.subtle.digest.bind(crypto.subtle);
    const source = await openImajevSource(base, {
      reader: ((_base: URL, entry: Parameters<ImajevReader>[1]) => {
        if (entry.path === 'imajev-metadata.json') return {bytes: copy(metadata)};
        if (stage === 'reader') {
          entered.resolve();
          return delayed.promise;
        }
        return {bytes: copy(calibration)};
      }) as ImajevReader
    });
    if (stage !== 'reader') {
      let calls = 0;
      t.mock.method(crypto.subtle, 'digest', (...args: Parameters<SubtleCrypto['digest']>) => {
        if (++calls === 1 && stage.startsWith('asset')) return original(...args);
        entered.resolve();
        return delayed.promise;
      });
    }
    const controller = new AbortController();
    const reason = Error(`cancel ${stage}`);
    const pending = source.readAsset('calibration.json', {signal: controller.signal});
    await entered.promise;
    if (stage.endsWith('close')) source.close();
    else controller.abort(reason);
    await assert.rejects(pending, (error) =>
      stage.endsWith('close') ? (error as {name?: unknown}).name === 'AbortError' : error === reason
    );
    delayed.reject(Error('late asset failure'));
    await nextTurn();
    source.close();
    t.mock.restoreAll();
  }
});

test('operation abort detaches the full-hash source listener before digest settles', async (t) => {
  const NativeController = AbortController;
  const controllers: AbortController[] = [];
  t.mock.method(
    globalThis,
    'AbortController',
    class extends NativeController {
      constructor() {
        super();
        controllers.push(this);
      }
    }
  );
  const source = await openImajevSource(base, {reader});
  const sourceSignal = controllers[0]!.signal;
  const original = crypto.subtle.digest.bind(crypto.subtle);
  const entered = deferred<void>();
  const delayed = deferred();
  let calls = 0;
  t.mock.method(crypto.subtle, 'digest', (...args: Parameters<SubtleCrypto['digest']>) => {
    if (++calls === 1) return original(...args);
    entered.resolve();
    return delayed.promise;
  });
  const controller = new NativeController();
  const reason = Error('cancel operation');
  const pending = source.readAsset('calibration.json', {signal: controller.signal});
  await entered.promise;
  assert.equal(getEventListeners(sourceSignal, 'abort').length, 1);
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
  assert.equal(getEventListeners(sourceSignal, 'abort').length, 0);
  assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
  delayed.reject(Error('late digest failure'));
  await nextTurn();
  source.close();
});
