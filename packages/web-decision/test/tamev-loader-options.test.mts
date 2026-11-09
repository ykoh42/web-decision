import assert from 'node:assert/strict';
import test from 'node:test';
import {TAMEV_MICRO_MODEL, TAMEV_MICRO_PROFILE} from '../src/tamev/contract.mjs';
import {WebDecisionError} from '../src/kev.js';
import {validateTamevLoadOptions} from '../src/tamev/loader.mjs';

test('TAMEV loader snapshots option values once before checking signal methods in order', () => {
  const events: string[] = [];
  const signal = new Proxy(
    {},
    {
      get(_target, key) {
        events.push(`signal:${String(key)}`);
        return () => {};
      }
    }
  );
  const target = {
    model: TAMEV_MICRO_MODEL,
    assetsUrl: 'https://example.test/tamev/',
    signal
  };
  const options = new Proxy(target, {
    ownKeys(object) {
      events.push('ownKeys');
      return Reflect.ownKeys(object);
    },
    getOwnPropertyDescriptor(object, key) {
      events.push(`descriptor:${String(key)}`);
      return Reflect.getOwnPropertyDescriptor(object, key);
    },
    get(object, key, receiver) {
      events.push(`value:${String(key)}`);
      return Reflect.get(object, key, receiver);
    }
  });

  const checked = validateTamevLoadOptions(options, TAMEV_MICRO_PROFILE);

  assert.equal(checked.base.href, 'https://example.test/tamev/');
  assert.deepEqual(events, [
    'ownKeys',
    'descriptor:model',
    'descriptor:assetsUrl',
    'descriptor:signal',
    'ownKeys',
    'value:model',
    'value:assetsUrl',
    'value:signal',
    'signal:throwIfAborted',
    'signal:addEventListener',
    'signal:removeEventListener'
  ]);
});

test('TAMEV loader rejects accessors before reading their value or snapshotting options', () => {
  let getterReads = 0;
  let ownKeysReads = 0;
  const options = new Proxy(
    Object.defineProperties(
      {},
      {
        model: {configurable: true, enumerable: true, value: TAMEV_MICRO_MODEL},
        assetsUrl: {
          configurable: true,
          enumerable: true,
          get() {
            getterReads++;
            throw new Error('option getter invoked');
          }
        }
      }
    ),
    {
      ownKeys(object) {
        ownKeysReads++;
        return Reflect.ownKeys(object);
      }
    }
  );

  assert.throws(() => validateTamevLoadOptions(options, TAMEV_MICRO_PROFILE), {code: 'INVALID_INPUT'});
  assert.equal(ownKeysReads, 1);
  assert.equal(getterReads, 0);
});

test('TAMEV loader preserves signal getter exception identity and short-circuits later methods', () => {
  const signalReads: string[] = [];
  const failure = new Error('signal method getter failed');
  const signal = new Proxy(
    {},
    {
      get(_target, key) {
        signalReads.push(String(key));
        if (key === 'throwIfAborted') throw failure;
        return () => {};
      }
    }
  );
  const options = {model: TAMEV_MICRO_MODEL, assetsUrl: 'https://example.test/tamev/', signal};

  assert.throws(
    () => validateTamevLoadOptions(options, TAMEV_MICRO_PROFILE),
    (error: unknown) => error === failure
  );
  assert.deepEqual(signalReads, ['throwIfAborted']);
});

test('TAMEV loader preserves malformed URL error code, message, and cause type', () => {
  assert.throws(
    () => validateTamevLoadOptions({model: TAMEV_MICRO_MODEL, assetsUrl: 'https://['}, TAMEV_MICRO_PROFILE),
    (error: unknown) => {
      assert.ok(error instanceof WebDecisionError);
      assert.equal(error.code, 'INVALID_INPUT');
      assert.equal(error.message, 'Invalid TAMEV Micro assetsUrl.');
      assert.ok(error.cause instanceof TypeError);
      return true;
    }
  );
});
