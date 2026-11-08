import assert from 'node:assert/strict';
import {test} from 'node:test';
import {watchLoadDeviceLoss} from '../src/load-device-loss.js';

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((ok, fail) => {
    resolve = ok;
    reject = fail;
  });
  return {promise, resolve, reject};
};

test('device loss aborts an active load with a model loading error', async () => {
  const lost = deferred(),
    info = {reason: 'unknown', message: 'device lost'};
  const watch = watchLoadDeviceLoss({lost: lost.promise});
  lost.resolve(info);
  await lost.promise;
  await Promise.resolve();
  assert.equal(watch.signal.aborted, true);
  assert.equal(watch.signal.reason.code, 'MODEL_LOAD_FAILED');
  assert.equal(watch.signal.reason.cause, info);
  watch.close();
});

test('preserves the caller abort reason and ignores later device loss', async () => {
  const lost = deferred(),
    controller = new AbortController(),
    reason = Error('user cancel');
  const watch = watchLoadDeviceLoss({lost: lost.promise}, controller.signal);
  controller.abort(reason);
  lost.resolve({reason: 'destroyed'});
  await lost.promise;
  await Promise.resolve();
  assert.equal(watch.signal.reason, reason);
  watch.close();
});

test('handles an already aborted caller signal', () => {
  const lost = deferred(),
    controller = new AbortController(),
    reason = Error('already canceled');
  controller.abort(reason);
  const watch = watchLoadDeviceLoss({lost: lost.promise}, controller.signal);
  assert.equal(watch.signal.reason, reason);
  watch.close();
  lost.resolve({reason: 'destroyed'});
});

test('closing the watch detaches the caller and ignores later device destruction', async () => {
  const lost = deferred(),
    controller = new AbortController();
  const watch = watchLoadDeviceLoss({lost: lost.promise}, controller.signal);
  watch.close();
  watch.close();
  controller.abort(Error('later'));
  lost.resolve({reason: 'destroyed'});
  await lost.promise;
  await Promise.resolve();
  assert.equal(watch.signal.aborted, false);
});

test('a rejected device loss promise becomes a loading error', async () => {
  const lost = deferred(),
    failure = Error('loss promise failed');
  const watch = watchLoadDeviceLoss({lost: lost.promise});
  lost.reject(failure);
  await lost.promise.catch(() => {});
  await Promise.resolve();
  assert.equal(watch.signal.reason.code, 'MODEL_LOAD_FAILED');
  assert.equal(watch.signal.reason.cause, failure);
  watch.close();
});

test('watch installation exceptions remove the caller listener and preserve the original cause', () => {
  for (const stage of ['lost-getter', 'then', 'add-listener']) {
    const controller = new AbortController(),
      reason = Error(stage);
    let adds = 0,
      removes = 0;
    const signal = controller.signal;
    const add = signal.addEventListener.bind(signal),
      remove = signal.removeEventListener.bind(signal);
    signal.addEventListener = (...args) => {
      adds++;
      add(...args);
      if (stage === 'add-listener') throw reason;
    };
    signal.removeEventListener = (...args) => {
      removes++;
      return remove(...args);
    };
    const badLost = new Proxy(
      {},
      {
        get() {
          throw reason;
        }
      }
    );
    const device =
      stage === 'lost-getter'
        ? {
            get lost() {
              throw reason;
            }
          }
        : {lost: badLost};
    assert.throws(
      () => watchLoadDeviceLoss(device, signal),
      (error) => error === reason
    );
    assert.equal(adds, 1);
    assert.equal(removes, 1);
  }
});
