import {WebDecisionError} from './kev.js';

/** Forward caller cancellation and GPU loss while model assets are loading. */
export function watchLoadDeviceLoss(device, callerSignal) {
  const controller = new AbortController();
  let closed = false;
  const forwardAbort = () => controller.abort(callerSignal.reason);
  const onLoss = (cause) => {
    if (!closed)
      controller.abort(
        new WebDecisionError('MODEL_LOAD_FAILED', 'The GPU device was lost while loading the model.', {cause})
      );
  };
  // A normal session disposal also resolves device.lost. close() ensures that
  // it cannot cancel an already completed load or retain the caller listener.
  try {
    callerSignal?.addEventListener('abort', forwardAbort, {once: true});
    if (callerSignal?.aborted) forwardAbort();
    device.lost.then(onLoss, onLoss);
  } catch (cause) {
    closed = true;
    try {
      callerSignal?.removeEventListener('abort', forwardAbort);
    } catch {}
    throw cause;
  }
  return {
    signal: controller.signal,
    close() {
      if (closed) return;
      closed = true;
      callerSignal?.removeEventListener('abort', forwardAbort);
    }
  };
}
