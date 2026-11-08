// H2O observes the device before TVM installs its own loss callback. TVM's
// dispose stays synchronous; only the enclosing H2O runner waits for loss.
export const H2O_TEARDOWN_TIMEOUT_MS = 5000;

export function observeH2oTeardown(gpu) {
  const lost = gpu.lost;
  let outcome;
  const settled = Promise.prototype.then.call(
    lost,
    (info) => {
      try {
        outcome = {reason: info?.reason};
      } catch {
        outcome = {};
      }
      return outcome;
    },
    (cause) => {
      outcome = {rejected: true, cause};
      return outcome;
    }
  );
  return {
    async beforeDispose() {
      // Observe promise reactions that were already queued before destruction.
      await Promise.resolve();
      if (outcome) throw new Error('GPU device was lost before disposal.');
    },
    async afterDispose(timeoutMs = H2O_TEARDOWN_TIMEOUT_MS) {
      let timer;
      try {
        const result = await Promise.race([
          settled,
          new Promise((_, reject) => {
            timer = setTimeout(() => reject(new Error('GPU device teardown did not settle.')), timeoutMs);
          })
        ]);
        if (result.rejected || result.reason !== 'destroyed')
          throw new Error(
            'Unexpected GPU device loss during teardown.',
            result.rejected ? {cause: result.cause} : undefined
          );
      } finally {
        clearTimeout(timer);
      }
    }
  };
}
