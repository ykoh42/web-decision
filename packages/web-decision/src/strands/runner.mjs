import {checkGpuDevice, createCausalInputs} from '../gpu-session-utils.js';
import {assertUsable, withStrandsScope} from './scope.mjs';
import {validateStrandsRow} from './choice.mjs';

/** One owned TVM runtime/device and its already verified fused parameter list. */
export class ExperimentalFusedRunner {
  constructor({runtime, device, gpu, fn, weights, lifecycle, uncaptured, disposalGuard}) {
    if (
      !runtime ||
      !device ||
      !gpu ||
      typeof fn !== 'function' ||
      !Array.isArray(weights) ||
      weights.length !== 328 ||
      !lifecycle ||
      !Array.isArray(uncaptured)
    )
      throw new TypeError('Invalid fused runtime ownership');
    Object.assign(this, {runtime, device, gpu, fn, weights, lifecycle, uncaptured, disposalGuard});
  }

  async run(row) {
    const snapshot = validateStrandsRow(row);
    return withStrandsScope(this, 'Strands inference failed.', async (lease) => {
      const {ids, positions, mask, valid} = createCausalInputs(this, snapshot.inputIds);
      const answer = this.runtime.empty([1], 'int32', this.device).copyFrom(Int32Array.of(snapshot.answerIndex));
      const options = this.runtime
        .empty([snapshot.optionIndices.length], 'int32', this.device)
        .copyFrom(Int32Array.from(snapshot.optionIndices));
      const output = this.fn(ids, positions, mask, valid, answer, options, ...this.weights);
      const host = this.runtime.empty([1, snapshot.options.length], 'float32', this.runtime.cpu());
      lease.prepareReadback();
      try {
        host.copyFrom(output);
      } finally {
        lease.captureReadback();
      }
      await lease.sync();
      checkGpuDevice(this);
      return host.toArray();
    });
  }

  async releaseCachedMemory() {
    return withStrandsScope(this, 'Could not release Strands cached memory.', async (lease) => {
      this.runtime.getGlobalFunc('vm.builtin.memory_manager.clear')();
      await lease.syncWithoutReadback();
    });
  }

  async dispose() {
    assertUsable(this);
    try {
      if (!this.lifecycle.disposed) {
        try {
          if (!this.lifecycle.loss) this.runtime.endScope();
        } finally {
          this.runtime.dispose();
        }
      }
    } finally {
      this.weights = [];
      this.runtime = this.device = this.gpu = this.fn = this.lifecycle = null;
      this.uncaptured = [];
    }
  }
}
