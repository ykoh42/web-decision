import {WebDecisionError} from '../kev.js';
import {checkGpuDevice} from '../gpu-session-utils.js';
import {retainStrandsResources} from '../strands/disposal-guard.mjs';
import {assertUsable, withStrandsScope} from '../strands/scope.mjs';
import {OPENDECIDER_NANO_MAX_TOKENS, OPENDECIDER_NANO_PARAMETER_COUNT} from './contract.mjs';
import {readOpenDeciderNanoMarkerScores} from './protocol.mjs';

const retained = new Set();
const shapeEq = (shape, expected) =>
  Array.isArray(shape) && shape.length === expected.length && expected.every((value, index) => shape[index] === value);
const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);

export class OpenDeciderNanoTvmRunner {
  constructor(owner) {
    if (
      !owner?.runtime ||
      !owner.device ||
      !owner.gpu ||
      typeof owner.fn !== 'function' ||
      !Array.isArray(owner.weights) ||
      owner.weights.length !== OPENDECIDER_NANO_PARAMETER_COUNT ||
      Array.from(
        {length: OPENDECIDER_NANO_PARAMETER_COUNT},
        (_, index) => !Object.hasOwn(owner.weights, index) || !owner.weights[index]
      ).some(Boolean) ||
      !owner.lifecycle ||
      !owner.disposalGuard ||
      !Array.isArray(owner.uncaptured) ||
      typeof owner.teardownObserver?.beforeDispose !== 'function' ||
      typeof owner.teardownObserver?.afterDispose !== 'function'
    )
      throw new TypeError('An owned OpenDecider-nano TVM runtime is required.');
    Object.assign(this, owner, {weights: [...owner.weights], busy: false, closed: false, disposal: null});
  }

  async exclusive(operation) {
    assertUsable(this);
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    if (this.busy) throw invalid('Concurrent direct runner use is not supported.');
    this.busy = true;
    try {
      return await operation();
    } finally {
      this.busy = false;
    }
  }

  async run(ids, markerPositions) {
    if (
      !Array.isArray(ids) ||
      ids.length < 1 ||
      ids.length > OPENDECIDER_NANO_MAX_TOKENS ||
      ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 50368)
    )
      throw invalid('Invalid OpenDecider-nano input IDs.');
    const inputIds = Int32Array.from(ids);
    const attentionMask = new Int32Array(ids.length).fill(1);
    return this.exclusive(() =>
      withStrandsScope(this, 'OpenDecider-nano inference failed.', async (lease) => {
        const idsTensor = this.runtime.empty([1, ids.length], 'int32', this.device).copyFrom(inputIds);
        const maskTensor = this.runtime.empty([1, ids.length], 'int32', this.device).copyFrom(attentionMask);
        const output = this.fn(idsTensor, maskTensor, ...this.weights);
        if (output?.dtype !== 'float32' || !shapeEq(output.shape, [1, ids.length]))
          throw new Error('Unexpected OpenDecider-nano graph output.');
        const host = this.runtime.empty([1, ids.length], 'float32', this.runtime.cpu());
        lease.prepareReadback();
        try {
          host.copyFrom(output);
        } finally {
          lease.captureReadback();
        }
        await lease.sync();
        checkGpuDevice(this);
        const logits = host.toArray();
        if (!(logits instanceof Float32Array) || logits.length !== ids.length || !logits.every(Number.isFinite))
          throw new Error('Invalid OpenDecider-nano score readback.');
        return readOpenDeciderNanoMarkerScores(ids, markerPositions, logits);
      })
    );
  }

  async releaseCachedMemory() {
    return this.exclusive(() =>
      withStrandsScope(this, 'Could not release OpenDecider-nano cached memory.', async (lease) => {
        this.runtime.getGlobalFunc('vm.builtin.memory_manager.clear')();
        await lease.syncWithoutReadback();
      })
    );
  }

  dispose() {
    if (this.disposal) return this.disposal;
    if (this.busy) return Promise.reject(invalid('Cannot dispose while direct runner work is active.'));
    this.closed = true;
    this.disposal = Promise.resolve().then(async () => {
      try {
        await this.teardownObserver.beforeDispose();
        assertUsable(this);
        if (this.lifecycle.disposed || this.lifecycle.loss || this.uncaptured.length)
          throw new Error('GPU device failed before OpenDecider-nano disposal.');
        this.runtime.endScope();
        await this.teardownObserver.beforeDispose();
        if (this.lifecycle.disposed || this.lifecycle.loss || this.uncaptured.length)
          throw new Error('GPU device failed during OpenDecider-nano disposal.');
        this.runtime.dispose();
        await this.teardownObserver.afterDispose(this.teardownTimeoutMs);
        if (this.lifecycle.loss || this.disposalGuard.blocked || this.uncaptured.length || this.poisoned)
          throw new Error('GPU device failed during OpenDecider-nano teardown.');
        this.gpuCompatibility?.restore();
        this.weights = [];
        this.runtime = this.device = this.gpu = this.fn = this.lifecycle = this.teardownObserver = null;
        this.gpuCompatibility = null;
        this.uncaptured = [];
      } catch (cause) {
        this.poisoned = true;
        retained.add(this);
        try {
          this.disposalGuard?.retain();
        } catch {
          retainStrandsResources(this.runtime, this.gpu);
        }
        if (cause instanceof WebDecisionError) throw cause;
        throw new WebDecisionError('INFERENCE_ERROR', 'Could not safely dispose OpenDecider-nano runtime.', {cause});
      }
    });
    return this.disposal;
  }
}
