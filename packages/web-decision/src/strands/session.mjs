import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {packStrandsRequest, unpackStrandsScores, validateStrandsRow} from './choice.mjs';
import {assertUsable} from './scope.mjs';

/** Independent experiment; the caller supplies an owned fused-graph runner. */
export class ExperimentalStrandsSession {
  constructor({tokenizer, runner}) {
    if (
      !tokenizer ||
      typeof runner?.run !== 'function' ||
      typeof runner?.releaseCachedMemory !== 'function' ||
      typeof runner?.dispose !== 'function'
    )
      throw new TypeError('Tokenizer and owned runner are required');
    this.tokenizer = tokenizer;
    this.runner = runner;
    this.queue = Promise.resolve();
    this.closed = false;
    this.disposal = null;
  }

  async decide(request, options = {}) {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    assertUsable(this.runner);
    const signal = options?.signal;
    signal?.throwIfAborted();
    validateDecisionFields(options);
    request = validateRequestFields(request, Infinity, Infinity, Infinity, 255);
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const packed = packStrandsRequest(this.tokenizer, request);
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    const schedule = () => {
      if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
      const task = this.queue.then(async () => {
        const scores = [];
        for (const row of packed.rows) {
          signal?.throwIfAborted();
          validateStrandsRow(row);
          // Once submitted, GPU work continues until synchronized even if the caller stops waiting.
          scores.push(await this.runner.run(row));
        }
        signal?.throwIfAborted();
        return unpackStrandsScores(scores, packed);
      });
      this.queue = task.catch(() => {});
      return task;
    };
    return signal ? waitForAbort(schedule, signal) : schedule();
  }

  async releaseCachedMemory() {
    if (this.closed) throw new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');
    assertUsable(this.runner);
    const task = this.queue.then(() => this.runner.releaseCachedMemory());
    this.queue = task.catch(() => {});
    return task;
  }

  async dispose() {
    if (this.disposal) return this.disposal;
    this.closed = true;
    this.disposal = (async () => {
      await this.queue;
      try {
        await this.runner.dispose();
      } finally {
        this.tokenizer = this.runner = null;
      }
    })();
    return this.disposal;
  }
}
