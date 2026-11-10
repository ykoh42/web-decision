import test from 'node:test';
import assert from 'node:assert/strict';
import {WebDecisionError} from '../src/curated/errors.ts';
import {mapKananaTvmResourceError} from '../src/kanana/tvm-resource-error.ts';

test('maps a structured TVM allocation failure to a public device resource limit', () => {
  const tvmError = Object.assign(new Error('GPU allocation failed'), {
    code: 'GPU_OUT_OF_MEMORY',
    errorScope: 'out-of-memory',
    size: 656_670_720
  });

  const result = mapKananaTvmResourceError(tvmError);

  assert.ok(result instanceof WebDecisionError);
  assert.equal(result.code, 'DEVICE_RESOURCE_LIMIT');
  assert.match(result.message, /656670720-byte WebGPU buffer/);
  assert.equal(result.cause, tvmError);
});

test('finds TVM allocation failures nested in cleanup errors', () => {
  const tvmError = Object.assign(new Error('GPU allocation failed'), {code: 'GPU_OUT_OF_MEMORY', size: 4096});
  const cause = new AggregateError([new Error('cleanup failed'), tvmError], 'inference and cleanup failed.');

  const result = mapKananaTvmResourceError(cause);

  assert.ok(result instanceof WebDecisionError);
  assert.equal(result.code, 'DEVICE_RESOURCE_LIMIT');
  assert.equal(result.cause, cause);
});

test('leaves unrelated TVM errors unchanged', () => {
  const cause = Object.assign(new Error('shader validation failed'), {code: 'GPU_VALIDATION_ERROR'});

  assert.equal(mapKananaTvmResourceError(cause), cause);
});

test('does not recurse forever through cyclic cause chains', () => {
  const cause: {code?: string; cause?: unknown} = {code: 'GPU_VALIDATION_ERROR'};
  cause.cause = cause;

  assert.equal(mapKananaTvmResourceError(cause), cause);
});
