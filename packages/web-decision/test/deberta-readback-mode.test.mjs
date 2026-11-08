import test from 'node:test';
import assert from 'node:assert/strict';
import {validateDebertaReadback} from '../src/deberta/readback-mode.mjs';
test('defaults to serial, accepts packed and rejects unsupported modes', () => {
  assert.equal(validateDebertaReadback(undefined), 'serial');
  assert.equal(validateDebertaReadback('serial'), 'serial');
  assert.equal(validateDebertaReadback('packed'), 'packed');
  for (const value of ['', 'auto', null, 0, true])
    assert.throws(() => validateDebertaReadback(value), {code: 'INVALID_INPUT'});
});
