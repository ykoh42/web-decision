import assert from 'node:assert/strict';
import test from 'node:test';
import {rotaryAllocationBytes} from './strands-v2-rotary-limits.ts';

test('accounts for the TVM allocation page around the sequence boundary', () => {
  assert.equal(rotaryAllocationBytes(131071), 134217728);
  assert.equal(rotaryAllocationBytes(131072), 134217728);
  assert.equal(rotaryAllocationBytes(131073), 134221824);
});
