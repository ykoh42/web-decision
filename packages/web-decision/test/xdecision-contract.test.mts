import assert from 'node:assert/strict';
import test from 'node:test';
import * as contract from '../src/xdecision/contract.mjs';

test('XDecision contract exposes pinned token limits without a fixed character cap', () => {
  assert.equal(contract.XDECISION_MAX_SEQUENCE_TOKENS, 1024);
  assert.equal(contract.XDECISION_MAX_HEAD_TOKENS, 256);
  assert.equal(contract.XDECISION_MAX_OPTION_TOKENS, 48);
  assert.equal(contract.XDECISION_MAX_OPTIONS, 122);
  assert.equal(contract.XDECISION_SPEC.maxSequenceTokens, contract.XDECISION_MAX_SEQUENCE_TOKENS);
  assert.equal(contract.XDECISION_SPEC.maxHeadTokens, contract.XDECISION_MAX_HEAD_TOKENS);
  assert.equal(contract.XDECISION_SPEC.maxOptionTokens, contract.XDECISION_MAX_OPTION_TOKENS);
  assert.equal(Object.hasOwn(contract.XDECISION_SPEC, 'maxQuestions'), false);
  assert.equal(Object.hasOwn(contract.XDECISION_SPEC, 'maxTextChars'), false);
  assert.equal(Object.hasOwn(contract, 'XDECISION_MAX_TEXT_CHARS'), false);
});
