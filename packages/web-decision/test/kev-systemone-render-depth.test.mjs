import test from 'node:test';
import assert from 'node:assert/strict';
import {prepareKevSystemOne} from '../src/kev/system-one.mjs';

function renderState(state, maxInputCharacters) {
  return prepareKevSystemOne(state, {check: {type: 'noul', instructions: null}}, maxInputCharacters).request.state;
}

function nestedObject(depth) {
  let value = 'leaf';
  for (let index = 0; index < depth; index++) value = {item: value};
  return value;
}

function nestedObjectText(depth) {
  let text = 'leaf';
  for (let level = depth - 1; level >= 0; level--)
    text = `${' '.repeat(level * 2)}item:${level === depth - 1 ? ' ' : '\n'}${text}`;
  return text;
}

function nestedArray(depth) {
  let value = 'leaf';
  for (let index = 0; index < depth; index++) value = [value];
  return value;
}

test('System One renders scalar state like pinned Kev', () => {
  assert.equal(renderState(null), '');
  assert.equal(renderState(true), 'True');
  assert.equal(renderState(false), 'False');
  assert.equal(renderState('plain text'), 'plain text');
});

test('System One renders JSON objects at and beyond the former depth boundary', () => {
  for (const depth of [1024, 1025, 1026]) assert.equal(renderState(nestedObject(depth)), nestedObjectText(depth));
});

test('System One renders deeply nested arrays without changing list formatting', () => {
  for (const depth of [1024, 1025, 1026, 10_000])
    assert.equal(renderState(nestedArray(depth)), `${'- '.repeat(depth)}leaf`);
});

test('System One preserves object key order', () => {
  const state = Object.assign(Object.create(null), {second: '2', first: '1'});
  assert.equal(renderState(state), 'second: 2\nfirst: 1');
});

test('System One strips array item prefixes using Python whitespace rules', () => {
  assert.equal(renderState({items: ['\u0085visible', '\ufeffvisible']}), 'items:\n  - visible\n  - \ufeffvisible');
});

test('System One rejects cycles and renders a shared acyclic value each time it appears', () => {
  const cycle = {};
  cycle.self = cycle;
  assert.throws(() => renderState(cycle), {code: 'INVALID_INPUT', message: /cycles/});

  const shared = {value: 'x'};
  assert.equal(renderState({left: shared, right: shared}), 'left:\n  value: x\nright:\n  value: x');
});

test('System One keeps its optional rendered-character budget', () => {
  const state = nestedObject(8);
  const rendered = nestedObjectText(8);
  const exactBudget = rendered.length + 'no'.length + 'yes'.length;
  assert.equal(renderState(state, exactBudget), rendered);
  assert.throws(() => renderState(state, exactBudget - 1), {code: 'INVALID_INPUT'});
});
