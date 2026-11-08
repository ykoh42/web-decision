import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test from 'node:test';
import decision from '../src/pplx/decision-config.json' with {type: 'json'};

test('answer codes retain the pinned upstream order, including skipped labels', () => {
  assert.equal(decision.codes.length, 255);
  assert.equal(decision.codes[68], 'BR');
  assert.equal(decision.codes[85], 'CK');
  assert.equal(
    createHash('sha256').update(JSON.stringify(decision.codes)).digest('hex'),
    'c493343e7a9a72aa93011d2e62c46b0b2cc3fae55cbd74ebbf2c57f93bb6c028'
  );
  assert.equal(decision.temperature, 1.5539092221462472);
});
