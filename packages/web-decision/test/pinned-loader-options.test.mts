import assert from 'node:assert/strict';
import test from 'node:test';
import {loadExperimentalKev08} from '../src/kev08/loader.js';
import {loadExperimentalMapika} from '../src/mapika/loader.js';

for (const [name, load] of [
  ['Kev 0.8B', loadExperimentalKev08],
  ['Mapika', loadExperimentalMapika]
] as const) {
  test(`${name} rejects missing asset locations before GPU initialization`, async () => {
    await assert.rejects(load({}), {code: 'INVALID_INPUT', message: 'assetsUrl is required.'});
  });

  test(`${name} rejects offline loading when cache is disabled`, async () => {
    await assert.rejects(load({assetsUrl: 'https://example.test/model/', offline: true, cache: false}), {
      code: 'INVALID_INPUT',
      message: 'Offline loading requires the model cache.'
    });
  });
}
