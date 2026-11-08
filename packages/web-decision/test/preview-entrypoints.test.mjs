import assert from 'node:assert/strict';
import * as nodeModule from 'node:module';
import test from 'node:test';
import * as decima from '../src/decima-base-clean-preview-entry.mjs';
import * as skejja from '../src/skejja-preview-entry.mjs';
import * as tinydecide from '../src/tinydecide-preview-entry.js';

const entrypoints = [decima, skejja, tinydecide];

const supportsHooks = typeof nodeModule.registerHooks === 'function';
let isolatedEntrypoints = [];
if (supportsHooks) {
  const wrapperUrls = new Set([
    new URL('../src/deberta-single-preview-entry.js', import.meta.url).href,
    new URL('../src/kev-preview-entry.js', import.meta.url).href,
    new URL('../src/kev-deberta-preview-entry.js', import.meta.url).href,
    new URL('../src/kev-deberta-reflexes-preview-entry.js', import.meta.url).href
  ]);
  const fakeModules = {
    './deberta/loader.mjs': `data:text/javascript,${encodeURIComponent('export async function loadExperimentalDeberta() { throw new Error("unexpected DeBERTa load"); }')}`,
    './kev-loader.js': `data:text/javascript,${encodeURIComponent('export async function loadKev06DecisionModel() { throw new Error("unexpected Kev load"); } export async function loadKevDecisionModel() { throw new Error("unexpected Reflexes load"); }')}`
  };
  const hooks = nodeModule.registerHooks({
    resolve(specifier, context, nextResolve) {
      if (wrapperUrls.has(context.parentURL) && fakeModules[specifier]) {
        return {url: fakeModules[specifier], shortCircuit: true};
      }
      return nextResolve(specifier, context);
    }
  });
  try {
    isolatedEntrypoints = await Promise.all([
      import('../src/kev-preview-entry.js'),
      import('../src/deberta-single-preview-entry.js'),
      import('../src/kev-deberta-preview-entry.js'),
      import('../src/kev-deberta-reflexes-preview-entry.js')
    ]);
  } finally {
    hooks.deregister();
  }
}

const [kev, deberta, kevDeberta, kevReflexes] = isolatedEntrypoints;
const allEntrypoints = [...entrypoints, ...isolatedEntrypoints];

test('preview entrypoints expose matching model descriptions and reject unsupported routes before loading', async () => {
  for (const entry of allEntrypoints) {
    const models = entry.listDecisionModels();
    assert.ok(models.length > 0);
    for (const model of models) {
      assert.equal(entry.isDecisionModel(model), true);
      assert.equal(entry.describeModel(model).model, model);
    }
    assert.equal(entry.isDecisionModel('unsupported-model'), false);
  }

  await assert.rejects(
    decima.loadDecisionModel({model: 'unsupported-model', assetsUrl: 'https://example.test/model/'}),
    {code: 'INVALID_INPUT'}
  );
  await assert.rejects(skejja.loadDecisionModel({model: 'unsupported-model'}), {code: 'INVALID_INPUT'});
  await assert.rejects(tinydecide.loadDecisionModel({model: 'unsupported-model'}), {code: 'INVALID_INPUT'});
  if (supportsHooks) {
    await assert.rejects(kev.loadDecisionModel({debertaReadback: true}), {code: 'INVALID_INPUT'});
    await assert.rejects(deberta.loadDecisionModel({model: 'unsupported-model'}), {code: 'INVALID_INPUT'});
    await assert.rejects(kevDeberta.loadDecisionModel({model: 'kev-0.6b-q4f16', debertaReadback: true}), {
      code: 'INVALID_INPUT'
    });
    await assert.rejects(kevReflexes.loadDecisionModel({model: 'kev-0.6b-reflexes-q4f16', debertaReadback: true}), {
      code: 'INVALID_INPUT'
    });
  }
});

test('preview entrypoints clear only the current-origin asset cache when available', async () => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'caches');
  try {
    delete globalThis.caches;
    for (const entry of allEntrypoints) assert.equal(await entry.clearModelCache(), false);

    const removed = [];
    Object.defineProperty(globalThis, 'caches', {
      configurable: true,
      writable: true,
      value: {
        async delete(name) {
          removed.push(name);
          return true;
        }
      }
    });
    for (const entry of allEntrypoints) assert.equal(await entry.clearModelCache(), true);
    assert.equal(removed.length, allEntrypoints.length);
    assert.ok(removed.every((name) => typeof name === 'string'));
  } finally {
    if (previous) Object.defineProperty(globalThis, 'caches', previous);
    else delete globalThis.caches;
  }
});
