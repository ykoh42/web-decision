import assert from 'node:assert/strict';
import test from 'node:test';
import {WebDecisionError} from '../src/kev.js';
import {createMapika4Tokenizer} from '../src/mapika4b/tokenizer.mjs';
import {MAPIKA4_LABELS_SHA256} from '../src/mapika4b/contract.mjs';

type EncodeOptions = Readonly<{add_special_tokens: false}>;

function expectedDigest(): ArrayBuffer {
  return Uint8Array.from(MAPIKA4_LABELS_SHA256.match(/../gu) ?? [], (part) => Number.parseInt(part, 16)).buffer;
}

function controlTokens(events: string[], ids: readonly number[]): object[] {
  return ids.map((id, index) =>
    Object.defineProperty({}, 'id', {
      get() {
        events.push(`id:${index}`);
        return id;
      }
    })
  );
}

test('reads control token getters before construction and hashes labels only after encoding', async (t) => {
  const events: string[] = [];
  const controls = controlTokens(
    events,
    Array.from({length: 33}, (_, index) => 246000 + index)
  );
  let addedTokenReads = 0;
  const data = Object.defineProperty({}, 'added_tokens', {
    get() {
      addedTokenReads++;
      events.push(`added_tokens:${addedTokenReads}`);
      return controls;
    }
  });
  const config = {};
  let nextId = 0;
  class FakeTokenizer {
    constructor(actualData: unknown, actualConfig: unknown) {
      events.push('constructor');
      assert.equal(actualData, data);
      assert.equal(actualConfig, config);
    }

    encode(text: string, options: EncodeOptions): {readonly ids: number[]} {
      assert.deepEqual(options, {add_special_tokens: false});
      events.push(`encode:${text}`);
      return {ids: text === '\n(' ? [247000] : [1000 + nextId++]};
    }
  }

  let digestedBinding: string | undefined;
  t.mock.method(crypto.subtle, 'digest', async (_algorithm: AlgorithmIdentifier, value: BufferSource) => {
    events.push('digest');
    digestedBinding = new TextDecoder().decode(new Uint8Array(value as ArrayBuffer));
    return expectedDigest();
  });

  const result = await createMapika4Tokenizer(FakeTokenizer, data, config);

  assert.equal(addedTokenReads, 2);
  assert.equal(result.labels.names.length, 255);
  assert.equal(result.labels.ids.length, 255);
  assert.deepEqual(result.labels.open, [247000]);
  assert.deepEqual(
    result.limits.controlIds,
    Array.from({length: 33}, (_, index) => 246000 + index)
  );
  assert.deepEqual(result.limits, {
    controlIds: Array.from({length: 33}, (_, index) => 246000 + index),
    maxInputTokens: 128,
    maxStateTokens: 128,
    temperature: 2.098
  });
  assert.equal(events.filter((event) => event.startsWith('encode:')).length, 256);
  assert.ok(events.indexOf('added_tokens:1') < events.indexOf('added_tokens:2'));
  assert.ok(events.indexOf('added_tokens:2') < events.indexOf('id:0'));
  assert.ok(events.indexOf('id:32') < events.indexOf('constructor'));
  assert.ok(events.indexOf('constructor') < events.indexOf('encode:A'));
  assert.ok(events.lastIndexOf('encode:\n(') < events.indexOf('digest'));
  assert.equal(
    digestedBinding,
    JSON.stringify([result.labels.names, result.labels.ids, result.labels.open, result.limits.controlIds])
  );
  assert.equal(events.at(-1), 'digest');
});

test('preserves validation and constructor errors before tokenizer encoding', async () => {
  const events: string[] = [];
  const cause = new Error('tokenizer constructor failed');
  class FailingTokenizer {
    constructor() {
      events.push('constructor');
      throw cause;
    }
  }
  const data = {
    added_tokens: controlTokens(
      events,
      Array.from({length: 33}, (_, index) => 246000 + index)
    )
  };

  await assert.rejects(createMapika4Tokenizer(FailingTokenizer, data, {}), (error: unknown) => error === cause);
  assert.equal(events.at(-1), 'constructor');
  assert.equal(
    events.some((event) => event.startsWith('encode:')),
    false
  );
  assert.equal(events.includes('digest'), false);

  await assert.rejects(createMapika4Tokenizer(FailingTokenizer, {added_tokens: []}, {}), (error: unknown) => {
    assert.ok(error instanceof WebDecisionError);
    assert.equal(error.code, 'INVALID_ASSET');
    assert.equal(error.message, 'Invalid Mapika 4B control tokens.');
    return true;
  });
  assert.equal(events.filter((event) => event === 'constructor').length, 1);
});
