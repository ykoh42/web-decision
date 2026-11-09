import assert from 'node:assert/strict';
import {Buffer} from 'node:buffer';
import test, {type TestContext} from 'node:test';
import type {H2oTokenizerConstructor} from '../src/h2o/tokenizer.mjs';
import {createH2oTokenizer} from '../src/h2o/tokenizer.mjs';
import {prepareCheckedH2oTokenizer} from '../src/h2o/tokenizer-adapter.mjs';

const pattern =
  "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const pins = [
  '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42',
  '316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8',
  '7eb96aaea6b7dbb495f0840b42f18014ca5a697697305f76f39d46cae890ccb2'
];

function fixture(t: TestContext, {beforeDigest}: {beforeDigest?: (index: number) => Promise<void>} = {}) {
  const names = Array.from({length: 33}, (_, index) => `token-${index}`);
  names[1] = '<|im_start|>';
  names[2] = '<|im_end|>';
  names[24] = '<think>';
  names[25] = '</think>';
  const token = (content: string) => ({
    content,
    normalized: false,
    special: false,
    single_word: false,
    lstrip: false,
    rstrip: false
  });
  const data = {
    normalizer: {type: 'NFC'},
    pre_tokenizer: {type: 'Sequence', pretokenizers: [{type: 'Split', pattern: {Regex: pattern}}, {type: 'ByteLevel'}]},
    added_tokens: names.slice(0, 26).map((name, index) => ({id: 248044 + index, ...token(name)}))
  };
  const config = {
    tokenizer_class: 'Qwen2Tokenizer',
    pretokenize_regex: pattern,
    added_tokens_decoder: Object.fromEntries(names.map((name, index) => [String(248044 + index), token(name)]))
  };
  const contract = {probes: Array.from({length: 24}, (_, index) => ({text: `probe-${index}`, token_ids: [7]}))};
  const encoder = new TextEncoder(),
    inputs: [Uint8Array, Uint8Array, Uint8Array] = [data, config, contract].map((value) =>
      encoder.encode(JSON.stringify(value))
    ) as [Uint8Array, Uint8Array, Uint8Array];
  const originals = inputs.map((bytes) => Buffer.from(bytes).toString('hex'));
  const tokenizerJson = new TextDecoder().decode(inputs[0]);
  const parse = JSON.parse;
  let tokenizerParses = 0;
  t.mock.method(
    JSON,
    'parse',
    (text: string, ...args: [reviver?: (this: unknown, key: string, value: unknown) => unknown]) => {
      if (text === tokenizerJson) tokenizerParses++;
      return parse(text, ...args);
    }
  );
  t.mock.method(crypto.subtle, 'digest', async (_algorithm: AlgorithmIdentifier, value: BufferSource) => {
    const index = originals.indexOf(Buffer.from(value as ArrayBuffer).toString('hex'));
    if (beforeDigest) await beforeDigest(index);
    return Uint8Array.from((index < 0 ? '00'.repeat(32) : pins[index]!).match(/../g)!, (part) => parseInt(part, 16))
      .buffer;
  });
  let probeCalls = 0;
  class Tokenizer {
    constructor(data: object, config: object) {
      const parsed = data as {added_tokens: {content: string}[]};
      const parsedConfig = config as {tokenizer_class: string; added_tokens_decoder: Record<string, {content: string}>};
      parsed.added_tokens[0]!.content = 'constructor-mutated';
      parsedConfig.tokenizer_class = 'constructor-mutated';
      parsedConfig.added_tokens_decoder['248044']!.content = 'constructor-mutated';
    }
    encode(text: string) {
      if (text.startsWith('probe-')) {
        probeCalls++;
        return {ids: [7]};
      }
      const marker = ['<|im_start|>', '<|im_end|>', '<think>', '</think>'].find((value) => text.includes(value));
      return {ids: marker ? [248044 + names.indexOf(marker)] : []};
    }
  }
  return {
    inputs,
    names,
    Tokenizer,
    typedTokenizer: Tokenizer as unknown as H2oTokenizerConstructor,
    probes: () => probeCalls,
    parses: () => tokenizerParses
  };
}

test('adapter keeps pre-constructor evidence and synchronous byte snapshots', async (t) => {
  const {inputs, names, typedTokenizer, probes, parses} = fixture(t);
  const pending = prepareCheckedH2oTokenizer(typedTokenizer, ...inputs);
  for (const input of inputs) input[0] = input[0]! ^ 1;
  const result = await pending;
  assert.deepEqual(result.addedTokens, names);
  assert.equal(probes(), 48);
  assert.equal(parses(), 1);
  assert.deepEqual(result.tokenizer.encode('probe-0', {add_special_tokens: false}).ids, [7]);
});

test('abort during tokenizer hashing prevents JSON parsing and tokenizer construction', async (t) => {
  let finish!: () => void;
  const pendingDigest = new Promise<void>((resolve) => {
    finish = resolve;
  });
  const {inputs, Tokenizer, parses} = fixture(t, {beforeDigest: () => pendingDigest}),
    controller = new AbortController(),
    reason = new Error('synthetic cancellation');
  let constructed = 0;
  class CountingTokenizer extends Tokenizer {
    constructor(...args: ConstructorParameters<typeof Tokenizer>) {
      super(...args);
      constructed++;
    }
  }
  const pending = prepareCheckedH2oTokenizer(
    CountingTokenizer as unknown as H2oTokenizerConstructor,
    ...inputs,
    controller.signal
  );
  controller.abort(reason);
  finish();
  await assert.rejects(pending, (error) => error === reason);
  assert.equal(parses(), 0);
  assert.equal(constructed, 0);
});

test('abort raised during a tokenizer probe prevents further probes', async (t) => {
  const {inputs, Tokenizer, probes} = fixture(t),
    controller = new AbortController(),
    reason = new Error('synthetic cancellation');
  class CancellingTokenizer extends Tokenizer {
    override encode(...args: Parameters<InstanceType<typeof Tokenizer>['encode']>) {
      const result = super.encode(...args);
      controller.abort(reason);
      return result;
    }
  }
  await assert.rejects(
    prepareCheckedH2oTokenizer(CancellingTokenizer as unknown as H2oTokenizerConstructor, ...inputs, controller.signal),
    (error) => error === reason
  );
  assert.equal(probes(), 1);
});

test('pinned input mismatch keeps the adapter error and direct builder keeps its return type', async (t) => {
  const {inputs, Tokenizer, typedTokenizer} = fixture(t);
  assert.ok((await createH2oTokenizer(typedTokenizer, ...inputs)) instanceof Tokenizer);
  inputs[0][0] = inputs[0][0]! ^ 1;
  await assert.rejects(prepareCheckedH2oTokenizer(typedTokenizer, ...inputs), {
    code: 'INVALID_ASSET',
    message: 'Pinned H2O tokenizer adapter verification failed.'
  });
});
