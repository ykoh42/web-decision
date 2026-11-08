import assert from 'node:assert/strict';
import test from 'node:test';
import {encodeKananaQuery} from '../src/kanana/tokenizer.mjs';
import {preprocessKananaImage} from '../src/kanana/image-processing.mjs';

const imageMarker = '<|vision_start|><|image_pad|><|vision_end|>';
const generationPrompt = '<|im_start|>assistant\n<think>\n\n</think>\n\n';
const queryMessages = (userText = 'State:\nA parcel is late.') =>
  Object.freeze([
    Object.freeze({role: 'system', content: Object.freeze([{type: 'text', text: 'Represent the state.'}])}),
    Object.freeze({role: 'user', content: Object.freeze([{type: 'text', text: userText}])})
  ]);

function tokenizerWithIds(ids = [128000, 128009, 128010]) {
  const prompts = [];
  return {
    prompts,
    ids,
    encode(prompt) {
      prompts.push(prompt);
      return {ids};
    }
  };
}

test('tokenizes the rendered text query once and returns a copied token count', () => {
  const tokenizer = tokenizerWithIds([128000, 10, 11, 128009]);
  const messages = queryMessages();
  const result = encodeKananaQuery(tokenizer, messages);
  const expectedPrompt =
    '<|im_start|>system\nRepresent the state.<|im_end|>\n' +
    '<|im_start|>user\nState:\nA parcel is late.<|im_end|>\n' +
    generationPrompt;

  assert.equal(tokenizer.prompts.length, 1);
  assert.equal(tokenizer.prompts[0], expectedPrompt);
  assert.equal(result.prompt, expectedPrompt);
  assert.deepEqual(result.inputIds, [128000, 10, 11, 128009]);
  assert.equal(result.tokenCount, 4);
  assert.notStrictEqual(result.inputIds, tokenizer.ids);
});

test('expands one image marker using the preprocessor token grid product', () => {
  const image = preprocessKananaImage(new Uint8Array(128 * 32 * 3), 128, 32);
  const tokenizer = tokenizerWithIds([128000, 128017, 128020, 128018]);
  const messages = [
    {role: 'system', content: [{type: 'text', text: 'Represent the state.'}]},
    {
      role: 'user',
      content: [
        {type: 'text', text: 'State:\n'},
        {type: 'image', image: 'photo.png'},
        {type: 'text', text: '\nChoose a label.'}
      ]
    }
  ];

  const result = encodeKananaQuery(tokenizer, messages, [image.imageTokenThw]);
  const imageTokenCount = image.imageTokenThw.reduce((product, dimension) => product * dimension, 1);
  const expectedMarker = `<|vision_start|>${'<|image_pad|>'.repeat(imageTokenCount)}<|vision_end|>`;

  assert.equal(result.prompt.includes(imageMarker), false);
  assert.ok(result.prompt.includes(expectedMarker));
  assert.deepEqual(image.imageTokenThw, [1, 4, 16]);
  assert.equal(tokenizer.prompts.length, 1);
});

test('expands multiple image grids in message order', () => {
  const tokenizer = tokenizerWithIds([1, 2, 3]);
  const messages = [
    {role: 'system', content: [{type: 'text', text: 'Describe the state.'}]},
    {
      role: 'user',
      content: [
        {type: 'text', text: 'State:\n'},
        {type: 'image', image: 'first.png'},
        {type: 'text', text: '\nBetween images\n'},
        {type: 'image', image: 'second.png'},
        {type: 'text', text: '\nEnd'}
      ]
    }
  ];
  const grids = [
    [1, 2, 2],
    [2, 1, 3]
  ];
  const result = encodeKananaQuery(tokenizer, messages, grids);
  const first = `<|vision_start|>${'<|image_pad|>'.repeat(4)}<|vision_end|>`;
  const second = `<|vision_start|>${'<|image_pad|>'.repeat(6)}<|vision_end|>`;

  assert.ok(result.prompt.indexOf(first) < result.prompt.indexOf(second));
  assert.ok(result.prompt.includes(`State:\n${first}\nBetween images\n${second}\nEnd`));
  assert.equal(tokenizer.prompts.length, 1);
});

test('rejects mismatched image marker and grid counts before tokenization', () => {
  const tokenizer = tokenizerWithIds();
  const messages = [
    {role: 'system', content: [{type: 'text', text: 'Represent the state.'}]},
    {role: 'user', content: [{type: 'text', text: `State: ${imageMarker}`}]}
  ];

  assert.throws(() => encodeKananaQuery(tokenizer, messages), /marker count does not match/);
  assert.throws(() => encodeKananaQuery(tokenizer, queryMessages(), [[1, 2, 2]]), /marker count does not match/);
  assert.equal(tokenizer.prompts.length, 0);
});

test('rejects orphan image tokens in text with and without image grids', () => {
  const textTokenizer = tokenizerWithIds();
  const imagePrefixTokenizer = tokenizerWithIds();
  const imageSuffixTokenizer = tokenizerWithIds();
  const textWithImageToken = queryMessages(`State: <|image_pad|>`);
  const imageWithOrphanPrefix = [
    {role: 'system', content: [{type: 'text', text: 'Represent the state.'}]},
    {
      role: 'user',
      content: [{type: 'text', text: `State: <|image_pad|> ${imageMarker}`}]
    }
  ];
  const imageWithOrphanSuffix = [
    {role: 'system', content: [{type: 'text', text: 'Represent the state.'}]},
    {
      role: 'user',
      content: [
        {type: 'text', text: 'State: '},
        {type: 'image', image: 'photo.png'},
        {type: 'text', text: ' <|image_pad|>'}
      ]
    }
  ];

  assert.throws(() => encodeKananaQuery(textTokenizer, textWithImageToken), /unpaired image token/);
  assert.throws(
    () => encodeKananaQuery(imagePrefixTokenizer, imageWithOrphanPrefix, [[1, 1, 1]]),
    /marker count does not match/
  );
  assert.throws(
    () => encodeKananaQuery(imageSuffixTokenizer, imageWithOrphanSuffix, [[1, 1, 1]]),
    /unpaired image token/
  );
  assert.equal(textTokenizer.prompts.length, 0);
  assert.equal(imagePrefixTokenizer.prompts.length, 0);
  assert.equal(imageSuffixTokenizer.prompts.length, 0);
});

test('preserves literal placeholders without images and mirrors upstream restoration with images', () => {
  const textTokenizer = tokenizerWithIds();
  const imageTokenizer = tokenizerWithIds();
  const textResult = encodeKananaQuery(textTokenizer, queryMessages('State: <|placeholder|>'));
  const imageMessages = [
    {role: 'system', content: [{type: 'text', text: 'Represent the state.'}]},
    {
      role: 'user',
      content: [
        {type: 'text', text: 'State: <|placeholder|> '},
        {type: 'image', image: 'photo.png'},
        {type: 'text', text: ' End'}
      ]
    }
  ];
  const imageResult = encodeKananaQuery(imageTokenizer, imageMessages, [[1, 1, 1]]);

  assert.ok(textResult.prompt.includes('State: <|placeholder|>'));
  assert.ok(imageResult.prompt.includes('State: <|image_pad|> <|vision_start|><|image_pad|><|vision_end|> End'));
  assert.equal(textTokenizer.prompts.length, 1);
  assert.equal(imageTokenizer.prompts.length, 1);
});

test('rejects invalid image token grids', () => {
  const tokenizer = tokenizerWithIds();
  const messages = [
    {role: 'system', content: [{type: 'text', text: 'Represent the state.'}]},
    {role: 'user', content: [{type: 'text', text: `State: ${imageMarker}`}]}
  ];

  for (const grid of [null, [], [1, 2], [1, 0, 2], [1, 1.5, 2], [1, Number.MAX_SAFE_INTEGER, 2]]) {
    assert.throws(() => encodeKananaQuery(tokenizer, messages, [grid]), /imageTokenThw|safe integer/);
  }
  assert.equal(tokenizer.prompts.length, 0);
});

test('returns all token IDs without applying truncation or an adapter length cap', () => {
  const ids = Array.from({length: 20_481}, (_, index) => index);
  const tokenizer = tokenizerWithIds(ids);
  const result = encodeKananaQuery(tokenizer, queryMessages());

  assert.equal(result.tokenCount, ids.length);
  assert.deepEqual(result.inputIds, ids);
  assert.notStrictEqual(result.inputIds, ids);
});

test('does not mutate frozen messages or image grids', () => {
  const tokenizer = tokenizerWithIds();
  const messages = Object.freeze([
    Object.freeze({role: 'system', content: Object.freeze([{type: 'text', text: 'Represent the state.'}])}),
    Object.freeze({
      role: 'user',
      content: Object.freeze(
        [
          {type: 'text', text: 'State:\n'},
          {type: 'image', image: 'photo.png'},
          {type: 'text', text: '\nEnd'}
        ].map(Object.freeze)
      )
    })
  ]);
  const grid = Object.freeze([1, 2, 3]);

  const result = encodeKananaQuery(tokenizer, messages, [grid]);

  assert.ok(result.prompt.includes('<|image_pad|>'.repeat(6)));
  assert.deepEqual(grid, [1, 2, 3]);
  assert.equal(messages[1].content[1].image, 'photo.png');
});

test('rejects missing tokenizers and malformed encoded IDs', () => {
  const messages = queryMessages();
  assert.throws(() => encodeKananaQuery(null, messages), /tokenizer with encode/);
  assert.throws(() => encodeKananaQuery({}, messages), /tokenizer with encode/);
  assert.throws(() => encodeKananaQuery({encode: () => null}, messages), /indexed token IDs/);
  assert.throws(() => encodeKananaQuery({encode: () => ({ids: [1, Number.NaN]})}, messages), /invalid token ID/);
  const sparseIds = new Array(2);
  sparseIds[1] = 1;
  assert.throws(() => encodeKananaQuery({encode: () => ({ids: sparseIds})}, messages), /sparse token ID/);
});
