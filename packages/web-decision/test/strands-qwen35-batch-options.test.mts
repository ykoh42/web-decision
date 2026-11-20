import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {
  finishQwen35AnswersFromLogits,
  QWEN35_GRAPH_MAX_BATCH,
  packQwen35Request,
  resolveQwen35BatchOptions
} from '../src/strands/qwen35-protocol.mts';
import {prepareQwen35GraphInput} from '../src/strands/qwen35-graph-input.mts';
import {prepareQwen35TextVisionInput} from '../src/strands/qwen35-text-vision-input.mts';

const utf8 = new TextEncoder();
const specialTokens = [
  ['<|image_pad|>', 248056],
  ['<|vision_end|>', 248054],
  ['<|endoftext|>', 248044],
  ['<bos>', 248043]
] as const;

function tokenizerFixture(onEncode?: () => void) {
  const byteDecoder = Object.fromEntries(
    Array.from({length: 256}, (_, byte) => [String.fromCharCode(byte + 0x100), byte])
  );
  const addedTokenIds = new Set(specialTokens.map(([, id]) => id));
  return {
    decoder: {byte_decoder: byteDecoder},
    get_added_tokens_decoder: () => addedTokenIds,
    encode(text: string, {add_special_tokens}: {readonly add_special_tokens: boolean}) {
      onEncode?.();
      const ids: number[] = [];
      const tokens: string[] = [];
      if (add_special_tokens) {
        ids.push(248043);
        tokens.push('<bos>');
      }
      let cursor = 0;
      while (cursor < text.length) {
        const special = specialTokens.find(([token]) => text.startsWith(token, cursor));
        if (special) {
          tokens.push(special[0]);
          ids.push(special[1]);
          cursor += special[0].length;
          continue;
        }
        let end = cursor + 1;
        while (end < text.length && !specialTokens.some(([token]) => text.startsWith(token, end))) end += 1;
        for (const byte of utf8.encode(text.slice(cursor, end).normalize('NFC'))) {
          tokens.push(String.fromCharCode(byte + 0x100));
          ids.push(byte + 1);
        }
        cursor = end;
      }
      return {ids, tokens};
    }
  };
}

function questions(count: number) {
  return Object.fromEntries(
    Array.from({length: count}, (_, index) => [`q${index}`, {type: 'noul', instructions: 'Is this true?'}])
  );
}

function choiceQuestion(optionCount: number) {
  return {
    type: 'choice',
    instructions: 'Pick the best option.',
    criteria: Object.fromEntries(
      Array.from({length: optionCount}, (_, index) => [`o${String(index).padStart(3, '0')}`, ''])
    )
  };
}

test('Qwen request packing keeps pinned defaults and forwards EngineConfig batch policies', () => {
  const tokenizer = tokenizerFixture();
  const request = {state: 'Shared state.', questions: questions(5)};
  const defaults = packQwen35Request(tokenizer, request);
  assert.equal(defaults.maxLength, 4096);
  const explicitDefaults = packQwen35Request(tokenizer, request, {
    maxBatch: 32,
    maxQuestionFraction: 0.75,
    maxBatchTokens: null
  });
  assert.deepEqual(explicitDefaults, defaults);
  assert.deepEqual(defaults.batchSizes, [5]);

  const maxBatch = packQwen35Request(tokenizer, request, {maxBatch: 2});
  assert.deepEqual(maxBatch.batchSizes, [2, 2, 1]);
  assert.deepEqual(
    maxBatch.rows.map(({inputIds}) => inputIds),
    defaults.rows.map(({inputIds}) => inputIds)
  );

  const rowCost = defaults.rows[0]!.inputIds.length;
  const maxBatchTokens = packQwen35Request(tokenizer, request, {maxBatch: 4, maxBatchTokens: rowCost * 2});
  assert.deepEqual(maxBatchTokens.batchSizes, [2, 2, 1]);
  assert.deepEqual(packQwen35Request(tokenizer, request, {maxBatch: 4, maxBatchTokens: 0}).batchSizes, [4, 1]);
  assert.deepEqual(
    packQwen35Request(tokenizer, request, {maxBatch: 4, maxBatchTokens: -1}).batchSizes,
    [1, 1, 1, 1, 1]
  );

  const longQuestion = {state: 'State.', questions: {q: {type: 'noul', instructions: 'i'.repeat(4500)}}};
  const defaultFit = packQwen35Request(tokenizer, longQuestion).rows[0]!;
  const customFit = packQwen35Request(tokenizer, longQuestion, {maxQuestionFraction: 0.25}).rows[0]!;
  assert.equal(defaultFit.inputIds.length - customFit.inputIds.length, 2048);
  assert.ok(customFit.optionIndices.every((index) => index < customFit.answerIndex));
});

test('Qwen context length may be lowered but cannot exceed the pinned checkpoint limit', async () => {
  const tokenizer = tokenizerFixture();
  const request = {
    state: 's'.repeat(6000),
    questions: {q: {type: 'noul', instructions: 'i'.repeat(5000)}}
  };
  const packed = packQwen35Request(tokenizer, request, {maxLength: 4096});

  assert.equal(packed.maxLength, 4096);
  assert.equal(packed.rows[0]!.inputIds.length, 4096);
  assert.throws(() => packQwen35Request(tokenizer, request, {maxLength: 0}), /positive safe integer/u);
  assert.throws(() => packQwen35Request(tokenizer, request, {maxLength: 1.5}), /positive safe integer/u);
  assert.throws(() => packQwen35Request(tokenizer, request, {maxLength: 4097}), /4096/u);

  const smaller = packQwen35Request(tokenizer, request, {maxLength: 2048});
  assert.equal(smaller.maxLength, 2048);
  assert.equal(smaller.rows[0]!.inputIds.length, 2048);

  let tokenizationCalls = 0;
  const noEncodeTokenizer = tokenizerFixture(() => {
    tokenizationCalls += 1;
  });
  assert.throws(() => packQwen35Request(noEncodeTokenizer, request, {maxLength: 4097}), /4096/u);
  assert.equal(tokenizationCalls, 0);

  const graph = await prepareQwen35GraphInput(tokenizer, request, {maxLength: 4096});
  assert.equal(graph.maxLength, 4096);
  assert.equal(graph.chunks[0]!.prefixInputShape[1] + graph.chunks[0]!.suffixInputShape[1], 4096);
  await assert.rejects(prepareQwen35GraphInput(tokenizer, request, {maxLength: 4097}), /4096/u);
});

test('Qwen graph input applies batch and question-fraction options to prepared chunks', async () => {
  const tokenizer = tokenizerFixture();
  const request = {state: 'Shared state.', questions: questions(5)};
  const defaults = await prepareQwen35GraphInput(tokenizer, request);
  const maxBatch = await prepareQwen35GraphInput(tokenizer, request, {maxBatch: 2});
  const maxBatchTokens = await prepareQwen35GraphInput(tokenizer, request, {
    maxBatch: 4,
    maxBatchTokens: 1
  });
  const graphBatchBoundary = await prepareQwen35GraphInput(
    tokenizer,
    {state: 'Shared state.', questions: questions(33)},
    {maxBatch: 33}
  );
  const contextBoundary = await prepareQwen35GraphInput(tokenizer, {
    state: 's'.repeat(5000),
    questions: {q: {type: 'noul', instructions: 'i'.repeat(2200)}}
  });

  // Source maxBatch groups are split again to fit the compiled R13 graph's three-row ABI.
  assert.deepEqual(
    defaults.chunks.map(({questions: rows}) => rows.length),
    [3, 2]
  );
  assert.deepEqual(
    maxBatch.chunks.map(({questions: rows}) => rows.length),
    [2, 2, 1]
  );
  assert.deepEqual(
    maxBatchTokens.chunks.map(({questions: rows}) => rows.length),
    [1, 1, 1, 1, 1]
  );
  // maxBatch remains caller-configurable; 33 source rows become eleven graph calls of three.
  assert.deepEqual(
    graphBatchBoundary.chunks.map(({fullAttentionMaskShape: [batch]}) => batch),
    Array.from({length: 11}, () => QWEN35_GRAPH_MAX_BATCH)
  );
  assert.deepEqual(contextBoundary.chunks[0]!.fullAttentionMaskShape, [1, 4096]);
  assert.ok(maxBatchTokens.inputTokenCount > defaults.inputTokenCount);

  const longQuestion = {state: 'State.', questions: {q: {type: 'noul', instructions: 'i'.repeat(4500)}}};
  const defaultFit = await prepareQwen35GraphInput(tokenizer, longQuestion);
  const customFit = await prepareQwen35GraphInput(tokenizer, longQuestion, {maxQuestionFraction: 0.25});
  assert.equal(defaultFit.chunks[0]!.suffixInputShape[1] - customFit.chunks[0]!.suffixInputShape[1], 2048);
});

test('Qwen images group by maxBatch even when text batches have a token budget', async () => {
  const tokenizer = tokenizerFixture();
  const image = (await readFile(new URL('./fixtures/qwen35-image-input/single-pixel.png', import.meta.url))).toString(
    'base64'
  );
  const request = {state: 'Inspect the image.', images: [image], questions: questions(5)};
  const control = await prepareQwen35GraphInput(tokenizer, request, {maxBatch: 2});
  const budget = await prepareQwen35GraphInput(tokenizer, request, {maxBatch: 2, maxBatchTokens: 1});

  assert.deepEqual(
    budget.chunks.map(({questions: rows}) => rows.length),
    [2, 2, 1]
  );
  assert.equal(budget.inputTokenCount, control.inputTokenCount);
});

test('text and vision helper bounds split graph batches after source context fitting', () => {
  const tokenizer = tokenizerFixture();
  const imageGrid = [1, 16, 16] as const;
  const imageInput = {
    imageGridThw: [imageGrid],
    pixelShape: [256, 1536] as const,
    pixelValues: new Float32Array(256 * 1536)
  };
  const config = {
    imageTokenId: 248056,
    visionEndTokenId: 248054,
    padTokenId: 248044,
    numPositionEmbeddings: 2304,
    spatialMergeSize: 2,
    maxBatch: 33
  };
  const sourceQuestions = questions(33);
  sourceQuestions.q0 = {type: 'noul', instructions: 'i'.repeat(2200)};
  const prepared = prepareQwen35TextVisionInput(
    tokenizer,
    {
      state: 'shared state. '.repeat(500),
      images: ['AA=='],
      questions: sourceQuestions
    },
    imageInput,
    config
  );

  // Vision uses the same fixed graph microbatch size and preserves source question order.
  assert.deepEqual(
    prepared.chunks.map(({questions: rows}) => rows.length),
    Array.from({length: 11}, () => QWEN35_GRAPH_MAX_BATCH)
  );
  assert.deepEqual(
    prepared.chunks.flatMap(({questions: rows}) => rows.map(({name}) => name)),
    Object.keys(sourceQuestions)
  );
  assert.deepEqual(prepared.chunks[0]!.prefixInputIds, prepared.chunks[1]!.prefixInputIds);
  for (const chunk of prepared.chunks) {
    const [batch, suffixWidth] = chunk.suffixInputShape;
    const sequenceWidth = chunk.prefixInputShape[1] + suffixWidth;
    assert.ok(batch >= 1 && batch <= QWEN35_GRAPH_MAX_BATCH);
    assert.ok(sequenceWidth >= 2 && sequenceWidth <= 4096);
    for (const question of chunk.questions)
      assert.ok(question.optionIndices.length >= 2 && question.optionIndices.length <= 255);
  }
  assert.equal(prepared.chunks[0]!.prefixInputShape[1] + prepared.chunks[0]!.suffixInputShape[1], 4096);

  for (const optionCount of [2, 255]) {
    const optionInput = prepareQwen35TextVisionInput(
      tokenizer,
      {state: 'state', images: ['AA=='], questions: {pick: choiceQuestion(optionCount)}},
      imageInput,
      config
    );
    const [chunk] = optionInput.chunks;
    assert.ok(chunk);
    assert.equal(chunk.questions[0]!.optionIndices.length, optionCount);
    assert.ok(chunk.questions[0]!.optionIndices.every((index) => index < chunk.questions[0]!.inputIds.length));
  }

  assert.throws(
    () =>
      prepareQwen35TextVisionInput(
        tokenizer,
        {state: 'state', images: ['AA=='], questions: {pick: choiceQuestion(256)}},
        imageInput,
        config
      ),
    /requires 2 to 255 options/u
  );
});

test('Qwen EngineConfig options reject invalid source-like values without clamping valid ranges', () => {
  const tokenizer = tokenizerFixture();
  const request = {state: 'State.', questions: questions(1)};
  assert.throws(() => packQwen35Request(tokenizer, request, {maxBatch: 0}), /positive integer/u);
  assert.throws(() => packQwen35Request(tokenizer, request, {maxBatch: 1.5}), /positive integer/u);
  assert.throws(() => packQwen35Request(tokenizer, request, {maxQuestionFraction: Number.NaN}), /finite number/u);
  assert.throws(() => packQwen35Request(tokenizer, request, {maxBatchTokens: 1.5}), /integer or null/u);

  assert.equal(resolveQwen35BatchOptions({maxQuestionFraction: -0.5}).maxQuestionFraction, -0.5);
  assert.throws(
    () => packQwen35Request(tokenizer, request, {maxQuestionFraction: -0.5}),
    /An option has no surviving token/u
  );
  assert.doesNotThrow(() => packQwen35Request(tokenizer, request, {maxQuestionFraction: 2}));
  assert.doesNotThrow(() => packQwen35Request(tokenizer, request, {maxBatchTokens: -1}));
});

test('Qwen output chunk validation follows a configured max_batch', () => {
  const request = {
    questions: Object.entries(questions(33)).map(([name]) => ({
      name,
      type: 'noul' as const,
      instructions: 'Is this true?',
      instructionText: 'Is this true?',
      options: [
        {label: 'false', sourceDescription: null, description: 'false'},
        {label: 'true', sourceDescription: null, description: 'true'}
      ]
    }))
  };
  const logits = Array.from({length: 33}, () => Float32Array.of(0, 1));

  assert.throws(() => finishQwen35AnswersFromLogits(request, [logits]), /1 to 32 rows/u);
  assert.equal(finishQwen35AnswersFromLogits(request, [logits], {maxBatch: 33}).length, 33);
});
