import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {prepareQwen35GraphInput} from '../src/strands/qwen35-graph-input.mts';
import {
  prepareQwen35R13GraphInputTuple,
  QWEN35_R13_RUNTIME_INPUT_NAMES
} from '../src/strands/qwen35-r13-pre-embed-bridge.mts';
import {QWEN35_TEMPERATURE_BY_KIND} from '../src/strands/qwen35-protocol.mts';

const utf8 = new TextEncoder();
const specialTokens = [
  ['<|image_pad|>', 248056],
  ['<|vision_end|>', 248054],
  ['<|endoftext|>', 248044],
  ['<bos>', 248043]
] as const;
const visionGoldens = JSON.parse(
  await readFile(new URL('./fixtures/qwen35-text-vision-goldens.json', import.meta.url), 'utf8')
) as {
  readonly source: {readonly checkpointRevision: string};
  readonly cases: readonly {
    readonly name: string;
    readonly files: readonly string[];
    readonly gridThw: readonly (readonly number[])[];
    readonly imageTokenCounts: readonly number[];
  }[];
};
const r13FixtureBytes = await readFile(new URL('../assets/qwen35/r13/fixture.json', import.meta.url));
const r13FixtureSha256 = createHash('sha256').update(r13FixtureBytes).digest('hex');
const r13Fixture = JSON.parse(r13FixtureBytes.toString('utf8')) as {
  readonly format: string;
  readonly mainParameterCount: number;
  readonly externalWeightCount: number;
  readonly runtimeInputNames: readonly string[];
  readonly parameterOrder: readonly {
    readonly index: number;
    readonly name: string;
    readonly external: boolean;
    readonly shape: readonly string[];
    readonly dtype: string;
  }[];
};
const fixtureDirectory = new URL('./fixtures/qwen35-image-input/', import.meta.url);
const require = createRequire(import.meta.url);
const magickWasm = await readFile(require.resolve('@imagemagick/magick-wasm/magick.wasm'));

function tokenizerFixture() {
  const byteDecoder = Object.fromEntries(
    Array.from({length: 256}, (_, byte) => [String.fromCharCode(byte + 0x100), byte])
  );
  const addedTokenIds = new Set(specialTokens.map(([, id]) => id));
  return {
    decoder: {byte_decoder: byteDecoder},
    get_added_tokens_decoder: () => addedTokenIds,
    encode(text: string, {add_special_tokens}: {readonly add_special_tokens: boolean}) {
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

async function imageData(file: string): Promise<string> {
  const bytes = await readFile(new URL(file, fixtureDirectory));
  return `data:image/png;base64,${bytes.toString('base64')}`;
}

function tokenEmbeddingValue(tokenId: number): number {
  // Every half-step below 64 is exactly representable as widened BF16.
  return (tokenId % 64) + 0.5;
}

function fillImageFeatureRows(imageIndex: number, rowCount: number): Float32Array {
  const values = new Float32Array(rowCount * 2048);
  for (let row = 0; row < rowCount; row += 1)
    values.fill(8 + imageIndex * 24 + (row % 8), row * 2048, (row + 1) * 2048);
  return values;
}

function assertPackedEmbeddingRows(
  chunk: Awaited<ReturnType<typeof prepareQwen35GraphInput>>['chunks'][number],
  input: Awaited<ReturnType<typeof prepareQwen35GraphInput>>,
  imageFeatures: Float32Array[],
  embeds: Awaited<ReturnType<typeof prepareQwen35R13GraphInputTuple>>[0],
  optionIndices: Awaited<ReturnType<typeof prepareQwen35R13GraphInputTuple>>[3],
  batch: number,
  sequence: number,
  hidden: number,
  optionWidth: number,
  prefix: number,
  suffix: number
) {
  for (let row = 0; row < batch; row += 1) {
    let imageIndex = 0;
    let featureRow = 0;
    for (let token = 0; token < prefix; token += 1) {
      const value = embeds.values[(row * sequence + token) * hidden]!;
      if (chunk.mmTokenTypeIds[token] !== 1) {
        assert.equal(value, tokenEmbeddingValue(chunk.prefixInputIds[token]!));
        continue;
      }
      assert.equal(value, imageFeatures[imageIndex]![featureRow * hidden]);
      featureRow += 1;
      if (featureRow === input.visionGeometry[imageIndex]!.featureRows) {
        imageIndex += 1;
        featureRow = 0;
      }
    }
    assert.equal(imageIndex, 2);

    const suffixOffset = row * suffix;
    const activeLength = chunk.questions[row]!.inputIds.length;
    for (let token = 0; token < suffix; token += 1) {
      const value = embeds.values[(row * sequence + prefix + token) * hidden]!;
      if (token < activeLength) assert.equal(value, tokenEmbeddingValue(chunk.suffixInputIds[suffixOffset + token]!));
      else assert.equal(value, 0, 'masked suffix padding has zero embedding data');
    }

    const optionOffset = row * optionWidth;
    const valid = chunk.questions[row]!.optionIndices;
    assert.deepEqual(
      Array.from(optionIndices.values.subarray(optionOffset, optionOffset + valid.length)),
      Array.from(valid, (index) => prefix + index)
    );
    assert.ok(optionIndices.values.slice(optionOffset + valid.length, optionOffset + 3).every((index) => index === -1));
  }
}

test('R13 pre-embed bridge orders six ABI inputs and composes pinned interleaved image rows', async () => {
  assert.equal(r13FixtureSha256, 'bde2174935a064392cadea8331c03752980fc99ef3f0de0807d4499b994b105b');
  assert.equal(r13Fixture.format, 'qwen35-full-pointer-tvmjs-fixture-v1');
  assert.equal(r13Fixture.mainParameterCount, 703);
  assert.equal(r13Fixture.externalWeightCount, 697);
  assert.deepEqual(r13Fixture.runtimeInputNames, QWEN35_R13_RUNTIME_INPUT_NAMES);
  assert.deepEqual(
    r13Fixture.parameterOrder.slice(0, 6).map(({index, name, external, shape, dtype}) => ({
      index,
      name,
      external,
      shape,
      dtype
    })),
    [
      {index: 0, name: 'inputs_embeds', external: false, shape: ['s0', 's22', 'T.int64(2048)'], dtype: 'float32'},
      {index: 1, name: 'attention_mask', external: false, shape: ['s0', 's22'], dtype: 'int32'},
      {index: 2, name: 'position_ids', external: false, shape: ['T.int64(4)', 's0', 's22'], dtype: 'int32'},
      {index: 3, name: 'option_indices', external: false, shape: ['s0', 's86'], dtype: 'int32'},
      {index: 4, name: 'temperatures', external: false, shape: ['s0'], dtype: 'float32'},
      {index: 5, name: 'n_slots', external: false, shape: ['s0'], dtype: 'int32'}
    ]
  );
  assert.equal(visionGoldens.source.checkpointRevision, '216d1ccb46a9e9b036eb9187b248db98480f77d4');
  const golden = visionGoldens.cases.find(({name}) => name === 'interleaved-two-images');
  assert.ok(golden, 'pinned two-image source geometry is present');
  assert.deepEqual(golden.files, ['small-upscale.png', 'vision-mid-p520.png']);

  const [smallImage, midImage] = await Promise.all(golden.files.map(imageData));
  const input = await prepareQwen35GraphInput(
    tokenizerFixture(),
    {
      state: 'Read the attachments in their listed order.',
      images: [smallImage, midImage],
      questions: {
        short: {type: 'choice', instructions: 'Pick one.', criteria: {first: null, second: null}},
        ordinal: {type: 'score', instructions: 'Rate the second attachment.', criteria: ['low', 'medium', 'high']},
        truth: {type: 'noul', instructions: 'Is the second image wider than the first?'}
      }
    },
    {magickWasm, maxBatch: 3}
  );
  const chunk = input.chunks[0]!;
  assert.equal(chunk.questions.length, 3);
  assert.deepEqual(
    Array.from({length: 2}, (_, image) => Array.from(input.imageGridThw.slice(image * 3, image * 3 + 3))),
    golden.gridThw
  );
  assert.deepEqual(
    input.visionGeometry.map(({featureRows}) => featureRows),
    golden.imageTokenCounts
  );

  const imageFeatures = input.visionGeometry.map(({featureRows}, imageIndex) =>
    fillImageFeatureRows(imageIndex, featureRows)
  );
  let lookedUpTokenIds = new Int32Array();
  const controller = new AbortController();
  let receivedSignal: AbortSignal | null | undefined;
  const tuple = await prepareQwen35R13GraphInputTuple(
    chunk,
    input,
    {
      imageFeatures,
      lookupTokenEmbeddings(tokenIds, options) {
        lookedUpTokenIds = new Int32Array(tokenIds);
        receivedSignal = options?.signal;
        const values = new Float32Array(tokenIds.length * 2048);
        tokenIds.forEach((tokenId, row) => values.fill(tokenEmbeddingValue(tokenId), row * 2048, (row + 1) * 2048));
        return values;
      }
    },
    {signal: controller.signal}
  );

  assert.equal(receivedSignal, controller.signal);
  assert.deepEqual(
    tuple.map(({name}) => name),
    QWEN35_R13_RUNTIME_INPUT_NAMES
  );
  assert.deepEqual(
    tuple.map(({dtype}) => dtype),
    ['float32', 'int32', 'int32', 'int32', 'float32', 'int32']
  );
  const [embeds, attention, positions, optionIndices, temperatures, nSlots] = tuple;
  const batch = embeds.shape[0]!;
  const sequence = embeds.shape[1]!;
  const hidden = embeds.shape[2]!;
  const optionWidth = optionIndices.shape[1]!;
  assert.deepEqual([batch, sequence, hidden], [3, chunk.prefixInputIds.length + chunk.suffixInputShape[1], 2048]);
  assert.deepEqual(attention.shape, [batch, sequence]);
  assert.deepEqual(positions.shape, [4, batch, sequence]);
  assert.deepEqual(optionIndices.shape, [batch, 3]);
  assert.deepEqual(temperatures.shape, [batch]);
  assert.deepEqual(nSlots.shape, [batch]);
  assert.deepEqual(attention.values, chunk.fullAttentionMask);
  assert.deepEqual(Array.from(nSlots.values), [2, 3, 2]);
  assert.deepEqual(Array.from(temperatures.values), [
    Math.fround(QWEN35_TEMPERATURE_BY_KIND.choice),
    Math.fround(QWEN35_TEMPERATURE_BY_KIND.score),
    Math.fround(QWEN35_TEMPERATURE_BY_KIND.noul)
  ]);
  assert.ok(lookedUpTokenIds.every((tokenId) => tokenId !== 248056 && tokenId !== 248044));
  assert.equal(
    lookedUpTokenIds.length,
    batch * chunk.mmTokenTypeIds.filter((modality) => modality === 0).length +
      chunk.questions.reduce((sum, question) => sum + question.inputIds.length, 0)
  );

  const prefix = chunk.prefixInputIds.length;
  const suffix = chunk.suffixInputShape[1];
  for (let axis = 0; axis < 4; axis += 1) {
    for (let row = 0; row < batch; row += 1) {
      const target = (axis * batch + row) * sequence;
      const prefixSource = axis * prefix;
      const suffixSource = (axis * batch + row) * suffix;
      assert.deepEqual(
        positions.values.subarray(target, target + prefix),
        chunk.prefixPositionIds.subarray(prefixSource, prefixSource + prefix)
      );
      assert.deepEqual(
        positions.values.subarray(target + prefix, target + sequence),
        chunk.suffixPositionIds.subarray(suffixSource, suffixSource + suffix)
      );
    }
  }

  assertPackedEmbeddingRows(
    chunk,
    input,
    imageFeatures,
    embeds,
    optionIndices,
    batch,
    sequence,
    hidden,
    optionWidth,
    prefix,
    suffix
  );
});

test('R13 bridge rejects non-source BF16 embeddings and mismatched per-image features', async () => {
  const input = await prepareQwen35GraphInput(tokenizerFixture(), {
    state: 'Text only.',
    questions: {route: {type: 'choice', instructions: 'Choose.', criteria: {left: null, right: null}}}
  });
  const chunk = input.chunks[0]!;
  await assert.rejects(
    prepareQwen35R13GraphInputTuple(chunk, input, {
      imageFeatures: [],
      lookupTokenEmbeddings(tokenIds) {
        return new Float32Array(tokenIds.length * 2048).fill(0.1);
      }
    }),
    /source BF16 values widened exactly/u
  );

  const image = await imageData('small-upscale.png');
  const imageInput = await prepareQwen35GraphInput(
    tokenizerFixture(),
    {
      state: 'Inspect the image.',
      images: [image],
      questions: {route: {type: 'choice', instructions: 'Choose.', criteria: {left: null, right: null}}}
    },
    {magickWasm}
  );
  await assert.rejects(
    prepareQwen35R13GraphInputTuple(imageInput.chunks[0]!, imageInput, {
      imageFeatures: [new Float32Array(2048)],
      lookupTokenEmbeddings(tokenIds) {
        const values = new Float32Array(tokenIds.length * 2048);
        values.fill(0.5);
        return values;
      }
    }),
    /pinned \[S,2048\] geometry/u
  );
  const features = new Float32Array(imageInput.visionGeometry[0]!.featureRows * 2048).fill(0.1);
  const sources = {
    imageFeatures: [features],
    lookupTokenEmbeddings: (tokenIds: Int32Array) => new Float32Array(tokenIds.length * 2048)
  };
  const imageChunk = imageInput.chunks[0]!;
  await assert.rejects(prepareQwen35R13GraphInputTuple(imageChunk, imageInput, sources), /source BF16/);
  const fp32Sources = {...sources, visionActivationPrecision: 'float32'} as const;
  const [embeddings] = await prepareQwen35R13GraphInputTuple(imageChunk, imageInput, fp32Sources);
  assert.equal(embeddings.values[imageChunk.mmTokenTypeIds.indexOf(1) * 2048], Math.fround(0.1));
  features[0] = NaN;
  await assert.rejects(prepareQwen35R13GraphInputTuple(imageChunk, imageInput, fp32Sources), /finite float32/);
});
