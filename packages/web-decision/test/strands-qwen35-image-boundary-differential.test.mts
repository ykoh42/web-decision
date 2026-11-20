import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {encode as encodePng} from 'fast-png';
import {
  preprocessQwen35Image,
  preprocessQwen35Images,
  type Qwen35ImageInputOptions
} from '../src/strands/qwen35-image-input.mts';
import {
  buildQwen35PerImageVisionGeometry,
  buildQwen35PackedVisionCuSeqlens
} from '../src/strands/qwen35-vision-geometry.mts';

const fixtureDirectory = new URL('./fixtures/qwen35-image-input/', import.meta.url);
const source = Object.freeze({
  strandsRevision: '13c05149265dc404c09d2469636f71b1320e1713',
  transformersVersion: '5.22.2',
  preprocessorConfigSha256: '27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516',
  minPixels: 65_536,
  maxPixels: 16_777_216,
  patchSize: 16,
  resizeFactor: 32,
  maxImages: 4
});

type ImageOracle = Readonly<{
  file: string;
  inputSha256: string;
  imageLongSide?: number;
  decodedSize: readonly [number, number];
  fittedSize: readonly [number, number];
  processedSize: readonly [number, number];
  gridThw: readonly [1, number, number];
  patchRows: number;
  pixelsSha256: string;
}>;

const imageOracle: Readonly<Record<string, ImageOracle>> = Object.freeze({
  singlePixel: {
    file: 'single-pixel.png',
    inputSha256: 'ac5a13ec1e039de5073aa38d8cd724bf1a92dab5707a90aa32b14c1e470d43f7',
    decodedSize: [1, 1],
    fittedSize: [1, 1],
    processedSize: [256, 256],
    gridThw: [1, 16, 16],
    patchRows: 256,
    pixelsSha256: '3def6c99409a5602a037b874dc95f4bd89bd33cad6e43f6e6bfcc97ce05f12d8'
  },
  smallUpscale: {
    file: 'small-upscale.png',
    inputSha256: 'da7d567a36e98800b47499daf3a5d0571f58848bb2c3d162ba8c1cfdc221dbfc',
    decodedSize: [128, 128],
    fittedSize: [128, 128],
    processedSize: [256, 256],
    gridThw: [1, 16, 16],
    patchRows: 256,
    pixelsSha256: '95128119edfca327e48450f7944d6744c461cb05ae222ccfd5d9777d9efe08ce'
  },
  portrait: {
    file: 'vision-mid-p520.png',
    inputSha256: '960d62ccd00835601f0dc7c54c4fd0d948691565c0e6b8e16691786e391c55ca',
    decodedSize: [320, 416],
    fittedSize: [320, 416],
    processedSize: [320, 416],
    gridThw: [1, 26, 20],
    patchRows: 520,
    pixelsSha256: 'd8eca5e4a659c0939f5cc3182c3fb11b966c8d4fb7f1c157540d28779ca86887'
  },
  landscape: {
    file: 'fit-and-resize.png',
    inputSha256: '7cad350fc799d524e44b02856c3262b8e8f97d116ed7c1e7f07a3745c02b59d6',
    decodedSize: [640, 480],
    fittedSize: [448, 336],
    processedSize: [448, 320],
    gridThw: [1, 20, 28],
    patchRows: 560,
    pixelsSha256: '982151025b3d6a43b04db336d49dd019c170ce24da137536f31c7cc1e2d20541'
  },
  orientedPortrait: {
    file: 'png-orientation-6.png',
    inputSha256: 'db783597f6619a72d8ff66bb29a1eda9fb02ca0fe42d43ffdb8e69e803a88093',
    decodedSize: [32, 64],
    fittedSize: [32, 64],
    processedSize: [192, 384],
    gridThw: [1, 24, 12],
    patchRows: 288,
    pixelsSha256: 'eab8e0626bee268f554a41aafc8b2152e464dab4cef0e970aa21e7bb73c3e2ef'
  },
  aspectLimit: {
    file: 'aspect-boundary-200.png',
    inputSha256: '802ccf41c7d1d8c3405f6b70c1b957f225c1396e5b0bb7e85c5878c9c998b3d1',
    decodedSize: [400, 2],
    fittedSize: [400, 2],
    processedSize: [3648, 32],
    gridThw: [1, 2, 228],
    patchRows: 456,
    pixelsSha256: '84ceb2335b0d1097b40384da825b38c0988ab90a01a94a75277bf33dbb3f298c'
  },
  tiesToEven: {
    file: 'ties-to-even-80x1024.png',
    inputSha256: 'ac9902a91c684cff77f9516f041736bb08d96de78125e70e7e07ba715a1b4850',
    imageLongSide: 0,
    decodedSize: [80, 1024],
    fittedSize: [80, 1024],
    processedSize: [64, 1024],
    gridThw: [1, 64, 4],
    patchRows: 256,
    pixelsSha256: '6f9e3ff8e3d5bab6e7d4be867628248da57452af4efdaf32650bb29f0273d944'
  }
});

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function float32LittleEndian(values: Float32Array): Uint8Array {
  const bytes = new Uint8Array(values.length * Float32Array.BYTES_PER_ELEMENT);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) view.setFloat32(index * 4, values[index]!, true);
  return bytes;
}

function int32LittleEndian(values: Int32Array): Uint8Array {
  const bytes = new Uint8Array(values.length * Int32Array.BYTES_PER_ELEMENT);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) view.setInt32(index * 4, values[index]!, true);
  return bytes;
}

async function encodedImage(oracle: ImageOracle): Promise<string> {
  const bytes = await readFile(new URL(oracle.file, fixtureDirectory));
  assert.equal(sha256(bytes), oracle.inputSha256, `${oracle.file} pinned input bytes`);
  return Buffer.from(bytes).toString('base64');
}

function imageOptions(oracle: ImageOracle): Qwen35ImageInputOptions {
  return oracle.imageLongSide === undefined ? {} : {imageLongSide: oracle.imageLongSide};
}

test('Qwen preprocessing differentials match pinned Transformers boundary outputs', async () => {
  assert.equal(source.strandsRevision, '13c05149265dc404c09d2469636f71b1320e1713');
  assert.equal(source.transformersVersion, '5.22.2');
  assert.equal(source.preprocessorConfigSha256, '27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516');
  assert.deepEqual(
    [source.minPixels, source.maxPixels, source.patchSize, source.resizeFactor, source.maxImages],
    [65_536, 16_777_216, 16, 32, 4]
  );

  for (const [name, expected] of Object.entries(imageOracle)) {
    const encoded = await encodedImage(expected);
    const actual = await preprocessQwen35Image(encoded, imageOptions(expected));
    assert.deepEqual(actual.decodedSize, expected.decodedSize, `${name} decoded size`);
    assert.deepEqual(actual.fittedSize, expected.fittedSize, `${name} fit size`);
    assert.deepEqual(actual.processedSize, expected.processedSize, `${name} processor size`);
    assert.deepEqual(actual.imageGridThw, expected.gridThw, `${name} grid`);
    assert.deepEqual(actual.pixelShape, [expected.patchRows, 1536], `${name} patch rows`);
    assert.equal(sha256(float32LittleEndian(actual.pixelValues)), expected.pixelsSha256, `${name} source pixels`);
  }
});

test('Qwen aspect-ratio edge matches pinned Transformers at and just above 200', async () => {
  const belowLimit = encodePng({
    width: 399,
    height: 2,
    data: new Uint8Array(399 * 2 * 3).fill(127),
    depth: 8,
    channels: 3
  });
  const below = await preprocessQwen35Image(Buffer.from(belowLimit).toString('base64'));
  assert.deepEqual(below.decodedSize, [399, 2]);
  assert.deepEqual(below.processedSize, [3616, 32]);
  assert.deepEqual(below.imageGridThw, [1, 2, 226]);

  const exact = await preprocessQwen35Image(await encodedImage(imageOracle.aspectLimit!));
  assert.deepEqual(exact.processedSize, [3648, 32]);
  assert.deepEqual(exact.imageGridThw, [1, 2, 228]);

  const aboveLimit = encodePng({
    width: 401,
    height: 2,
    data: new Uint8Array(401 * 2 * 3).fill(127),
    depth: 8,
    channels: 3
  });
  await assert.rejects(
    preprocessQwen35Image(Buffer.from(aboveLimit).toString('base64')),
    /aspect ratio must not exceed 200/
  );
});

test('Qwen vision geometry matches source outputs at minimum and maximum processor pixel limits', () => {
  assert.equal(source.minPixels / (source.patchSize * source.patchSize), 256);
  assert.equal(source.maxPixels / (source.patchSize * source.patchSize), 65_536);

  const boundaryGoldens = [
    {
      grid: [1, 16, 16] as const,
      patchRows: 256,
      featureRows: 64,
      interpIndicesSha256: 'e7ef0a4c70bdc4a75fc172d776ed2646e41ff72d47715046354fe4bb87c4b12e',
      interpWeightsSha256: 'e115a98a3fc37defe2b405e58b6075970225f457022f65e2e10c88ec2ba7a77f',
      positionIdsSha256: '1bab81fecd461ffe4d43b3b6a7677838c54a1cb9e0cac096a891b8b19d4f6b6e'
    },
    {
      grid: [1, 256, 256] as const,
      patchRows: 65_536,
      featureRows: 16_384,
      interpIndicesSha256: 'b5337019fc13d3b4f5c3c8ed632ae160240d594da85c021b0d6d11fe57073636',
      interpWeightsSha256: 'fcecdb1668e3fd3ec032fb813bd34c5b5dac4443a775fb877b3dada28af73217',
      positionIdsSha256: 'e60480fcd786d0106e79c272e8f2ed7931597eb32bd2e69dc380ca36e36d7850'
    }
  ];

  for (const expected of boundaryGoldens) {
    const [geometry] = buildQwen35PerImageVisionGeometry([expected.grid], {
      numPositionEmbeddings: 2304,
      spatialMergeSize: 2
    });
    assert.ok(geometry);
    assert.equal(geometry.patchRows, expected.patchRows);
    assert.equal(geometry.featureRows, expected.featureRows);
    assert.equal(sha256(int32LittleEndian(geometry.interpIndices)), expected.interpIndicesSha256);
    assert.equal(sha256(float32LittleEndian(geometry.interpWeights)), expected.interpWeightsSha256);
    assert.equal(sha256(int32LittleEndian(geometry.positionIdsByToken)), expected.positionIdsSha256);
    assert.deepEqual([...buildQwen35PackedVisionCuSeqlens([geometry])], [0, expected.patchRows]);
  }
});

test('Qwen one-to-four image batches preserve pinned source order and concatenated pixels', async () => {
  const batches = [
    {
      name: 'one',
      keys: ['singlePixel'],
      sourcePixelsSha256: '3def6c99409a5602a037b874dc95f4bd89bd33cad6e43f6e6bfcc97ce05f12d8'
    },
    {
      name: 'two',
      keys: ['smallUpscale', 'portrait'],
      sourcePixelsSha256: '292dce2308508def640eee8ea310419806a48f8113cf800d6933b653c1e83703'
    },
    {name: 'three', keys: ['singlePixel', 'landscape', 'orientedPortrait']},
    {
      name: 'four',
      keys: ['singlePixel', 'landscape', 'orientedPortrait', 'aspectLimit'],
      sourcePixelsSha256: '69dfd2c81eea76c7993e619a4b3de7fdb4be0846951327f1708bcdcbd150d34d'
    }
  ] as const;

  for (const batch of batches) {
    const expectedImages = await Promise.all(
      batch.keys.map(async (key) => {
        const oracle = imageOracle[key];
        assert.ok(oracle, `${key} has a pinned source oracle`);
        return {oracle, encoded: await encodedImage(oracle)};
      })
    );
    const individual = await Promise.all(
      expectedImages.map(({encoded, oracle}) => preprocessQwen35Image(encoded, imageOptions(oracle)))
    );
    const actual = await preprocessQwen35Images(expectedImages.map(({encoded}) => encoded));
    assert.deepEqual(
      actual.imageGridThw,
      individual.map(({imageGridThw}) => imageGridThw),
      `${batch.name} grid order`
    );
    assert.equal(
      actual.pixelShape[0],
      individual.reduce((sum, image) => sum + image.pixelShape[0], 0)
    );

    const expectedPixels = new Float32Array(actual.pixelValues.length);
    let offset = 0;
    for (const image of individual) {
      expectedPixels.set(image.pixelValues, offset);
      offset += image.pixelValues.length;
    }
    assert.deepEqual(actual.pixelValues, expectedPixels, `${batch.name} pixel concatenation`);
    if ('sourcePixelsSha256' in batch)
      assert.equal(
        sha256(float32LittleEndian(actual.pixelValues)),
        batch.sourcePixelsSha256,
        `${batch.name} source batch`
      );
  }
});
