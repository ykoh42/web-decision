import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test from 'node:test';
import {
  buildQwen35PackedVisionCuSeqlens,
  buildQwen35PerImageVisionGeometry,
  getQwen35MaximumPackedPatchRows
} from '../src/strands/qwen35-vision-geometry.mts';

const config = {numPositionEmbeddings: 2304, spatialMergeSize: 2};

test('vision geometry accepts custom resize shapes and the pinned processor maximum', () => {
  const geometries = buildQwen35PerImageVisionGeometry(
    [
      [1, 26, 34],
      [1, 30, 40],
      [1, 256, 256]
    ],
    config
  );

  assert.deepEqual(
    geometries.map(({patchRows, featureRows}) => [patchRows, featureRows]),
    [
      [884, 221],
      [1200, 300],
      [65_536, 16_384]
    ]
  );
});

test('packed cu_seqlens keeps each of one to four images in its own ordered segment', () => {
  const geometries = buildQwen35PerImageVisionGeometry(
    [
      [1, 16, 16],
      [1, 26, 34],
      [1, 30, 40],
      [1, 256, 256]
    ],
    config
  );
  const expected = [
    [0, 256],
    [0, 256, 1140],
    [0, 256, 1140, 2340],
    [0, 256, 1140, 2340, 67_876]
  ];

  for (let imageCount = 1; imageCount <= 4; imageCount += 1)
    assert.deepEqual([...buildQwen35PackedVisionCuSeqlens(geometries.slice(0, imageCount))], expected[imageCount - 1]);
});

test('packed maximum four-image boundary arithmetic stays in int32 without padding', () => {
  const maximumImage = buildQwen35PerImageVisionGeometry([[1, 256, 256]], config)[0]!;
  assert.deepEqual(
    [...buildQwen35PackedVisionCuSeqlens([maximumImage, maximumImage, maximumImage, maximumImage])],
    [0, 65_536, 131_072, 196_608, 262_144]
  );
});

test('packed P maximum follows the effective maxImages option', () => {
  assert.equal(getQwen35MaximumPackedPatchRows(1), 65_536);
  assert.equal(getQwen35MaximumPackedPatchRows(4), 262_144);
  assert.equal(getQwen35MaximumPackedPatchRows(5), 327_680);

  const fiveImageConfig = {...config, maxImages: 5};
  const maximumImage = buildQwen35PerImageVisionGeometry(
    [
      [1, 256, 256],
      [1, 256, 256],
      [1, 256, 256],
      [1, 256, 256],
      [1, 256, 256]
    ],
    fiveImageConfig
  );
  assert.equal(
    maximumImage.reduce((sum, image) => sum + image.patchRows, 0),
    327_680
  );
  assert.deepEqual(
    [...buildQwen35PackedVisionCuSeqlens(maximumImage, 5)],
    [0, 65_536, 131_072, 196_608, 262_144, 327_680]
  );
});

test('packed boundary builder rejects an invalid image segment', () => {
  const geometry = buildQwen35PerImageVisionGeometry([[1, 16, 16]], config)[0]!;
  geometry.cuSeqlens[1] = 128;
  assert.throws(() => buildQwen35PackedVisionCuSeqlens([geometry]), /must carry the source \[0,P\] segment/);
});

test('interpolation and position geometry matches pinned Transformers 5.22.2 source tensors', () => {
  // Reference hashes come from transformers.vision_utils helpers with 48x48
  // source positions, bilinear interpolation, align_corners=True, and merge=2.
  const sourceHashes = [
    {
      grid: [1, 28, 28] as const,
      interpIndices: '092a77bf20c84d741a343889db77d4c41a64c8d16703f0c320c4d7e9840c21f0',
      interpWeights: 'd3e0d4f02cb6ab6e89dff5b4d6e6855fb92c3c3ee75388e1030aa423ba6b2959',
      positionIds: 'd72674d53ca91310e9ef481c7d9d7e5e4ddfd25fb133e24cd4a90508425eacff'
    },
    {
      grid: [1, 26, 20] as const,
      interpIndices: '0719971c95a2405c2e9e200484802f762cb6f52d2155f4d858ffc2467018a76d',
      interpWeights: '76e10cb8c67871741f98ebb72a936bd6a3d476e9a88ffd3e5b7af7f87a0d8b0b',
      positionIds: '867e21cc14731abe78e01083b1abbce0ed224bd4a72adfd967ee1d4045b8dc71'
    },
    {
      grid: [1, 20, 26] as const,
      interpIndices: '87b567e38f68d4772489910dd19ed54741456c10b2ae2298c68c262031d96a00',
      interpWeights: 'ccbe139ea12feb9da613dc8b1c5d32997f7024b07e5edbc32039943e423c0cb8',
      positionIds: 'daa53643d8ec7ed1502de8e27921b5f3ea2f25324bd95c66c004db5acc8a90eb'
    },
    {
      grid: [1, 16, 16] as const,
      interpIndices: 'e7ef0a4c70bdc4a75fc172d776ed2646e41ff72d47715046354fe4bb87c4b12e',
      interpWeights: 'e115a98a3fc37defe2b405e58b6075970225f457022f65e2e10c88ec2ba7a77f',
      positionIds: '1bab81fecd461ffe4d43b3b6a7677838c54a1cb9e0cac096a891b8b19d4f6b6e'
    },
    {
      grid: [1, 256, 256] as const,
      interpIndices: 'b5337019fc13d3b4f5c3c8ed632ae160240d594da85c021b0d6d11fe57073636',
      interpWeights: 'fcecdb1668e3fd3ec032fb813bd34c5b5dac4443a775fb877b3dada28af73217',
      positionIds: 'e60480fcd786d0106e79c272e8f2ed7931597eb32bd2e69dc380ca36e36d7850'
    }
  ];
  const digest = (value: ArrayBufferView) =>
    createHash('sha256')
      .update(new Uint8Array(value.buffer, value.byteOffset, value.byteLength))
      .digest('hex');

  for (const expected of sourceHashes) {
    const geometry = buildQwen35PerImageVisionGeometry([expected.grid], config)[0]!;
    assert.equal(digest(geometry.interpIndices), expected.interpIndices, `${expected.grid} interpolation indices`);
    assert.equal(digest(geometry.interpWeights), expected.interpWeights, `${expected.grid} interpolation weights`);
    assert.equal(digest(geometry.positionIdsByToken), expected.positionIds, `${expected.grid} position IDs`);
  }
});
