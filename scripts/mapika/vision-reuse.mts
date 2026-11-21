import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {open, readFile, stat} from 'node:fs/promises';
import {join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {parseSafetensorsHeader} from '../../packages/web-decision/src/kanana/safetensors.mjs';
import {
  QWEN35_VISION_BINDINGS_FILE_PIN,
  QWEN35_VISION_WASM_PIN
} from '../../packages/web-decision/src/strands/qwen35-vision-contract.mts';
import type {Qwen35VisionWeightBinding} from '../../packages/web-decision/src/strands/qwen35-vision-weights.mts';
import {
  MAPIKA_2B_METADATA_PINS,
  MAPIKA_2B_REPOSITORY,
  MAPIKA_2B_REVISION,
  MAPIKA_2B_WEIGHT_PINS
} from './assets-inventory.mts';

const qwen35Base = {
  repository: 'Qwen/Qwen3.5-2B-Base',
  revision: 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c',
  weights: 'model.safetensors-00001-of-00001.safetensors',
  bytes: 4_548_221_488,
  headerRangeBytes: 76_656,
  headerJsonSha256: 'ccba2c1f645fe59268f89ee7ea552e02ce6bf0c8a087bf332fb0e3bd99bfee9d',
  config: {bytes: 2_908, sha256: 'ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4'}
};
const graphAssetRoot = new URL('../../packages/web-decision/assets/qwen35/vision/', import.meta.url);
const graphName = 'qwen35-dynamic-vision.webgpu.wasm';
const imageTokenFields = ['image_token_id', 'vision_start_token_id', 'vision_end_token_id'] as const;

type FilePin = Readonly<{bytes: number; sha256: string}>;
type Tensor = ReturnType<typeof parseSafetensorsHeader>['tensors'][number];
type VisionConfiguration = Readonly<
  {vision_config: Record<string, unknown>} & Record<(typeof imageTokenFields)[number], number>
>;

function assertDigest(data: Uint8Array, digest: string, label: string): void {
  assert.equal(createHash('sha256').update(data).digest('hex'), digest, label + ' differs from the pinned SHA-256.');
}

async function readPinnedFile(file: string | URL, pin: FilePin): Promise<Buffer> {
  assert.equal((await stat(file)).size, pin.bytes, 'Pinned asset length differs.');
  const bytes = await readFile(file);
  assert.equal(bytes.length, pin.bytes, 'Pinned asset length differs.');
  assertDigest(bytes, pin.sha256, String(file));
  return bytes;
}

async function readBaseHeader(file: string): Promise<Tensor[]> {
  const handle = await open(file, 'r');
  try {
    assert.equal((await handle.stat()).size, qwen35Base.bytes, 'Qwen3.5 base file size differs.');
    const header = Buffer.alloc(qwen35Base.headerRangeBytes);
    let offset = 0;
    while (offset < header.length) {
      const {bytesRead} = await handle.read(header, offset, header.length - offset, offset);
      assert.ok(bytesRead > 0, 'Qwen3.5 base header ended early.');
      offset += bytesRead;
    }
    assertDigest(header.subarray(8), qwen35Base.headerJsonSha256, 'Qwen3.5 base header');
    return parseSafetensorsHeader(header, qwen35Base.bytes).tensors;
  } finally {
    await handle.close();
  }
}

async function verifyVisionConfiguration(baseSnapshot: string, mapikaConfigPath: string) {
  const base = await readPinnedFile(join(baseSnapshot, 'config.json'), qwen35Base.config);
  const target = await readPinnedFile(mapikaConfigPath, MAPIKA_2B_METADATA_PINS['vision/config.json']);
  const from = JSON.parse(base.toString('utf8')) as VisionConfiguration;
  const to = JSON.parse(target.toString('utf8')) as VisionConfiguration;
  // The pinned Transformers Qwen3.5 vision model supplies these defaults for the base config.
  const normalizedBase = {
    ...from.vision_config,
    model_type: 'qwen3_5_vision',
    dtype: 'bfloat16',
    rope_parameters: {rope_theta: 10_000, rope_type: 'axial'}
  };
  assert.deepEqual(to.vision_config, normalizedBase, 'Mapika and Qwen3.5 vision architectures differ.');
  for (const key of imageTokenFields) assert.equal(to[key], from[key], 'Image token differs: ' + key);
  return {sourceConfig: MAPIKA_2B_METADATA_PINS['vision/config.json'], rope: normalizedBase.rope_parameters};
}

function checkLayout(target: Tensor, source: Tensor, binding: Qwen35VisionWeightBinding): void {
  assert.equal(target.dtype, binding.sourceDtype, 'Mapika dtype differs: ' + target.name);
  assert.equal(source.dtype, target.dtype, 'Qwen dtype differs: ' + target.name);
  assert.deepEqual(target.shape, binding.shape, 'Mapika shape differs: ' + target.name);
  assert.deepEqual(source.shape, target.shape, 'Qwen shape differs: ' + target.name);
  assert.equal(target.length, binding.sourceByteCount, 'Mapika byte length differs: ' + target.name);
  assert.equal(source.length, target.length, 'Qwen byte length differs: ' + target.name);
}

async function hashTensor(
  file: string,
  tensor: Tensor,
  completeDigest: ReturnType<typeof createHash>
): Promise<string> {
  const digest = createHash('sha256');
  const stream = createReadStream(file, {start: tensor.offset, end: tensor.offset + tensor.length - 1});
  for await (const chunk of stream) {
    completeDigest.update(chunk);
    digest.update(chunk);
  }
  return digest.digest('hex');
}

/**
 * Recreate the published Mapika vision file in its own header order, directly
 * from Qwen vision tensors. Verify both the individual graph pins and the LFS hash.
 */
export async function proveMapikaVisionReuse(
  baseSnapshot: string,
  mapikaHeaderPath: string,
  mapikaVisionConfigPath: string
) {
  const pin = MAPIKA_2B_WEIGHT_PINS.vision;
  const header = await readPinnedFile(mapikaHeaderPath, {bytes: pin.headerRangeBytes, sha256: pin.headerRangeSha256});
  const configuration = await verifyVisionConfiguration(baseSnapshot, mapikaVisionConfigPath);
  const targets = parseSafetensorsHeader(header, pin.bytes).tensors.sort((a, b) => a.offset - b.offset);
  const sourceFile = join(baseSnapshot, qwen35Base.weights);
  const sources = new Map((await readBaseHeader(sourceFile)).map((tensor) => [tensor.name, tensor]));
  const bindingsBytes = await readPinnedFile(
    new URL('weight-bindings.json', graphAssetRoot),
    QWEN35_VISION_BINDINGS_FILE_PIN
  );
  const bindings = JSON.parse(bindingsBytes.toString('utf8')) as {bindings: Qwen35VisionWeightBinding[]};
  const bindingMap = new Map(bindings.bindings.map((binding) => [binding.sourceName, binding]));
  assert.equal(targets.length, pin.tensorCount, 'Mapika vision tensor count differs.');
  assert.equal(bindingMap.size, targets.length, 'Qwen vision parameter count differs.');
  const completeDigest = createHash('sha256').update(header);
  for (const target of targets) {
    const source = sources.get(target.name);
    const binding = bindingMap.get(target.name);
    assert.ok(source && binding, 'No Qwen graph binding for ' + target.name);
    checkLayout(target, source, binding);
    assert.equal(await hashTensor(sourceFile, source, completeDigest), binding.sourcePayloadSha256, target.name);
  }
  const reconstructedSha256 = completeDigest.digest('hex');
  assert.equal(
    reconstructedSha256,
    pin.sha256,
    'Reconstructed Mapika vision file differs from the official LFS asset.'
  );
  await readPinnedFile(new URL(graphName, graphAssetRoot), QWEN35_VISION_WASM_PIN);
  return {
    format: 'web-decision-mapika-2b-v12-vision-reuse-v1',
    repository: MAPIKA_2B_REPOSITORY,
    revision: MAPIKA_2B_REVISION,
    visionWeights: {...pin, reconstructedSha256},
    qwenBase: {repository: qwen35Base.repository, revision: qwen35Base.revision},
    graph: {file: graphName, ...QWEN35_VISION_WASM_PIN, bindings: QWEN35_VISION_BINDINGS_FILE_PIN},
    configuration
  };
}

async function main(): Promise<void> {
  const [baseSnapshot, mapikaHeader, mapikaVisionConfig, ...extra] = process.argv.slice(2);
  assert.ok(
    baseSnapshot && mapikaHeader && mapikaVisionConfig && extra.length === 0,
    'Usage: node --import tsx scripts/mapika/vision-reuse.mts <qwen-base-snapshot> <mapika-vision-header> <mapika-vision-config>'
  );
  process.stdout.write(
    JSON.stringify(await proveMapikaVisionReuse(baseSnapshot, mapikaHeader, mapikaVisionConfig)) + '\n'
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();
