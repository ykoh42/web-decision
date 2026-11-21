import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {stat} from 'node:fs/promises';
import {join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {hashSafetensorsFile, SAFETENSORS_HASH_CHUNK_BYTES} from '../kanana/hash_safetensors_ranges.mts';

export const MAPIKA_2B_REPOSITORY = 'Mapika/decider-2b';
export const MAPIKA_2B_REVISION = '7614a6c7ea5a047aa7d7bcfb1a0015469381c3e3';

export const MAPIKA_2B_WEIGHT_PINS = {
  text: {
    file: 'model.safetensors',
    bytes: 3_763_692_048,
    sha256: '1265b014ce710ef219eb137792705bc0e15a59f1c68460cf26394ae5416876e1',
    headerRangeBytes: 41_872,
    headerRangeSha256: '0457e3386fa22ac175a865c1f01490744fd2c087f9ae9145e085cc0d4a92820e',
    tensorCount: 320
  },
  vision: {
    file: 'vision/visual.safetensors',
    bytes: 662_866_152,
    sha256: 'bf7a6edb1196c9478f63736079aef71d4b0faa98c3c37fcc09f0239cc53d1c97',
    headerRangeBytes: 33_000,
    headerRangeSha256: '56398711c854e1e7b81d4f729f6f90fb00ef0e14effea04b2c556cdb75fb11d6',
    tensorCount: 297
  }
} as const;

const tokenizerPin = {
  bytes: 19_989_325,
  sha256: '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'
};
const tokenizerConfigPin = {
  bytes: 1_127,
  sha256: '171ecbe7ddae98d11840698f7df2b8d5b4722139db0f0620d3bbf429bd656250'
};
const chatTemplatePin = {
  bytes: 7_755,
  sha256: '273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80'
};

export const MAPIKA_2B_METADATA_PINS = {
  'README.md': {
    bytes: 9_876,
    sha256: 'ef48857e1de19444845090ed3253da6a9557ddfa98fa6434ca738948c7b84bbb'
  },
  'config.json': {bytes: 1_790, sha256: '6cb8daca9fb653c61485ff7452fc068bacd5c27cbee659ecd24b47186b0d1b52'},
  'decider_config.json': {
    bytes: 1_868,
    sha256: 'bbe0f382362b23a5ab9115e23d01a1b0511bf334c9355092a3b4d2ea0ddb5e1d'
  },
  'tokenizer.json': tokenizerPin,
  'tokenizer_config.json': tokenizerConfigPin,
  'chat_template.jinja': chatTemplatePin,
  'vision/config.json': {
    bytes: 2_733,
    sha256: '1eb0581a3232412f874acbeb2725292a22be2feebdce04b15363f4255464c07e'
  },
  'vision/processor_config.json': {
    bytes: 1_221,
    sha256: '9ad79dcc3fc51f5eaa519b43adcb3ff5b748a5a0af959237ad7b68b710b9e041'
  },
  'vision/tokenizer.json': tokenizerPin,
  'vision/tokenizer_config.json': tokenizerConfigPin,
  'vision/chat_template.jinja': chatTemplatePin
} as const;

async function verifyFile(snapshot: string, name: string, expected: {bytes: number; sha256: string}): Promise<void> {
  const file = join(snapshot, name);
  const info = await stat(file);
  assert.ok(info.isFile(), name + ' must be a regular source file.');
  assert.equal(info.size, expected.bytes, name + ' size differs from the pinned source.');
  const digest = createHash('sha256');
  for await (const chunk of createReadStream(file, {highWaterMark: SAFETENSORS_HASH_CHUNK_BYTES})) {
    digest.update(chunk);
  }
  assert.equal(digest.digest('hex'), expected.sha256, name + ' digest differs from the pinned source.');
}

async function collectWeights(
  snapshot: string,
  pin: (typeof MAPIKA_2B_WEIGHT_PINS)[keyof typeof MAPIKA_2B_WEIGHT_PINS]
) {
  await verifyFile(snapshot, pin.file, pin);
  const manifest = await hashSafetensorsFile(join(snapshot, pin.file), {path: pin.file});
  assert.equal(manifest.headerLength + 8, pin.headerRangeBytes, pin.file + ' header length changed.');
  assert.equal(manifest.headerSha256, pin.headerRangeSha256, pin.file + ' header hash changed.');
  assert.equal(manifest.tensorCount, pin.tensorCount, pin.file + ' tensor count changed.');
  assert.ok(
    manifest.tensors.every(({dtype}) => dtype === 'BF16'),
    pin.file + ' contains an unexpected dtype.'
  );
  return {...pin, tensors: manifest.tensors};
}

/** Verify the local snapshot and hash bounded tensor ranges for browser loading. */
export async function mapikaSourceInventory(snapshot: string) {
  for (const [name, pin] of Object.entries(MAPIKA_2B_METADATA_PINS)) await verifyFile(snapshot, name, pin);
  return {
    format: 'web-decision-mapika-2b-v12-source-inventory-v1',
    repository: MAPIKA_2B_REPOSITORY,
    revision: MAPIKA_2B_REVISION,
    maxRangeBytes: SAFETENSORS_HASH_CHUNK_BYTES,
    files: MAPIKA_2B_METADATA_PINS,
    weights: {
      text: await collectWeights(snapshot, MAPIKA_2B_WEIGHT_PINS.text),
      vision: await collectWeights(snapshot, MAPIKA_2B_WEIGHT_PINS.vision)
    }
  };
}

async function main(): Promise<void> {
  const [snapshot, ...extra] = process.argv.slice(2);
  assert.ok(snapshot && extra.length === 0, 'Usage: node --import tsx scripts/mapika/assets-inventory.mts <snapshot>');
  process.stdout.write(JSON.stringify(await mapikaSourceInventory(resolve(snapshot))) + '\n');
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();
