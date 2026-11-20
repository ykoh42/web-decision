import {sha256, waitForAbort} from '../assets.js';
import type {AssetCache} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {RangeCacheTransaction} from './range-cache.mjs';
import type {RangeCacheReadOptions, RangeCacheSpec} from './range-cache.mjs';

export const QWEN35_WORD_MANIFEST_PATH = 'manifest.json';
export const QWEN35_WORD_MANIFEST_BYTES = 761512;
export const QWEN35_WORD_MANIFEST_SHA256 = '54256df07cde4990833d232acac1d6962d3945f1ccdd1957ff1a6fab773dac69';

const MAX_RANGE_BYTES = 8 * 1024 * 1024;
const FORMAT = 'web-decision-qwen35-pinned-word-assets-v2';
const invalid = (message: string) => new WebDecisionError('INVALID_ASSET', message);

export type Qwen35WordAssetId = 'base' | 'lora' | 'head';

export interface Qwen35WordRange {
  readonly offset: number;
  readonly bytes: number;
  readonly sha256: string;
}

export interface Qwen35WordTensor {
  readonly name: string;
  readonly shape: readonly number[];
  readonly dtype: 'BF16' | 'F32';
  readonly elementCount: number;
  readonly sourceDataByteOffset: number;
  readonly storageByteOffset: number;
  readonly storageWordOffset: number;
  readonly sourceByteCount: number;
  readonly packedByteCount: number;
  readonly paddingByteCount: number;
  readonly sourcePayloadSha256: string;
  readonly packedTensorSha256: string;
}

export interface Qwen35WordAsset {
  readonly source: Readonly<{
    repository: string;
    revision: string;
    file: string;
    bytes: number;
    sha256: string;
    pinnedSha256: string;
    headerBytes: number;
    headerSha256: string;
    payloadBytes: number;
  }>;
  readonly packed: Readonly<{
    file: string;
    bytes: number;
    sha256: string;
    storageDtype: 'uint32';
    endianness: 'little';
    rangeChunkBytes: number;
    rangeChunks: readonly Qwen35WordRange[];
  }>;
  readonly tensorCount: number;
  readonly dtypeCounts: Readonly<Record<string, number>>;
  readonly tensors: readonly Qwen35WordTensor[];
}

export interface Qwen35WordAssetManifest {
  readonly format: typeof FORMAT;
  readonly storage: Readonly<{
    wordDtype: 'uint32';
    endianness: 'little';
    bf16Layout: string;
    f32Layout: string;
    tensorBoundaryPadding: string;
    maxChunkBytes: number;
  }>;
  readonly assets: Readonly<Record<Qwen35WordAssetId, Qwen35WordAsset>>;
}

interface AssetPin {
  readonly repository: string;
  readonly revision: string;
  readonly sourceFile: string;
  readonly sourceBytes: number;
  readonly sourceSha256: string;
  readonly headerBytes: number;
  readonly headerSha256: string;
  readonly packedFile: string;
  readonly packedBytes: number;
  readonly packedSha256: string;
  readonly tensorCount: number;
  readonly dtypeCounts: Readonly<Record<string, number>>;
  readonly tensorInventorySha256: string;
  readonly rangeInventorySha256: string;
}

const ASSET_PINS: Readonly<Record<Qwen35WordAssetId, AssetPin>> = Object.freeze({
  base: Object.freeze({
    repository: 'Qwen/Qwen3.5-2B-Base',
    revision: 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c',
    sourceFile: 'model.safetensors-00001-of-00001.safetensors',
    sourceBytes: 4548221488,
    sourceSha256: '928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d',
    headerBytes: 76648,
    headerSha256: 'ccba2c1f645fe59268f89ee7ea552e02ce6bf0c8a087bf332fb0e3bd99bfee9d',
    packedFile: 'base.words.u32le',
    packedBytes: 4548144832,
    packedSha256: '8b923cb255cbf1a0967a881f5fb97540aed6a06dd4dac91d869caa7547c42a01',
    tensorCount: 632,
    dtypeCounts: Object.freeze({BF16: 596, F32: 36}),
    tensorInventorySha256: '5999424a1caedde013fbc1c897d37ae04413afdd4324dd66e72fb69ccfa346aa',
    rangeInventorySha256: '67a0e7cea78002dcd6cb64b6460a638f1fc59339882c9fc40175f04789bd3a60'
  }),
  lora: Object.freeze({
    repository: 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610',
    revision: '216d1ccb46a9e9b036eb9187b248db98480f77d4',
    sourceFile: 'lora/adapter_model.safetensors',
    sourceBytes: 201878888,
    sourceSha256: 'ae1d0cd1ac3edec6df0a519a3ad18f1c382099032d493f8f98c96a80cfebb4c7',
    headerBytes: 48480,
    headerSha256: 'e04b556003455c6b7409a241a4560c04d9bf3963086adeba8beed890e3aac92f',
    packedFile: 'lora-adapter.words.u32le',
    packedBytes: 201830400,
    packedSha256: 'b6782ad0da77c127be3611a9b4ffff727b74986919107544729c2492ec25f7db',
    tensorCount: 372,
    dtypeCounts: Object.freeze({F32: 372}),
    tensorInventorySha256: '97f0112a0eb86b28d4bb0c6e430cdba4ae0b3690ba90b72452e485b734211957',
    rangeInventorySha256: '615faa8a8e6b910524fe9f3741ba91a44a751313485fbc23f8dd47c52aa48246'
  }),
  head: Object.freeze({
    repository: 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610',
    revision: '216d1ccb46a9e9b036eb9187b248db98480f77d4',
    sourceFile: 'head.safetensors',
    sourceBytes: 12605928,
    sourceSha256: 'ace56742545b505e6fae34ada85c8edbeedb23ba8974d8680aebe0380d6083a1',
    headerBytes: 480,
    headerSha256: 'c6519609a373c365dca59de5749fc286bf2dfbe27e3e5a5d7c73b88b40c86941',
    packedFile: 'pointer-head.words.u32le',
    packedBytes: 12605440,
    packedSha256: '2f78e52281c5d85ef1e749f202ef82f4f5515e228d8231981352c6bfc2034f62',
    tensorCount: 6,
    dtypeCounts: Object.freeze({F32: 6}),
    tensorInventorySha256: 'e0759e098f2c44c06306deb6d627be39d10e07ecc828589cb38ca5df40a10d5c',
    rangeInventorySha256: '33658c5bb0ad6158ed1b9607e3ad88461c7cbaa4b7e8b5e2cf22af494774996f'
  })
});

const MANIFEST_RANGE: RangeCacheSpec = Object.freeze({
  path: QWEN35_WORD_MANIFEST_PATH,
  totalBytes: QWEN35_WORD_MANIFEST_BYTES,
  offset: 0,
  length: QWEN35_WORD_MANIFEST_BYTES,
  sha256: QWEN35_WORD_MANIFEST_SHA256
});

function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw invalid('Invalid Qwen3.5 ' + label + '.');
  return value as Record<string, unknown>;
}

function integer(value: unknown, label: string, minimum = 0): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum) throw invalid('Invalid Qwen3.5 ' + label + '.');
  return value as number;
}

function digest(value: unknown, label: string): string {
  if (typeof value !== 'string' || !/^[0-9a-f]{64}$/.test(value)) throw invalid('Invalid Qwen3.5 ' + label + '.');
  return value;
}

function equal(value: unknown, expected: string | number, label: string): void {
  if (value !== expected) throw invalid('Unexpected Qwen3.5 ' + label + '.');
}

function validateDtypeCounts(
  value: unknown,
  expected: Readonly<Record<string, number>>
): Readonly<Record<string, number>> {
  const counts = record(value, 'dtype counts');
  const names = Object.keys(counts).sort();
  const expectedNames = Object.keys(expected).sort();
  if (names.length !== expectedNames.length || names.some((name, index) => name !== expectedNames[index]))
    throw invalid('Unexpected Qwen3.5 dtype inventory.');
  for (const name of expectedNames) equal(counts[name], expected[name]!, 'dtype count');
  return Object.freeze({...expected});
}

function parseRanges(value: unknown, asset: string, totalBytes: number): readonly Qwen35WordRange[] {
  if (!Array.isArray(value) || !value.length) throw invalid('Missing Qwen3.5 range hashes for ' + asset + '.');
  let offset = 0;
  const chunks = value.map((candidate) => {
    const chunk = record(candidate, 'range metadata');
    equal(chunk.offset, offset, 'range offset for ' + asset);
    const bytes = integer(chunk.bytes, 'range byte count', 1);
    if (bytes > MAX_RANGE_BYTES || bytes % 4 !== 0 || offset % 4 !== 0 || offset > totalBytes - bytes)
      throw invalid('Qwen3.5 range exceeds its packed asset: ' + asset + '.');
    const parsed = Object.freeze({offset, bytes, sha256: digest(chunk.sha256, 'range hash')});
    offset += bytes;
    return parsed;
  });
  if (offset !== totalBytes) throw invalid('Qwen3.5 range hashes do not cover ' + asset + '.');
  return Object.freeze(chunks);
}

function parseTensors(
  value: unknown,
  asset: Qwen35WordAssetId,
  packedBytes: number,
  payloadBytes: number,
  expectedCount: number
): readonly Qwen35WordTensor[] {
  if (!Array.isArray(value) || value.length !== expectedCount)
    throw invalid('Unexpected Qwen3.5 tensor count for ' + asset + '.');
  let sourceOffset = 0;
  let storageOffset = 0;
  const names = new Set<string>();
  const tensors = value.map((candidate) => {
    const item = record(candidate, 'tensor metadata');
    if (typeof item.name !== 'string' || !item.name || names.has(item.name))
      throw invalid('Invalid or duplicate Qwen3.5 tensor name.');
    names.add(item.name);
    if (!Array.isArray(item.shape) || item.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 1))
      throw invalid('Invalid Qwen3.5 tensor shape: ' + item.name + '.');
    if (item.dtype !== 'BF16' && item.dtype !== 'F32') throw invalid('Unsupported Qwen3.5 tensor dtype.');
    const shape = Object.freeze((item.shape as number[]).slice());
    const elementCount = shape.reduce((count, dimension) => count * dimension, 1);
    if (!Number.isSafeInteger(elementCount)) throw invalid('Invalid Qwen3.5 tensor element count.');
    equal(item.elementCount, elementCount, 'tensor element count');
    const sourceByteCount = elementCount * (item.dtype === 'BF16' ? 2 : 4);
    equal(item.sourceByteCount, sourceByteCount, 'tensor source byte count');
    equal(item.sourceDataByteOffset, sourceOffset, 'tensor source offset');
    const paddingByteCount = -sourceByteCount % 4;
    equal(item.paddingByteCount, paddingByteCount, 'tensor padding');
    equal(item.packedByteCount, sourceByteCount + paddingByteCount, 'tensor packed byte count');
    equal(item.storageByteOffset, storageOffset, 'tensor storage offset');
    equal(item.storageWordOffset, storageOffset / 4, 'tensor storage word offset');
    const parsed = Object.freeze({
      name: item.name,
      shape,
      dtype: item.dtype,
      elementCount,
      sourceDataByteOffset: sourceOffset,
      storageByteOffset: storageOffset,
      storageWordOffset: storageOffset / 4,
      sourceByteCount,
      packedByteCount: sourceByteCount + paddingByteCount,
      paddingByteCount,
      sourcePayloadSha256: digest(item.sourcePayloadSha256, 'tensor source hash'),
      packedTensorSha256: digest(item.packedTensorSha256, 'tensor packed hash')
    });
    if (paddingByteCount === 0 && parsed.sourcePayloadSha256 !== parsed.packedTensorSha256)
      throw invalid('Qwen3.5 tensor payload hashes do not match: ' + item.name + '.');
    sourceOffset += sourceByteCount;
    storageOffset += sourceByteCount + paddingByteCount;
    return parsed;
  });
  if (sourceOffset !== payloadBytes || storageOffset !== packedBytes)
    throw invalid('Qwen3.5 tensor metadata does not cover packed asset ' + asset + '.');
  return Object.freeze(tensors);
}

/** Validate the converter manifest against the single curated Qwen3.5 Strands checkpoint. */
export async function validateQwen35WordAssetManifest(value: unknown): Promise<Qwen35WordAssetManifest> {
  const manifest = record(value, 'word-asset manifest');
  equal(manifest.format, FORMAT, 'word-asset format');
  const storage = record(manifest.storage, 'storage format');
  equal(storage.wordDtype, 'uint32', 'storage dtype');
  equal(storage.endianness, 'little', 'storage endianness');
  equal(storage.bf16Layout, 'flatten row-major; element 2n in low16, element 2n+1 in high16', 'BF16 layout');
  equal(storage.f32Layout, 'one unchanged 32-bit source value per word', 'F32 layout');
  equal(
    storage.tensorBoundaryPadding,
    'zero bytes to next 4-byte boundary; excluded by logical shape',
    'tensor padding'
  );
  equal(storage.maxChunkBytes, MAX_RANGE_BYTES, 'maximum range size');

  const rawAssets = record(manifest.assets, 'asset set');
  const labels = Object.keys(rawAssets).sort();
  if (labels.join(',') !== 'base,head,lora') throw invalid('Unexpected Qwen3.5 asset set.');
  const assets = {} as Record<Qwen35WordAssetId, Qwen35WordAsset>;
  for (const id of ['base', 'lora', 'head'] as const) {
    const pin = ASSET_PINS[id];
    const raw = record(rawAssets[id], id + ' asset');
    const source = record(raw.source, id + ' source');
    equal(source.repository, pin.repository, id + ' source repository');
    equal(source.revision, pin.revision, id + ' source revision');
    equal(source.file, pin.sourceFile, id + ' source path');
    equal(source.bytes, pin.sourceBytes, id + ' source size');
    equal(source.sha256, pin.sourceSha256, id + ' source hash');
    equal(source.pinnedSha256, pin.sourceSha256, id + ' source pin');
    equal(source.headerBytes, pin.headerBytes, id + ' source header size');
    equal(source.headerSha256, pin.headerSha256, id + ' source header hash');
    const payloadBytes = integer(source.payloadBytes, id + ' source payload size');
    if (payloadBytes + pin.headerBytes + 8 !== pin.sourceBytes) throw invalid('Invalid Qwen3.5 source payload size.');

    const packed = record(raw.packed, id + ' packed asset');
    equal(packed.file, pin.packedFile, id + ' packed file');
    equal(packed.bytes, pin.packedBytes, id + ' packed size');
    equal(packed.sha256, pin.packedSha256, id + ' packed hash');
    equal(packed.storageDtype, 'uint32', id + ' packed dtype');
    equal(packed.endianness, 'little', id + ' packed endianness');
    equal(packed.rangeChunkBytes, MAX_RANGE_BYTES, id + ' range chunk size');
    const rangeChunks = parseRanges(packed.rangeChunks, id, pin.packedBytes);
    const rangeInventory = rangeChunks.map(({offset, bytes, sha256}) => [offset, bytes, sha256]);
    const rangeInventoryDigest = await sha256(new TextEncoder().encode(JSON.stringify(rangeInventory)));
    if (rangeInventoryDigest !== pin.rangeInventorySha256)
      throw invalid('Unexpected Qwen3.5 ' + id + ' range hash inventory.');

    equal(raw.tensorCount, pin.tensorCount, id + ' tensor count');
    const dtypeCounts = validateDtypeCounts(raw.dtypeCounts, pin.dtypeCounts);
    const tensors = parseTensors(raw.tensors, id, pin.packedBytes, payloadBytes, pin.tensorCount);
    const inventoryRecords = tensors.map(({name, shape, dtype, sourceDataByteOffset}) => [
      name,
      shape,
      dtype,
      sourceDataByteOffset
    ]);
    const inventoryDigest = await sha256(new TextEncoder().encode(JSON.stringify(inventoryRecords)));
    if (inventoryDigest !== pin.tensorInventorySha256)
      throw invalid('Unexpected Qwen3.5 tensor inventory for ' + id + '.');
    const actualDtypeCounts: Record<string, number> = {};
    for (const tensor of tensors) actualDtypeCounts[tensor.dtype] = (actualDtypeCounts[tensor.dtype] ?? 0) + 1;
    validateDtypeCounts(actualDtypeCounts, pin.dtypeCounts);
    assets[id] = Object.freeze({
      source: Object.freeze({
        repository: pin.repository,
        revision: pin.revision,
        file: pin.sourceFile,
        bytes: pin.sourceBytes,
        sha256: pin.sourceSha256,
        pinnedSha256: pin.sourceSha256,
        headerBytes: pin.headerBytes,
        headerSha256: pin.headerSha256,
        payloadBytes
      }),
      packed: Object.freeze({
        file: pin.packedFile,
        bytes: pin.packedBytes,
        sha256: pin.packedSha256,
        storageDtype: 'uint32',
        endianness: 'little',
        rangeChunkBytes: MAX_RANGE_BYTES,
        rangeChunks
      }),
      tensorCount: pin.tensorCount,
      dtypeCounts,
      tensors
    });
  }
  return Object.freeze({
    format: FORMAT,
    storage: Object.freeze({
      wordDtype: 'uint32',
      endianness: 'little',
      bf16Layout: storage.bf16Layout as string,
      f32Layout: storage.f32Layout as string,
      tensorBoundaryPadding: storage.tensorBoundaryPadding as string,
      maxChunkBytes: MAX_RANGE_BYTES
    }),
    assets: Object.freeze(assets)
  });
}

export async function parsePinnedQwen35WordAssetManifest(
  bytes: Uint8Array,
  signal?: AbortSignal | null
): Promise<Qwen35WordAssetManifest> {
  if (!(bytes instanceof Uint8Array) || bytes.byteLength !== QWEN35_WORD_MANIFEST_BYTES)
    throw invalid('Unexpected Qwen3.5 word-asset manifest size.');
  const actualDigest = await waitForAbort(() => sha256(bytes), signal);
  signal?.throwIfAborted();
  if (actualDigest !== QWEN35_WORD_MANIFEST_SHA256) throw invalid('Qwen3.5 word-asset manifest hash mismatch.');
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(bytes)) as unknown;
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Could not parse the Qwen3.5 word-asset manifest.', {cause});
  }
  signal?.throwIfAborted();
  const manifest = await validateQwen35WordAssetManifest(value);
  signal?.throwIfAborted();
  return manifest;
}

export interface OpenQwen35WordAssetsOptions {
  readonly base: URL;
  readonly cache?: AssetCache | null | undefined;
  readonly offline?: boolean | undefined;
  readonly signal?: AbortSignal | null | undefined;
}

export interface Qwen35WordAssetSource {
  readonly manifest: Qwen35WordAssetManifest;
  readRange(asset: Qwen35WordAssetId, rangeIndex: number, options?: RangeCacheReadOptions): Promise<ArrayBuffer>;
  commit(options?: RangeCacheReadOptions): Promise<boolean>;
}

/** Open hash-pinned word assets and mark offline reuse complete only after every range is cached. */
export async function openQwen35WordAssets({
  base,
  cache,
  offline = false,
  signal
}: OpenQwen35WordAssetsOptions): Promise<Qwen35WordAssetSource> {
  const transaction = new RangeCacheTransaction({
    base,
    cache,
    offline,
    signal,
    markerId: 'strands-qwen35-word-assets-v2:' + QWEN35_WORD_MANIFEST_SHA256
  });
  await transaction.open();
  transaction.require([MANIFEST_RANGE]);
  const manifestResult = await transaction.read(base, MANIFEST_RANGE, {signal});
  const manifest = await parsePinnedQwen35WordAssetManifest(new Uint8Array(manifestResult.bytes), signal);
  const specs = {} as Record<Qwen35WordAssetId, readonly RangeCacheSpec[]>;
  for (const id of ['base', 'lora', 'head'] as const) {
    const asset = manifest.assets[id];
    specs[id] = Object.freeze(
      asset.packed.rangeChunks.map((chunk) =>
        Object.freeze({
          path: asset.packed.file,
          totalBytes: asset.packed.bytes,
          offset: chunk.offset,
          length: chunk.bytes,
          sha256: chunk.sha256
        })
      )
    );
  }
  transaction.require([...specs.base, ...specs.lora, ...specs.head]);
  transaction.seal();
  return Object.freeze({
    manifest,
    async readRange(asset: Qwen35WordAssetId, rangeIndex: number, options: RangeCacheReadOptions = {}) {
      const selected = Object.hasOwn(specs, asset) ? specs[asset] : undefined;
      if (!selected || !Number.isSafeInteger(rangeIndex) || rangeIndex < 0 || rangeIndex >= selected.length)
        throw invalid('Invalid Qwen3.5 asset range selection.');
      const result = await transaction.read(base, selected[rangeIndex]!, options);
      return result.bytes;
    },
    commit(options: RangeCacheReadOptions = {}) {
      return transaction.commit(options);
    }
  });
}
