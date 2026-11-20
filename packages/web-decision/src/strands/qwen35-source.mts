import {sha256} from '../assets.js';
import type {AssetCache} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {RangeCacheTransaction} from './range-cache.mjs';
import type {RangeCacheReadOptions, RangeCacheSpec} from './range-cache.mjs';
import {QWEN35_WORD_MANIFEST_SHA256, validateQwen35WordAssetManifest} from './qwen35-assets.mjs';
import type {Qwen35WordAssetId, Qwen35WordAssetManifest, Qwen35WordTensor} from './qwen35-assets.mjs';
import pinnedWordManifest from '../../assets/qwen35/pinned-word-assets-manifest.json' with {type: 'json'};
import pinnedRangeInventory from '../../assets/qwen35/source-range-inventory.json' with {type: 'json'};

const FORMAT = 'web-decision-qwen35-source-range-inventory-v1';
export const QWEN35_SOURCE_MAX_RANGE_BYTES = 8 * 1024 * 1024;
const MAX_RANGE_BYTES = QWEN35_SOURCE_MAX_RANGE_BYTES;
const RANGE_READ_CONCURRENCY = 4;
const RANGE_INVENTORY_SHA256 = '2af34ff0913cb5e3a10123ea59f55d74360e753576da84e64947b0b2af9a8d45';
const HF_BASE = new URL('https://huggingface.co/');
const invalid = (message: string) => new WebDecisionError('INVALID_ASSET', message);

async function runBounded<T>(items: readonly T[], limit: number, run: (item: T) => Promise<void>): Promise<void> {
  let next = 0;
  let failed = false;
  let failure: unknown;
  async function worker(): Promise<void> {
    while (!failed) {
      const index = next++;
      if (index >= items.length) return;
      try {
        await run(items[index]!);
      } catch (cause) {
        if (!failed) {
          failed = true;
          failure = cause;
        }
        return;
      }
    }
  }
  await Promise.all(Array.from({length: Math.min(limit, items.length)}, () => worker()));
  if (failed) throw failure;
}

export interface Qwen35SourceRange {
  readonly offset: number;
  readonly bytes: number;
  readonly sha256: string;
}

export interface Qwen35SourceRangeAsset {
  readonly source: Readonly<{
    repository: string;
    revision: string;
    file: string;
    bytes: number;
    sha256: string;
    headerBytes: number;
    headerSha256: string;
    payloadBytes: number;
  }>;
  readonly ranges: readonly Qwen35SourceRange[];
}

export interface Qwen35SourceRangeInventory {
  readonly format: typeof FORMAT;
  readonly wordManifestSha256: string;
  readonly maxRangeBytes: number;
  readonly assets: Readonly<Partial<Record<Qwen35WordAssetId, Qwen35SourceRangeAsset>>>;
}

export interface Qwen35SourceOpenOptions {
  readonly cache?: AssetCache | null | undefined;
  readonly offline?: boolean | undefined;
  readonly signal?: AbortSignal | null | undefined;
}

export interface Qwen35SourceTensorReadOptions extends RangeCacheReadOptions {}

export interface Qwen35SourcePrefetchOptions extends RangeCacheReadOptions {
  readonly onProgress?: ((bytes: number) => void) | undefined;
}

export interface Qwen35PinnedSource {
  readonly manifest: Qwen35WordAssetManifest;
  /** Sum of the exact source range windows required by this model scope. */
  readonly requiredRangeBytes?: number;
  /** Read at most one pinned range window from a tensor without materializing the full tensor. */
  readTensorRange(
    asset: Qwen35WordAssetId,
    tensorName: string,
    tensorOffset: number,
    length: number,
    options?: Qwen35SourceTensorReadOptions
  ): Promise<Uint8Array>;
  /** Convenience for tensors that fit in memory; large tensors should use readTensorRange. */
  readTensor(
    asset: Qwen35WordAssetId,
    tensorName: string,
    options?: Qwen35SourceTensorReadOptions
  ): Promise<Uint8Array>;
  commit(options?: RangeCacheReadOptions): Promise<boolean>;
  /** Detach the load-only abort signal after the model has fully loaded. */
  finishLoading?(): void;
  dispose?(): void;
  /** Fetch the exact source ranges required by this model scope for offline reuse. */
  prefetchRequiredRanges?(options?: Qwen35SourcePrefetchOptions): Promise<void>;
}

export interface Qwen35SourceLocation {
  readonly base: URL;
  readonly path: string;
}

export type Qwen35SourceResolver = (source: Qwen35SourceRangeAsset['source']) => Qwen35SourceLocation;

interface Qwen35SourceLoaderDependencies {
  readonly manifest: Qwen35WordAssetManifest;
  readonly rangeInventory: unknown;
  readonly expectedRangeInventorySha256: string;
  readonly requiredTensors?: Readonly<Partial<Record<Qwen35WordAssetId, readonly string[]>>> | undefined;
  readonly resolveSource?: Qwen35SourceResolver | undefined;
}

/** Source tensors needed by the decision graph and every possible tokenizer row lookup. */
export function requiredQwen35DecisionSourceTensors(
  manifest: Qwen35WordAssetManifest
): Readonly<Partial<Record<Qwen35WordAssetId, readonly string[]>>> {
  return Object.freeze(
    Object.fromEntries(
      (Object.keys(manifest.assets) as Qwen35WordAssetId[]).map((assetId) => [
        assetId,
        Object.freeze(
          manifest.assets[assetId].tensors
            .filter(({name}) => assetId !== 'base' || !name.startsWith('mtp.'))
            .map(({name}) => name)
        )
      ])
    )
  );
}

/** Order source reads by asset and Safetensors offset while retaining binding objects. */
export function orderQwen35SourceTensorReads<
  Binding extends Readonly<{
    asset: Qwen35WordAssetId;
    sourceName: string;
  }>
>(manifest: Qwen35WordAssetManifest, bindings: readonly Binding[]): readonly Binding[] {
  const assetOrder: Readonly<Record<Qwen35WordAssetId, number>> = {lora: 0, head: 1, base: 2};
  const ordered = bindings.map((binding, order) => {
    const tensor = manifest.assets[binding.asset].tensors.find(({name}) => name === binding.sourceName);
    if (!tensor) throw invalid('Unknown Qwen3.5 source tensor: ' + binding.sourceName + '.');
    return {binding, order, assetOrder: assetOrder[binding.asset], sourceOffset: tensor.sourceDataByteOffset};
  });
  ordered.sort(
    (left, right) =>
      left.assetOrder - right.assetOrder || left.sourceOffset - right.sourceOffset || left.order - right.order
  );
  return Object.freeze(ordered.map(({binding}) => binding));
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw invalid('Invalid Qwen3.5 ' + label + '.');
  return value as Record<string, unknown>;
}

function sameKeys(value: Record<string, unknown>, expected: readonly string[], label: string): void {
  const actual = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  if (actual.length !== sortedExpected.length || actual.some((key, index) => key !== sortedExpected[index]))
    throw invalid('Unexpected Qwen3.5 ' + label + ' fields.');
}

function safeSourcePath(value: unknown, label: string): string {
  if (
    typeof value !== 'string' ||
    !value ||
    value.startsWith('/') ||
    /[\\?#]/.test(value) ||
    value.split('/').some((part) => !part || part === '.' || part === '..')
  )
    throw invalid('Invalid Qwen3.5 ' + label + '.');
  return value;
}

function defaultSourceResolver(source: Qwen35SourceRangeAsset['source']): Qwen35SourceLocation {
  const path = [...source.repository.split('/'), 'resolve', source.revision, ...source.file.split('/')]
    .map((part) => encodeURIComponent(part))
    .join('/');
  return Object.freeze({base: HF_BASE, path});
}

/** Fixed public Hugging Face source URL; callers cannot provide repository or revision text. */
export function resolvePinnedQwen35SourceUrl(asset: Qwen35WordAssetId): URL {
  const source = (
    pinnedWordManifest as unknown as {
      readonly assets: Record<Qwen35WordAssetId, {readonly source: Qwen35SourceRangeAsset['source']}>;
    }
  ).assets[asset]?.source;
  if (!source) throw invalid('Unknown Qwen3.5 source asset.');
  const resolved = defaultSourceResolver(source);
  return new URL(resolved.path, resolved.base);
}

async function validateRangeInventory(
  value: unknown,
  manifest: Qwen35WordAssetManifest,
  expectedSha256: string
): Promise<Readonly<Record<Qwen35WordAssetId, Qwen35SourceRangeAsset>>> {
  const inventory = record(value, 'source-range inventory');
  const actualSha256 = await sha256(new TextEncoder().encode(JSON.stringify(value)));
  if (actualSha256 !== expectedSha256) throw invalid('Qwen3.5 source-range inventory hash mismatch.');
  if (inventory.format !== FORMAT) throw invalid('Unexpected Qwen3.5 source-range inventory format.');
  if (inventory.wordManifestSha256 !== QWEN35_WORD_MANIFEST_SHA256)
    throw invalid('Unexpected Qwen3.5 source word-manifest hash.');
  if (inventory.maxRangeBytes !== MAX_RANGE_BYTES) throw invalid('Unexpected Qwen3.5 source range size.');

  const rawAssets = record(inventory.assets, 'source-range assets');
  const manifestAssets = manifest.assets as Readonly<
    Record<Qwen35WordAssetId, Qwen35WordAssetManifest['assets'][Qwen35WordAssetId]>
  >;
  const ids = Object.keys(manifestAssets).sort() as Qwen35WordAssetId[];
  sameKeys(rawAssets, ids, 'source-range asset set');
  const parsed = {} as Record<Qwen35WordAssetId, Qwen35SourceRangeAsset>;
  for (const id of ids) {
    const asset = record(rawAssets[id], id + ' source-range asset');
    sameKeys(asset, ['source', 'ranges'], id + ' source-range asset');
    const rawSource = record(asset.source, id + ' source-range metadata');
    const source = manifestAssets[id].source;
    sameKeys(
      rawSource,
      ['repository', 'revision', 'file', 'bytes', 'sha256', 'headerBytes', 'headerSha256', 'payloadBytes'],
      id + ' source-range metadata'
    );
    for (const field of [
      'repository',
      'revision',
      'file',
      'bytes',
      'sha256',
      'headerBytes',
      'headerSha256',
      'payloadBytes'
    ] as const) {
      if (rawSource[field] !== source[field]) throw invalid('Unexpected Qwen3.5 ' + id + ' source metadata.');
    }
    safeSourcePath(source.repository, id + ' source repository');
    safeSourcePath(source.revision, id + ' source revision');
    safeSourcePath(source.file, id + ' source file');
    if (!Array.isArray(asset.ranges) || !asset.ranges.length)
      throw invalid('Missing Qwen3.5 source ranges for ' + id + '.');
    let offset = 0;
    const ranges = asset.ranges.map((candidate) => {
      const range = record(candidate, id + ' source range');
      sameKeys(range, ['offset', 'bytes', 'sha256'], id + ' source range');
      if (
        range.offset !== offset ||
        !Number.isSafeInteger(range.bytes) ||
        (range.bytes as number) < 1 ||
        (range.bytes as number) > MAX_RANGE_BYTES ||
        offset > source.bytes - (range.bytes as number) ||
        typeof range.sha256 !== 'string' ||
        !/^[0-9a-f]{64}$/.test(range.sha256)
      )
        throw invalid('Invalid Qwen3.5 source range for ' + id + '.');
      const chunk = Object.freeze({
        offset,
        bytes: range.bytes as number,
        sha256: range.sha256
      });
      offset += chunk.bytes;
      return chunk;
    });
    if (offset !== source.bytes) throw invalid('Qwen3.5 source ranges do not cover ' + id + '.');
    parsed[id] = Object.freeze({source, ranges: Object.freeze(ranges)});
  }
  return Object.freeze(parsed);
}

function tensorDtype(value: unknown): string | null {
  return value === 'BF16' || value === 'F32' ? value : null;
}

function verifySafetensorsHeader(header: Uint8Array, source: Qwen35SourceRangeAsset['source']): void {
  if (header.byteLength !== 8 + source.headerBytes) throw invalid('Unexpected Qwen3.5 Safetensors header size.');
  const headerLength = new DataView(header.buffer, header.byteOffset, 8).getBigUint64(0, true);
  if (headerLength !== BigInt(source.headerBytes)) throw invalid('Unexpected Qwen3.5 Safetensors header length.');
}

function validateSafetensorsHeaderValue(value: unknown, tensors: readonly Qwen35WordTensor[]): void {
  const header = record(value, 'Safetensors header');
  const names = Object.keys(header)
    .filter((name) => name !== '__metadata__')
    .sort();
  const expected = tensors.map((tensor) => tensor.name).sort();
  if (names.length !== expected.length || names.some((name, index) => name !== expected[index]))
    throw invalid('Qwen3.5 Safetensors header tensor names differ from the pinned word manifest.');
  if (Object.hasOwn(header, '__metadata__') && (!header.__metadata__ || typeof header.__metadata__ !== 'object'))
    throw invalid('Invalid Qwen3.5 Safetensors metadata header.');
  for (const tensor of tensors) {
    const value = record(header[tensor.name], 'Safetensors tensor header');
    sameKeys(value, ['dtype', 'shape', 'data_offsets'], tensor.name + ' Safetensors tensor');
    if (
      tensorDtype(value.dtype) !== tensor.dtype ||
      !Array.isArray(value.shape) ||
      value.shape.length !== tensor.shape.length ||
      value.shape.some((dimension, index) => dimension !== tensor.shape[index]) ||
      !Array.isArray(value.data_offsets) ||
      value.data_offsets.length !== 2 ||
      value.data_offsets[0] !== tensor.sourceDataByteOffset ||
      value.data_offsets[1] !== tensor.sourceDataByteOffset + tensor.sourceByteCount
    )
      throw invalid('Qwen3.5 Safetensors header offset or tensor contract mismatch: ' + tensor.name + '.');
  }
}

/** Stop futile CacheStorage writes once the browser reports exhausted quota. */
function stopRepeatedQuotaWrites(
  cache: AssetCache | null | undefined,
  onQuotaFailure: (() => void) | undefined
): AssetCache | null | undefined {
  if (!cache) return cache;
  const pendingWrites = new Map<string, Promise<void>>();
  let quotaExceeded = false;
  let quotaFailure: unknown;

  function discard(response: Response): void {
    try {
      void response.body?.cancel().catch(() => {});
    } catch {}
  }

  return {
    match(key) {
      return cache.match(key);
    },
    async put(key, response) {
      const digest = new URL(key).searchParams.get('digest');
      if (quotaExceeded) {
        discard(response);
        throw quotaFailure;
      }
      if (!digest) {
        try {
          await cache.put(key, response);
        } catch (cause) {
          if ((cause as {readonly name?: unknown} | null | undefined)?.name === 'QuotaExceededError') {
            quotaExceeded = true;
            quotaFailure = cause;
            onQuotaFailure?.();
          }
          throw cause;
        }
        return;
      }
      const pending = pendingWrites.get(digest);
      if (pending) {
        discard(response);
        return pending;
      }

      const write = cache.put(key, response);
      pendingWrites.set(digest, write);
      try {
        await write;
      } catch (cause) {
        if ((cause as {readonly name?: unknown} | null | undefined)?.name === 'QuotaExceededError') {
          quotaExceeded = true;
          quotaFailure = cause;
          onQuotaFailure?.();
        }
        throw cause;
      } finally {
        if (pendingWrites.get(digest) === write) pendingWrites.delete(digest);
      }
    }
  };
}

function selectRequiredRanges(
  source: Qwen35SourceRangeAsset['source'],
  allSpecs: readonly RangeCacheSpec[],
  tensors: readonly Qwen35WordTensor[],
  requiredNames: readonly string[]
): RangeCacheSpec[] {
  const tensorsByName = new Map(tensors.map((tensor) => [tensor.name, tensor]));
  if (!requiredNames.length || new Set(requiredNames).size !== requiredNames.length)
    throw invalid('Qwen3.5 required source tensor set is empty or contains duplicates.');
  const selectedSpecs = new Map<string, RangeCacheSpec>();

  function includeBytes(offset: number, length: number): void {
    const end = offset + length;
    if (
      !Number.isSafeInteger(offset) ||
      !Number.isSafeInteger(length) ||
      offset < 0 ||
      length < 1 ||
      !Number.isSafeInteger(end) ||
      end > source.bytes
    )
      throw invalid('Qwen3.5 required source tensor range is invalid.');
    let cursor = offset;
    for (const spec of allSpecs) {
      const overlapStart = Math.max(offset, spec.offset);
      const overlapEnd = Math.min(end, spec.offset + spec.length);
      if (overlapStart >= overlapEnd) continue;
      if (overlapStart > cursor) throw invalid('Qwen3.5 required source ranges have a gap.');
      cursor = Math.max(cursor, overlapEnd);
      selectedSpecs.set(JSON.stringify([spec.path, spec.offset, spec.length, spec.sha256]), spec);
    }
    if (cursor !== end) throw invalid('Qwen3.5 required source ranges do not cover the requested tensor.');
  }

  const payloadOffset = 8 + source.headerBytes;
  includeBytes(0, payloadOffset);
  for (const name of requiredNames) {
    const tensor = tensorsByName.get(name);
    if (!tensor) throw invalid('Unknown Qwen3.5 required source tensor: ' + name + '.');
    includeBytes(payloadOffset + tensor.sourceDataByteOffset, tensor.sourceByteCount);
  }
  return [...selectedSpecs.values()].sort((left, right) => left.offset - right.offset);
}

/** Build the internal reader with injected metadata/resolver for deterministic tests. */
export function createQwen35SourceLoader({
  manifest,
  rangeInventory,
  expectedRangeInventorySha256,
  requiredTensors,
  resolveSource = defaultSourceResolver
}: Qwen35SourceLoaderDependencies): (options?: Qwen35SourceOpenOptions) => Promise<Qwen35PinnedSource> {
  return async function openQwen35Source({cache, offline = false, signal}: Qwen35SourceOpenOptions = {}) {
    if (typeof offline !== 'boolean') throw new WebDecisionError('INVALID_INPUT', 'Invalid Qwen3.5 source options.');
    const inventory = await validateRangeInventory(rangeInventory, manifest, expectedRangeInventorySha256);
    const sourcePlans = new Map<
      Qwen35WordAssetId,
      Readonly<{asset: Qwen35SourceRangeAsset; location: Qwen35SourceLocation; specs: readonly RangeCacheSpec[]}>
    >();
    let base: URL | undefined;
    const requiredByKey = new Map<string, RangeCacheSpec>();
    for (const assetId of Object.keys(inventory) as Qwen35WordAssetId[]) {
      const asset = inventory[assetId]!;
      const location = resolveSource(asset.source);
      if (!(location.base instanceof URL)) throw invalid('Invalid Qwen3.5 source URL resolver base.');
      if (base && base.href !== location.base.href) throw invalid('Qwen3.5 source ranges must share a cache origin.');
      base ??= location.base;
      const path = safeSourcePath(location.path, assetId + ' resolved source path');
      const allSpecs = asset.ranges.map((range) =>
        Object.freeze({
          path,
          totalBytes: asset.source.bytes,
          offset: range.offset,
          length: range.bytes,
          sha256: range.sha256
        })
      );
      const requiredNames = requiredTensors?.[assetId];
      const specs = requiredNames
        ? selectRequiredRanges(asset.source, allSpecs, manifest.assets[assetId].tensors, requiredNames)
        : allSpecs;
      sourcePlans.set(assetId, Object.freeze({asset, location, specs: Object.freeze(specs)}));
      for (const spec of specs)
        requiredByKey.set(JSON.stringify([spec.path, spec.totalBytes, spec.offset, spec.length, spec.sha256]), spec);
    }
    if (!base) throw invalid('Qwen3.5 source-range inventory is empty.');
    const required = [...requiredByKey.values()];
    const requiredRangeBytes = required.reduce((sum, {length}) => sum + length, 0);
    const scopeSha256 = await sha256(
      new TextEncoder().encode(
        JSON.stringify(
          required.map(({path, totalBytes, offset, length, sha256: digest}) => [
            path,
            totalBytes,
            offset,
            length,
            digest
          ])
        )
      )
    );
    let cacheQuotaExceeded = false;
    const transaction = new RangeCacheTransaction({
      base,
      cache: stopRepeatedQuotaWrites(cache, () => (cacheQuotaExceeded = true)),
      offline,
      signal,
      markerId: 'qwen35-source-ranges-v2:' + expectedRangeInventorySha256 + ':' + scopeSha256,
      markerPath: 'qwen35-source-offline-complete-v1.json'
    });
    let loadSignal = signal;
    await transaction.open();
    transaction.require(required);
    transaction.seal();

    const verifiedHeaders = new Map<Qwen35WordAssetId, Promise<void>>();
    async function readFileBytes(
      assetId: Qwen35WordAssetId,
      fileOffset: number,
      length: number,
      readOptions: Qwen35SourceTensorReadOptions
    ): Promise<Uint8Array<ArrayBuffer>> {
      const plan = sourcePlans.get(assetId);
      if (!plan || !Number.isSafeInteger(fileOffset) || fileOffset < 0 || !Number.isSafeInteger(length) || length < 1)
        throw invalid('Invalid Qwen3.5 source byte range.');
      if (fileOffset > plan.asset.source.bytes - length)
        throw invalid('Qwen3.5 source byte range exceeds the pinned file.');
      const output = new Uint8Array(new ArrayBuffer(length));
      const fileEnd = fileOffset + length;
      const overlaps: Array<Readonly<{spec: RangeCacheSpec; begin: number; end: number}>> = [];
      let coveredUntil = fileOffset;
      for (let index = 0; index < plan.specs.length; index++) {
        const spec = plan.specs[index]!;
        const begin = Math.max(fileOffset, spec.offset);
        const end = Math.min(fileEnd, spec.offset + spec.length);
        if (begin >= end) continue;
        if (begin !== coveredUntil) throw invalid('Qwen3.5 source ranges left a tensor byte gap.');
        overlaps.push({spec, begin, end});
        coveredUntil = end;
      }
      if (coveredUntil !== fileEnd) throw invalid('Qwen3.5 source ranges left a tensor byte gap.');

      await runBounded(overlaps, RANGE_READ_CONCURRENCY, async (overlap) => {
        readOptions.signal?.throwIfAborted();
        const result = await transaction.read(base!, overlap.spec, {signal: readOptions.signal});
        const sourceBytes = new Uint8Array(result.bytes);
        output.set(
          sourceBytes.subarray(overlap.begin - overlap.spec.offset, overlap.end - overlap.spec.offset),
          overlap.begin - fileOffset
        );
      });
      return output;
    }
    async function verifyHeader(assetId: Qwen35WordAssetId, readOptions: Qwen35SourceTensorReadOptions): Promise<void> {
      const prior = verifiedHeaders.get(assetId);
      if (prior) return prior;
      const plan = sourcePlans.get(assetId)!;
      const metadata = manifest.assets[assetId];
      const pending = (async () => {
        const header = await readFileBytes(assetId, 0, 8 + plan.asset.source.headerBytes, readOptions);
        const jsonBytes = header.subarray(8);
        if ((await sha256(jsonBytes)) !== plan.asset.source.headerSha256)
          throw invalid('Qwen3.5 Safetensors header hash mismatch.');
        let value: unknown;
        try {
          value = JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(jsonBytes)) as unknown;
        } catch (cause) {
          throw new WebDecisionError('INVALID_ASSET', 'Could not parse the Qwen3.5 Safetensors header.', {cause});
        }
        verifySafetensorsHeader(header, plan.asset.source);
        validateSafetensorsHeaderValue(value, metadata.tensors);
      })();
      verifiedHeaders.set(assetId, pending);
      try {
        await pending;
      } catch (cause) {
        verifiedHeaders.delete(assetId);
        throw cause;
      }
    }

    return Object.freeze({
      manifest,
      requiredRangeBytes,
      async readTensorRange(
        assetId: Qwen35WordAssetId,
        tensorName: string,
        tensorOffset: number,
        length: number,
        readOptions: Qwen35SourceTensorReadOptions = {}
      ) {
        loadSignal?.throwIfAborted();
        readOptions.signal?.throwIfAborted();
        const metadata = manifest.assets[assetId];
        const plan = sourcePlans.get(assetId);
        const tensor = metadata?.tensors.find((candidate) => candidate.name === tensorName);
        if (!plan || !tensor) throw invalid('Unknown Qwen3.5 source tensor.');
        if (
          !Number.isSafeInteger(tensorOffset) ||
          tensorOffset < 0 ||
          !Number.isSafeInteger(length) ||
          length < 1 ||
          length > MAX_RANGE_BYTES ||
          tensorOffset > tensor.sourceByteCount - length
        )
          throw invalid('Invalid Qwen3.5 tensor byte range.');
        await verifyHeader(assetId, readOptions);
        const offset = 8 + plan.asset.source.headerBytes + tensor.sourceDataByteOffset + tensorOffset;
        return readFileBytes(assetId, offset, length, readOptions);
      },
      async readTensor(
        assetId: Qwen35WordAssetId,
        tensorName: string,
        readOptions: Qwen35SourceTensorReadOptions = {}
      ) {
        loadSignal?.throwIfAborted();
        readOptions.signal?.throwIfAborted();
        const metadata = manifest.assets[assetId];
        const plan = sourcePlans.get(assetId);
        const tensor = metadata?.tensors.find((candidate) => candidate.name === tensorName);
        if (!plan || !tensor) throw invalid('Unknown Qwen3.5 source tensor.');
        await verifyHeader(assetId, readOptions);
        const offset = 8 + plan.asset.source.headerBytes + tensor.sourceDataByteOffset;
        const bytes = await readFileBytes(assetId, offset, tensor.sourceByteCount, readOptions);
        if ((await sha256(bytes)) !== tensor.sourcePayloadSha256)
          throw invalid('Qwen3.5 source tensor hash mismatch: ' + tensor.name + '.');
        return bytes;
      },
      commit(options: RangeCacheReadOptions = {}) {
        return transaction.commit(options);
      },
      finishLoading() {
        loadSignal?.throwIfAborted();
        transaction.detachSignal();
        loadSignal = undefined;
      },
      dispose() {
        loadSignal = undefined;
        transaction.dispose();
        verifiedHeaders.clear();
      },
      async prefetchRequiredRanges(readOptions: Qwen35SourcePrefetchOptions = {}) {
        if (!cache || offline || cacheQuotaExceeded) return;
        let cacheFailed = false;
        await runBounded(required, RANGE_READ_CONCURRENCY, async (spec) => {
          if (cacheFailed || cacheQuotaExceeded) return;
          readOptions.signal?.throwIfAborted();
          if (transaction.hasRead(spec) && transaction.isCached(spec)) {
            readOptions.onProgress?.(spec.length);
            return;
          }
          for (let attempt = 0; attempt < 2; attempt += 1) {
            if (cacheQuotaExceeded) return;
            const result = await transaction.read(base!, spec, readOptions);
            if (result.cachedByHash) {
              readOptions.onProgress?.(spec.length);
              return;
            }
          }
          cacheFailed = true;
        });
      }
    });
  };
}

let pinnedLoaderPromise: Promise<(options?: Qwen35SourceOpenOptions) => Promise<Qwen35PinnedSource>> | undefined;

/** Open original pinned Safetensors by fixed Hugging Face URLs; no assetsUrl is accepted. */
export async function openPinnedQwen35Source(options?: Qwen35SourceOpenOptions): Promise<Qwen35PinnedSource> {
  pinnedLoaderPromise ??= (async () => {
    const manifest = await validateQwen35WordAssetManifest(pinnedWordManifest);
    return createQwen35SourceLoader({
      manifest,
      rangeInventory: pinnedRangeInventory,
      expectedRangeInventorySha256: RANGE_INVENTORY_SHA256,
      requiredTensors: requiredQwen35DecisionSourceTensors(manifest)
    });
  })();
  return (await pinnedLoaderPromise)(options);
}
