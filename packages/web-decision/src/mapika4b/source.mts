import type {AssetFileMetadata} from '../assets.js';
import {assetUrl, sha256, waitForAbort} from '../assets.js';
import {readAssetBatch} from '../asset-batch.js';
import {cancelResponse, readBoundedResponse} from '../bounded-response.js';
import {parseStrictJson} from '../h2o/strict-json.mjs';
import {WebDecisionError} from '../kev.js';
import {readVerifiedRange} from '../strands/range-reader.mjs';
import {
  MAPIKA4_DLIGHT_ALL_GRAPH_SHA256,
  MAPIKA4_DLIGHT_ALL_MODEL,
  MAPIKA4_DLIGHT_GRAPH_SHA256,
  MAPIKA4_DLIGHT_MODEL,
  MAPIKA4_GRAPH_SHA256,
  MAPIKA4_INVENTORY_SHA256,
  MAPIKA4_MODEL
} from './contract.mjs';

export interface Mapika4RangePart {
  readonly offset: number;
  readonly bytes: number;
  readonly sha256: string;
}

interface Mapika4AssetInventory {
  readonly state?: unknown;
  readonly parameters?: unknown;
}

interface Mapika4InventoryParameter {
  readonly position: unknown;
  readonly target: unknown;
  readonly shape: unknown;
  readonly dtype: unknown;
  readonly kind: unknown;
  readonly values: unknown;
  readonly path: unknown;
  readonly fileOffset: unknown;
  readonly bytes: unknown;
  readonly ranges: unknown;
}

interface Mapika4CommonEntry {
  readonly position: number;
  readonly target: unknown;
  readonly shape: number[];
  readonly dtype: string;
}

export interface Mapika4RotaryEntry extends Mapika4CommonEntry {
  readonly bindingKind: 'derived-rotary';
  readonly values: unknown;
}

export interface Mapika4FileEntry extends Mapika4CommonEntry {
  readonly bindingKind: 'q4-candidate';
  readonly outputBytes: number;
  readonly ranges: unknown;
  readonly outputPath: string;
  readonly fileOffset: number;
}

export type Mapika4ParameterEntry = Mapika4RotaryEntry | Mapika4FileEntry;

interface IndexedMapika4Files {
  readonly entries: Mapika4ParameterEntry[];
  readonly files: Map<string, number>;
}

export interface Mapika4Source {
  readonly entries: Mapika4ParameterEntry[];
  readonly wasm: Uint8Array;
  readonly tokenizer: unknown;
  readonly tokenizerConfig: unknown;
  readRange(
    entry: Mapika4FileEntry,
    index: number,
    part: Mapika4RangePart,
    signal?: AbortSignal | null | undefined
  ): Promise<ArrayBuffer>;
}

type Mapika4Model = typeof MAPIKA4_MODEL | typeof MAPIKA4_DLIGHT_MODEL | typeof MAPIKA4_DLIGHT_ALL_MODEL;

const FILES: Readonly<Record<string, AssetFileMetadata>> = Object.freeze({
  'inventory.json': {bytes: 382491, sha256: MAPIKA4_INVENTORY_SHA256},
  'tokenizer.json': {bytes: 19989325, sha256: '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'},
  'tokenizer_config.json': {bytes: 1127, sha256: '171ecbe7ddae98d11840698f7df2b8d5b4722139db0f0620d3bbf429bd656250'}
});
const invalid = () => new WebDecisionError('INVALID_ASSET', 'Unexpected Mapika 4B asset inventory.');

export function mapika4Files(model: unknown): Readonly<Record<string, AssetFileMetadata>> {
  const graphs: Readonly<Record<string, AssetFileMetadata>> = {
    [MAPIKA4_MODEL]: {bytes: 4963401, sha256: MAPIKA4_GRAPH_SHA256},
    [MAPIKA4_DLIGHT_MODEL]: {bytes: 4996669, sha256: MAPIKA4_DLIGHT_GRAPH_SHA256},
    [MAPIKA4_DLIGHT_ALL_MODEL]: {bytes: 5168538, sha256: MAPIKA4_DLIGHT_ALL_GRAPH_SHA256}
  };
  const graph = Object.hasOwn(graphs, model as PropertyKey) ? graphs[model as string] : null;
  if (!graph) throw invalid();
  return {...FILES, 'model.wasm': graph};
}

async function readFile(
  base: URL,
  name: string,
  expected: AssetFileMetadata,
  signal?: AbortSignal | null
): Promise<Uint8Array> {
  try {
    const response = await waitForAbort(
      () => fetch(assetUrl(base, name), {signal} as RequestInit),
      signal,
      cancelResponse
    );
    if (response.status !== 200) {
      cancelResponse(response);
      throw new Error(`Unexpected HTTP status ${response.status}`);
    }
    const bytes = await readBoundedResponse(response, expected.bytes, {exact: true, signal} as NonNullable<
      Parameters<typeof readBoundedResponse>[2]
    >);
    if ((await waitForAbort(() => sha256(bytes), signal)) !== expected.sha256) throw invalid();
    signal?.throwIfAborted();
    return new Uint8Array(bytes);
  } catch (cause) {
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError) throw cause;
    throw new WebDecisionError('ASSET_LOAD_FAILED', `Could not load ${name}.`, {cause});
  }
}

export function indexMapika4Files(inventory: unknown): IndexedMapika4Files {
  if (
    (inventory as Mapika4AssetInventory | null | undefined)?.state !== 'local-ranges-pending-review' ||
    !Array.isArray((inventory as Mapika4AssetInventory).parameters) ||
    (inventory as {readonly parameters: unknown[]}).parameters.length !== 682
  )
    throw invalid();
  const files = new Map<string, number>();
  const entries = (inventory as {readonly parameters: Mapika4InventoryParameter[]}).parameters.map(
    (item, index): Mapika4ParameterEntry => {
      if (
        item.position !== index + 5 ||
        !Array.isArray(item.shape) ||
        item.shape.some((value: unknown) => typeof value !== 'string' || !/^[1-9]\d*$/.test(value))
      )
        throw invalid();
      const common: Mapika4CommonEntry = {
        position: item.position as number,
        target: item.target,
        shape: item.shape.map(Number),
        dtype: typeof item.dtype === 'string' ? item.dtype.replace('torch.', '') : ''
      };
      if (item.kind === 'rotary') return {...common, bindingKind: 'derived-rotary', values: item.values};
      if (
        item.kind !== 'file' ||
        typeof item.path !== 'string' ||
        !/^[A-Za-z0-9_.-]+$/.test(item.path) ||
        !Number.isSafeInteger(item.fileOffset) ||
        (item.fileOffset as number) < 0 ||
        !Number.isSafeInteger(item.bytes) ||
        (item.bytes as number) < 1 ||
        !Number.isSafeInteger((item.fileOffset as number) + (item.bytes as number))
      )
        throw invalid();
      files.set(item.path, Math.max(files.get(item.path) ?? 0, (item.fileOffset as number) + (item.bytes as number)));
      return {
        ...common,
        bindingKind: 'q4-candidate',
        outputBytes: item.bytes as number,
        ranges: item.ranges,
        outputPath: item.path,
        fileOffset: item.fileOffset as number
      };
    }
  );
  if (files.size !== 676) throw invalid();
  return {entries, files};
}

interface OpenMapika4SourceOptions {
  readonly signal?: AbortSignal | null | undefined;
  readonly model?: Mapika4Model | undefined;
}

export async function openMapika4Source(
  base: URL,
  {signal, model = MAPIKA4_MODEL}: OpenMapika4SourceOptions = {}
): Promise<Mapika4Source> {
  const files = mapika4Files(model);
  const [inventoryBytes, wasm, tokenizerBytes, configBytes] = await readAssetBatch(
    ['inventory.json', 'model.wasm', 'tokenizer.json', 'tokenizer_config.json'],
    (name, localSignal) => readFile(base, name, files[name]!, localSignal),
    signal
  );
  signal?.throwIfAborted();
  let inventory: unknown, tokenizer: unknown, tokenizerConfig: unknown;
  try {
    inventory = parseStrictJson(inventoryBytes);
    tokenizer = parseStrictJson(tokenizerBytes, {maxBytes: FILES['tokenizer.json']!.bytes});
    tokenizerConfig = parseStrictJson(configBytes);
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Mapika 4B metadata.', {cause});
  }
  const {entries, files: weightFiles} = indexMapika4Files(inventory);
  signal?.throwIfAborted();
  return {
    entries,
    wasm: wasm!,
    tokenizer,
    tokenizerConfig,
    async readRange(entry, index, part, readSignal) {
      if ((entry.ranges as readonly unknown[])[index] !== part) throw invalid();
      return (
        await (
          readVerifiedRange as unknown as (
            rangeBase: URL,
            spec: {
              readonly path: string;
              readonly totalBytes: number | undefined;
              readonly offset: number;
              readonly length: number;
              readonly sha256: string;
            },
            options: {readonly signal?: AbortSignal | null | undefined}
          ) => Promise<{readonly bytes: ArrayBuffer}>
        )(
          base,
          {
            path: `weights/${entry.outputPath}`,
            totalBytes: weightFiles.get(entry.outputPath),
            offset: entry.fileOffset + part.offset,
            length: part.bytes,
            sha256: part.sha256
          },
          {signal: readSignal}
        )
      ).bytes;
    }
  };
}
