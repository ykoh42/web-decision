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

const FILES = Object.freeze({
  'inventory.json': {bytes: 382491, sha256: MAPIKA4_INVENTORY_SHA256},
  'tokenizer.json': {bytes: 19989325, sha256: '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'},
  'tokenizer_config.json': {bytes: 1127, sha256: '171ecbe7ddae98d11840698f7df2b8d5b4722139db0f0620d3bbf429bd656250'}
});
const invalid = () => new WebDecisionError('INVALID_ASSET', 'Unexpected Mapika 4B asset inventory.');

export function mapika4Files(model) {
  const graphs = {
    [MAPIKA4_MODEL]: {bytes: 4963401, sha256: MAPIKA4_GRAPH_SHA256},
    [MAPIKA4_DLIGHT_MODEL]: {bytes: 4996669, sha256: MAPIKA4_DLIGHT_GRAPH_SHA256},
    [MAPIKA4_DLIGHT_ALL_MODEL]: {bytes: 5168538, sha256: MAPIKA4_DLIGHT_ALL_GRAPH_SHA256}
  };
  const graph = Object.hasOwn(graphs, model) ? graphs[model] : null;
  if (!graph) throw invalid();
  return {...FILES, 'model.wasm': graph};
}

async function readFile(base, name, expected, signal) {
  try {
    const response = await waitForAbort(() => fetch(assetUrl(base, name), {signal}), signal, cancelResponse);
    if (response.status !== 200) {
      cancelResponse(response);
      throw new Error(`Unexpected HTTP status ${response.status}`);
    }
    const bytes = await readBoundedResponse(response, expected.bytes, {exact: true, signal});
    if ((await waitForAbort(() => sha256(bytes), signal)) !== expected.sha256) throw invalid();
    signal?.throwIfAborted();
    return new Uint8Array(bytes);
  } catch (cause) {
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError) throw cause;
    throw new WebDecisionError('ASSET_LOAD_FAILED', `Could not load ${name}.`, {cause});
  }
}

export function indexMapika4Files(inventory) {
  if (
    inventory?.state !== 'local-ranges-pending-review' ||
    !Array.isArray(inventory.parameters) ||
    inventory.parameters.length !== 682
  )
    throw invalid();
  const files = new Map();
  const entries = inventory.parameters.map((item, index) => {
    if (
      item.position !== index + 5 ||
      !Array.isArray(item.shape) ||
      item.shape.some((value) => typeof value !== 'string' || !/^[1-9]\d*$/.test(value))
    )
      throw invalid();
    const common = {
      position: item.position,
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
      item.fileOffset < 0 ||
      !Number.isSafeInteger(item.bytes) ||
      item.bytes < 1 ||
      !Number.isSafeInteger(item.fileOffset + item.bytes)
    )
      throw invalid();
    files.set(item.path, Math.max(files.get(item.path) ?? 0, item.fileOffset + item.bytes));
    return {
      ...common,
      bindingKind: 'q4-candidate',
      outputBytes: item.bytes,
      ranges: item.ranges,
      outputPath: item.path,
      fileOffset: item.fileOffset
    };
  });
  if (files.size !== 676) throw invalid();
  return {entries, files};
}

export async function openMapika4Source(base, {signal, model = MAPIKA4_MODEL} = {}) {
  const files = mapika4Files(model);
  const [inventoryBytes, wasm, tokenizerBytes, configBytes] = await readAssetBatch(
    ['inventory.json', 'model.wasm', 'tokenizer.json', 'tokenizer_config.json'],
    (name, localSignal) => readFile(base, name, files[name], localSignal),
    signal
  );
  signal?.throwIfAborted();
  let inventory, tokenizer, tokenizerConfig;
  try {
    inventory = parseStrictJson(inventoryBytes);
    tokenizer = parseStrictJson(tokenizerBytes, {maxBytes: FILES['tokenizer.json'].bytes});
    tokenizerConfig = parseStrictJson(configBytes);
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Mapika 4B metadata.', {cause});
  }
  const {entries, files: weightFiles} = indexMapika4Files(inventory);
  signal?.throwIfAborted();
  return {
    entries,
    wasm,
    tokenizer,
    tokenizerConfig,
    async readRange(entry, index, part, readSignal) {
      if (entry.ranges[index] !== part) throw invalid();
      return (
        await readVerifiedRange(
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
