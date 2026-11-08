import {assetUrl, sha256, waitForAbort} from '../assets.js';
import {readAssetBatch} from '../asset-batch.js';
import {cancelResponse, readBoundedResponse} from '../bounded-response.js';
import {parseStrictJson} from '../h2o/strict-json.mjs';
import {WebDecisionError} from '../kev.js';
import {readVerifiedRange} from '../strands/range-reader.mjs';
import {elephantVariant, ELEPHANT_INVENTORY_SHA256, ELEPHANT_MODEL} from './contract.mjs';

const FILES = Object.freeze({
  'inventory.json': {bytes: 431398, sha256: ELEPHANT_INVENTORY_SHA256},
  'tokenizer.json': {bytes: 19989339, sha256: 'a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f'},
  'tokenizer_config.json': {bytes: 1505, sha256: '2984c03425a28d242744e234a5003786fd289b3713db79fe0958c65eb59d85b2'}
});
const invalid = () => new WebDecisionError('INVALID_ASSET', 'Unexpected Elephant asset inventory.');

export function elephantFiles(model) {
  const variant = elephantVariant(model);
  if (!variant) throw invalid();
  return {...FILES, 'model.wasm': {bytes: variant.graphBytes, sha256: variant.graphSha256}};
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

export function indexElephantFiles(inventory) {
  if (
    inventory?.state !== 'local-ranges-pending-review' ||
    !Array.isArray(inventory.parameters) ||
    inventory.parameters.length !== 684
  )
    throw invalid();
  const names = new Set();
  return inventory.parameters.map((item, index) => {
    const path = `${String(index).padStart(4, '0')}.bin`;
    if (
      item?.position !== index + 5 ||
      typeof item.target !== 'string' ||
      names.has(item.target) ||
      item.kind !== 'file' ||
      item.path !== path ||
      item.fileOffset !== 0 ||
      !Array.isArray(item.shape) ||
      item.shape.some((value) => !Number.isSafeInteger(value) || value < 1) ||
      !Number.isSafeInteger(item.bytes) ||
      item.bytes < 1 ||
      !Array.isArray(item.ranges)
    )
      throw invalid();
    names.add(item.target);
    return {
      position: item.position,
      target: item.target,
      shape: item.shape,
      dtype: item.dtype,
      bindingKind: 'q4-candidate',
      outputBytes: item.bytes,
      ranges: item.ranges,
      outputPath: path
    };
  });
}

export async function openElephantSource(base, {signal, model = ELEPHANT_MODEL} = {}) {
  const files = elephantFiles(model);
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
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Elephant metadata.', {cause});
  }
  const entries = indexElephantFiles(inventory);
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
            totalBytes: entry.outputBytes,
            offset: part.offset,
            length: part.bytes,
            sha256: part.sha256
          },
          {signal: readSignal}
        )
      ).bytes;
    }
  };
}
