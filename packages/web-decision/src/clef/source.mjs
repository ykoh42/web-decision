import {assetUrl, sha256, waitForAbort} from '../assets.js';
import {readAssetBatch} from '../asset-batch.js';
import {cancelResponse, readBoundedResponse} from '../bounded-response.js';
import {parseStrictJson} from '../h2o/strict-json.mjs';
import {WebDecisionError} from '../kev.js';
import {CLEF_ABI_SHA256, CLEF_GRAPH_SHA256, CLEF_INVENTORY_SHA256, CLEF_MODEL} from './contract.mjs';
import {readClefRange} from './range.mjs';

const FILES = Object.freeze({
  'inventory.json': {bytes: 325498, sha256: CLEF_INVENTORY_SHA256},
  'model.wasm': {bytes: 6233593, sha256: CLEF_GRAPH_SHA256},
  'tokenizer.json': {bytes: 19989325, sha256: '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'},
  'tokenizer_config.json': {bytes: 1075, sha256: '91a08f825d370d085d692e04cf117cdd7faad7bf18e996f1e6031b6dab03db72'}
});
const WIDTH = {uint32: 4, float16: 2, float32: 4},
  HEX = /^[a-f0-9]{64}$/;
const PATH = /^(?:source\/t\d{4}(?:-b\d{4})?\.(?:q4|scales|fp32)|binding\/p\d{4}\.(?:fp16|fp32))$/;
const invalid = () => new WebDecisionError('INVALID_ASSET', 'Unexpected Clef asset inventory.');

async function readFile(base, name, expected, signal) {
  let response;
  try {
    response = await waitForAbort(
      () => fetch(assetUrl(base, name), {signal, redirect: 'error'}),
      signal,
      cancelResponse
    );
    if (response.status !== 200 || response.redirected) throw new Error('Unexpected asset response');
    const data = await readBoundedResponse(response, expected.bytes, {exact: true, signal});
    if ((await waitForAbort(() => sha256(data), signal)) !== expected.sha256) throw invalid();
    signal?.throwIfAborted();
    return new Uint8Array(data);
  } catch (cause) {
    cancelResponse(response);
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError) throw cause;
    throw new WebDecisionError('ASSET_LOAD_FAILED', `Could not load ${name}.`, {cause});
  }
}

function validateEntry(entry, targets, paths) {
  if (
    !entry ||
    typeof entry.target !== 'string' ||
    targets.has(entry.target) ||
    typeof entry.path !== 'string' ||
    !PATH.test(entry.path) ||
    paths.has(entry.path) ||
    !Object.hasOwn(WIDTH, entry.dtype) ||
    !Array.isArray(entry.shape) ||
    entry.shape.some((n) => !Number.isSafeInteger(n) || n < 1) ||
    !Array.isArray(entry.chunks) ||
    !entry.chunks.length
  )
    throw invalid();
  const bytes = entry.shape.reduce((n, dim) => n * dim, WIDTH[entry.dtype]);
  if (!Number.isSafeInteger(bytes) || bytes !== entry.bytes || bytes < 1 || bytes > 128 * 1024 * 1024) throw invalid();
  let offset = 0;
  for (const [index, part] of entry.chunks.entries()) {
    if (
      part.offset !== offset ||
      !Number.isSafeInteger(part.length) ||
      part.length < 1 ||
      part.length > 4 * 1024 * 1024 ||
      typeof part.sha256 !== 'string' ||
      !HEX.test(part.sha256) ||
      part.offset % 4 ||
      (part.length % 4 && !(index === entry.chunks.length - 1 && entry.dtype === 'float16' && part.length % 4 === 2))
    )
      throw invalid();
    offset += part.length;
  }
  if (offset !== bytes) throw invalid();
  targets.add(entry.target);
  paths.add(entry.path);
  return entry;
}

function validateBanks(inventory, targets, paths) {
  for (const [prefix, banks] of [
    ['body.embed_tokens', inventory.inputBanks],
    ['output_table', inventory.outputBanks]
  ]) {
    for (const [index, bank] of banks.entries()) {
      const rows = [65536, 65536, 65536, 51712][index];
      if (bank?.rows !== rows) throw invalid();
      for (const [field, dtype, width] of [
        ['packed', 'uint32', 512],
        ['scales', 'float16', 128]
      ]) {
        const entry = validateEntry(bank[field], targets, paths);
        if (
          entry.target !== `${prefix}.${field}_${index}` ||
          entry.dtype !== dtype ||
          entry.shape.length !== 2 ||
          entry.shape[0] !== rows ||
          entry.shape[1] !== width ||
          typeof entry.sha256 !== 'string' ||
          !HEX.test(entry.sha256)
        )
          throw invalid();
      }
    }
  }
}

export function indexClefInventory(value) {
  const inventory = structuredClone(value);
  if (
    inventory?.state !== 'clef-selected-q4-inventory-v1' ||
    inventory.model !== CLEF_MODEL ||
    inventory.abiSha256 !== CLEF_ABI_SHA256 ||
    inventory.graphSha256 !== CLEF_GRAPH_SHA256 ||
    !Array.isArray(inventory.parameters) ||
    inventory.parameters.length !== 797 ||
    !Array.isArray(inventory.inputBanks) ||
    inventory.inputBanks.length !== 4 ||
    !Array.isArray(inventory.outputBanks) ||
    inventory.outputBanks.length !== 4
  )
    throw invalid();
  const targets = new Set(),
    paths = new Set();
  let total = 0;
  for (const [index, entry] of inventory.parameters.entries()) {
    validateEntry(entry, targets, paths);
    if (entry.position !== index + 17) throw invalid();
    total += entry.bytes;
  }
  if (total !== 4380840720) throw invalid();
  validateBanks(inventory, targets, paths);
  return {entries: inventory.parameters, inputBanks: inventory.inputBanks, outputBanks: inventory.outputBanks};
}

export async function openClefSource(base, {signal} = {}) {
  const [inventoryBytes, wasm, tokenizerBytes, configBytes] = await readAssetBatch(
    Object.entries(FILES),
    ([name, expected], batchSignal) => readFile(base, name, expected, batchSignal),
    signal
  );
  let inventory, tokenizer, tokenizerConfig;
  try {
    inventory = parseStrictJson(inventoryBytes);
    tokenizer = parseStrictJson(tokenizerBytes, {maxBytes: FILES['tokenizer.json'].bytes});
    tokenizerConfig = parseStrictJson(configBytes);
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Clef metadata.', {cause});
  }
  const indexed = indexClefInventory(inventory);
  if (!WebAssembly.validate(wasm)) throw invalid();
  signal?.throwIfAborted();
  return {
    ...indexed,
    wasm,
    tokenizer,
    tokenizerConfig,
    readRange: (entry, part, readSignal) => readClefRange(base, entry, part, readSignal)
  };
}
