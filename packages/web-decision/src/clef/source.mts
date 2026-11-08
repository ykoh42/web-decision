import {assetUrl, sha256, waitForAbort} from '../assets.js';
import {readAssetBatch} from '../asset-batch.js';
import {cancelResponse, readBoundedResponse} from '../bounded-response.js';
import {parseStrictJson} from '../h2o/strict-json.mjs';
import {WebDecisionError} from '../kev.js';
import {CLEF_ABI_SHA256, CLEF_GRAPH_SHA256, CLEF_INVENTORY_SHA256, CLEF_MODEL} from './contract.mjs';
import {readClefRange} from './range.mjs';
import type {ClefRangePart, ClefSource, ClefSourceIndex, ClefTensorEntry} from './types.mjs';

const FILES = Object.freeze({
  'inventory.json': {bytes: 325498, sha256: CLEF_INVENTORY_SHA256},
  'model.wasm': {bytes: 6233593, sha256: CLEF_GRAPH_SHA256},
  'tokenizer.json': {bytes: 19989325, sha256: '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'},
  'tokenizer_config.json': {bytes: 1075, sha256: '91a08f825d370d085d692e04cf117cdd7faad7bf18e996f1e6031b6dab03db72'}
} satisfies Record<string, {bytes: number; sha256: string}>);
const WIDTH = {uint32: 4, float16: 2, float32: 4},
  HEX = /^[a-f0-9]{64}$/;
const PATH = /^(?:source\/t\d{4}(?:-b\d{4})?\.(?:q4|scales|fp32)|binding\/p\d{4}\.(?:fp16|fp32))$/;
const invalid = () => new WebDecisionError('INVALID_ASSET', 'Unexpected Clef asset inventory.');

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

async function readFile(
  base: URL,
  name: string,
  expected: {bytes: number; sha256: string},
  signal?: AbortSignal
): Promise<Uint8Array> {
  let response;
  try {
    response = await waitForAbort(
      () => fetch(assetUrl(base, name), {redirect: 'error', ...(signal ? {signal} : {})}),
      signal,
      cancelResponse
    );
    if (response.status !== 200 || response.redirected) throw new Error('Unexpected asset response');
    const data = await readBoundedResponse(response, expected.bytes, {exact: true, ...(signal ? {signal} : {})});
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

function validateEntry(value: unknown, targets: Set<string>, paths: Set<string>): ClefTensorEntry {
  const entry = isRecord(value) ? value : null;
  const dtype = entry?.dtype;
  if (
    !entry ||
    typeof entry.target !== 'string' ||
    targets.has(entry.target) ||
    typeof entry.path !== 'string' ||
    !PATH.test(entry.path) ||
    paths.has(entry.path) ||
    typeof dtype !== 'string' ||
    !Object.hasOwn(WIDTH, dtype) ||
    !Array.isArray(entry.shape) ||
    entry.shape.some((n) => !Number.isSafeInteger(n) || n < 1) ||
    !Array.isArray(entry.chunks) ||
    !entry.chunks.length
  )
    throw invalid();
  const entryShape = entry.shape as number[];
  const entryDtype = dtype as keyof typeof WIDTH;
  const bytes = entryShape.reduce((n, dim) => n * dim, WIDTH[entryDtype]);
  if (!Number.isSafeInteger(bytes) || bytes !== entry.bytes || bytes < 1 || bytes > 128 * 1024 * 1024) throw invalid();
  let offset = 0;
  const chunks = entry.chunks as unknown[];
  for (const [index, partValue] of chunks.entries()) {
    const part = isRecord(partValue) ? partValue : null;
    if (
      !part ||
      part.offset !== offset ||
      typeof part.length !== 'number' ||
      !Number.isSafeInteger(part.length) ||
      part.length < 1 ||
      part.length > 4 * 1024 * 1024 ||
      typeof part.sha256 !== 'string' ||
      !HEX.test(part.sha256) ||
      part.offset % 4 ||
      ((part.length as number) % 4 &&
        !(index === chunks.length - 1 && entryDtype === 'float16' && (part.length as number) % 4 === 2))
    )
      throw invalid();
    offset += part.length as number;
  }
  if (offset !== bytes) throw invalid();
  targets.add(entry.target);
  paths.add(entry.path);
  return entry as unknown as ClefTensorEntry;
}

function validateBankStream(
  bank: Record<string, unknown>,
  prefix: string,
  field: 'packed' | 'scales',
  dtype: 'uint32' | 'float16',
  width: number,
  index: number,
  rows: number,
  targets: Set<string>,
  paths: Set<string>
): void {
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

function validateBank(value: unknown, prefix: string, index: number, targets: Set<string>, paths: Set<string>): void {
  const bank = isRecord(value) ? value : null;
  const rows = [65536, 65536, 65536, 51712][index];
  if (!bank || rows === undefined || bank.rows !== rows) throw invalid();
  validateBankStream(bank, prefix, 'packed', 'uint32', 512, index, rows, targets, paths);
  validateBankStream(bank, prefix, 'scales', 'float16', 128, index, rows, targets, paths);
}

function validateBanks(inventory: Record<string, unknown>, targets: Set<string>, paths: Set<string>): void {
  const inputBanks = inventory.inputBanks as unknown[];
  const outputBanks = inventory.outputBanks as unknown[];
  for (const [prefix, banks] of [
    ['body.embed_tokens', inputBanks],
    ['output_table', outputBanks]
  ] as const) {
    for (const [index, bank] of banks.entries()) validateBank(bank, prefix, index, targets, paths);
  }
}

export function indexClefInventory(value: unknown): ClefSourceIndex {
  const inventoryValue = structuredClone(value);
  const inventory = isRecord(inventoryValue) ? inventoryValue : null;
  if (
    !inventory ||
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
  const targets = new Set<string>(),
    paths = new Set<string>();
  let total = 0;
  const parameters = inventory.parameters as unknown[];
  for (const [index, entryValue] of parameters.entries()) {
    const entry = validateEntry(entryValue, targets, paths);
    if (entry.position !== index + 17) throw invalid();
    total += entry.bytes;
  }
  if (total !== 4380840720) throw invalid();
  validateBanks(inventory, targets, paths);
  return {
    entries: parameters.map((entry) => entry as ClefTensorEntry),
    inputBanks: inventory.inputBanks as ClefSourceIndex['inputBanks'],
    outputBanks: inventory.outputBanks as ClefSourceIndex['outputBanks']
  };
}

export async function openClefSource(base: URL, {signal}: {signal?: AbortSignal} = {}): Promise<ClefSource> {
  const fileBytes = await readAssetBatch(
    Object.keys(FILES),
    (name, batchSignal) => readFile(base, name, FILES[name as keyof typeof FILES], batchSignal),
    signal
  );
  const inventoryBytes = fileBytes[0]!;
  const wasm = fileBytes[1]!;
  const tokenizerBytes = fileBytes[2]!;
  const configBytes = fileBytes[3]!;
  let inventory: unknown, tokenizer: unknown, tokenizerConfig: unknown;
  try {
    inventory = parseStrictJson(inventoryBytes);
    tokenizer = parseStrictJson(tokenizerBytes, {maxBytes: FILES['tokenizer.json'].bytes});
    tokenizerConfig = parseStrictJson(configBytes);
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Clef metadata.', {cause});
  }
  const indexed = indexClefInventory(inventory);
  if (!WebAssembly.validate(wasm.slice().buffer)) throw invalid();
  signal?.throwIfAborted();
  return {
    ...indexed,
    wasm,
    tokenizer,
    tokenizerConfig,
    readRange: (entry: ClefTensorEntry, part: ClefRangePart, readSignal?: AbortSignal) =>
      readClefRange(base, entry, part, readSignal)
  };
}
