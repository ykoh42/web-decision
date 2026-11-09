import {bytesFor} from '../nari/upload-plan.mjs';

interface ImajevChunk {
  offset: number;
  bytes: number;
  sha256: string;
}
interface ImajevFile {
  path: string;
  bytes: number;
  sha256: string;
  chunks: ImajevChunk[];
}
interface ImajevBindingBase {
  argument: string;
  target: string;
  dtype: string;
  shape: number[];
}
interface ImajevRotaryBinding extends ImajevBindingBase {
  sha256: string;
  bytesHex: string;
}
interface ImajevFileBinding extends ImajevBindingBase {
  bytes: number;
  offset: number;
  path: string;
  fileSha256: string;
  rangeSha256: string;
  fileBytes: number;
  chunks: ImajevChunk[];
}
type ImajevBinding = ImajevRotaryBinding | ImajevFileBinding;
interface ImajevMetadata {
  format: string;
  provenanceSigned: boolean;
  bindingSha256: string;
  maxInputTokens: number;
  bindings: ImajevBinding[];
  assets: ImajevFile[];
}
type ImajevFileMap = Map<string, {bytes: number; chunks: Map<string, string>}>;

const SHA = /^[a-f0-9]{64}$/;
const PATH = /^[A-Za-z0-9_.-]+$/;
const MAX_CHUNK = 4 * 1024 * 1024;
const MAX_TENSOR = 134184960;
const ROTARY_HASH = 'ef51962de52263845c53966465eb225b734f9bf294e3e382229ec4630404077f';
const ROTARY = ['model.rotary_emb.inv_freq', 'model.rotary_emb.original_inv_freq'];
const ASSETS = ['model.wasm', 'tokenizer.json', 'tokenizer_config.json', 'decision_readout.json', 'calibration.json'];
const invalid: () => never = () => {
  throw Error('Invalid Imajev inventory.');
};
const positive = (n: number) => Number.isSafeInteger(n) && n > 0;

function addChunk(files: ImajevFileMap, path: string, total: number, chunk: ImajevChunk) {
  if (
    typeof path !== 'string' ||
    !PATH.test(path) ||
    !positive(total) ||
    !chunk ||
    !Number.isSafeInteger(chunk.offset) ||
    chunk.offset < 0 ||
    !positive(chunk.bytes) ||
    chunk.bytes > MAX_CHUNK ||
    chunk.offset > total - chunk.bytes ||
    typeof chunk.sha256 !== 'string' ||
    !SHA.test(chunk.sha256)
  )
    invalid();
  let file = files.get(path);
  if (!file) files.set(path, (file = {bytes: total, chunks: new Map()}));
  if (file.bytes !== total) invalid();
  const key = `${chunk.offset}:${chunk.bytes}`;
  if (file.chunks.has(key)) invalid();
  file.chunks.set(key, chunk.sha256);
}

function checkChunks(
  files: ImajevFileMap,
  path: string,
  total: number,
  chunks: ImajevChunk[],
  offset: number,
  size: number
) {
  if (!Array.isArray(chunks) || chunks.length < 1 || !Number.isSafeInteger(offset) || offset < 0) invalid();
  let position = offset;
  for (const chunk of chunks) {
    if (chunk?.offset !== position || chunk.offset % 4 !== 0) invalid();
    addChunk(files, path, total, chunk);
    position += chunk.bytes;
    if (!Number.isSafeInteger(position) || position > offset + size) invalid();
  }
  if (position !== offset + size) invalid();
}

function checkRotary(row: ImajevRotaryBinding, index: number) {
  if (
    row.target !== ROTARY[index] ||
    row.dtype !== 'float32' ||
    row.shape.length !== 1 ||
    row.shape[0] !== 32 ||
    row.sha256 !== ROTARY_HASH ||
    typeof row.bytesHex !== 'string' ||
    !/^[a-f0-9]{256}$/.test(row.bytesHex)
  )
    invalid();
}

function checkFileRow(row: ImajevFileBinding, size: number, files: ImajevFileMap, fileHashes: Map<string, string>) {
  if (
    row.bytes !== size ||
    !Number.isSafeInteger(row.offset) ||
    row.offset < 0 ||
    row.offset % 4 !== 0 ||
    typeof row.fileSha256 !== 'string' ||
    !SHA.test(row.fileSha256) ||
    typeof row.rangeSha256 !== 'string' ||
    !SHA.test(row.rangeSha256)
  )
    invalid();
  const previous = fileHashes.get(row.path);
  if (previous !== undefined && previous !== row.fileSha256) invalid();
  fileHashes.set(row.path, row.fileSha256);
  checkChunks(files, row.path, row.fileBytes, row.chunks, row.offset, size);
}

function checkBindings(rows: ImajevBinding[], files: ImajevFileMap) {
  if (!Array.isArray(rows) || rows.length !== 1082) invalid();
  const names = new Set<string>(),
    targets = new Set<string>(),
    fileHashes = new Map<string, string>();
  for (let index = 0; index < rows.length; index++) {
    const row = rows[index];
    if (
      !row ||
      typeof row.argument !== 'string' ||
      !row.argument ||
      names.has(row.argument) ||
      typeof row.target !== 'string' ||
      !row.target ||
      targets.has(row.target) ||
      !['float16', 'float32', 'uint32'].includes(row.dtype)
    )
      invalid();
    names.add(row.argument);
    targets.add(row.target);
    const size = bytesFor(row);
    if (size > MAX_TENSOR) invalid();
    if (index >= 1080) {
      if (Object.hasOwn(row, 'path')) invalid();
      checkRotary(row as ImajevRotaryBinding, index - 1080);
    } else checkFileRow(row as ImajevFileBinding, size, files, fileHashes);
  }
  if (files.size !== 677) invalid();
}

function checkAssets(assets: ImajevFile[], files: ImajevFileMap) {
  if (!Array.isArray(assets) || assets.length !== ASSETS.length) invalid();
  for (const [index, asset] of assets.entries()) {
    if (
      !asset ||
      asset.path !== ASSETS[index] ||
      files.has(asset.path) ||
      !positive(asset.bytes) ||
      asset.bytes > 16 * 1024 * 1024 ||
      typeof asset.sha256 !== 'string' ||
      !SHA.test(asset.sha256)
    )
      invalid();
    checkChunks(files, asset.path, asset.bytes, asset.chunks, 0, asset.bytes);
  }
}

/** Validate the shape and range plan after its exact metadata bytes are pinned. */
export function prepareImajevPlan(metadata: ImajevMetadata) {
  if (
    metadata?.format !== 'web-decision-imajev-q4-dlight-local-v1' ||
    metadata.provenanceSigned !== false ||
    metadata.bindingSha256 !== 'd6984fe4ca6514a9b2a3df9ffc3d3117302624d9a399f860a94adee2b6f17563' ||
    metadata.maxInputTokens !== 128
  )
    invalid();
  const files: ImajevFileMap = new Map();
  checkBindings(metadata.bindings, files);
  checkAssets(metadata.assets, files);
  return {files, bindings: metadata.bindings, assets: metadata.assets, maxInputTokens: 128};
}
