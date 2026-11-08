import {bytesFor} from '../nari/upload-plan.mjs';

const SHA = /^[a-f0-9]{64}$/;
const PATH = /^[A-Za-z0-9_.-]+$/;
const MAX_CHUNK = 4 * 1024 * 1024;
const MAX_TENSOR = 134184960;
const ROTARY_HASH = 'ef51962de52263845c53966465eb225b734f9bf294e3e382229ec4630404077f';
const ROTARY = ['model.rotary_emb.inv_freq', 'model.rotary_emb.original_inv_freq'];
const ASSETS = ['model.wasm', 'tokenizer.json', 'tokenizer_config.json', 'decision_readout.json', 'calibration.json'];
const invalid = () => {
  throw Error('Invalid Imajev inventory.');
};
const positive = (n) => Number.isSafeInteger(n) && n > 0;

function addChunk(files, path, total, chunk) {
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

function checkChunks(files, path, total, chunks, offset, size) {
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

function checkRotary(row, index) {
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

function checkFileRow(row, size, files, fileHashes) {
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

function checkBindings(rows, files) {
  if (!Array.isArray(rows) || rows.length !== 1082) invalid();
  const names = new Set(),
    targets = new Set(),
    fileHashes = new Map();
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
      checkRotary(row, index - 1080);
    } else checkFileRow(row, size, files, fileHashes);
  }
  if (files.size !== 677) invalid();
}

function checkAssets(assets, files) {
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
export function prepareImajevPlan(metadata) {
  if (
    metadata?.format !== 'web-decision-imajev-q4-dlight-local-v1' ||
    metadata.provenanceSigned !== false ||
    metadata.bindingSha256 !== 'd6984fe4ca6514a9b2a3df9ffc3d3117302624d9a399f860a94adee2b6f17563' ||
    metadata.maxInputTokens !== 128
  )
    invalid();
  const files = new Map();
  checkBindings(metadata.bindings, files);
  checkAssets(metadata.assets, files);
  return {files, bindings: metadata.bindings, assets: metadata.assets, maxInputTokens: 128};
}
