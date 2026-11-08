import {assetUrl, sha256, waitForAbort} from '../assets.js';
import {readAssetBatch} from '../asset-batch.js';
import {cancelResponse, readBoundedResponse} from '../bounded-response.js';
import {parseStrictJson} from '../h2o/strict-json.mjs';
import {WebDecisionError} from '../kev.js';
import {readVerifiedRange} from '../strands/range-reader.mjs';
import {FLASH_GRAPH_SHA256, FLASH_MANIFEST_SHA256} from './contract.mjs';
import type {FlashManifest, FlashManifestEntry, FlashQ4CandidateEntry, FlashRange, FlashSource} from './types.js';

const FILES = Object.freeze({
  'manifest.pending.json': {bytes: 527849, sha256: FLASH_MANIFEST_SHA256},
  'model.pending.wasm': {bytes: 4952155, sha256: FLASH_GRAPH_SHA256},
  'tokenizer.json': {bytes: 19989492, sha256: '6f32ce20dc35f57a7f9ad1eac03525bd7d30f9df8cea6507e958279cc3657706'},
  'tokenizer_config.json': {bytes: 1268, sha256: '188cf0b9db5cfd6e92f1b63c646f5975be6f8383c6c9a2fa5dc2b48a8b229402'}
});
const fail = (message: string): WebDecisionError => new WebDecisionError('INVALID_ASSET', message);

async function readFile(base: URL, name: keyof typeof FILES, signal?: AbortSignal): Promise<Uint8Array> {
  const expected = FILES[name];
  let response;
  try {
    response = await waitForAbort(() => fetch(assetUrl(base, name), signal ? {signal} : {}), signal, cancelResponse);
    if (response.status !== 200) {
      cancelResponse(response);
      throw new Error(`Unexpected HTTP status ${response.status}`);
    }
    const bytes = await readBoundedResponse(response, expected.bytes, {exact: true, ...(signal ? {signal} : {})});
    if ((await waitForAbort(() => sha256(bytes), signal)) !== expected.sha256) throw fail('Pinned asset hash differs.');
    signal?.throwIfAborted();
    return new Uint8Array(bytes);
  } catch (cause) {
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError) throw cause;
    throw new WebDecisionError('ASSET_LOAD_FAILED', `Could not load ${name}.`, {cause});
  }
}

export function indexFlashFiles(manifest: unknown): Map<string, number> {
  const candidate = manifest as {state?: unknown; entries?: unknown};
  if (
    candidate?.state !== 'pplx-q4-joint-assets-pending' ||
    !Array.isArray(candidate.entries) ||
    candidate.entries.length !== 682
  )
    throw fail('Unexpected Perplexity asset inventory.');
  const files = new Map();
  for (const entry of candidate.entries as FlashManifestEntry[]) {
    if (entry.bindingKind === 'derived-rotary') continue;
    if (entry.bindingKind !== 'q4-candidate' || !/^(candidate|vectors)\/[A-Za-z0-9_.-]+$/.test(entry.outputPath))
      throw fail('Invalid Perplexity asset path.');
    const base = entry.conversion === 'none' ? entry.sourceOffset : entry.conversion === 'fp32-to-fp16' ? 0 : NaN;
    if (
      !Number.isSafeInteger(base) ||
      base < 0 ||
      !Number.isSafeInteger(entry.outputBytes) ||
      entry.outputBytes < 1 ||
      !Number.isSafeInteger(base + entry.outputBytes)
    )
      throw fail('Invalid Perplexity file range.');
    files.set(entry.outputPath, Math.max(files.get(entry.outputPath) ?? 0, base + entry.outputBytes));
  }
  if (files.size !== 676) throw fail('Incomplete Perplexity asset inventory.');
  return files;
}

export async function openFlashSource(base: URL, {signal}: {signal?: AbortSignal} = {}): Promise<FlashSource> {
  const fileBytes = await readAssetBatch(
    Object.keys(FILES),
    (name, localSignal) => readFile(base, name as keyof typeof FILES, localSignal),
    signal
  );
  const manifestBytes = fileBytes[0]!;
  const wasm = fileBytes[1]!;
  const tokenizerBytes = fileBytes[2]!;
  const configBytes = fileBytes[3]!;
  signal?.throwIfAborted();
  let manifest: unknown, tokenizer: unknown, tokenizerConfig: unknown;
  try {
    manifest = parseStrictJson(manifestBytes);
    tokenizer = parseStrictJson(tokenizerBytes, {maxBytes: FILES['tokenizer.json'].bytes});
    tokenizerConfig = parseStrictJson(configBytes);
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Perplexity metadata.', {cause});
  }
  const typedManifest = manifest as FlashManifest;
  const files = indexFlashFiles(manifest);
  signal?.throwIfAborted();
  return {
    manifest: typedManifest,
    wasm,
    tokenizer,
    tokenizerConfig,
    async readRange(entry: FlashQ4CandidateEntry, index: number, part: FlashRange, readSignal?: AbortSignal) {
      if (entry.ranges[index] !== part) throw fail('Unexpected Perplexity range.');
      const baseOffset = entry.conversion === 'none' ? entry.sourceOffset : 0;
      const totalBytes = files.get(entry.outputPath)!;
      return (
        await readVerifiedRangeWithSignal(
          base,
          {
            path: entry.outputPath,
            totalBytes,
            offset: baseOffset + part.offset,
            length: part.bytes,
            sha256: part.sha256
          },
          {signal: readSignal}
        )
      ).bytes;
    }
  };
}

const readVerifiedRangeWithSignal = readVerifiedRange as unknown as (
  base: URL,
  descriptor: {path: string; totalBytes: number; offset: number; length: number; sha256: string},
  options: {signal?: AbortSignal | undefined; offline?: boolean}
) => Promise<{bytes: ArrayBuffer; cachedByHash: boolean}>;
