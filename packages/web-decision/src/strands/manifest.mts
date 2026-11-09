import {sha256, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {STRANDS_FUSED_GRAPH_MAX_TOKENS} from './contract.mjs';

export const MANIFEST_SHA256 = '74eb7f501ab64d5906a498dac20e161044616cfb67607ddd5c24030887132209';
export const WASM_SHA256 = '020720bae0d99b592b3985130daac08a2c41adadad76143ab649f4afe109bc6e';
export const TOKENIZER_SHA256 = 'a2cdd2e108566b09079afa8d266e9e65e7c280f27b771218237a06de5ba9cd86';
export const TOKENIZER_CONFIG_SHA256 = '8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45';
export const MANIFEST_BYTES = 258565;
export const WASM_BYTES = 5199127;
const TOTAL_BYTES = 3776255872;
export const LARGEST_BINDING = 1017118720;
const MAX_CHUNK = 8 * 1024 * 1024;
const verified = new WeakSet();
interface StrandsChunk {
  readonly offset: number;
  readonly bytes: number;
  readonly sha256: string;
}

interface StrandsRecord {
  readonly index: number;
  readonly dtype: string;
  readonly shape: number[];
  readonly bytes: number;
  readonly sourceOffset: number;
  readonly sourceFileBytes: number;
  readonly sourcePath: string;
  readonly chunks: StrandsChunk[];
  readonly target: string;
}

interface StrandsInputContract {
  readonly sequence: string;
  readonly batch: number;
  readonly ids: string;
  readonly positions: string;
  readonly mask: string;
  readonly valid_mask: string;
  readonly answer_index: string;
  readonly option_indices: string;
}

interface StrandsManifest {
  readonly status: string;
  readonly wasmSha256: string;
  readonly inputContract: StrandsInputContract;
  readonly totalBytes: number;
  readonly largestBindingBytes: number;
  readonly maxRangeBytes: number;
  readonly records: StrandsRecord[];
}

interface AdapterCandidate {
  readonly isFallbackAdapter?: unknown;
  readonly info?: {readonly isFallbackAdapter?: unknown} | null | undefined;
  readonly features?: {readonly has?: ((feature: string) => unknown) | undefined} | null | undefined;
  readonly limits?:
    | {
        readonly maxBufferSize?: number;
        readonly maxStorageBufferBindingSize?: number;
        readonly maxStorageBuffersPerShaderStage?: number;
      }
    | null
    | undefined;
}

interface ProgressUpdate {
  readonly loadedBytes: number;
  readonly totalBytes: number;
  readonly fraction: number;
}

interface UploadOptions {
  readonly allocate?: ((record: StrandsRecord) => unknown | PromiseLike<unknown>) | undefined;
  readonly upload?:
    | ((
        record: StrandsRecord,
        tensor: unknown,
        signal?: AbortSignal | null | undefined
      ) => unknown | PromiseLike<unknown>)
    | undefined;
  readonly signal?: AbortSignal | null | undefined;
  readonly onProgress?: ((progress: ProgressUpdate) => void) | undefined;
}

const invalid = (message: string): Error => new WebDecisionError('INVALID_ASSET', message);

function validateChunks(rec: StrandsRecord): void {
  let offset = 0;
  for (const chunk of rec.chunks) {
    if (
      chunk.offset !== offset ||
      !Number.isSafeInteger(chunk.bytes) ||
      chunk.bytes < 1 ||
      chunk.bytes > MAX_CHUNK ||
      chunk.bytes % 4 ||
      !/^[0-9a-f]{64}$/.test(chunk.sha256)
    )
      throw invalid('Invalid Strands chunk metadata.');
    offset += chunk.bytes;
    Object.freeze(chunk);
  }
  if (offset !== rec.bytes) throw invalid('Incomplete Strands tensor ranges.');
}

/** Parse only the exact compiled graph's pinned 328-parameter source manifest. */
export async function parsePinnedManifest(
  bytes: unknown,
  signal?: AbortSignal | null | undefined
): Promise<StrandsManifest> {
  signal?.throwIfAborted();
  if (!(bytes instanceof Uint8Array) || !bytes.byteLength || bytes.byteLength > 1024 * 1024)
    throw invalid('Invalid Strands manifest bytes.');
  const digest: string = await waitForAbort(() => sha256(bytes), signal);
  signal?.throwIfAborted();
  if (digest !== MANIFEST_SHA256) throw invalid('Strands manifest hash mismatch.');
  let manifest: StrandsManifest;
  try {
    manifest = JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(bytes));
  } catch {
    throw invalid('Invalid Strands manifest JSON.');
  }
  if (
    manifest.status !== 'success' ||
    manifest.wasmSha256 !== WASM_SHA256 ||
    manifest.inputContract?.sequence !== `2..${STRANDS_FUSED_GRAPH_MAX_TOKENS}` ||
    manifest.inputContract?.batch !== 1 ||
    manifest.inputContract?.ids !== '[1,n] int32' ||
    manifest.inputContract?.positions !== '[1,n] int32' ||
    manifest.inputContract?.mask !== '[1,1,n,n] float16' ||
    manifest.inputContract?.valid_mask !== '[1,n] float16' ||
    manifest.inputContract?.answer_index !== '[1] int32, exactly n-1' ||
    manifest.inputContract?.option_indices !== '[m] int32, each in [0,n-1)' ||
    manifest.totalBytes !== TOTAL_BYTES ||
    manifest.largestBindingBytes !== LARGEST_BINDING ||
    manifest.maxRangeBytes !== MAX_CHUNK ||
    manifest.records?.length !== 328
  )
    throw invalid('Unexpected Strands manifest contract.');
  let total = 0;
  for (const [index, rec] of manifest.records.entries()) {
    const width = rec.dtype === 'float16' ? 2 : rec.dtype === 'float32' ? 4 : 0;
    if (
      rec.index !== index ||
      !width ||
      !Array.isArray(rec.shape) ||
      !rec.shape.length ||
      rec.shape.some((value) => !Number.isSafeInteger(value) || value < 1) ||
      !Number.isSafeInteger(rec.bytes) ||
      rec.bytes <= 0 ||
      rec.bytes > LARGEST_BINDING ||
      rec.shape.reduce((count, value) => count * value, width) !== rec.bytes ||
      !Number.isSafeInteger(rec.sourceOffset) ||
      rec.sourceOffset < 0 ||
      rec.sourceOffset % 4 ||
      !Number.isSafeInteger(rec.sourceFileBytes) ||
      rec.sourceFileBytes < rec.sourceOffset + rec.bytes ||
      typeof rec.sourcePath !== 'string' ||
      !rec.sourcePath ||
      rec.sourcePath.startsWith('/') ||
      /[\\?#]/.test(rec.sourcePath) ||
      rec.sourcePath.split('/').some((part) => !part || part === '.' || part === '..') ||
      !Array.isArray(rec.chunks) ||
      !rec.chunks.length
    )
      throw invalid(`Invalid Strands parameter metadata at index ${index}.`);
    validateChunks(rec);
    total += rec.bytes;
    Object.freeze(rec.chunks);
    Object.freeze(rec.shape);
    Object.freeze(rec);
  }
  if (
    total !== TOTAL_BYTES ||
    manifest.records[0]!.bytes !== LARGEST_BINDING ||
    manifest.records.slice(320, 326).some((r) => !r.target.startsWith('pointer.head.') || r.dtype !== 'float32')
  )
    throw invalid('Unexpected Strands parameter totals or pointer head.');
  Object.freeze(manifest.records);
  Object.freeze(manifest.inputContract);
  Object.freeze(manifest);
  signal?.throwIfAborted();
  verified.add(manifest);
  return manifest;
}

/** Validate actual adapter limits; this does not prove sufficient free memory. */
export function checkFusedAdapter(adapter: unknown): void {
  if (
    !adapter ||
    (adapter as AdapterCandidate).isFallbackAdapter ||
    (adapter as AdapterCandidate).info?.isFallbackAdapter ||
    typeof (adapter as AdapterCandidate).features?.has !== 'function' ||
    !(adapter as AdapterCandidate).features!.has!('shader-f16') ||
    ((adapter as AdapterCandidate).limits?.maxBufferSize as number) < LARGEST_BINDING ||
    ((adapter as AdapterCandidate).limits?.maxStorageBufferBindingSize as number) < LARGEST_BINDING ||
    ((adapter as AdapterCandidate).limits?.maxStorageBuffersPerShaderStage as number) < 8 ||
    !Number.isFinite((adapter as AdapterCandidate).limits?.maxBufferSize) ||
    !Number.isFinite((adapter as AdapterCandidate).limits?.maxStorageBufferBindingSize) ||
    !Number.isFinite((adapter as AdapterCandidate).limits?.maxStorageBuffersPerShaderStage)
  )
    throw new WebDecisionError('UNSUPPORTED_ENVIRONMENT', 'Strands needs shader-f16 and a 1,017,118,720-byte binding.');
}

/** Upload in exact graph order; caller owns all allocations and failure cleanup. */
export async function uploadPinnedWeights(
  manifest: unknown,
  {allocate, upload, signal, onProgress}: UploadOptions = {}
): Promise<unknown[]> {
  if (!verified.has(manifest as object) || typeof allocate !== 'function' || typeof upload !== 'function')
    throw invalid('A verified Strands manifest and upload functions are required.');
  const weights: unknown[] = [];
  let loaded = 0;
  for (const record of (manifest as StrandsManifest).records) {
    signal?.throwIfAborted();
    const tensor = await allocate(record);
    signal?.throwIfAborted();
    await upload(record, tensor, signal);
    signal?.throwIfAborted();
    weights.push(tensor);
    loaded += record.bytes;
    onProgress?.({loadedBytes: loaded, totalBytes: TOTAL_BYTES, fraction: loaded / TOTAL_BYTES});
    signal?.throwIfAborted();
  }
  return weights;
}
