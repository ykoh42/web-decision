import {readVerifiedRange} from '../strands/range-reader.mjs';
import {waitForAbort} from '../abort.js';
import {parseStrictJson} from '../h2o/strict-json.mjs';
import {WebDecisionError} from '../kev.js';
import {NARI_FULL, NARI_MIXED} from './variants.mjs';

interface NariChunk {
  offset: number;
  bytes: number;
  sha256: string;
}
interface NariFile {
  path: string;
  bytes: number;
  sha256: string;
  chunks: NariChunk[];
}
interface NariMetadata {
  format: string;
  provenanceSigned: boolean;
  ordered: unknown[];
  inventory: NariFile[];
  assets: NariFile[];
  letterFile: unknown;
  sourceSha256: string;
}
interface NariRange {
  path: string;
  totalBytes: number;
  offset: number;
  length: number;
  sha256: string;
}
type NariReader = (
  base: URL,
  entry: NariRange,
  options: {cache: false; offline: false; signal: AbortSignal | undefined}
) => {bytes: ArrayBuffer} | PromiseLike<{bytes: ArrayBuffer}>;
interface SourceOptions {
  signal?: AbortSignal | undefined;
  reader?: NariReader | undefined;
  variant?: typeof NARI_FULL | typeof NARI_MIXED | undefined;
}
interface OperationOptions {
  signal?: AbortSignal | undefined;
}

const METADATA_PATH = 'nari-metadata.json';
const invalid = (message: string) => new WebDecisionError('INVALID_ASSET', message);
const digest = async (bytes: Uint8Array<ArrayBuffer>) =>
  [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
function freeze<T>(value: T): T {
  if (value && typeof value === 'object') {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}

function assetBase(value: string) {
  let base;
  try {
    base = new URL(value);
  } catch {
    throw invalid('Invalid Nari asset URL.');
  }
  if (
    !['http:', 'https:'].includes(base.protocol) ||
    base.username ||
    base.password ||
    base.search ||
    base.hash ||
    !base.pathname.endsWith('/')
  )
    throw invalid('Expected an HTTP(S) Nari asset directory.');
  return base;
}

// The pin authenticates this exact projection, not the upstream publisher.
export async function openNariSource(
  baseValue: string,
  {signal, reader = readVerifiedRange, variant = NARI_FULL}: SourceOptions = {}
) {
  if (variant !== NARI_FULL && variant !== NARI_MIXED) throw invalid('Unknown Nari variant.');
  if (typeof reader !== 'function') throw invalid('Invalid Nari range reader.');
  const base = assetBase(baseValue);
  signal?.throwIfAborted();
  const result = await waitForAbort(
    () =>
      reader(
        base,
        {
          path: METADATA_PATH,
          totalBytes: variant.metadataBytes,
          offset: 0,
          length: variant.metadataBytes,
          sha256: variant.metadataSha256
        },
        {cache: false, offline: false, signal}
      ),
    signal
  );
  signal?.throwIfAborted();
  if (!(result?.bytes instanceof ArrayBuffer) || result.bytes.byteLength !== variant.metadataBytes)
    throw invalid('Invalid Nari metadata.');
  const metadataBytes = Uint8Array.from(new Uint8Array(result.bytes));
  const metadataHash = await waitForAbort(() => digest(metadataBytes), signal);
  signal?.throwIfAborted();
  if (metadataHash !== variant.metadataSha256) throw invalid('Invalid Nari metadata.');
  let metadata: NariMetadata;
  try {
    metadata = freeze(parseStrictJson(metadataBytes)) as NariMetadata;
  } catch {
    throw invalid('Invalid Nari metadata.');
  }
  if (
    metadata.format !== variant.format ||
    metadata.provenanceSigned !== false ||
    metadata.ordered?.length !== variant.bindings ||
    metadata.inventory?.length !== variant.bindings - 1 ||
    metadata.assets?.length !== 4
  )
    throw invalid('Unexpected Nari metadata.');
  const files = new Map([...metadata.inventory, ...metadata.assets].map((file) => [file.path, file]));
  if (files.size !== variant.bindings + 3) throw invalid('Duplicate Nari asset path.');
  const controller = new AbortController();
  const forward = () => controller.abort(signal!.reason);
  signal?.addEventListener('abort', forward, {once: true});
  if (signal?.aborted) forward();
  let closed = false;

  async function readRange(
    path: string,
    offset: number,
    length: number,
    {signal: operationSignal}: OperationOptions = {}
  ) {
    if (closed) throw invalid('Nari source is closed.');
    const file = files.get(path);
    const chunk = file?.chunks.find((part) => part.offset === offset && part.bytes === length);
    if (!chunk) throw invalid('Range is not in the Nari inventory.');
    operationSignal?.throwIfAborted();
    controller.signal.throwIfAborted();
    const local = new AbortController();
    const cancelSource = () => local.abort(controller.signal.reason);
    const cancelOperation = () => local.abort(operationSignal!.reason);
    controller.signal.addEventListener('abort', cancelSource, {once: true});
    operationSignal?.addEventListener('abort', cancelOperation, {once: true});
    if (controller.signal.aborted) cancelSource();
    if (operationSignal?.aborted) cancelOperation();
    try {
      const result = await waitForAbort(
        () =>
          reader(
            base,
            {path, totalBytes: file!.bytes, offset, length, sha256: chunk.sha256},
            {cache: false, offline: false, signal: local.signal}
          ),
        local.signal
      );
      local.signal.throwIfAborted();
      if (!(result?.bytes instanceof ArrayBuffer) || result.bytes.byteLength !== length)
        throw invalid('Nari range hash mismatch.');
      const bytes = Uint8Array.from(new Uint8Array(result.bytes));
      const hash = await waitForAbort(() => digest(bytes), local.signal);
      local.signal.throwIfAborted();
      if (hash !== chunk.sha256) throw invalid('Nari range hash mismatch.');
      return bytes;
    } finally {
      controller.signal.removeEventListener('abort', cancelSource);
      operationSignal?.removeEventListener('abort', cancelOperation);
    }
  }

  async function readAsset(path: string, options?: OperationOptions) {
    if (closed) throw invalid('Nari source is closed.');
    const file = metadata.assets.find((entry) => entry.path === path);
    if (!file) throw invalid('Unknown Nari small asset.');
    const bytes = new Uint8Array(file.bytes);
    for (const chunk of file.chunks) bytes.set(await readRange(path, chunk.offset, chunk.bytes, options), chunk.offset);
    controller.signal.throwIfAborted();
    options?.signal?.throwIfAborted();
    const local = new AbortController();
    const cancelSource = () => local.abort(controller.signal.reason);
    const cancelOperation = () => local.abort(options!.signal!.reason);
    controller.signal.addEventListener('abort', cancelSource, {once: true});
    options?.signal?.addEventListener('abort', cancelOperation, {once: true});
    if (controller.signal.aborted) cancelSource();
    if (options?.signal?.aborted) cancelOperation();
    try {
      const hash = await waitForAbort(() => digest(bytes), local.signal);
      local.signal.throwIfAborted();
      if (hash !== file.sha256) throw invalid('Nari asset hash mismatch.');
      return bytes;
    } finally {
      controller.signal.removeEventListener('abort', cancelSource);
      options?.signal?.removeEventListener('abort', cancelOperation);
    }
  }

  return {
    plan: {ordered: metadata.ordered, letterFile: metadata.letterFile, sourceSha256: metadata.sourceSha256},
    inventory: metadata.inventory,
    readRange,
    readAsset,
    close() {
      if (closed) return;
      closed = true;
      signal?.removeEventListener('abort', forward);
      controller.abort(new DOMException('Nari source closed', 'AbortError'));
    }
  };
}
