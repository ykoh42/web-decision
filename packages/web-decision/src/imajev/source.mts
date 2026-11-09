import {readVerifiedRange} from '../strands/range-reader.mjs';
import {waitForAbort} from '../abort.js';
import {parseStrictJson} from '../h2o/strict-json.mjs';
import {WebDecisionError} from '../kev.js';
import {prepareImajevPlan} from './plan.mjs';

interface ImajevRange {
  path: string;
  totalBytes: number;
  offset: number;
  length: number;
  sha256: string;
}
type ImajevReader = (
  base: URL,
  entry: ImajevRange,
  options: {cache: false; offline: false; signal: AbortSignal | undefined}
) => Promise<{bytes: ArrayBuffer}>;
interface SourceOptions {
  signal?: AbortSignal | undefined;
  reader?: ImajevReader | undefined;
}
interface OperationOptions {
  signal?: AbortSignal | undefined;
}

const IMAJEV_METADATA_BYTES = 695014;
export const IMAJEV_METADATA_SHA256 = '0741c89f225e04ef1d2b72c88d7b3913dcc006ca5975bfc64ee34386ceea73a1';
const METADATA_PATH = 'imajev-metadata.json';
const invalid = () => new WebDecisionError('INVALID_ASSET', 'Invalid Imajev asset or inventory.');
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

function assetBase(value: string | URL) {
  let base;
  try {
    base = new URL(value);
  } catch {
    throw invalid();
  }
  if (
    !['http:', 'https:'].includes(base.protocol) ||
    base.username ||
    base.password ||
    base.search ||
    base.hash ||
    !base.pathname.endsWith('/')
  )
    throw invalid();
  return base;
}

async function verifiedBytes(reader: ImajevReader, base: URL, entry: ImajevRange, signal: AbortSignal | undefined) {
  signal?.throwIfAborted();
  const result = await waitForAbort(() => reader(base, entry, {cache: false, offline: false, signal}), signal);
  signal?.throwIfAborted();
  if (!(result?.bytes instanceof ArrayBuffer) || result.bytes.byteLength !== entry.length) throw invalid();
  const bytes = Uint8Array.from(new Uint8Array(result.bytes));
  const hash = await waitForAbort(() => digest(bytes), signal);
  signal?.throwIfAborted();
  if (hash !== entry.sha256) throw invalid();
  return bytes;
}

/** Open only the locally pinned experimental metadata; hashes do not authenticate a publisher. */
export async function openImajevSource(
  baseValue: string | URL,
  {signal, reader = readVerifiedRange}: SourceOptions = {}
) {
  if (typeof reader !== 'function') throw invalid();
  const base = assetBase(baseValue);
  const bytes = await verifiedBytes(
    reader,
    base,
    {
      path: METADATA_PATH,
      totalBytes: IMAJEV_METADATA_BYTES,
      offset: 0,
      length: IMAJEV_METADATA_BYTES,
      sha256: IMAJEV_METADATA_SHA256
    },
    signal
  );
  let metadata: unknown, plan: ReturnType<typeof prepareImajevPlan>;
  try {
    metadata = freeze(parseStrictJson(bytes));
    plan = prepareImajevPlan(metadata as Parameters<typeof prepareImajevPlan>[0]);
  } catch {
    throw invalid();
  }
  signal?.throwIfAborted();
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
    if (closed) throw invalid();
    const file = plan.files.get(path);
    const sha256 = file?.chunks.get(`${offset}:${length}`);
    if (!sha256) throw invalid();
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
      return await verifiedBytes(reader, base, {path, totalBytes: file!.bytes, offset, length, sha256}, local.signal);
    } finally {
      controller.signal.removeEventListener('abort', cancelSource);
      operationSignal?.removeEventListener('abort', cancelOperation);
    }
  }

  async function readAsset(path: string, options?: OperationOptions) {
    if (closed) throw invalid();
    const file = plan.assets.find((entry) => entry.path === path);
    if (!file) throw invalid();
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
      if (hash !== file.sha256) throw invalid();
      return bytes;
    } finally {
      controller.signal.removeEventListener('abort', cancelSource);
      options?.signal?.removeEventListener('abort', cancelOperation);
    }
  }

  return {
    plan: Object.freeze({bindings: plan.bindings, maxInputTokens: plan.maxInputTokens}),
    readRange,
    readAsset,
    close() {
      if (closed) return;
      closed = true;
      signal?.removeEventListener('abort', forward);
      controller.abort(new DOMException('Imajev source closed', 'AbortError'));
    }
  };
}
