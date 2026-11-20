import {assetUrl, commitCachedManifest, waitForAbort} from '../assets.js';
import type {AssetCache} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {readVerifiedRange as readVerifiedRangeRuntime} from './range-reader.mjs';

const MAX_RANGE = 8 * 1024 * 1024;
const MAX_MEMORY_RANGES = 4;
const invalid = (message: string) => new WebDecisionError('INVALID_ASSET', message);

export interface RangeCacheSpec {
  readonly path: string;
  readonly totalBytes: number;
  readonly offset: number;
  readonly length: number;
  readonly sha256: string;
}

export interface RangeCacheReadOptions {
  readonly signal?: AbortSignal | null | undefined;
}

export interface RangeCacheReadResult {
  readonly bytes: ArrayBuffer;
  readonly cachedByHash: boolean;
}

interface RangeTransactionOptions {
  readonly base: URL;
  readonly cache?: AssetCache | null | undefined;
  readonly offline?: boolean | undefined;
  readonly signal?: AbortSignal | null | undefined;
  readonly markerId: string;
  readonly markerPath?: string | undefined;
}

interface CancellableBody {
  cancel(): unknown;
}

interface CombinedSignals {
  readonly signal: AbortSignal | null | undefined;
  close(): void;
}

interface OfflineMarker {
  readonly format?: unknown;
  readonly markerId?: unknown;
}

interface ReadVerifiedRangeOptions {
  readonly cache: AssetCache | null | undefined;
  readonly offline: boolean;
  readonly signal: AbortSignal | null | undefined;
}

const readVerifiedRange: (
  base: URL,
  spec: RangeCacheSpec,
  options: ReadVerifiedRangeOptions
) => Promise<RangeCacheReadResult> = readVerifiedRangeRuntime;

function keyOf(spec: RangeCacheSpec): string {
  if (
    !spec ||
    typeof spec.path !== 'string' ||
    !spec.path ||
    !Number.isSafeInteger(spec.totalBytes) ||
    spec.totalBytes < 1 ||
    !Number.isSafeInteger(spec.offset) ||
    spec.offset < 0 ||
    !Number.isSafeInteger(spec.length) ||
    spec.length < 1 ||
    spec.length > MAX_RANGE ||
    spec.offset > spec.totalBytes - spec.length ||
    !/^[0-9a-f]{64}$/.test(spec.sha256)
  )
    throw invalid('Invalid required range.');
  return JSON.stringify([spec.path, spec.totalBytes, spec.offset, spec.length, spec.sha256]);
}

function cancelBody(body: CancellableBody | null | undefined): void {
  try {
    Promise.resolve(body?.cancel()).catch(() => {});
  } catch {}
}

function combineSignals(...signals: (AbortSignal | null | undefined)[]): CombinedSignals {
  const active = [
    ...new Set(signals.filter((signal): signal is AbortSignal => signal !== null && signal !== undefined))
  ];
  if (!active.length) return {signal: undefined, close() {}};
  if (active.length === 1) return {signal: active[0], close() {}};
  const controller = new AbortController();
  const listeners = active.map((signal) => {
    const abort = () => controller.abort(signal.reason);
    signal.addEventListener('abort', abort, {once: true});
    return {signal, abort};
  });
  const aborted = active.find((signal) => signal.aborted);
  if (aborted) controller.abort(aborted.reason);
  return {
    signal: controller.signal,
    close() {
      for (const {signal, abort} of listeners) signal.removeEventListener('abort', abort);
    }
  };
}

async function readMarker(response: Response, signal: AbortSignal | null | undefined): Promise<unknown> {
  if (signal?.aborted) {
    cancelBody(response?.body);
    signal.throwIfAborted();
  }
  const reader = response.body?.getReader();
  if (!reader) throw invalid('Offline completion marker has no body.');
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for (;;) {
      const {done, value} = await waitForAbort(() => reader.read(), signal);
      if (done) break;
      size += value.byteLength;
      if (size > 4096) throw invalid('Offline completion marker is too large.');
      chunks.push(value);
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }
    return JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(bytes)) as unknown;
  } finally {
    cancelBody(reader);
  }
}

/** Track the exact expected ranges before publishing an offline-complete marker. */
export class RangeCacheTransaction {
  #base: URL;
  #cache: AssetCache | null | undefined;
  #offline: boolean;
  #signal: AbortSignal | null | undefined;
  #disposed = false;
  #disposeController = new AbortController();
  #onAbort = () => this.#disposeWithReason(this.#signal?.reason);
  #marker: Readonly<{format: 1; markerId: string}>;
  #markerUrl: string;
  #required = new Set<string>();
  #specs = new Map<string, RangeCacheSpec>();
  #read = new Set<string>();
  #cached = new Set<string>();
  #uncached = new Set<string>();
  #memory = new Map<string, RangeCacheReadResult>();
  #sealed = false;
  #ready = false;

  constructor({
    base,
    cache,
    offline = false,
    signal,
    markerId,
    markerPath = 'strands-offline-complete-v1.json'
  }: RangeTransactionOptions) {
    if (
      !(base instanceof URL) ||
      typeof markerId !== 'string' ||
      !markerId ||
      typeof offline !== 'boolean' ||
      typeof markerPath !== 'string' ||
      !markerPath
    )
      throw invalid('Invalid range cache transaction options.');
    if (offline && !cache) throw new WebDecisionError('ASSET_LOAD_FAILED', 'Offline range cache is unavailable.');
    this.#base = new URL(base.href);
    this.#cache = cache;
    this.#offline = offline;
    this.#signal = signal;
    this.#marker = Object.freeze({format: 1, markerId});
    this.#markerUrl = assetUrl(base, markerPath);
    this.#signal?.addEventListener('abort', this.#onAbort, {once: true});
    if (this.#signal?.aborted) this.dispose();
  }

  #assertOpen(): void {
    if (this.#disposed) throw new WebDecisionError('SESSION_CLOSED', 'The verified range transaction is disposed.');
  }

  dispose(): void {
    this.#disposeWithReason(new WebDecisionError('SESSION_CLOSED', 'The verified range transaction is disposed.'));
  }

  #disposeWithReason(reason: unknown): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#disposeController.abort(reason);
    this.#memory.clear();
    this.#required.clear();
    this.#specs.clear();
    this.#read.clear();
    this.#cached.clear();
    this.#uncached.clear();
    this.#signal?.removeEventListener('abort', this.#onAbort);
  }

  async open(): Promise<void> {
    this.#signal?.throwIfAborted();
    this.#assertOpen();
    const combined = combineSignals(this.#signal, this.#disposeController.signal);
    try {
      combined.signal?.throwIfAborted();
      if (this.#ready) return;
      if (this.#offline) {
        let marker: OfflineMarker | null | undefined;
        try {
          const response = await waitForAbort(
            () => this.#cache!.match(this.#markerUrl),
            combined.signal,
            (value) => cancelBody(value?.body)
          );
          if (combined.signal?.aborted) {
            cancelBody(response?.body);
            combined.signal.throwIfAborted();
          }
          if (response) marker = (await readMarker(response, combined.signal)) as OfflineMarker;
        } catch (cause) {
          combined.signal?.throwIfAborted();
          throw new WebDecisionError('ASSET_LOAD_FAILED', 'Could not read the offline completion marker.', {cause});
        }
        if (
          !marker ||
          Object.keys(marker).length !== 2 ||
          marker.format !== 1 ||
          marker.markerId !== this.#marker.markerId
        )
          throw new WebDecisionError('ASSET_LOAD_FAILED', 'The complete verified model is not available offline.');
      }
      combined.signal?.throwIfAborted();
      this.#assertOpen();
      this.#ready = true;
    } finally {
      combined.close();
    }
  }

  require(specs: readonly RangeCacheSpec[]): void {
    this.#assertOpen();
    if (this.#sealed) throw invalid('Range requirements are already sealed.');
    for (const spec of specs) {
      const required = Object.freeze({...spec});
      assetUrl(this.#base, required.path);
      const key = keyOf(required);
      this.#required.add(key);
      this.#specs.set(key, required);
    }
  }

  seal(): void {
    this.#assertOpen();
    if (!this.#ready || !this.#required.size) throw invalid('Cannot seal an empty or unopened range transaction.');
    this.#sealed = true;
  }

  hasRead(spec: RangeCacheSpec): boolean {
    return !this.#disposed && this.#read.has(keyOf(spec));
  }

  isCached(spec: RangeCacheSpec): boolean {
    return !this.#disposed && this.#cached.has(keyOf(spec));
  }

  hasUncachedReads(): boolean {
    return !this.#disposed && this.#uncached.size > 0;
  }

  #remember(key: string, result: RangeCacheReadResult): void {
    this.#memory.delete(key);
    this.#memory.set(key, {bytes: result.bytes.slice(0), cachedByHash: result.cachedByHash});
    while (this.#memory.size > MAX_MEMORY_RANGES) this.#memory.delete(this.#memory.keys().next().value!);
  }

  #record(key: string, result: RangeCacheReadResult): void {
    this.#read.add(key);
    if (result.cachedByHash) {
      this.#cached.add(key);
      this.#uncached.delete(key);
    } else {
      this.#cached.delete(key);
      this.#uncached.add(key);
    }
    this.#remember(key, result);
  }

  #cacheKey(spec: RangeCacheSpec): string {
    const url = new URL('/.web-decision-cache/sha256/', assetUrl(this.#base, spec.path));
    url.searchParams.set('digest', spec.sha256);
    return url.href;
  }

  /** Stop using the load-only signal while keeping the verified source transaction open. */
  detachSignal(): void {
    this.#signal?.throwIfAborted();
    this.#assertOpen();
    this.#signal?.removeEventListener('abort', this.#onAbort);
    this.#signal = undefined;
  }

  async read(base: URL, spec: RangeCacheSpec, {signal}: RangeCacheReadOptions = {}): Promise<RangeCacheReadResult> {
    this.#signal?.throwIfAborted();
    this.#assertOpen();
    if (!this.#ready || base.href !== this.#base.href) throw invalid('Unexpected range transaction base.');
    const key = keyOf(spec);
    if (!this.#required.has(key)) throw invalid('The requested range is not in the verified requirements.');
    const combined = combineSignals(this.#signal, signal, this.#disposeController.signal);
    try {
      combined.signal?.throwIfAborted();
      const inMemory = this.#memory.get(key);
      if (inMemory) {
        this.#memory.delete(key);
        this.#memory.set(key, inMemory);
        let result = inMemory;
        if (!result.cachedByHash && this.#cache && !this.#offline) {
          try {
            await waitForAbort(
              () => this.#cache!.put(this.#cacheKey(spec), new Response(result.bytes.slice(0))),
              combined.signal
            );
            result = {bytes: result.bytes, cachedByHash: true};
          } catch {
            combined.signal?.throwIfAborted();
          }
        }
        combined.signal?.throwIfAborted();
        this.#assertOpen();
        this.#record(key, result);
        return {bytes: result.bytes.slice(0), cachedByHash: result.cachedByHash};
      }
      try {
        const result = await readVerifiedRange(base, spec, {
          cache: this.#cache,
          offline: this.#offline,
          signal: combined.signal
        });
        combined.signal?.throwIfAborted();
        this.#assertOpen();
        this.#record(key, result);
        return result;
      } catch (cause) {
        this.#cached.delete(key);
        this.#uncached.delete(key);
        throw cause;
      }
    } finally {
      combined.close();
    }
  }

  async commit({signal}: RangeCacheReadOptions = {}): Promise<boolean> {
    this.#signal?.throwIfAborted();
    this.#assertOpen();
    const combined = combineSignals(this.#signal, signal, this.#disposeController.signal);
    try {
      combined.signal?.throwIfAborted();
      if (!this.#ready || !this.#sealed) throw invalid('The range transaction was not sealed.');
      if (this.#offline || !this.#cache || [...this.#required].some((key) => !this.#cached.has(key))) return false;
      for (const key of this.#required) {
        const spec = this.#specs.get(key);
        if (!spec) return false;
        let response: Response | null | undefined;
        try {
          response = await waitForAbort(
            () => this.#cache!.match(this.#cacheKey(spec)),
            combined.signal,
            (value) => cancelBody(value?.body)
          );
        } catch {
          combined.signal?.throwIfAborted();
          return false;
        }
        if (!response) return false;
        cancelBody(response.body);
      }
      try {
        await commitCachedManifest(this.#cache, this.#markerUrl, this.#marker, combined.signal);
      } catch {
        combined.signal?.throwIfAborted();
        return false;
      }
      combined.signal?.throwIfAborted();
      return true;
    } finally {
      combined.close();
    }
  }
}
