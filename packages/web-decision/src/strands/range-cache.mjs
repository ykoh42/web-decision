import {assetUrl, commitCachedManifest, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {readVerifiedRange} from './range-reader.mjs';

const MAX_RANGE = 8 * 1024 * 1024;
const invalid = (message) => new WebDecisionError('INVALID_ASSET', message);

function keyOf(spec) {
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

function cancelBody(body) {
  try {
    Promise.resolve(body?.cancel()).catch(() => {});
  } catch {}
}

function combineSignals(first, second) {
  if (!first || !second || first === second) return {signal: first || second, close() {}};
  const controller = new AbortController();
  const abortFirst = () => controller.abort(first.reason),
    abortSecond = () => controller.abort(second.reason);
  first.addEventListener('abort', abortFirst, {once: true});
  second.addEventListener('abort', abortSecond, {once: true});
  if (first.aborted) abortFirst();
  else if (second.aborted) abortSecond();
  return {
    signal: controller.signal,
    close() {
      first.removeEventListener('abort', abortFirst);
      second.removeEventListener('abort', abortSecond);
    }
  };
}

async function readMarker(response, signal) {
  if (signal?.aborted) {
    cancelBody(response?.body);
    signal.throwIfAborted();
  }
  const reader = response.body?.getReader();
  if (!reader) throw invalid('Offline completion marker has no body.');
  const chunks = [];
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
    return JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(bytes));
  } finally {
    cancelBody(reader);
  }
}

/** Track the exact expected ranges before publishing an offline-complete marker. */
export class RangeCacheTransaction {
  #base;
  #cache;
  #offline;
  #signal;
  #marker;
  #markerUrl;
  #required = new Set();
  #cached = new Set();
  #sealed = false;
  #ready = false;

  constructor({base, cache, offline = false, signal, markerId}) {
    if (!(base instanceof URL) || typeof markerId !== 'string' || !markerId || typeof offline !== 'boolean')
      throw invalid('Invalid range cache transaction options.');
    if (offline && !cache) throw new WebDecisionError('ASSET_LOAD_FAILED', 'Offline range cache is unavailable.');
    this.#base = new URL(base.href);
    this.#cache = cache;
    this.#offline = offline;
    this.#signal = signal;
    this.#marker = Object.freeze({format: 1, markerId});
    this.#markerUrl = assetUrl(base, 'strands-offline-complete-v1.json');
  }

  async open() {
    this.#signal?.throwIfAborted();
    if (this.#ready) return;
    if (this.#offline) {
      let marker;
      try {
        const response = await waitForAbort(
          () => this.#cache.match(this.#markerUrl),
          this.#signal,
          (value) => cancelBody(value?.body)
        );
        if (this.#signal?.aborted) {
          cancelBody(response?.body);
          this.#signal.throwIfAborted();
        }
        if (response) marker = await readMarker(response, this.#signal);
      } catch (cause) {
        this.#signal?.throwIfAborted();
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
    this.#signal?.throwIfAborted();
    this.#ready = true;
  }

  require(specs) {
    if (this.#sealed) throw invalid('Range requirements are already sealed.');
    for (const spec of specs) {
      assetUrl(this.#base, spec.path);
      this.#required.add(keyOf(spec));
    }
  }

  seal() {
    if (!this.#ready || !this.#required.size) throw invalid('Cannot seal an empty or unopened range transaction.');
    this.#sealed = true;
  }

  async read(base, spec, {signal} = {}) {
    if (!this.#ready || base.href !== this.#base.href) throw invalid('Unexpected range transaction base.');
    const key = keyOf(spec);
    if (!this.#required.has(key)) throw invalid('The requested range is not in the verified requirements.');
    const combined = combineSignals(this.#signal, signal);
    try {
      combined.signal?.throwIfAborted();
      const result = await readVerifiedRange(base, spec, {
        cache: this.#cache,
        offline: this.#offline,
        signal: combined.signal
      });
      combined.signal?.throwIfAborted();
      if (result.cachedByHash) this.#cached.add(key);
      else this.#cached.delete(key);
      return result;
    } finally {
      combined.close();
    }
  }

  async commit({signal} = {}) {
    const combined = combineSignals(this.#signal, signal);
    try {
      combined.signal?.throwIfAborted();
      if (!this.#ready || !this.#sealed) throw invalid('The range transaction was not sealed.');
      if (this.#offline || !this.#cache || [...this.#required].some((key) => !this.#cached.has(key))) return false;
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
