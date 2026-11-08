import {openAssetCache, readVerifiedAsset, sha256, waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {SKEJJA_ARCHIVE, SKEJJA_INT8_PROFILE} from './contract.mjs';

const fail = (code, message, cause) => new WebDecisionError(code, message, cause ? {cause} : undefined);
const invalidArchive = (cause) => fail('INVALID_ASSET', 'The pinned Skejja archive is malformed.', cause);
const MAX_TAR_TAIL_BYTES = 10 * 1024;
const MAX_ARCHIVE_FILE_BYTES = 128 * 1024 * 1024;

function ensureGzipSupport() {
  if (typeof globalThis.DecompressionStream !== 'function' || typeof globalThis.ReadableStream !== 'function')
    throw fail('UNSUPPORTED_ENVIRONMENT', 'Skejja ID-only loading requires browser gzip decompression support.');
  try {
    new globalThis.DecompressionStream('gzip');
  } catch (cause) {
    throw fail('UNSUPPORTED_ENVIRONMENT', 'Skejja ID-only loading requires browser gzip decompression support.', cause);
  }
}

function isZero(bytes) {
  for (const byte of bytes) if (byte !== 0) return false;
  return true;
}

function readAsciiField(header, start, length) {
  let end = start + length;
  for (let index = start; index < start + length; index++) {
    if (header[index] === 0) {
      end = index;
      for (let padding = index + 1; padding < start + length; padding++) {
        if (header[padding] !== 0) throw invalidArchive();
      }
      break;
    }
    if (header[index] < 0x20 || header[index] > 0x7e) throw invalidArchive();
  }
  return String.fromCharCode(...header.subarray(start, end));
}

function readOctalField(header, start, length) {
  let end = start + length;
  for (let index = start; index < start + length; index++) {
    if (header[index] === 0) {
      end = index;
      for (let padding = index + 1; padding < start + length; padding++) {
        if (header[padding] !== 0 && header[padding] !== 0x20) throw invalidArchive();
      }
      break;
    }
  }
  const value = String.fromCharCode(...header.subarray(start, end)).trim();
  if (!/^[0-7]+$/.test(value)) throw invalidArchive();
  const parsed = Number.parseInt(value, 8);
  if (!Number.isSafeInteger(parsed) || parsed < 0) throw invalidArchive();
  return parsed;
}

function validateTarHeader(header) {
  if (header.byteLength !== 512 || String.fromCharCode(...header.subarray(257, 263)) !== 'ustar\0')
    throw invalidArchive();
  if (String.fromCharCode(...header.subarray(263, 265)) !== '00') throw invalidArchive();
  const checksum = readOctalField(header, 148, 8);
  let actual = 0;
  for (let index = 0; index < header.length; index++) actual += index >= 148 && index < 156 ? 0x20 : header[index];
  if (checksum !== actual) throw invalidArchive();

  const name = readAsciiField(header, 0, 100);
  const prefix = readAsciiField(header, 345, 155);
  if (!name) throw invalidArchive();
  return {
    path: prefix ? `${prefix}/${name}` : name,
    type: header[156] === 0 ? '\0' : String.fromCharCode(header[156]),
    size: readOctalField(header, 124, 12)
  };
}

function createTarReader(reader, signal, maxExpandedBytes) {
  let chunk = null;
  let offset = 0;
  let produced = 0;
  let ended = false;

  async function nextChunk() {
    if (chunk && offset < chunk.byteLength) return true;
    chunk = null;
    offset = 0;
    if (ended) return false;
    const item = await waitForAbort(() => reader.read(), signal);
    if (item.done) {
      ended = true;
      return false;
    }
    if (!(item.value instanceof Uint8Array) || item.value.BYTES_PER_ELEMENT !== 1) throw invalidArchive();
    produced += item.value.byteLength;
    if (produced > maxExpandedBytes) throw invalidArchive();
    chunk = item.value;
    return chunk.byteLength > 0 ? true : nextChunk();
  }

  async function readInto(target, targetOffset, length) {
    let written = 0;
    while (written < length) {
      if (!(await nextChunk())) throw invalidArchive();
      const count = Math.min(length - written, chunk.byteLength - offset);
      target.set(chunk.subarray(offset, offset + count), targetOffset + written);
      offset += count;
      written += count;
    }
  }

  async function readExact(length) {
    const target = new Uint8Array(length);
    await readInto(target, 0, length);
    return target;
  }

  async function skip(length) {
    let remaining = length;
    while (remaining > 0) {
      if (!(await nextChunk())) throw invalidArchive();
      const count = Math.min(remaining, chunk.byteLength - offset);
      offset += count;
      remaining -= count;
    }
  }

  async function verifyZeroTail(maxBytes) {
    let length = 0;
    while (await nextChunk()) {
      const tail = chunk.subarray(offset);
      length += tail.byteLength;
      if (length > maxBytes || !isZero(tail)) throw invalidArchive();
      offset = chunk.byteLength;
    }
  }

  return {readExact, readInto, skip, verifyZeroTail};
}

function asArchiveBytes(archiveBytes) {
  if (archiveBytes instanceof ArrayBuffer) return new Uint8Array(archiveBytes);
  if (ArrayBuffer.isView(archiveBytes))
    return new Uint8Array(archiveBytes.buffer, archiveBytes.byteOffset, archiveBytes.byteLength);
  return null;
}

function validateProfileFile(name, metadata, archiveSpec) {
  if (archiveSpec.files[name] !== metadata.bytes) throw invalidArchive();
  if (!Number.isSafeInteger(metadata.bytes) || metadata.bytes < 0 || metadata.bytes > MAX_ARCHIVE_FILE_BYTES)
    throw invalidArchive();
  if (!/^[0-9a-f]{64}$/.test(metadata.sha256)) throw invalidArchive();
}

function validateArchiveInputs(bytes, profile, archiveSpec) {
  if (!bytes || bytes.byteLength !== archiveSpec.bytes || !archiveSpec.root || !archiveSpec.files)
    throw invalidArchive();
  for (const [name, metadata] of Object.entries(profile.files)) validateProfileFile(name, metadata, archiveSpec);
  if (!Number.isSafeInteger(archiveSpec.maxExpandedBytes)) throw invalidArchive();
  if (archiveSpec.maxExpandedBytes < 1024 || archiveSpec.maxExpandedBytes > SKEJJA_ARCHIVE.maxExpandedBytes)
    throw invalidArchive();
}

function createArchiveParseState(profile, archiveSpec) {
  return {
    expected: new Map(Object.entries(archiveSpec.files)),
    extracted: new Map(),
    seen: new Set(),
    rootSeen: false,
    profile
  };
}

function validateArchiveFileMember(member, state, archiveSpec) {
  if (!['\0', '0'].includes(member.type) || !member.path.startsWith(`${archiveSpec.root}/`)) throw invalidArchive();
  const name = member.path.slice(archiveSpec.root.length + 1);
  if (!name || name.includes('/') || !state.expected.has(name) || state.seen.has(name)) throw invalidArchive();
  if (member.size !== state.expected.get(name) || member.size > MAX_ARCHIVE_FILE_BYTES) throw invalidArchive();
  return name;
}

async function readArchiveFilePayload(name, member, tar, state, signal, hashBytes) {
  const metadata = state.profile.files[name];
  if (!metadata) {
    await tar.skip(member.size);
    return null;
  }

  const payload = new Uint8Array(member.size);
  await tar.readInto(payload, 0, member.size);
  signal?.throwIfAborted();
  const digest = await waitForAbort(() => hashBytes(payload.buffer), signal);
  if (digest !== metadata.sha256) throw fail('INVALID_ASSET', `Integrity check failed for ${name}.`);
  return payload.buffer;
}

async function consumeArchiveFile(member, tar, state, archiveSpec, signal, hashBytes) {
  const name = validateArchiveFileMember(member, state, archiveSpec);
  state.seen.add(name);
  const payload = await readArchiveFilePayload(name, member, tar, state, signal, hashBytes);
  if (payload) state.extracted.set(name, payload);
  const padding = (512 - (member.size % 512)) % 512;
  if (padding && !isZero(await tar.readExact(padding))) throw invalidArchive();
}

async function consumeArchiveMember(member, tar, state, archiveSpec, signal, hashBytes) {
  if (!state.rootSeen) {
    if (member.path !== `${archiveSpec.root}/` || member.type !== '5' || member.size !== 0) throw invalidArchive();
    state.rootSeen = true;
    return;
  }
  await consumeArchiveFile(member, tar, state, archiveSpec, signal, hashBytes);
}

async function consumeArchiveFooter(tar) {
  if (!isZero(await tar.readExact(512))) throw invalidArchive();
  await tar.verifyZeroTail(MAX_TAR_TAIL_BYTES);
}

function validateExtractedFiles(state, footerSeen) {
  if (!footerSeen || !state.rootSeen || state.seen.size !== state.expected.size) throw invalidArchive();
  if (state.extracted.size !== Object.keys(state.profile.files).length) throw invalidArchive();
  for (const name of Object.keys(state.profile.files)) {
    if (!state.extracted.has(name)) throw invalidArchive();
  }
}

async function readArchiveEntries(tar, profile, archiveSpec, signal, hashBytes) {
  const state = createArchiveParseState(profile, archiveSpec);
  let footerSeen = false;
  for (let memberIndex = 0; memberIndex <= state.expected.size + 1; memberIndex++) {
    signal?.throwIfAborted();
    const header = await tar.readExact(512);
    if (isZero(header)) {
      await consumeArchiveFooter(tar);
      footerSeen = true;
      break;
    }
    await consumeArchiveMember(validateTarHeader(header), tar, state, archiveSpec, signal, hashBytes);
  }
  validateExtractedFiles(state, footerSeen);
  signal?.throwIfAborted();
  return state.extracted;
}

function throwArchiveFailure(cause, signal) {
  signal?.throwIfAborted();
  if (cause instanceof WebDecisionError || cause?.name === 'AbortError') throw cause;
  throw invalidArchive(cause);
}

async function releaseArchiveReader(reader, done) {
  if (!reader) return;
  if (!done) {
    try {
      await reader.cancel();
    } catch {}
  }
  try {
    reader.releaseLock();
  } catch {}
}

/** Parse the pinned USTAR archive and retain only files used by the selected model. */
export async function parseSkejjaArchive(
  archiveBytes,
  profile,
  {signal, archiveSpec = SKEJJA_ARCHIVE, hashBytes = sha256} = {}
) {
  signal?.throwIfAborted();
  const bytes = asArchiveBytes(archiveBytes);
  validateArchiveInputs(bytes, profile, archiveSpec);
  ensureGzipSupport();

  let reader;
  let done = false;
  try {
    const compressed = new globalThis.ReadableStream({
      start(controller) {
        signal?.throwIfAborted();
        controller.enqueue(bytes);
        controller.close();
      }
    });
    const decompressed = compressed.pipeThrough(new globalThis.DecompressionStream('gzip'));
    reader = decompressed.getReader();
    const tar = createTarReader(reader, signal, archiveSpec.maxExpandedBytes);
    const extracted = await readArchiveEntries(tar, profile, archiveSpec, signal, hashBytes);
    done = true;
    return extracted;
  } catch (cause) {
    throwArchiveFailure(cause, signal);
  } finally {
    await releaseArchiveReader(reader, done);
  }
}

export async function openSkejjaSource(
  {base, signal, cacheEnabled, offline, onProgress, useArchive = false},
  profile = SKEJJA_INT8_PROFILE,
  dependencies = {}
) {
  const openCache = dependencies.openAssetCache ?? openAssetCache;
  const readAsset = dependencies.readVerifiedAsset ?? readVerifiedAsset;
  const parseArchive = dependencies.parseSkejjaArchive ?? parseSkejjaArchive;
  const archiveSpec = dependencies.archiveSpec ?? SKEJJA_ARCHIVE;
  if (useArchive) ensureGzipSupport();
  const cache = await openCache(cacheEnabled, signal);
  let loadedBytes = 0;

  if (!useArchive) {
    return {
      async read(name) {
        const buffer = await readAsset(base, {files: profile.files}, name, {
          cache,
          signal,
          offline,
          onBytes(bytes) {
            loadedBytes += bytes;
            onProgress?.({
              loadedBytes,
              totalBytes: profile.totalBytes,
              fraction: loadedBytes / profile.totalBytes
            });
          }
        });
        return buffer;
      }
    };
  }

  let archivePromise;
  const loadArchive = () => {
    if (!archivePromise) {
      archivePromise = (async () => {
        let archiveBytes = await readAsset(
          base,
          {files: {[archiveSpec.filename]: {bytes: archiveSpec.bytes, sha256: archiveSpec.sha256}}},
          archiveSpec.filename,
          {cache, signal, offline, maxAssetBytes: archiveSpec.bytes}
        );
        loadedBytes += archiveSpec.bytes;
        onProgress?.({loadedBytes, totalBytes: archiveSpec.bytes, fraction: loadedBytes / archiveSpec.bytes});
        try {
          return await parseArchive(archiveBytes, profile, {signal, archiveSpec});
        } finally {
          archiveBytes = null;
        }
      })();
    }
    return archivePromise;
  };

  return {
    async read(name) {
      if (!Object.hasOwn(profile.files, name)) throw fail('INVALID_ASSET', `Asset not listed for Skejja: ${name}.`);
      const assets = await loadArchive();
      const buffer = assets.get(name);
      if (!buffer) throw fail('INVALID_ASSET', `The pinned Skejja archive is missing ${name}.`);
      return buffer;
    },
    async release(names) {
      if (!archivePromise) return;
      try {
        const assets = await loadArchive();
        for (const name of names) assets.delete(name);
      } catch {}
    }
  };
}
