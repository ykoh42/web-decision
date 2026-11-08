import {sha256, waitForAbort} from '../assets.js';

const ROWS = [65536, 65536, 65536, 51712],
  MAX_RANGE = 4 * 1024 * 1024;
const HEX = /^[a-f0-9]{64}$/;

function checkStream(entry, rows, width, dtype) {
  if (
    !entry ||
    entry.dtype !== dtype ||
    !Array.isArray(entry.shape) ||
    entry.shape.length !== 2 ||
    entry.shape[0] !== rows ||
    entry.shape[1] !== width ||
    !Number.isSafeInteger(entry.bytes) ||
    entry.bytes !== rows * width * (dtype === 'uint32' ? 4 : 2) ||
    typeof entry.path !== 'string' ||
    !Array.isArray(entry.chunks) ||
    !entry.chunks.length
  )
    throw new Error('Invalid Clef bank metadata');
  let offset = 0;
  for (const part of entry.chunks) {
    if (
      part.offset !== offset ||
      !Number.isSafeInteger(part.length) ||
      part.length < 1 ||
      part.length > MAX_RANGE ||
      typeof part.sha256 !== 'string' ||
      !HEX.test(part.sha256)
    )
      throw new Error('Invalid Clef bank ranges');
    offset += part.length;
  }
  if (offset !== entry.bytes) throw new Error('Incomplete Clef bank ranges');
}

function validateBanks(banks, layout, columns) {
  if (!Array.isArray(banks) || banks.length !== 4) throw new Error('Invalid Clef selected IDs');
  const paths = new Set();
  for (let index = 0; index < 4; index++) {
    const bank = banks[index];
    if (bank?.rows !== layout[index]) throw new Error('Invalid Clef bank row count');
    checkStream(bank.packed, bank.rows, columns / 8, 'uint32');
    checkStream(bank.scales, bank.rows, columns / 32, 'float16');
    for (const stream of [bank.packed, bank.scales]) {
      if (paths.has(stream.path)) throw new Error('Duplicate Clef bank source');
      paths.add(stream.path);
    }
  }
}

function copySelected(bytes, part, selected, rowBytes, target, copied) {
  const end = part.offset + part.length;
  for (const [row, index] of selected) {
    const start = row * rowBytes,
      left = Math.max(start, part.offset),
      right = Math.min(start + rowBytes, end);
    if (left < right) {
      target.set(bytes.subarray(left - part.offset, right - part.offset), index * rowBytes + left - start);
      copied[index] += right - left;
    }
  }
}

async function collectStream(stream, rowBytes, target, selected, readRange, signal) {
  const copied = new Uint32Array(target.length / rowBytes);
  for (const part of stream.chunks) {
    const end = part.offset + part.length;
    if (!selected.some(([row]) => row * rowBytes < end && (row + 1) * rowBytes > part.offset)) continue;
    const expected = {offset: part.offset, length: part.length, sha256: part.sha256};
    signal?.throwIfAborted();
    const received = await readRange(structuredClone(stream), {...expected}, signal);
    signal?.throwIfAborted();
    const raw = received instanceof ArrayBuffer ? new Uint8Array(received) : received;
    if (!(raw instanceof Uint8Array) || raw.byteLength !== expected.length) throw new Error('Wrong Clef range length');
    const bytes = Uint8Array.from(raw);
    const digest = await waitForAbort(() => sha256(bytes), signal);
    signal?.throwIfAborted();
    if (digest !== expected.sha256) throw new Error('Clef range hash mismatch');
    copySelected(bytes, expected, selected, rowBytes, target, copied);
  }
  if (selected.some(([, index]) => copied[index] !== rowBytes)) throw new Error('Incomplete Clef selected row');
}

async function selectRows(inputIds, inputBanks, {readRange, signal} = {}, layout, columns) {
  if (typeof readRange !== 'function') throw new TypeError('A range reader is required');
  const ids = Array.from(inputIds ?? []),
    banks = structuredClone(inputBanks);
  const maximum = layout.reduce((a, b) => a + b, 0);
  if (!ids.length || ids.length > 512 || ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= maximum))
    throw new Error('Invalid Clef selected IDs');
  validateBanks(banks, layout, columns);
  const unique = [],
    lookup = new Map(),
    local = [];
  for (const id of ids) {
    if (!lookup.has(id)) {
      lookup.set(id, unique.length);
      unique.push(id);
    }
    local.push(lookup.get(id));
  }
  const packedRow = (columns / 8) * 4,
    scaleRow = (columns / 32) * 2;
  const rowBanks = new Int32Array(unique.length),
    packed = new Uint8Array(unique.length * packedRow),
    scales = new Uint8Array(unique.length * scaleRow);
  let base = 0;
  for (let bank = 0; bank < 4; bank++) {
    const entry = banks[bank],
      rows = layout[bank],
      selected = [];
    unique.forEach((id, index) => {
      if (id >= base && id < base + rows) {
        rowBanks[index] = bank;
        selected.push([id - base, index]);
      }
    });
    await collectStream(entry.packed, packedRow, packed, selected, readRange, signal);
    await collectStream(entry.scales, scaleRow, scales, selected, readRange, signal);
    base += rows;
  }
  const words = new Uint32Array((unique.length * columns) / 8),
    half = new Uint16Array((unique.length * columns) / 32);
  const pv = new DataView(packed.buffer),
    sv = new DataView(scales.buffer);
  for (let i = 0; i < words.length; i++) words[i] = pv.getUint32(i * 4, true);
  for (let i = 0; i < half.length; i++) {
    const value = sv.getUint16(i * 2, true);
    if ((value & 0x7c00) === 0x7c00) throw new Error('Nonfinite Clef scale');
    half[i] = value;
  }
  return {unique, local: Int32Array.from(local), rowBanks, packed: words, scales: half};
}

/** Authenticate every range intersecting selected rows; other ranges are not read. */
export function selectClefRows(ids, banks, options) {
  return selectRows(ids, banks, options, ROWS, 4096);
}

/** Bounded synthetic seam; it cannot represent real model banks. */
export function selectFakeClefRows(ids, banks, {fake, layout, columns, ...options} = {}) {
  if (
    fake !== true ||
    !Array.isArray(layout) ||
    layout.length !== 4 ||
    layout.some((value) => !Number.isSafeInteger(value) || value < 1 || value > 64) ||
    ![32, 64].includes(columns)
  )
    throw new TypeError('Invalid synthetic Clef bank layout');
  return selectRows(ids, banks, options, [...layout], columns);
}
