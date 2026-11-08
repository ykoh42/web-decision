// Prepare a validated batch-one joint input for the pinned Q4 graph.
import {WebDecisionError} from '../kev.js';
import type {FlashGraphInput, PackedFlashRequest} from './types.js';
const VOCABULARY = 248320;
const MAX_TOKENS = 512;
const MAX_MARKERS = 255;

function invalid(): never {
  throw new WebDecisionError('INVALID_INPUT', 'Invalid Perplexity graph inputs.');
}

export function createFlashGraphInputs(packed: unknown, markerId: number): FlashGraphInput[] {
  const candidate = packed as PackedFlashRequest | null | undefined;
  if (!candidate || !Array.isArray(candidate.ids) || !Array.isArray(candidate.groups)) invalid();
  const ids = Array.from(candidate.ids);
  const groups = Array.from(candidate.groups);
  const n = ids.length;
  if (
    n < 2 ||
    n > MAX_TOKENS ||
    groups.length < 1 ||
    groups.length > MAX_MARKERS ||
    !Number.isSafeInteger(markerId) ||
    markerId < 0 ||
    markerId >= VOCABULARY ||
    ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= VOCABULARY)
  )
    invalid();
  const markers = groups.map((group) => group?.index);
  if (
    markers.some(
      (index, i) =>
        !Number.isSafeInteger(index) ||
        index < 0 ||
        index >= n ||
        ids[index as number] !== markerId ||
        (i > 0 && markers[i - 1]! >= index)
    )
  )
    invalid();
  const actual = [];
  for (let i = 0; i < n; i++) if (ids[i] === markerId) actual.push(i);
  if (actual.length !== markers.length || actual.some((index, i) => index !== markers[i])) invalid();
  return [
    {dtype: 'int32', shape: [1, n], data: Int32Array.from(ids)},
    {dtype: 'int32', shape: [1, n], data: Int32Array.from({length: n}, (_, i) => i)},
    {dtype: 'float16', shape: [1, 1, n, n], data: new Uint16Array(n * n)},
    {dtype: 'float16', shape: [1, n], data: new Uint16Array(n).fill(0x3c00)},
    {dtype: 'int32', shape: [markers.length], data: Int32Array.from(markers)}
  ];
}
