// Internal distribution contract for the unchanged, pinned range inventory.
import {WebDecisionError} from '../kev.js';
import {readVerifiedRange} from '../strands/range-reader.mjs';
import {PINS} from './binding.mjs';
import {validateH2oSignal} from './signal.mjs';

export async function readH2oInventory(baseValue, signal) {
  validateH2oSignal(signal);
  signal?.throwIfAborted();
  let base;
  try {
    base = new URL(baseValue);
  } catch {
    throw new WebDecisionError('INVALID_ASSET', 'Invalid asset base');
  }
  if (
    !['https:', 'http:'].includes(base.protocol) ||
    base.username ||
    base.password ||
    base.search ||
    base.hash ||
    !base.pathname.endsWith('/')
  )
    throw new WebDecisionError('INVALID_ASSET', 'Expected an HTTP(S) asset directory');
  const bytes = 187681;
  const result = await readVerifiedRange(
    base,
    {path: 'evidence/inventory.json', totalBytes: bytes, offset: 0, length: bytes, sha256: PINS.inventory},
    {cache: false, offline: false, signal}
  );
  return new Uint8Array(result.bytes);
}
