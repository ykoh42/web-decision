// Experimental upload against the pinned range inventory and runtime owner.
import {WebDecisionError} from '../kev.js';
import {assertUsable, withStrandsScope} from '../strands/scope.mjs';
import {validateRangeInventory} from './plan.mjs';

const sha = async (bytes) =>
  [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
    .map((x) => x.toString(16).padStart(2, '0'))
    .join('');
const invalid = (message) => new WebDecisionError('INVALID_ASSET', message);

function prepareParameters(owner, plan, inventory, {readRange, makeHasher}) {
  if (
    owner?.phase !== 'prepared' ||
    !Array.isArray(plan?.parameters) ||
    plan.parameters.length !== 682 ||
    !inventory ||
    typeof makeHasher !== 'function' ||
    typeof readRange !== 'function'
  )
    throw invalid('Missing ordered H2O load inputs.');
  const parameters = Array.from(plan.parameters, (entry) => {
    if (!entry || !Array.isArray(entry.shape)) throw invalid('Invalid H2O parameter.');
    return {...entry, shape: [...entry.shape], values: Array.isArray(entry.values) ? [...entry.values] : entry.values};
  });
  const files = parameters.slice(0, 680),
    rotary = parameters.slice(680);
  if (
    files.some(
      (entry) =>
        entry.kind !== 'file' || typeof entry.path !== 'string' || !/^[A-Za-z0-9_.-]+\.pending$/.test(entry.path)
    ) ||
    new Set(files.map((entry) => entry.path)).size !== 680 ||
    rotary.some(
      (entry, i) =>
        entry.kind !== 'rotary' || entry.target !== ['body.rotary_emb.inv_freq', 'body.rotary_emb.original_inv_freq'][i]
    )
  )
    throw invalid('Expected 680 distinct files followed by two ordered rotary inputs.');
  return parameters;
}

async function uploadFile(entry, ranges, {base, readRange, makeHasher, signal}, write) {
  const hasher = makeHasher();
  for (const range of ranges) {
    signal?.throwIfAborted();
    if (range.offset % 4 || range.length % 4) throw invalid('Unaligned H2O weight range.');
    const response = await readRange(base, range, {signal, cache: false, offline: false});
    if (!(response?.bytes instanceof ArrayBuffer) || response.bytes.byteLength !== range.length)
      throw invalid('H2O range length mismatch.');
    const bytes = Uint8Array.from(new Uint8Array(response.bytes));
    if ((await sha(bytes)) !== range.sha256) throw invalid('H2O range hash mismatch.');
    hasher.update(bytes);
    signal?.throwIfAborted();
    await write(range.offset, bytes);
  }
  if ((await hasher.digest()) !== entry.sha256) throw invalid('H2O full weight hash mismatch.');
}

async function uploadEntry(owner, entry, inventory, options) {
  options.signal?.throwIfAborted();
  assertUsable(owner);
  if (owner.lifecycle.loss || owner.lifecycle.disposed) throw new Error('Device lost during upload');
  const tensor = owner.runtime.empty(entry.shape, entry.dtype, owner.device);
  const context = owner.runtime.lib?.webGPUContext;
  if (
    typeof context?.copyRawBytesToBuffer !== 'function' ||
    typeof tensor.byteOffset !== 'number' ||
    tensor.byteOffset % 4
  )
    throw new Error('Pinned TVM upload API unavailable.');
  const write = async (offset, bytes) =>
    withStrandsScope(owner, 'H2O weight upload failed.', async (lease) => {
      context.copyRawBytesToBuffer(bytes, tensor.getDataPtr(), tensor.byteOffset + offset, bytes.byteLength);
      await lease.syncWithoutReadback();
    });
  if (entry.kind === 'file') {
    await uploadFile(entry, validateRangeInventory(entry, inventory[entry.path]), options, write);
  } else if (entry.kind === 'rotary') {
    if (
      entry.dtype !== 'float32' ||
      entry.shape?.length !== 1 ||
      entry.shape[0] !== 32 ||
      !Array.isArray(entry.values) ||
      entry.values.length !== 32 ||
      entry.values.some((v) => !Number.isFinite(v))
    )
      throw invalid('Invalid H2O rotary values.');
    await write(0, new Uint8Array(Float32Array.from(entry.values).buffer));
  } else throw invalid('Invalid H2O parameter source.');
  return tensor;
}

async function upload(owner, plan, inventory, options) {
  const parameters = prepareParameters(owner, plan, inventory, options);
  owner.phase = 'loading';
  const weights = [];
  try {
    for (const entry of parameters) {
      weights.push(await uploadEntry(owner, entry, inventory, options));
    }
    options.signal?.throwIfAborted();
    assertUsable(owner);
    if (owner.lifecycle.loss || owner.lifecycle.disposed) throw new Error('Device lost during upload');
    owner.uploadedWeights = Object.freeze(weights);
    owner.phase = 'prepared';
    return owner.uploadedWeights;
  } catch (error) {
    owner.phase = 'failed';
    // An upload may have submitted GPU work. Conservatively retain the entire
    // root scope/runtime on every failed load, even when no pending work is proved.
    owner.disposalGuard.retain();
    throw error;
  }
}

export async function uploadH2oParameters({owner, plan, inventory, base, readRange, makeHasher, signal} = {}) {
  return upload(owner, plan, inventory, {base, readRange, makeHasher, signal});
}

/** Tests inject fake TVM and range reader; the fake marker is not a security boundary. */
export function uploadFakeH2oParameters({
  fake,
  owner,
  plan,
  inventory,
  base = 'https://example.invalid/',
  readRange,
  makeHasher,
  signal
}) {
  if (fake !== true || typeof readRange !== 'function') throw new TypeError('Fake adapters required');
  return upload(owner, plan, inventory, {base, readRange, makeHasher, signal});
}
