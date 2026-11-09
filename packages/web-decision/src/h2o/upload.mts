// Experimental upload against the pinned range inventory and runtime owner.
import {WebDecisionError} from '../kev.js';
import {assertUsable, withStrandsScope} from '../strands/scope.mjs';
import {validateRangeInventory} from './plan.mjs';
import type {H2oRuntimeOwner, H2oTensor} from './owner.mjs';
import type {H2oOpenedSource} from './source.mjs';

type H2oRangeChunk = H2oOpenedSource['inventory'][string][number];
type H2oRangeSpec = Parameters<H2oOpenedSource['readRange']>[1];
type H2oRangeReader = H2oOpenedSource['readRange'];
type H2oHasher = {update(bytes: unknown): void; digest(): string | Promise<string>};
type H2oHasherFactory = () => H2oHasher;
type H2oInventory = Record<string, H2oRangeChunk[] | undefined>;
type H2oFileParameter = {
  kind: 'file';
  path: string;
  dtype: string;
  shape: number[];
  sha256: string;
  values?: unknown;
};
type H2oRotaryParameter = {
  kind: 'rotary';
  target: string;
  dtype: string;
  shape: number[];
  sha256?: string;
  values?: unknown;
};
type H2oParameter = H2oFileParameter | H2oRotaryParameter;
type H2oPlan = {parameters: H2oParameter[]};
type H2oUploadOptions = {
  owner?: H2oRuntimeOwner | null | undefined;
  plan?: H2oPlan | null | undefined;
  inventory?: H2oInventory | null | undefined;
  base?: string | URL | undefined;
  readRange?: H2oRangeReader | undefined;
  makeHasher?: H2oHasherFactory | undefined;
  signal?: AbortSignal | null | undefined;
};
type H2oUploadRuntimeOptions = {
  base: string | URL | undefined;
  readRange: H2oRangeReader | undefined;
  makeHasher: H2oHasherFactory | undefined;
  signal: AbortSignal | null | undefined;
};
type H2oValidatedUploadRuntimeOptions = H2oUploadRuntimeOptions & {
  readRange: H2oRangeReader;
  makeHasher: H2oHasherFactory;
};
type H2oWebGPUContext = {
  copyRawBytesToBuffer(bytes: Uint8Array, data: unknown, offset: number, length: number): unknown;
};
type ValidateRangeInventory = (file: H2oFileParameter, ranges: H2oRangeChunk[] | undefined) => H2oRangeSpec[];

const sha = async (bytes: Uint8Array): Promise<string> =>
  [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes as BufferSource))]
    .map((x) => x.toString(16).padStart(2, '0'))
    .join('');
const invalid = (message: string): WebDecisionError => new WebDecisionError('INVALID_ASSET', message);

function prepareParameters(
  owner: H2oRuntimeOwner | null | undefined,
  plan: H2oPlan | null | undefined,
  inventory: H2oInventory | null | undefined,
  {readRange, makeHasher}: H2oUploadRuntimeOptions
): H2oParameter[] {
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
    new Set(files.map((entry) => (entry as H2oFileParameter).path)).size !== 680 ||
    rotary.some(
      (entry, i) =>
        entry.kind !== 'rotary' || entry.target !== ['body.rotary_emb.inv_freq', 'body.rotary_emb.original_inv_freq'][i]
    )
  )
    throw invalid('Expected 680 distinct files followed by two ordered rotary inputs.');
  return parameters;
}

async function uploadFile(
  entry: H2oFileParameter,
  ranges: H2oRangeSpec[],
  {base, readRange, makeHasher, signal}: H2oValidatedUploadRuntimeOptions,
  write: (offset: number, bytes: Uint8Array) => Promise<unknown>
): Promise<void> {
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

async function uploadEntry(
  owner: H2oRuntimeOwner,
  entry: H2oParameter,
  inventory: H2oInventory,
  options: H2oValidatedUploadRuntimeOptions
): Promise<H2oTensor> {
  options.signal?.throwIfAborted();
  assertUsable(owner);
  if (owner.lifecycle.loss || owner.lifecycle.disposed) throw new Error('Device lost during upload');
  const tensor = owner.runtime.empty(entry.shape, entry.dtype, owner.device);
  const context = owner.runtime.lib?.webGPUContext as H2oWebGPUContext | null | undefined;
  if (
    typeof context?.copyRawBytesToBuffer !== 'function' ||
    typeof tensor.byteOffset !== 'number' ||
    tensor.byteOffset % 4
  )
    throw new Error('Pinned TVM upload API unavailable.');
  const write = async (offset: number, bytes: Uint8Array) =>
    withStrandsScope(owner, 'H2O weight upload failed.', async (lease) => {
      context.copyRawBytesToBuffer(
        bytes,
        tensor.getDataPtr(),
        (tensor.byteOffset as number) + offset,
        bytes.byteLength
      );
      await lease.syncWithoutReadback();
    });
  if (entry.kind === 'file') {
    await uploadFile(
      entry,
      (validateRangeInventory as ValidateRangeInventory)(entry, inventory[entry.path]),
      options,
      write
    );
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

async function upload(
  owner: H2oRuntimeOwner | null | undefined,
  plan: H2oPlan | null | undefined,
  inventory: H2oInventory | null | undefined,
  options: H2oUploadRuntimeOptions
): Promise<readonly H2oTensor[]> {
  const parameters = prepareParameters(owner, plan, inventory, options);
  owner!.phase = 'loading';
  const weights: H2oTensor[] = [];
  try {
    for (const entry of parameters) {
      weights.push(await uploadEntry(owner!, entry, inventory!, options as H2oValidatedUploadRuntimeOptions));
    }
    options.signal?.throwIfAborted();
    assertUsable(owner!);
    if (owner!.lifecycle.loss || owner!.lifecycle.disposed) throw new Error('Device lost during upload');
    owner!.uploadedWeights = Object.freeze(weights);
    owner!.phase = 'prepared';
    return owner!.uploadedWeights;
  } catch (error) {
    owner!.phase = 'failed';
    // An upload may have submitted GPU work. Conservatively retain the entire
    // root scope/runtime on every failed load, even when no pending work is proved.
    owner!.disposalGuard.retain();
    throw error;
  }
}

export async function uploadH2oParameters({
  owner,
  plan,
  inventory,
  base,
  readRange,
  makeHasher,
  signal
}: H2oUploadOptions = {}): Promise<readonly H2oTensor[]> {
  return upload(owner, plan, inventory, {base, readRange, makeHasher, signal});
}

/** Tests inject fake TVM and range reader; the fake marker is not a security boundary. */
type FakeH2oUploadOptions = Omit<H2oUploadOptions, 'base'> & {
  fake: unknown;
  base?: string | URL | undefined;
};

export function uploadFakeH2oParameters({
  fake,
  owner,
  plan,
  inventory,
  base = 'https://example.invalid/',
  readRange,
  makeHasher,
  signal
}: FakeH2oUploadOptions): Promise<readonly H2oTensor[]> {
  if (fake !== true || typeof readRange !== 'function') throw new TypeError('Fake adapters required');
  return upload(owner, plan, inventory, {base, readRange, makeHasher, signal});
}
