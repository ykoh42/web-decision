import {createIncrementalSha256} from '../h2o/hasher.mjs';
import {waitForAbort} from '../abort.js';
import {bytesFor, aligned} from '../nari/upload-plan.mjs';
import {withNariScope} from '../nari/scope.mjs';

interface ImajevChunk {
  offset: number;
  bytes: number;
  sha256: string;
}
interface ImajevRowBase {
  dtype: string;
  shape: number[];
}
interface ImajevFileRow extends ImajevRowBase {
  path: string;
  offset: number;
  bytes: number;
  chunks: ImajevChunk[];
  rangeSha256: string;
  sha256?: never;
  bytesHex?: never;
}
interface ImajevRotaryRow extends ImajevRowBase {
  path?: undefined;
  sha256: string;
  bytesHex: string;
}
type ImajevRow = ImajevFileRow | ImajevRotaryRow;
interface UploadOwner {
  phase: string;
  lost: unknown;
  blocked: unknown;
  uncaptured: unknown[];
  hold(): {release(): unknown};
  retain(): unknown;
}
interface ImajevTensor {
  getDataPtr(): unknown;
  byteOffset: number;
}
interface ImajevRuntimeOwner extends UploadOwner {
  runtime: {
    lib?: {
      webGPUContext?: {
        copyRawBytesToBuffer(bytes: Uint8Array, pointer: unknown, offset: number, length: number): unknown;
        gpuBufferFromPtr(pointer: unknown): {size: number};
      };
    };
    empty(shape: number[], dtype: string, device: unknown): ImajevTensor;
  };
  device: unknown;
  resources: ImajevTensor[];
}
type ReadRange = (
  path: string,
  offset: number,
  length: number,
  options: {signal: AbortSignal | undefined}
) => Promise<unknown>;
interface Hasher {
  update(bytes: Uint8Array): unknown;
  digest(): string;
}
interface UploadParams<T> {
  owner: UploadOwner;
  rows: ImajevRow[];
  readRange: ReadRange;
  makeHasher(): Hasher;
  allocate(row: ImajevRow, needed: number): {tensor: T; capacityBytes: number};
  writeAndSettle(tensor: T, offset: number, bytes: Uint8Array): Promise<unknown>;
  signal?: AbortSignal | undefined;
}
interface UploadSource {
  plan: {bindings: ImajevRow[]};
  readRange: ReadRange;
}

const sha = async (bytes: Uint8Array<ArrayBuffer>) =>
  [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
const invalid: () => never = () => {
  throw Error('Invalid Imajev upload.');
};
const unhex = (value: string) => Uint8Array.from(value.match(/../g)!, (byte) => Number.parseInt(byte, 16));

function checkOwner(owner: UploadOwner, signal: AbortSignal | undefined) {
  signal?.throwIfAborted();
  if (owner.phase !== 'uploading' || owner.lost || owner.blocked || owner.uncaptured.length) invalid();
}

function rowChunks(row: ImajevFileRow, size: number) {
  if (!Array.isArray(row.chunks) || !row.chunks.length || row.bytes !== size) invalid();
  let position = row.offset;
  for (const chunk of row.chunks) {
    if (
      !chunk ||
      chunk.offset !== position ||
      !Number.isSafeInteger(chunk.offset) ||
      chunk.offset % 4 !== 0 ||
      !Number.isSafeInteger(chunk.bytes) ||
      chunk.bytes < 1 ||
      chunk.bytes > 4 * 1024 * 1024 ||
      !/^[a-f0-9]{64}$/.test(chunk.sha256)
    )
      invalid();
    position += chunk.bytes;
    if (!Number.isSafeInteger(position) || position > row.offset + size) invalid();
  }
  if (position !== row.offset + size) invalid();
  return row.chunks;
}

async function readChunk(
  row: ImajevRow,
  chunk: ImajevChunk,
  size: number,
  readRange: ReadRange,
  signal: AbortSignal | undefined
) {
  let bytes;
  if (row.path) {
    const value = await readRange(row.path, chunk.offset, chunk.bytes, {signal});
    if (!(value instanceof Uint8Array) || value.byteLength !== chunk.bytes) invalid();
    bytes = Uint8Array.from(value);
  } else {
    if (typeof row.bytesHex !== 'string' || !/^[a-f0-9]+$/.test(row.bytesHex) || row.bytesHex.length !== size * 2)
      invalid();
    bytes = unhex(row.bytesHex);
  }
  signal?.throwIfAborted();
  const hash = await waitForAbort(() => sha(bytes), signal);
  signal?.throwIfAborted();
  if (hash !== chunk.sha256) throw Error('Imajev chunk hash mismatch.');
  return bytes;
}

async function writeRow<T>({
  owner,
  row,
  readRange,
  makeHasher,
  allocate,
  writeAndSettle,
  signal
}: Omit<UploadParams<T>, 'rows'> & {row: ImajevRow}) {
  checkOwner(owner, signal);
  const size = bytesFor(row);
  const allocation = allocate(row, aligned(size));
  if (
    !allocation?.tensor ||
    !Number.isSafeInteger(allocation.capacityBytes) ||
    allocation.capacityBytes < aligned(size)
  )
    invalid();
  const hasher = makeHasher();
  if (!hasher || typeof hasher.update !== 'function' || typeof hasher.digest !== 'function') invalid();
  const chunks = row.path ? rowChunks(row, size) : [{offset: 0, bytes: size, sha256: row.sha256!}];
  for (const chunk of chunks) {
    checkOwner(owner, signal);
    const bytes = await readChunk(row, chunk, size, readRange, signal);
    checkOwner(owner, signal);
    hasher.update(bytes);
    const offset = row.path ? chunk.offset - row.offset : 0;
    const padded = new Uint8Array(aligned(bytes.byteLength));
    padded.set(bytes);
    if (offset < 0 || offset % 4 || offset + padded.byteLength > allocation.capacityBytes) invalid();
    await writeAndSettle(allocation.tensor, offset, padded);
    checkOwner(owner, signal);
  }
  if (hasher.digest() !== (row.path ? row.rangeSha256 : row.sha256)) throw Error('Imajev tensor hash mismatch.');
  checkOwner(owner, signal);
  return allocation.tensor;
}

/** Upload a previously validated snapshot. The injected I/O also supports isolated fake tests. */
export async function uploadImajevRows<T>({
  owner,
  rows,
  readRange,
  makeHasher,
  allocate,
  writeAndSettle,
  signal
}: UploadParams<T>) {
  if (
    !owner ||
    owner.phase !== 'prepared' ||
    !Array.isArray(owner.uncaptured) ||
    typeof owner.hold !== 'function' ||
    typeof owner.retain !== 'function' ||
    !Array.isArray(rows) ||
    !rows.length ||
    typeof readRange !== 'function' ||
    typeof makeHasher !== 'function' ||
    typeof allocate !== 'function' ||
    typeof writeAndSettle !== 'function'
  )
    invalid();
  const snapshot = structuredClone(rows),
    weights: T[] = [];
  try {
    signal?.throwIfAborted();
    const lease = owner.hold();
    if (!lease || typeof lease.release !== 'function') invalid();
    owner.phase = 'uploading';
    for (const row of snapshot)
      weights.push(await writeRow({owner, row, readRange, makeHasher, allocate, writeAndSettle, signal}));
    checkOwner(owner, signal);
    await lease.release();
    checkOwner(owner, signal);
    owner.phase = 'prepared';
    return weights;
  } catch (error) {
    owner.phase = 'failed';
    try {
      owner.retain();
    } catch (cause) {
      throw new AggregateError([error, cause], 'Imajev upload and retention failed.', {cause});
    }
    throw error;
  }
}

/** Allocate with TVM and wait for each submitted WebGPU write before advancing. */
export function uploadImajevFromSource(
  owner: ImajevRuntimeOwner,
  source: UploadSource,
  {
    HashClass,
    signal
  }: {HashClass?: Parameters<typeof createIncrementalSha256>[0]; signal?: AbortSignal | undefined} = {}
) {
  if (!source?.plan || source.plan.bindings?.length !== 1082 || typeof source.readRange !== 'function') invalid();
  const context = owner.runtime.lib?.webGPUContext;
  if (typeof context?.copyRawBytesToBuffer !== 'function' || typeof context?.gpuBufferFromPtr !== 'function') invalid();
  return uploadImajevRows({
    owner,
    rows: source.plan.bindings,
    readRange: source.readRange.bind(source),
    makeHasher: () => createIncrementalSha256(HashClass),
    signal,
    allocate(row, needed) {
      const tensor = owner.runtime.empty(row.shape, row.dtype, owner.device);
      owner.resources.push(tensor);
      if (typeof tensor.getDataPtr !== 'function' || !Number.isSafeInteger(tensor.byteOffset) || tensor.byteOffset % 4)
        invalid();
      const buffer = context.gpuBufferFromPtr(tensor.getDataPtr());
      if (!Number.isSafeInteger(buffer?.size) || buffer.size < tensor.byteOffset + needed) invalid();
      return {tensor, capacityBytes: buffer.size - tensor.byteOffset};
    },
    async writeAndSettle(tensor, offset, bytes) {
      await withNariScope(
        owner,
        'Imajev parameter upload failed.',
        async (lease: {syncWithoutReadback(): Promise<unknown>}) => {
          context.copyRawBytesToBuffer(bytes, tensor.getDataPtr(), tensor.byteOffset + offset, bytes.byteLength);
          await lease.syncWithoutReadback();
        }
      );
    }
  });
}
