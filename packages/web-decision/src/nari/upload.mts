// Ordered verified range uploader. The caller must supply a pinned source and guarded runtime owner.
import {aligned, bytesFor, prepareUpload} from './upload-plan.mjs';
import {waitForAbort} from '../abort.js';
type UploadPlan = Parameters<typeof prepareUpload>[0];
type UploadInventory = Parameters<typeof prepareUpload>[1];
type UploadRotary = Parameters<typeof prepareUpload>[2];
type UploadFile = UploadPlan['letterFile'];
interface UploadOwner {
  phase: string;
  lost: unknown;
  retain(): unknown;
  hold(): {release(): unknown};
}
interface UploadDiagnostic {
  phase(value: string): void;
  upload(phase: string, tensorIndex: number, rangeIndex?: number | null): void;
  rangeComplete(): void;
  tensorComplete(): void;
}
interface UploadOptions<T> {
  owner: UploadOwner;
  plan: UploadPlan;
  inventory: UploadInventory;
  rotary: UploadRotary;
  readRange(path: string, offset: number, length: number, options: {signal: AbortSignal | undefined}): Promise<unknown>;
  sha256(data: Uint8Array): string | PromiseLike<string>;
  makeHasher(): {update(data: Uint8Array): unknown; digest(): string | PromiseLike<string>};
  allocate(entry: {shape: number[]; dtype: string}, size: number): {tensor: T; capacityBytes: number};
  writeAndSettle(tensor: T, offset: number, bytes: Uint8Array): PromiseLike<unknown>;
  beforeUpload?: () => PromiseLike<unknown> | unknown;
  signal?: AbortSignal | undefined;
  diagnostic?: UploadDiagnostic | undefined;
  expectedBindings?: number | undefined;
}

const fail: () => never = () => {
  throw new Error('invalid Nari upload');
};
const aborted = (signal: AbortSignal | undefined) => signal?.throwIfAborted();
const safe = (n: number) => Number.isSafeInteger(n) && n >= 0;

export async function uploadParameters<T>({
  owner,
  plan,
  inventory,
  rotary,
  readRange,
  sha256,
  makeHasher,
  allocate,
  writeAndSettle,
  beforeUpload = async () => {},
  signal,
  diagnostic,
  expectedBindings = 730
}: UploadOptions<T>) {
  if (
    !owner ||
    owner.phase !== 'prepared' ||
    typeof owner.retain !== 'function' ||
    typeof owner.hold !== 'function' ||
    typeof readRange !== 'function' ||
    typeof sha256 !== 'function' ||
    typeof makeHasher !== 'function' ||
    typeof allocate !== 'function' ||
    typeof writeAndSettle !== 'function' ||
    typeof beforeUpload !== 'function'
  )
    fail();
  const parameters: {target: string; graphArgument: string; tensor: T; fileSha256?: string; source?: string}[] = [];
  try {
    diagnostic?.phase('upload-validate');
    aborted(signal);
    const {plan: p, byFile, rotaryBytes} = prepareUpload(plan, inventory, rotary, expectedBindings);
    const check = () => {
      aborted(signal);
      if (owner.lost || owner.phase !== 'uploading') throw new Error('owner unavailable');
    };
    const lease = owner.hold();
    if (!lease || typeof lease.release !== 'function') fail();
    owner.phase = 'uploading';
    async function tensor(
      entry: {shape: number[]; dtype: string},
      file: UploadFile | null | undefined,
      constant: Uint8Array | null | undefined,
      tensorIndex: number
    ) {
      diagnostic?.upload('upload-guard', tensorIndex);
      check();
      await beforeUpload();
      check();
      const size = bytesFor(entry);
      diagnostic?.upload('upload-allocate', tensorIndex);
      const allocated = allocate(entry, aligned(size));
      if (!allocated || !allocated.tensor || !safe(allocated.capacityBytes) || allocated.capacityBytes < aligned(size))
        fail();
      async function write(offset: number, bytes: Uint8Array, rangeIndex: number | null = null) {
        diagnostic?.upload('upload-guard', tensorIndex, rangeIndex);
        check();
        await beforeUpload();
        check();
        if (offset % 4 !== 0) fail();
        let padded = bytes;
        if (bytes.byteLength % 4 !== 0) {
          padded = new Uint8Array(aligned(bytes.byteLength));
          padded.set(bytes);
        }
        if (offset + padded.byteLength > allocated.capacityBytes) fail();
        // The runtime adapter must resolve only after submitted work has settled.
        diagnostic?.upload('gpu-write', tensorIndex, rangeIndex);
        await writeAndSettle(allocated.tensor, offset, padded);
        check();
      }
      if (constant) {
        await write(0, constant);
      } else {
        const hasher = makeHasher();
        if (!hasher || typeof hasher.update !== 'function' || typeof hasher.digest !== 'function') fail();
        let rangeIndex = 0;
        for (const range of byFile.get(file!.path)!.chunks) {
          diagnostic?.upload('range-read', tensorIndex, rangeIndex);
          check();
          const received = await readRange(file!.path, range.offset, range.bytes, {signal});
          check();
          if (!(received instanceof Uint8Array) || received.byteLength !== range.bytes) fail();
          const data = Uint8Array.from(received);
          diagnostic?.upload('range-hash', tensorIndex, rangeIndex);
          const hash = await waitForAbort(() => sha256(data), signal);
          check();
          if (hash !== range.sha256) throw new Error('range hash mismatch');
          hasher.update(data);
          await write(range.offset, data, rangeIndex);
          diagnostic?.rangeComplete();
          rangeIndex++;
        }
        diagnostic?.upload('file-hash', tensorIndex);
        const hash = await waitForAbort(() => hasher.digest(), signal);
        check();
        if (hash !== file!.sha256) throw new Error('whole file hash mismatch');
      }
      check();
      diagnostic?.upload('tensor-complete', tensorIndex);
      diagnostic?.tensorComplete();
      return allocated.tensor;
    }
    const fileCount = expectedBindings - 2;
    for (let i = 0; i < expectedBindings; i++) {
      const entry = p.ordered[i]!,
        t = await tensor(
          entry,
          i < fileCount ? entry.file : null,
          i >= fileCount ? rotaryBytes[i - fileCount] : null,
          i
        );
      parameters.push({
        target: entry.target,
        graphArgument: entry.graphArgument,
        tensor: t,
        ...(i < fileCount ? {fileSha256: entry.file!.sha256} : {source: 'pinned-config-rotary'})
      });
    }
    const letterRows = await tensor({shape: [2, 26, 2560], dtype: 'float32'}, p.letterFile, null, expectedBindings);
    diagnostic?.phase('upload-finalize');
    check();
    await lease.release();
    check();
    owner.phase = 'prepared';
    return {parameters, letterRows, letterSha256: p.letterFile.sha256, signed: false};
  } catch (error) {
    owner.phase = 'failed';
    try {
      owner.retain();
    } catch (retainError) {
      throw new AggregateError([error, retainError], 'upload and retain failed', {cause: retainError});
    }
    throw error;
  }
}
