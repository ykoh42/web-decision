// Ordered verified range uploader. The caller must supply a pinned source and guarded runtime owner.
import {aligned, bytesFor, prepareUpload} from './upload-plan.mjs';
import {waitForAbort} from '../abort.js';
const fail = () => {
  throw new Error('invalid Nari upload');
};
const aborted = (signal) => signal?.throwIfAborted();
const safe = (n) => Number.isSafeInteger(n) && n >= 0;

export async function uploadParameters({
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
}) {
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
  const parameters = [];
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
    async function tensor(entry, file, constant, tensorIndex) {
      diagnostic?.upload('upload-guard', tensorIndex);
      check();
      await beforeUpload();
      check();
      const size = bytesFor(entry);
      diagnostic?.upload('upload-allocate', tensorIndex);
      const allocated = allocate(entry, aligned(size));
      if (!allocated || !allocated.tensor || !safe(allocated.capacityBytes) || allocated.capacityBytes < aligned(size))
        fail();
      async function write(offset, bytes, rangeIndex = null) {
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
        for (const range of byFile.get(file.path).chunks) {
          diagnostic?.upload('range-read', tensorIndex, rangeIndex);
          check();
          const received = await readRange(file.path, range.offset, range.bytes, {signal});
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
        if (hash !== file.sha256) throw new Error('whole file hash mismatch');
      }
      check();
      diagnostic?.upload('tensor-complete', tensorIndex);
      diagnostic?.tensorComplete();
      return allocated.tensor;
    }
    const fileCount = expectedBindings - 2;
    for (let i = 0; i < expectedBindings; i++) {
      const entry = p.ordered[i],
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
        ...(i < fileCount ? {fileSha256: entry.file.sha256} : {source: 'pinned-config-rotary'})
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
