interface FileDescriptor {
  path: string;
  bytes: number;
  sha256: string;
}
interface ChunkDescriptor {
  offset: number;
  bytes: number;
  sha256: string;
}
interface InventoryRow extends FileDescriptor {
  chunks: ChunkDescriptor[];
}
interface UploadEntry {
  target: string;
  graphArgument: string;
  source: string;
  dtype: string;
  shape: number[];
  file: FileDescriptor;
}
interface UploadPlan {
  ordered: UploadEntry[];
  letterFile: FileDescriptor;
}

const MAX = 4 * 1024 * 1024;
const ROTARY = ['body.rotary_emb.inv_freq', 'body.rotary_emb.original_inv_freq'];
const SHA = /^[a-f0-9]{64}$/;
const safe = (value: number) => Number.isSafeInteger(value) && value >= 0;
const dense = (array: unknown, length: number) =>
  Array.isArray(array) &&
  array.length === length &&
  Array.from({length}, (_, i) => Object.hasOwn(array, i)).every(Boolean);
const invalid: () => never = () => {
  throw Error('invalid Nari upload');
};
export const aligned = (value: number) => Math.ceil(value / 4) * 4;

export function bytesFor(entry: {dtype: string; shape: number[]}) {
  const width = ({float16: 2, float32: 4, int32: 4, uint32: 4} as Record<string, number>)[entry.dtype];
  if (
    !width ||
    !Array.isArray(entry.shape) ||
    !entry.shape.length ||
    !dense(entry.shape, entry.shape.length) ||
    entry.shape.some((n) => !safe(n) || n === 0)
  )
    invalid();
  const count = entry.shape.reduce((a, b) => a * b, 1),
    bytes = count * width;
  if (!safe(count) || !safe(bytes) || !safe(aligned(bytes))) invalid();
  return bytes;
}

function validateDescriptor(file: FileDescriptor, seen: Set<string>) {
  if (
    !file ||
    typeof file.path !== 'string' ||
    !/^[A-Za-z0-9_.-]+\.pending$/.test(file.path) ||
    !safe(file.bytes) ||
    file.bytes === 0 ||
    typeof file.sha256 !== 'string' ||
    !SHA.test(file.sha256) ||
    seen.has(file.path)
  )
    invalid();
  seen.add(file.path);
}

function validateEntries(plan: UploadPlan, bindings: number) {
  if (!plan || !dense(plan.ordered, bindings) || !plan.letterFile) invalid();
  const fileCount = bindings - 2;
  const targets = new Set<string>(),
    argumentsSeen = new Set<string>(),
    files = new Set<string>();
  for (let i = 0; i < bindings; i++) {
    const entry = plan.ordered[i];
    if (
      !entry ||
      typeof entry.target !== 'string' ||
      !entry.target ||
      typeof entry.graphArgument !== 'string' ||
      !entry.graphArgument ||
      targets.has(entry.target) ||
      argumentsSeen.has(entry.graphArgument)
    )
      invalid();
    targets.add(entry.target);
    argumentsSeen.add(entry.graphArgument);
    const size = bytesFor(entry);
    if (i < fileCount) {
      if (entry.source !== 'planned-asset') invalid();
      validateDescriptor(entry.file, files);
      if (entry.file.bytes !== size) invalid();
    } else if (
      entry.source !== 'pinned-config-rotary' ||
      entry.target !== ROTARY[i - fileCount] ||
      entry.dtype !== 'float32' ||
      entry.shape.length !== 1 ||
      entry.shape[0] !== 32 ||
      Object.hasOwn(entry, 'file')
    )
      invalid();
  }
  validateDescriptor(plan.letterFile, files);
  if (plan.letterFile.bytes !== 52 * 2560 * 4) invalid();
  return [...plan.ordered.slice(0, fileCount).map((entry) => entry.file), plan.letterFile];
}

function validateRanges(row: InventoryRow, file: FileDescriptor) {
  if (
    !file ||
    row.bytes !== file.bytes ||
    row.sha256 !== file.sha256 ||
    !Array.isArray(row.chunks) ||
    !row.chunks.length
  )
    invalid();
  let position = 0;
  for (const part of row.chunks) {
    if (
      !part ||
      !safe(part.offset) ||
      part.offset !== position ||
      !safe(part.bytes) ||
      part.bytes === 0 ||
      part.bytes > MAX ||
      typeof part.sha256 !== 'string' ||
      !SHA.test(part.sha256) ||
      part.offset % 4 !== 0
    )
      invalid();
    position += part.bytes;
    if (!safe(position) || position > file.bytes || (position < file.bytes && part.bytes % 4 !== 0)) invalid();
  }
  if (position !== file.bytes) invalid();
}

function validateInventory(inventory: InventoryRow[], expected: FileDescriptor[], bindings: number) {
  if (!dense(inventory, bindings - 1)) invalid();
  const expectedByPath = new Map(expected.map((file) => [file.path, file])),
    byFile = new Map();
  for (const row of inventory) {
    const file = expectedByPath.get(row?.path);
    if (!file || byFile.has(file.path)) invalid();
    validateRanges(row, file);
    byFile.set(file.path, row);
  }
  if (byFile.size !== expectedByPath.size) invalid();
  return byFile;
}

function snapshotRotary(rotary: Record<string, Float32Array>) {
  if (
    !rotary ||
    !ROTARY.every(
      (key) =>
        rotary[key] instanceof Float32Array &&
        rotary[key]!.length === 32 &&
        Array.from(rotary[key]!).every(Number.isFinite)
    ) ||
    !rotary[ROTARY[0]!]!.every((value, i) => Object.is(value, rotary[ROTARY[1]!]![i]))
  )
    invalid();
  return ROTARY.map((key) => {
    const bytes = new Uint8Array(128),
      view = new DataView(bytes.buffer);
    for (let i = 0; i < 32; i++) view.setFloat32(i * 4, rotary[key]![i]!, true);
    return bytes;
  });
}

export function prepareUpload(
  plan: UploadPlan,
  inventory: InventoryRow[],
  rotary: Record<string, Float32Array>,
  bindings = 730
) {
  if (bindings !== 730 && bindings !== 606) invalid();
  const snapshot = structuredClone(plan),
    rows = structuredClone(inventory);
  const byFile = validateInventory(rows, validateEntries(snapshot, bindings), bindings);
  return {plan: snapshot, byFile, rotaryBytes: snapshotRotary(rotary)};
}
