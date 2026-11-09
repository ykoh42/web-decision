// Experimental browser source for the pinned H2O asset contract.
import {WebDecisionError} from '../kev.js';
import {waitForAbort} from '../abort.js';
import {readVerifiedRange} from '../strands/range-reader.mjs';
import {parseStrictJson} from './strict-json.mjs';
import {
  preparePublicCandidatePlan,
  PUBLIC_METADATA_SHA256,
  H2O_BASELINE_MODEL_ID,
  h2oCandidateSpec,
  h2oGraphIdentity
} from './plan.mjs';
import {createIncrementalSha256, MAX_CHUNK_BYTES, MAX_FILE_BYTES} from './hasher.mjs';

type RangeSpec = {path: string; totalBytes: number; offset: number; length: number; sha256: string};
type RangeChunk = Pick<RangeSpec, 'offset' | 'length' | 'sha256'>;
type FileDescriptor = {path: string; bytes: number; sha256: string};
type InventoryFile = FileDescriptor & {chunks: RangeChunk[]};
type ParsedInventory = {
  format: string;
  manifestSha256: string;
  auditReceiptSha256: string;
  files: Array<InventoryFile | null | undefined>;
};
type FileParameter = {kind: 'file'; path: string; bytes: number; sha256: string};
type RotaryParameter = {kind: 'rotary'; target: string};
type H2oPlan = {parameters: Array<FileParameter | RotaryParameter>};
type PreparePlan = (
  metadataBytes: Uint8Array,
  manifestBytes: Uint8Array,
  model: string,
  candidateBytes: Uint8Array | undefined,
  signal: AbortSignal | undefined
) => Promise<H2oPlan>;
type IncrementalHasher = {update(bytes: unknown): void; digest(): string | Promise<string>};
type RangeResponse = {bytes?: unknown} | null | undefined;
export type H2oHashConstructor = new () => {
  update(bytes: Uint8Array): unknown;
  digest(): unknown;
  clean(): unknown;
};
type RangeReader = (
  base: URL,
  spec: RangeSpec,
  options: {cache: false; offline: false; signal: AbortSignal}
) => Promise<RangeResponse>;
type H2oSourcePolicy = {
  manifestSha256: string;
  stage: FileDescriptor;
  manifest: FileDescriptor;
  wasm: FileDescriptor;
  tokenizer: FileDescriptor;
  tokenizerConfig: FileDescriptor;
  tokenizerContract: FileDescriptor;
  candidateMetadata?: FileDescriptor;
  weightCount: number;
  preparePlan: (
    stage: Uint8Array,
    manifest: Uint8Array,
    candidateMetadata: Uint8Array | undefined,
    signal: AbortSignal | undefined
  ) => H2oPlan | Promise<H2oPlan>;
};
type OpenOptions = {
  inventoryBytes: unknown;
  inventorySha256: unknown;
  auditReceiptSha256: unknown;
  signal?: AbortSignal | undefined;
};
export type H2oOpenedSource = {
  stage: Uint8Array;
  manifest: Uint8Array;
  candidateMetadata?: Uint8Array;
  wasm: Uint8Array;
  tokenizer: Uint8Array;
  tokenizerConfig: Uint8Array;
  tokenizerContract: Uint8Array;
  inventory: Record<string, RangeChunk[]>;
  readRange(
    requestBase: unknown,
    spec: RangeSpec,
    options?: {cache?: false; offline?: false; signal?: AbortSignal | null | undefined}
  ): Promise<{bytes: ArrayBuffer; cachedByHash: false}>;
  close(): void;
};
type H2oModel = string;
type H2oFactoryOptions = {inventoryBytes?: Uint8Array; HashClass?: H2oHashConstructor; model?: H2oModel};
type H2oFakeSourceOptions = {
  fake?: unknown;
  base: unknown;
  inventoryBytes: unknown;
  inventorySha256: unknown;
  auditReceiptSha256: unknown;
  signal?: AbortSignal | undefined;
  policy: H2oSourcePolicy;
  reader: RangeReader;
  makeHasher: () => IncrementalHasher;
};

const hash = async (bytes: Uint8Array) =>
  [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes as BufferSource))]
    .map((x) => x.toString(16).padStart(2, '0'))
    .join('');
const bad = (message: string): WebDecisionError => new WebDecisionError('INVALID_ASSET', message);
const SHA = /^[a-f0-9]{64}$/;
const pinned = Object.freeze({
  manifestSha256: '27f83988992c9d9fe2febe924848259890bcb4d39e1c57cb225dd8cb22610e9a',
  stage: {
    path: 'h2o-stage.log',
    bytes: 198540,
    sha256: '0d0da47d26dc200934a9ee8bb57192284190a15e6886a6a83a055efeb83e90fb'
  },
  manifest: {
    path: 'manifest.pending.json',
    bytes: 260294,
    sha256: '27f83988992c9d9fe2febe924848259890bcb4d39e1c57cb225dd8cb22610e9a'
  },
  wasm: {
    path: 'model.pending.wasm',
    bytes: 4850370,
    sha256: '62fb8ddb2b98106cb24f925a585f930f87d9f6df25e32deb342aa3ae37850a50'
  },
  tokenizer: {
    path: 'tokenizer/tokenizer.json',
    bytes: 12807982,
    sha256: '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
  },
  tokenizerConfig: {
    path: 'tokenizer/tokenizer_config.json',
    bytes: 16710,
    sha256: '316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8'
  },
  tokenizerContract: {
    path: 'tokenizer_contract.json',
    bytes: 10320,
    sha256: '7eb96aaea6b7dbb495f0840b42f18014ca5a697697305f76f39d46cae890ccb2'
  }
});

function safeBase(base: unknown): URL {
  let value;
  try {
    value = new URL(base as string | URL);
  } catch {
    throw bad('Invalid asset base');
  }
  if (
    !['https:', 'http:'].includes(value.protocol) ||
    value.username ||
    value.password ||
    value.search ||
    value.hash ||
    !value.pathname.endsWith('/')
  )
    throw bad('Expected an HTTP(S) asset directory');
  return value;
}

function verifyRanges(item: InventoryFile | null | undefined, expected: FileDescriptor, align: boolean): InventoryFile {
  if (
    !item ||
    item.path !== expected.path ||
    item.bytes !== expected.bytes ||
    item.sha256 !== expected.sha256 ||
    !Array.isArray(item.chunks) ||
    !item.chunks.length ||
    !Number.isSafeInteger(item.bytes) ||
    item.bytes < 1 ||
    item.bytes > MAX_FILE_BYTES ||
    !SHA.test(item.sha256)
  )
    throw bad('Inventory file mismatch');
  let offset = 0;
  const chunks = item.chunks.map((part) => {
    if (
      !part ||
      Object.keys(part).sort().join(',') !== 'length,offset,sha256' ||
      !Number.isSafeInteger(part.offset) ||
      part.offset !== offset ||
      !Number.isSafeInteger(part.length) ||
      part.length < 1 ||
      part.length > MAX_CHUNK_BYTES ||
      part.length > item.bytes - offset ||
      typeof part.sha256 !== 'string' ||
      !SHA.test(part.sha256) ||
      (align && (part.offset % 4 || part.length % 4))
    )
      throw bad('Invalid inventory ranges');
    offset += part.length;
    return {offset: part.offset, length: part.length, sha256: part.sha256};
  });
  if (offset !== item.bytes) throw bad('Incomplete inventory ranges');
  return {path: item.path, bytes: item.bytes, sha256: item.sha256, chunks};
}

async function readInventory(
  inventoryBytes: unknown,
  inventorySha256: unknown,
  auditReceiptSha256: unknown,
  policy: H2oSourcePolicy,
  signal: AbortSignal | undefined
): Promise<Map<string, InventoryFile>> {
  if (
    !(inventoryBytes instanceof Uint8Array) ||
    inventoryBytes.length > 1_000_000 ||
    typeof inventorySha256 !== 'string' ||
    !SHA.test(inventorySha256) ||
    typeof auditReceiptSha256 !== 'string' ||
    !SHA.test(auditReceiptSha256)
  )
    throw bad('Missing pinned inventory evidence');
  const invBytes = Uint8Array.from(inventoryBytes);
  signal?.throwIfAborted();
  const inventoryHash = await waitForAbort(() => hash(invBytes), signal);
  signal?.throwIfAborted();
  if (inventoryHash !== inventorySha256) throw bad('Inventory hash mismatch');
  let inventory;
  try {
    inventory = parseStrictJson(invBytes) as ParsedInventory;
  } catch {
    throw bad('Invalid inventory JSON');
  }
  if (
    inventory.format !== 'h2o-range-inventory-v1' ||
    inventory.manifestSha256 !== policy.manifestSha256 ||
    inventory.auditReceiptSha256 !== auditReceiptSha256 ||
    !Array.isArray(inventory.files)
  )
    throw bad('Inventory evidence mismatch');
  const declared = new Map<string, InventoryFile>();
  for (const file of inventory.files) {
    if (!file || typeof file.path !== 'string' || declared.has(file.path)) throw bad('Duplicate inventory path');
    declared.set(file.path, file);
  }
  return declared;
}

async function open(
  baseValue: unknown,
  {inventoryBytes, inventorySha256, auditReceiptSha256, signal}: OpenOptions,
  policy: H2oSourcePolicy,
  reader: RangeReader,
  makeHasher: () => IncrementalHasher
): Promise<H2oOpenedSource> {
  const base = safeBase(baseValue);
  const declared = await readInventory(inventoryBytes, inventorySha256, auditReceiptSha256, policy, signal);
  const controller = new AbortController();
  const forward = () => controller.abort(signal!.reason);
  signal?.addEventListener('abort', forward, {once: true});
  if (signal?.aborted) forward();
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    signal?.removeEventListener('abort', forward);
    controller.abort(new DOMException('Asset source closed', 'AbortError'));
  };
  const read = async (spec: RangeSpec): Promise<Uint8Array> => {
    controller.signal.throwIfAborted();
    const result = await waitForAbort(
      () => reader(base, spec, {cache: false, offline: false, signal: controller.signal}),
      controller.signal
    );
    controller.signal.throwIfAborted();
    if (!(result?.bytes instanceof ArrayBuffer) || result.bytes.byteLength !== spec.length)
      throw bad('Range response length mismatch');
    const bytes = Uint8Array.from(new Uint8Array(result.bytes));
    const digest = await waitForAbort(() => hash(bytes), controller.signal);
    controller.signal.throwIfAborted();
    if (digest !== spec.sha256) throw bad('Range response hash mismatch');
    return bytes;
  };
  const small = async (descriptor: FileDescriptor): Promise<Uint8Array> => {
    if (descriptor.bytes > MAX_CHUNK_BYTES) throw bad('Small pinned file too large');
    return read({
      path: descriptor.path,
      totalBytes: descriptor.bytes,
      offset: 0,
      length: descriptor.bytes,
      sha256: descriptor.sha256
    });
  };
  try {
    const stage = await small(policy.stage),
      manifest = await small(policy.manifest);
    const candidateMetadata = policy.candidateMetadata ? await small(policy.candidateMetadata) : undefined;
    const plan = await waitForAbort(
      () => policy.preparePlan(stage, manifest, candidateMetadata, controller.signal),
      controller.signal
    );
    controller.signal.throwIfAborted();
    const weightFiles = plan.parameters.filter((entry) => entry.kind === 'file');
    if (weightFiles.length !== policy.weightCount || declared.size !== policy.weightCount + 1)
      throw bad('Inventory file count mismatch');
    const allowed = new Map<string, InventoryFile>();
    for (const file of weightFiles) {
      if (allowed.has(file.path)) throw bad('Duplicate ordered weight path');
      allowed.set(file.path, verifyRanges(declared.get(file.path), file, true));
    }
    const token = verifyRanges(declared.get(policy.tokenizer.path), policy.tokenizer, false);
    if (allowed.has(token.path)) throw bad('Duplicate tokenizer path');
    const tokenizer = new Uint8Array(token.bytes),
      fullHash = makeHasher();
    for (const part of token.chunks) {
      const data = await read({path: token.path, totalBytes: token.bytes, ...part});
      fullHash.update(data);
      tokenizer.set(data, part.offset);
    }
    if ((await fullHash.digest()) !== token.sha256) throw bad('Tokenizer full hash mismatch');
    const wasm = await small(policy.wasm),
      tokenizerConfig = await small(policy.tokenizerConfig),
      tokenizerContract = await small(policy.tokenizerContract);
    controller.signal.throwIfAborted();
    const inventoryByPath: Record<string, RangeChunk[]> = Object.create(null);
    for (const [path, value] of allowed) inventoryByPath[path] = value.chunks.map((part) => ({...part}));
    return {
      stage,
      manifest,
      ...(candidateMetadata === undefined ? {} : {candidateMetadata}),
      wasm,
      tokenizer,
      tokenizerConfig,
      tokenizerContract,
      inventory: inventoryByPath,
      async readRange(
        requestBase: unknown,
        spec: unknown,
        options: unknown = {}
      ): Promise<{
        bytes: ArrayBuffer;
        cachedByHash: false;
      }> {
        if (closed) throw bad('Asset source is closed');
        if (
          !options ||
          typeof options !== 'object' ||
          Array.isArray(options) ||
          ![null, Object.prototype].includes(Object.getPrototypeOf(options))
        )
          throw bad('Invalid range source options');
        const values = Object.create(null);
        for (const key of Reflect.ownKeys(options)) {
          const descriptor = Object.getOwnPropertyDescriptor(options, key);
          if (
            typeof key !== 'string' ||
            !['cache', 'offline', 'signal'].includes(key) ||
            !descriptor ||
            !Object.hasOwn(descriptor, 'value')
          )
            throw bad('Invalid range source options');
          values[key] = descriptor.value;
        }
        if (
          String(safeBase(requestBase)) !== String(base) ||
          (values.cache !== undefined && values.cache !== false) ||
          (values.offline !== undefined && values.offline !== false)
        )
          throw bad('Unsupported range source options');
        if (
          !spec ||
          typeof spec !== 'object' ||
          Array.isArray(spec) ||
          Object.keys(spec).sort().join(',') !== 'length,offset,path,sha256,totalBytes'
        )
          throw bad('Invalid range request');
        const file = allowed.get((spec as RangeSpec | null | undefined)?.path as string);
        const match = file?.chunks.find(
          (part) =>
            part.offset === (spec as RangeSpec).offset &&
            part.length === (spec as RangeSpec).length &&
            part.sha256 === (spec as RangeSpec).sha256
        );
        if (!file || !match || (spec as RangeSpec).totalBytes !== file.bytes)
          throw bad('Range is not pinned by inventory');
        const operationSignal = values.signal as AbortSignal | null | undefined;
        operationSignal?.throwIfAborted();
        // Forward caller cancellation without mutating the source-wide signal.
        const local = new AbortController();
        const cancel = (event: Event) => local.abort((event.target as AbortSignal).reason);
        controller.signal.addEventListener('abort', cancel, {once: true});
        operationSignal?.addEventListener('abort', cancel, {once: true});
        if (controller.signal.aborted) local.abort(controller.signal.reason);
        if (operationSignal?.aborted) local.abort(operationSignal.reason);
        try {
          const response = await waitForAbort(
            () =>
              reader(
                base,
                {path: file.path, totalBytes: file.bytes, ...match},
                {cache: false, offline: false, signal: local.signal}
              ),
            local.signal
          );
          local.signal.throwIfAborted();
          if (!(response?.bytes instanceof ArrayBuffer) || response.bytes.byteLength !== match.length)
            throw bad('Range response length mismatch');
          const bytes = Uint8Array.from(new Uint8Array(response.bytes));
          const digest = await waitForAbort(() => hash(bytes), local.signal);
          local.signal.throwIfAborted();
          if (digest !== match.sha256) throw bad('Range response hash mismatch');
          return {bytes: bytes.buffer, cachedByHash: false};
        } finally {
          controller.signal.removeEventListener('abort', cancel);
          operationSignal?.removeEventListener('abort', cancel);
        }
      },
      close
    };
  } catch (error) {
    close();
    signal?.throwIfAborted();
    throw error;
  }
}

/** Separate path-free projection; its pins are unsigned consistency evidence. */
export function createH2oPublicSourceFactory({
  inventoryBytes,
  HashClass,
  model = H2O_BASELINE_MODEL_ID
}: H2oFactoryOptions = {}): (options: {
  base: string | URL;
  signal?: AbortSignal | undefined;
}) => Promise<H2oOpenedSource> {
  h2oGraphIdentity(model);
  const candidate = model === H2O_BASELINE_MODEL_ID ? undefined : h2oCandidateSpec(model);
  const policy: H2oSourcePolicy = {
    ...pinned,
    weightCount: 680,
    preparePlan: (stage, manifest, candidate, signal) =>
      (preparePublicCandidatePlan as PreparePlan)(stage, manifest, model, candidate, signal),
    stage: {path: 'h2o-public-metadata.json', bytes: 178642, sha256: PUBLIC_METADATA_SHA256},
    ...(candidate
      ? {
          wasm: {path: candidate.graphPath, bytes: candidate.graphBytes, sha256: candidate.graphSha256},
          candidateMetadata: {
            path: candidate.metadataPath,
            bytes: candidate.metadataBytes,
            sha256: candidate.metadataSha256
          }
        }
      : {})
  };
  return ({base, signal}: {base: string | URL; signal?: AbortSignal | undefined}) =>
    open(
      base,
      {
        inventoryBytes,
        inventorySha256: '48080bff4785648f5b49fabde8ab84b341e0b4f2a6d20151f4c6a8051dab423d',
        auditReceiptSha256: '100e7cb766d5d50f3ae503ade53dd751a9c3253cc44d02f2a06cb42026b33b41',
        signal
      },
      policy,
      readVerifiedRange,
      () => createIncrementalSha256(HashClass)
    );
}

/** Fixture-only policy seam; no network when the supplied reader is fake. */
export function openFakeH2oSource({
  fake,
  base,
  inventoryBytes,
  inventorySha256,
  auditReceiptSha256,
  signal,
  policy,
  reader,
  makeHasher
}: H2oFakeSourceOptions): Promise<H2oOpenedSource> {
  if (fake !== true || typeof reader !== 'function' || typeof makeHasher !== 'function')
    throw new TypeError('Fake source dependencies required');
  return open(base, {inventoryBytes, inventorySha256, auditReceiptSha256, signal}, policy, reader, makeHasher);
}
