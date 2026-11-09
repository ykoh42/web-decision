// Metadata bridge only. It never opens weight files or runs a model.
import {WebDecisionError} from '../kev.js';

type H2oCandidateSpec = {
  graphPath: string;
  graphBytes: number;
  graphSha256: string;
  metadataPath: string;
  metadataBytes: number;
  metadataSha256: string;
  format: string;
  stageSha256: string;
  linkSha256: string;
  graphReportSha256?: string | undefined;
  conversionCounts: Readonly<Record<string, number>>;
};
type H2oStageParameter = {
  position: unknown;
  parameterIndex: unknown;
  shape: unknown;
  source: unknown;
  dtype: unknown;
  values?: unknown;
  target?: unknown;
  file?: unknown;
  bytes?: unknown;
};
type H2oOrderedCandidate = {
  userInputs: unknown;
  stateInputs: unknown;
  rotaryInputs: unknown;
  parameterInputs: H2oStageParameter[];
};
type H2oStageRecord = {revision: unknown; orderedCandidate?: H2oOrderedCandidate | null | undefined};
type H2oManifestFile = {path: string; bytes: number; sha256: string};
type H2oManifest = {
  state: unknown;
  sourceSha256: unknown;
  headSourceSha256: unknown;
  productionReady: unknown;
  graphOrderVerified: unknown;
  artifacts: unknown[];
  head: {file: unknown; labels: Iterable<string>; tokenIds: Iterable<number>};
};
type H2oCandidateGraph = Record<string, unknown>;
type H2oCandidateMetadata = {
  [key: string]: unknown;
  candidate?: H2oCandidateGraph | null | undefined;
  baseline?: Record<string, unknown> | null | undefined;
  sequenceContract?: Record<string, unknown> | null | undefined;
};
type H2oCandidateParameter =
  | {
      kind: 'file';
      target: string;
      shape: number[];
      dtype: string;
      path: string;
      bytes: number;
      sha256: string;
    }
  | {
      kind: 'rotary';
      target: string;
      shape: number[];
      dtype: string;
      values: number[];
    };
export type H2oCandidatePlan = {
  revision: string;
  wasmSha256: string;
  userInputs: string[];
  parameters: H2oCandidateParameter[];
  labels: string[];
  labelIds: number[];
  state: 'metadata-only-not-executable';
  sourceAndWeightsAudited: false;
};
export type H2oRangeSpec = {offset: number; length: number; sha256: string; path: string; totalBytes: number};

export const H2O_BASELINE_MODEL_ID = 'experimental-h2o-lightning-4b-q4f16';
export const H2O_DLIGHT_MODEL_ID = 'experimental-h2o-lightning-4b-q4f16-dlight';
export const H2O_DUAL_MODEL_ID = 'experimental-h2o-lightning-4b-q4f16-dual';
export const DLIGHT_WASM_SHA256 = '32da4bd915dec178f29828acd2a57a202df069f4c6c0f13b0d9e7c55ff782336';
const DLIGHT_METADATA_SHA256 = '91e0dc3c2500e463584ec30e0e1a27a778310a9aace313cdca67bc4dc6908b84';
export const DUAL_WASM_SHA256 = 'd4531ab4a4690b192ba205c42a4e9ca0117d09e9454ff90ee9958306f35ce9b8';
const DUAL_METADATA_SHA256 = '931e7999629ecba1c705e5f74220090417be3cd7dd7651d63a42aa87894af41e';
export const REVISION = '052bb7d48c71abcd769d91994792bbc664b7681e';
export const WASM_SHA256 = '62fb8ddb2b98106cb24f925a585f930f87d9f6df25e32deb342aa3ae37850a50';
const STAGE_SHA256 = '0d0da47d26dc200934a9ee8bb57192284190a15e6886a6a83a055efeb83e90fb';
export const PUBLIC_METADATA_SHA256 = 'e97329081dafc32b4429790f241e3ba0e2861529dba4076fad10ed2d858ee8ee';
const MANIFEST_SHA256 = '27f83988992c9d9fe2febe924848259890bcb4d39e1c57cb225dd8cb22610e9a';
const SOURCE_SHA256 = 'c9b6290d4c0c4338c37ce37425a978027473e4d45b84e038da2369b76ff8a377';
const HEAD_SHA256 = '319a8a6be2f05f0a2d1d39f66a41295f6fc97472aff71bdf7cc127fb9d013fe9';
const USER_INPUTS = ['ids', 'positions', 'full_mask', 'valid', 'slot'];
const candidates: Readonly<Record<string, H2oCandidateSpec>> = Object.freeze({
  [H2O_DLIGHT_MODEL_ID]: Object.freeze({
    graphPath: 'model.dlight.pending.wasm',
    graphBytes: 4883626,
    graphSha256: DLIGHT_WASM_SHA256,
    metadataPath: 'h2o-dlight-metadata.json',
    metadataBytes: 178971,
    metadataSha256: DLIGHT_METADATA_SHA256,
    format: 'h2o-dlight-private-provenance-v1',
    stageSha256: '401c2c6bfb45dc7050f5dff8b419d4dda416bec9c86cfc0e63ef37941fb9afc8',
    linkSha256: '325ba4ea133623cda4d9aa61e9e6ed574cb767ab13be7c846286b97a02f25402',
    conversionCounts: Object.freeze({selected: 64, fallback: 184})
  }),
  [H2O_DUAL_MODEL_ID]: Object.freeze({
    graphPath: 'model.dual.pending.wasm',
    graphBytes: 4916749,
    graphSha256: DUAL_WASM_SHA256,
    metadataPath: 'h2o-dual-metadata.json',
    metadataBytes: 179056,
    metadataSha256: DUAL_METADATA_SHA256,
    format: 'h2o-dual-private-provenance-v1',
    graphReportSha256: '98514a47043429ad4bf24990fb908ca3674cfb2e151eb9e0a1784bf1f7910bfd',
    stageSha256: '3a0c2e5d7e6c090c4106f0c32b9e280c188aac5c3133f36b523bf3836559e032',
    linkSha256: '563f8948d96cf3ad2f625f5e4ca2c48c9983c5d9097e5df8b18d597431e20544',
    conversionCounts: Object.freeze({up: 64, down: 32, fallback: 152})
  })
});
const hash = async (bytes: Uint8Array, signal: AbortSignal | null | undefined): Promise<string> => {
  signal?.throwIfAborted();
  const digest = await crypto.subtle.digest('SHA-256', bytes as BufferSource);
  signal?.throwIfAborted();
  return [...new Uint8Array(digest)].map((x) => x.toString(16).padStart(2, '0')).join('');
};
const fail = (message: string): never => {
  throw new WebDecisionError('INVALID_ASSET', message);
};

export function h2oGraphIdentity(model: string = H2O_BASELINE_MODEL_ID): {sha256: string; bytes: number} {
  if (model === H2O_BASELINE_MODEL_ID) return {sha256: WASM_SHA256, bytes: 4850370};
  const candidate = h2oCandidateSpec(model);
  return {sha256: candidate.graphSha256, bytes: candidate.graphBytes};
}

export function h2oCandidateSpec(model: string): H2oCandidateSpec {
  const candidate = Object.hasOwn(candidates, model) ? candidates[model] : undefined;
  if (!candidate) fail('Unknown H2O model.');
  return candidate!;
}

export async function verifyWasm(
  bytes: unknown,
  model: string = H2O_BASELINE_MODEL_ID,
  signal?: AbortSignal | null | undefined
): Promise<Uint8Array> {
  signal?.throwIfAborted();
  const identity = h2oGraphIdentity(model);
  if (!(bytes instanceof Uint8Array) || bytes.length !== identity.bytes) fail('Unexpected H2O pending WASM.');
  const owned = Uint8Array.from(bytes as Uint8Array);
  if ((await hash(owned, signal)) !== identity.sha256) fail('Unexpected H2O pending WASM.');
  return owned;
}

export async function prepareCandidatePlan(
  stageBytes: unknown,
  manifestBytes: unknown,
  signal?: AbortSignal | null | undefined
): Promise<H2oCandidatePlan> {
  return preparePlan(stageBytes, manifestBytes, STAGE_SHA256, signal);
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object')
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonical((value as Record<string, unknown>)[key])])
    );
  return value;
}

/** Candidate graphs share the baseline ABI and carry their own pinned provenance. */
export async function preparePublicCandidatePlan(
  metadataBytes: unknown,
  manifestBytes: unknown,
  model: string = H2O_BASELINE_MODEL_ID,
  candidateBytes?: unknown,
  signal?: AbortSignal | null | undefined
): Promise<H2oCandidatePlan> {
  signal?.throwIfAborted();
  const identity = h2oGraphIdentity(model);
  if (!(metadataBytes instanceof Uint8Array) || !(manifestBytes instanceof Uint8Array)) fail('Invalid H2O metadata.');
  const baseline = Uint8Array.from(metadataBytes as Uint8Array),
    manifest = Uint8Array.from(manifestBytes as Uint8Array);
  const candidate = candidateBytes instanceof Uint8Array ? Uint8Array.from(candidateBytes) : candidateBytes;
  const plan = await preparePlan(baseline, manifest, PUBLIC_METADATA_SHA256, signal);
  signal?.throwIfAborted();
  if (model === H2O_BASELINE_MODEL_ID) {
    if (candidate !== undefined) fail('Unexpected H2O candidate metadata.');
    return plan;
  }
  const spec = h2oCandidateSpec(model);
  if (
    !(candidate instanceof Uint8Array) ||
    candidate.length !== spec.metadataBytes ||
    (await hash(candidate, signal)) !== spec.metadataSha256
  )
    fail('Unexpected H2O candidate metadata.');
  let value!: H2oCandidateMetadata, original!: H2oCandidateMetadata;
  try {
    const decoder = new TextDecoder('utf-8', {fatal: true});
    value = JSON.parse(decoder.decode(candidate as Uint8Array)) as H2oCandidateMetadata;
    original = JSON.parse(decoder.decode(baseline)) as H2oCandidateMetadata;
  } catch {
    fail('Invalid H2O candidate metadata.');
  }
  const graph = value?.candidate,
    source = value?.baseline;
  if (
    value?.format !== spec.format ||
    value.variant !== model ||
    value.provenanceSigned !== false ||
    value.weightsRead !== false ||
    value.gpuExecuted !== false ||
    graph?.graphPath !== spec.graphPath ||
    graph.graphBytes !== identity.bytes ||
    graph.graphSha256 !== identity.sha256 ||
    graph.stageSha256 !== spec.stageSha256 ||
    graph.linkSha256 !== spec.linkSha256 ||
    graph.graphReportSha256 !== spec.graphReportSha256 ||
    source?.publicMetadataSha256 !== PUBLIC_METADATA_SHA256 ||
    source.stageSha256 !== STAGE_SHA256 ||
    source.manifestSha256 !== MANIFEST_SHA256 ||
    source.inventorySha256 !== '48080bff4785648f5b49fabde8ab84b341e0b4f2a6d20151f4c6a8051dab423d' ||
    source.auditReceiptSha256 !== '100e7cb766d5d50f3ae503ade53dd751a9c3253cc44d02f2a06cb42026b33b41' ||
    JSON.stringify(canonical(value.conversionCounts)) !== JSON.stringify(canonical(spec.conversionCounts)) ||
    value.sequenceContract?.minimum !== 2 ||
    value.sequenceContract?.maximum !== 512 ||
    value.sequenceContract?.runtimeEnforcementRequired !== true ||
    JSON.stringify(canonical(value.orderedCandidate)) !== JSON.stringify(canonical(original.orderedCandidate))
  )
    fail('H2O candidate provenance or ordered input mismatch.');
  return {...plan, wasmSha256: identity.sha256};
}

async function preparePlan(
  stageBytes: unknown,
  manifestBytes: unknown,
  expectedStageSha256: string,
  signal: AbortSignal | null | undefined
): Promise<H2oCandidatePlan> {
  signal?.throwIfAborted();
  if (!(stageBytes instanceof Uint8Array) || !(manifestBytes instanceof Uint8Array)) fail('Invalid H2O metadata.');
  stageBytes = Uint8Array.from(stageBytes as Uint8Array);
  manifestBytes = Uint8Array.from(manifestBytes as Uint8Array);
  if (
    !(stageBytes instanceof Uint8Array) ||
    stageBytes.length > 1_000_000 ||
    (await hash(stageBytes, signal)) !== expectedStageSha256 ||
    !(manifestBytes instanceof Uint8Array) ||
    manifestBytes.length > 1_000_000 ||
    (await hash(manifestBytes, signal)) !== MANIFEST_SHA256
  )
    fail('Unexpected H2O stage or conversion manifest.');
  let manifest!: H2oManifest, stages!: H2oStageRecord[];
  try {
    const decoder = new TextDecoder('utf-8', {fatal: true});
    manifest = JSON.parse(decoder.decode(manifestBytes as Uint8Array)) as H2oManifest;
    stages = decoder
      .decode(stageBytes as Uint8Array)
      .split('\n')
      .flatMap((line) => {
        try {
          return [JSON.parse(line) as H2oStageRecord];
        } catch {
          return [];
        }
      })
      .filter((value) => value.orderedCandidate);
  } catch {
    fail('Invalid candidate metadata.');
  }
  if (
    stages.length !== 1 ||
    stages[0]!.revision !== REVISION ||
    manifest.state !== 'verified-pending-review' ||
    manifest.sourceSha256 !== SOURCE_SHA256 ||
    manifest.headSourceSha256 !== HEAD_SHA256 ||
    manifest.productionReady !== false ||
    manifest.graphOrderVerified !== false ||
    !Array.isArray(manifest.artifacts) ||
    manifest.artifacts.length !== 428
  )
    fail('Unexpected candidate metadata.');
  const ordered = stages[0]!.orderedCandidate!;
  if (
    JSON.stringify(ordered.userInputs) !== JSON.stringify(USER_INPUTS) ||
    ordered.stateInputs !== 680 ||
    ordered.rotaryInputs !== 2 ||
    ordered.parameterInputs?.length !== 682
  )
    fail('Unexpected compiled input contract.');
  const files = new Map<string, H2oManifestFile>();
  const add = (file: unknown): void => {
    if (
      !file ||
      typeof (file as H2oManifestFile).path !== 'string' ||
      !/^[A-Za-z0-9_.-]+\.pending$/.test((file as H2oManifestFile).path) ||
      !Number.isSafeInteger((file as H2oManifestFile).bytes) ||
      (file as H2oManifestFile).bytes <= 0 ||
      !/^[a-f0-9]{64}$/.test((file as H2oManifestFile).sha256) ||
      files.has((file as H2oManifestFile).path)
    )
      fail('Invalid or duplicate candidate file.');
    files.set((file as H2oManifestFile).path, file as H2oManifestFile);
  };
  for (const entry of manifest.artifacts)
    for (const key of ['file', 'packed', 'scales'])
      if ((entry as Record<string, unknown>)[key]) add((entry as Record<string, unknown>)[key]);
  add(manifest.head.file);
  if (files.size !== 680) fail('Candidate file count mismatch.');
  const seen = new Set<string>();
  const parameters: H2oCandidateParameter[] = ordered.parameterInputs.map((entry, i) => {
    if (
      entry.position !== i + 5 ||
      entry.parameterIndex !== i ||
      !Array.isArray(entry.shape) ||
      entry.shape.some((n) => !Number.isSafeInteger(n) || n < 1)
    )
      fail('Invalid parameter order or shape.');
    if (i >= 680) {
      if (
        entry.source !== 'checked-config' ||
        entry.dtype !== 'float32' ||
        JSON.stringify(entry.shape) !== '[32]' ||
        !Array.isArray(entry.values) ||
        entry.values.length !== 32 ||
        entry.values.some((n) => !Number.isFinite(n))
      )
        fail('Invalid rotary input.');
      return {
        kind: 'rotary',
        target: entry.target as string,
        shape: [...(entry.shape as number[])],
        dtype: entry.dtype as string,
        values: [...(entry.values as number[])]
      };
    }
    const file = files.get(entry.file as string);
    const bytesPer = {float16: 2, float32: 4, uint32: 4}[entry.dtype as 'float16' | 'float32' | 'uint32'];
    if (
      entry.source !== 'pending-weight' ||
      !file ||
      seen.has(entry.file as string) ||
      !bytesPer ||
      entry.bytes !== file.bytes ||
      (entry.shape as number[]).reduce((a, b) => a * b, bytesPer) !== file.bytes
    )
      fail('Parameter and file mismatch.');
    seen.add(entry.file as string);
    return {
      kind: 'file',
      target: entry.target as string,
      shape: [...(entry.shape as number[])],
      dtype: entry.dtype as string,
      path: (file as H2oManifestFile).path,
      bytes: (file as H2oManifestFile).bytes,
      sha256: (file as H2oManifestFile).sha256
    };
  });
  if (
    seen.size !== 680 ||
    parameters[680]!.target !== 'body.rotary_emb.inv_freq' ||
    parameters[681]!.target !== 'body.rotary_emb.original_inv_freq'
  )
    fail('Incomplete parameter coverage.');
  return {
    revision: REVISION,
    wasmSha256: WASM_SHA256,
    userInputs: [...USER_INPUTS],
    parameters,
    labels: [...manifest.head.labels],
    labelIds: [...manifest.head.tokenIds],
    state: 'metadata-only-not-executable',
    sourceAndWeightsAudited: false
  } satisfies H2oCandidatePlan;
}

// The converter stored whole-file hashes. For a future range loader, accept an
// independently produced range inventory only after checking contiguous coverage.
// This validates geometry, not that supplied chunk hashes belong to the file.
export function validateRangeInventory(file: unknown, ranges: unknown): H2oRangeSpec[] {
  const path = (file as {path?: unknown} | null | undefined)?.path as string,
    totalBytes = (file as {bytes?: unknown} | null | undefined)?.bytes as number;
  if (
    typeof path !== 'string' ||
    !/^[A-Za-z0-9_.-]+\.pending$/.test(path) ||
    !Number.isSafeInteger(totalBytes) ||
    totalBytes < 1
  )
    fail('Invalid range file.');
  if (!Array.isArray(ranges) || !ranges.length) fail('Missing independent range hashes.');
  let offset = 0;
  const verified: H2oRangeSpec[] = [];
  for (const part of ranges as unknown[]) {
    if (
      !part ||
      typeof part !== 'object' ||
      Array.isArray(part) ||
      Object.keys(part).sort().join(',') !== 'length,offset,sha256'
    )
      fail('Invalid range inventory.');
    const {offset: start, length, sha256} = part as {offset: number; length: number; sha256: string};
    if (
      !Number.isSafeInteger(start) ||
      start !== offset ||
      !Number.isSafeInteger(length) ||
      length < 1 ||
      length > 8 * 1024 * 1024 ||
      typeof sha256 !== 'string' ||
      !/^[a-f0-9]{64}$/.test(sha256) ||
      length > totalBytes - offset
    )
      fail('Invalid range inventory.');
    verified.push({offset: start, length, sha256, path, totalBytes});
    offset += length;
  }
  if (offset !== totalBytes) fail('Incomplete range inventory.');
  return verified;
}
