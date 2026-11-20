import manifestUrl from '../../assets/qwen35/r19/manifest.json?url&no-inline';
import fixtureUrl from '../../assets/qwen35/r19/fixture.json?url&no-inline';
import wasmUrl from '../../assets/qwen35/r19/qwen35-text-pointer.webgpu.wasm?url&no-inline';
import bindingUrl from '../../assets/qwen35/r19/weight-bindings.json?url&no-inline';
import {parseJson, readVerifiedAsset, sha256} from '../assets.js';
import type {AssetCache, AssetFileMetadata} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import type {Qwen35PackedGraphWeightBinding} from './qwen35-r13-runtime.mts';
import {QWEN35_R19_TEXT} from './qwen35-r19-text-contract.mts';

const {manifest: MANIFEST_PIN, fixture: FIXTURE_PIN, wasm: WASM_PIN, inputNames: RUNTIME_INPUT_NAMES} = QWEN35_R19_TEXT;
const BINDINGS_PIN = Object.freeze({
  bytes: 392846,
  sha256: '03604675d3bdbf6f8a7cef7dcc372c0626a0f4d0b9964ec06b200dc5918d5526'
});
const SOURCE_REPORT_SHA256 = 'd278da94de41ba53c4a42c922f67193ff3762937ece55e3a7f77a1051d80e899';
const SOURCE_RELAX_SHA256 = '624cc890ac1ffb92caac65fd27b9c924f2d0b71bb009325bf16aa77fe8f6830a';
const PACKED_ABI_SHA256 = 'accc30973a0612f8b30da8e416ea082d6aa3e0c7b3eee80eb3ab17bfdc3dd783';
const SOURCE_BINDINGS_SHA256 = '0107466fdfe1eaa8ab35caa3f22f0a382838c58c9ebb852270a5001a5fd987dd';
const ARGUMENT_NAMES_SHA256 = 'f2a0e40d162bcd064e9f924c7ee91f50dd91604ed362b0a2e9cab9187d4e313f';
const PARAMETER_INVENTORY_SHA256 = 'a27d37aef07c1fa67d78653dc4da01945ef54af473f465e98795e72f16ab5f2b';
const STRANDS_SOURCE_REVISION = '13c05149265dc404c09d2469636f71b1320e1713';
const CHECKPOINT_REVISION = '216d1ccb46a9e9b036eb9187b248db98480f77d4';
const PARAMETER_COUNT = RUNTIME_INPUT_NAMES.length + 697;
const RUNTIME_INPUT_DTYPES = ['float32', 'int32', 'int32', 'int32', 'float32', 'int32', 'shape'] as const;

export interface Qwen35R19ArtifactManifest {
  readonly format: 'web-decision-qwen35-r19-text-artifact-manifest-v1';
  readonly generation: 'R19';
  readonly modelId: 'strands-decider-2b-qwen3.5-v1-2610';
  readonly source: Readonly<Record<string, unknown>>;
  readonly abi: Readonly<
    Record<string, unknown> & {
      readonly bindingsSha256: string;
      readonly argumentNamesSha256: string;
    }
  >;
  readonly files: Readonly<Record<string, AssetFileMetadata>>;
}

export interface Qwen35R19Artifacts {
  readonly manifest: Qwen35R19ArtifactManifest;
  readonly manifestBytes: Uint8Array;
  readonly fixtureBytes: Uint8Array;
  readonly wasmBytes: Uint8Array;
  readonly bindings: readonly Qwen35PackedGraphWeightBinding[];
}

export interface OpenQwen35R19ArtifactOptions {
  readonly cache?: AssetCache | null | undefined;
  readonly offline?: boolean | undefined;
  readonly signal?: AbortSignal | null | undefined;
  readonly onBytes?: ((bytes: number) => void) | undefined;
}

interface PinnedBindingFile {
  readonly format: 'qwen35-r19-verified-packed-weight-bindings-v1';
  readonly sourceReportSha256: string;
  readonly sourceRelaxSha256: string;
  readonly strandsDeciderSourceRevision: string;
  readonly checkpointRevision: string;
  readonly sourceBindingsSha256: string;
  readonly bindingsSha256: string;
  readonly argumentNamesSha256: string;
  readonly bindings: readonly Qwen35PackedGraphWeightBinding[];
}

function invalid(message: string): WebDecisionError {
  return new WebDecisionError('INVALID_ASSET', message);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function assetLocation(assetUrl: string): Readonly<{base: URL}> {
  let url: URL;
  try {
    url = new URL(assetUrl, import.meta.url);
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Invalid packaged Qwen3.5 R19 artifact URL.', {cause});
  }
  if ([...url.searchParams.keys()].some((key) => !['import', 'url', 'no-inline'].includes(key)) || url.hash)
    throw invalid('Packaged Qwen3.5 R19 artifact URL has unsupported URL components.');
  return Object.freeze({base: new URL('.', url)});
}

function pinnedManifest(bytes: Uint8Array): Qwen35R19ArtifactManifest {
  const manifest = parseJson(bytes, 'manifest.json') as Qwen35R19ArtifactManifest;
  if (
    manifest?.format !== 'web-decision-qwen35-r19-text-artifact-manifest-v1' ||
    manifest.generation !== 'R19' ||
    manifest.modelId !== 'strands-decider-2b-qwen3.5-v1-2610' ||
    !isRecord(manifest.source) ||
    manifest.source.importReportSha256 !== SOURCE_REPORT_SHA256 ||
    manifest.source.relaxModuleSha256 !== SOURCE_RELAX_SHA256 ||
    manifest.source.packedAbiSha256 !== PACKED_ABI_SHA256 ||
    manifest.source.parameterInventorySha256 !== PARAMETER_INVENTORY_SHA256 ||
    manifest.source.strandsDeciderSourceRevision !== STRANDS_SOURCE_REVISION ||
    manifest.source.checkpointRevision !== CHECKPOINT_REVISION ||
    !isRecord(manifest.abi) ||
    manifest.abi.mainParameterCount !== PARAMETER_COUNT ||
    manifest.abi.fullParameterNamesSha256 !== QWEN35_R19_TEXT.parameterNamesSha256 ||
    manifest.abi.externalWeightCount !== 697 ||
    manifest.abi.bindingsSha256 !== 'bfdd259943568b0b8a08594a75ba35d47133a60b2418c3fa8f2c8bce4ece801f' ||
    manifest.abi.argumentNamesSha256 !== ARGUMENT_NAMES_SHA256 ||
    manifest.abi.packedBf16ParameterCount !== 319 ||
    manifest.abi.packedInt32ParameterCount !== 319 ||
    manifest.abi.retainedFp32ParameterCount !== 378 ||
    !Array.isArray(manifest.abi.runtimeInputNames) ||
    JSON.stringify(manifest.abi.runtimeInputNames) !== JSON.stringify(RUNTIME_INPUT_NAMES) ||
    !isRecord(manifest.files) ||
    JSON.stringify(manifest.files['fixture.json']) !== JSON.stringify(FIXTURE_PIN) ||
    JSON.stringify(manifest.files['qwen35-text-pointer.webgpu.wasm']) !== JSON.stringify(WASM_PIN) ||
    JSON.stringify(manifest.files['weight-bindings.json']) !== JSON.stringify(BINDINGS_PIN)
  )
    throw invalid('Packaged Qwen3.5 R19 manifest differs from its exact source, files, or ABI pins.');
  return Object.freeze(manifest);
}

function canonicalBindings(bindings: readonly Qwen35PackedGraphWeightBinding[]): string {
  return JSON.stringify(
    bindings.map((binding) => ({
      index: binding.index,
      asset: binding.asset,
      sourceName: binding.sourceName,
      targetName: binding.targetName,
      relaxArgumentName: binding.relaxArgumentName,
      shape: binding.shape,
      sourceDtype: binding.sourceDtype,
      loadedDtype: binding.loadedDtype,
      relaxDtype: binding.relaxDtype,
      loadedPayloadSha256: binding.loadedPayloadSha256 ?? null,
      relaxShape: binding.relaxShape
    }))
  );
}

function pinnedBindings(
  bytes: Uint8Array,
  manifest: Qwen35R19ArtifactManifest
): readonly Qwen35PackedGraphWeightBinding[] {
  const file = parseJson(bytes, 'weight-bindings.json') as PinnedBindingFile;
  if (
    file?.format !== 'qwen35-r19-verified-packed-weight-bindings-v1' ||
    file.sourceReportSha256 !== SOURCE_REPORT_SHA256 ||
    file.sourceRelaxSha256 !== SOURCE_RELAX_SHA256 ||
    file.strandsDeciderSourceRevision !== STRANDS_SOURCE_REVISION ||
    file.checkpointRevision !== CHECKPOINT_REVISION ||
    file.sourceBindingsSha256 !== SOURCE_BINDINGS_SHA256 ||
    file.bindingsSha256 !== manifest.abi.bindingsSha256 ||
    file.argumentNamesSha256 !== manifest.abi.argumentNamesSha256 ||
    !Array.isArray(file.bindings) ||
    file.bindings.length !== 697
  )
    throw invalid('Packaged Qwen3.5 R19 weight bindings differ from their source and ABI pins.');

  const bindings = file.bindings.map((binding, index) => {
    if (
      !binding ||
      binding.index !== index ||
      !['base', 'lora', 'head'].includes(binding.asset) ||
      typeof binding.sourceName !== 'string' ||
      typeof binding.targetName !== 'string' ||
      typeof binding.relaxArgumentName !== 'string' ||
      !Array.isArray(binding.shape) ||
      !Array.isArray(binding.relaxShape) ||
      binding.shape.length === 0 ||
      binding.relaxShape.length === 0 ||
      binding.shape.some((dimension: number) => !Number.isSafeInteger(dimension) || dimension < 1) ||
      binding.relaxShape.some((dimension: number) => !Number.isSafeInteger(dimension) || dimension < 1) ||
      !['float32', 'bfloat16'].includes(binding.sourceDtype) ||
      !['float32', 'bfloat16'].includes(binding.loadedDtype) ||
      !['float32', 'int32'].includes(binding.relaxDtype) ||
      (binding.loadedPayloadSha256 !== null &&
        (typeof binding.loadedPayloadSha256 !== 'string' || !/^[0-9a-f]{64}$/u.test(binding.loadedPayloadSha256)))
    )
      throw invalid('Packaged Qwen3.5 R19 weight bindings are malformed.');
    return Object.freeze({
      ...binding,
      shape: Object.freeze([...binding.shape]),
      relaxShape: Object.freeze([...binding.relaxShape]),
      loadedPayloadSha256: binding.loadedPayloadSha256 ?? undefined
    });
  });
  return Object.freeze(bindings);
}

function staticDimension(value: unknown): number {
  if (typeof value !== 'string') throw invalid('Qwen3.5 R19 fixture contains a non-string shape dimension.');
  const match = /^T\.int64\((\d+)\)$/u.exec(value);
  if (!match) throw invalid(`Qwen3.5 R19 fixture contains a non-static dimension: ${value}.`);
  const dimension = Number(match[1]);
  if (!Number.isSafeInteger(dimension) || dimension < 1)
    throw invalid(`Qwen3.5 R19 fixture contains an invalid dimension: ${value}.`);
  return dimension;
}

async function verifyFixture(
  bytes: Uint8Array,
  wasmBytes: Uint8Array,
  bindings: readonly Qwen35PackedGraphWeightBinding[]
) {
  const fixture = parseJson(bytes, 'fixture.json') as Record<string, unknown>;
  const parameters = fixture.parameterOrder;
  const wasm = fixture.wasm;
  if (
    fixture.format !== 'qwen35-full-pointer-tvmjs-fixture-v1' ||
    fixture.mainParameterCount !== PARAMETER_COUNT ||
    fixture.externalWeightCount !== 697 ||
    fixture.deviceExecutionPerformed !== false ||
    fixture.modelNumericalInferencePerformed !== false ||
    !Array.isArray(fixture.runtimeInputNames) ||
    JSON.stringify(fixture.runtimeInputNames) !== JSON.stringify(RUNTIME_INPUT_NAMES) ||
    !Array.isArray(parameters) ||
    parameters.length !== PARAMETER_COUNT ||
    !isRecord(wasm) ||
    wasm.file !== 'qwen35-text-pointer.webgpu.wasm' ||
    wasm.bytes !== WASM_PIN.bytes ||
    wasm.sha256 !== WASM_PIN.sha256 ||
    !isRecord(fixture.sourceEvidence) ||
    !isRecord(fixture.sourceEvidence.source) ||
    fixture.sourceReport !== 'qwen35-inputs-embeds-packed-bf16-full-tvmjs-batch3-2026-11-19.json' ||
    fixture.savedRelaxSha256 !== SOURCE_RELAX_SHA256 ||
    fixture.sourceEvidence.relaxScriptSha256 !== SOURCE_RELAX_SHA256 ||
    fixture.sourceEvidence.source.strandsDeciderSourceRevision !== STRANDS_SOURCE_REVISION
  )
    throw invalid('Packaged Qwen3.5 R19 fixture differs from its exact source graph ABI.');

  const names: string[] = [];
  for (let index = 0; index < parameters.length; index += 1) {
    const parameter = parameters[index];
    if (!isRecord(parameter) || parameter.index !== index || typeof parameter.name !== 'string')
      throw invalid(`Qwen3.5 R19 graph parameter ${index} is malformed.`);
    names.push(parameter.name);
    if (index < RUNTIME_INPUT_NAMES.length) {
      const expectedDtype = RUNTIME_INPUT_DTYPES[index];
      if (
        parameter.name !== RUNTIME_INPUT_NAMES[index] ||
        parameter.external !== false ||
        parameter.dtype !== expectedDtype
      )
        throw invalid(`Qwen3.5 R19 runtime input ${index} differs from its pinned ABI.`);
      continue;
    }
    const binding = bindings[index - RUNTIME_INPUT_NAMES.length]!;
    if (
      parameter.external !== true ||
      parameter.role !== 'external-checkpoint-weight' ||
      parameter.name !== binding.relaxArgumentName ||
      parameter.dtype !== binding.relaxDtype ||
      !Array.isArray(parameter.shape) ||
      JSON.stringify(parameter.shape.map(staticDimension)) !== JSON.stringify(binding.relaxShape)
    )
      throw invalid(`Qwen3.5 R19 graph parameter ${index} differs from its pinned weight binding.`);
  }
  if ((await sha256(new TextEncoder().encode(names.join('\n')))) !== QWEN35_R19_TEXT.parameterNamesSha256)
    throw invalid('Qwen3.5 R19 graph parameter-name order differs from its pinned ABI.');
  if (
    (await sha256(new TextEncoder().encode(canonicalBindings(bindings)))) !==
    'bfdd259943568b0b8a08594a75ba35d47133a60b2418c3fa8f2c8bce4ece801f'
  )
    throw invalid('Qwen3.5 R19 packed weight binding hashes differ from their pinned ABI.');
  if (
    (await sha256(new TextEncoder().encode(bindings.map(({relaxArgumentName}) => relaxArgumentName).join('\n')))) !==
    ARGUMENT_NAMES_SHA256
  )
    throw invalid('Qwen3.5 R19 weight argument names differ from their pinned ABI.');
  if (wasmBytes.byteLength !== WASM_PIN.bytes) throw invalid('Packaged Qwen3.5 R19 WASM length differs from its pin.');
}

async function readAsset(
  assetUrl: string,
  name: string,
  metadata: AssetFileMetadata,
  options: OpenQwen35R19ArtifactOptions
) {
  const {base} = assetLocation(assetUrl);
  const buffer = await readVerifiedAsset(base, {files: {[name]: metadata}}, name, options);
  return new Uint8Array(buffer);
}

/** Load and verify the exact packed R19 text graph for the pinned Qwen3.5 checkpoint. */
export async function openPinnedQwen35R19Artifacts({
  cache,
  offline = false,
  signal,
  onBytes
}: OpenQwen35R19ArtifactOptions = {}): Promise<Qwen35R19Artifacts> {
  if (typeof offline !== 'boolean')
    throw new WebDecisionError('INVALID_INPUT', 'Invalid Qwen3.5 R19 artifact options.');
  signal?.throwIfAborted();
  const options = {cache, offline, signal, onBytes};
  const manifestBytes = await readAsset(manifestUrl, 'manifest.json', MANIFEST_PIN, options);
  const manifest = pinnedManifest(manifestBytes);
  const [fixtureBytes, wasmBytes, bindingBytes] = await Promise.all([
    readAsset(fixtureUrl, 'fixture.json', FIXTURE_PIN, options),
    readAsset(wasmUrl, 'qwen35-text-pointer.webgpu.wasm', WASM_PIN, options),
    readAsset(bindingUrl, 'weight-bindings.json', BINDINGS_PIN, options)
  ]);
  signal?.throwIfAborted();
  const bindings = pinnedBindings(bindingBytes, manifest);
  await verifyFixture(fixtureBytes, wasmBytes, bindings);
  return Object.freeze({manifest, manifestBytes, fixtureBytes, wasmBytes, bindings});
}
