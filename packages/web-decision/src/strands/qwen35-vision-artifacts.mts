import fixtureUrl from '../../assets/qwen35/vision/fixture.json?url&no-inline';
import wasmUrl from '../../assets/qwen35/vision/qwen35-dynamic-vision.webgpu.wasm?url&no-inline';
import bindingUrl from '../../assets/qwen35/vision/weight-bindings.json?url&no-inline';
import {parseJson, readVerifiedAsset} from '../assets.js';
import type {AssetCache, AssetFileMetadata} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {
  QWEN35_VISION_ARGUMENT_NAMES_SHA256,
  QWEN35_VISION_BINDINGS_FILE_PIN,
  QWEN35_VISION_BINDINGS_SHA256,
  QWEN35_VISION_FIXTURE_PIN,
  QWEN35_VISION_WASM_PIN,
  verifyQwen35VisionFixtureWasm
} from './qwen35-vision-contract.mts';
import type {Qwen35VisionArtifacts} from './qwen35-vision-contract.mts';
import {
  isPinnedQwen35VisionRopeInitializationEvidence,
  type Qwen35VisionArtifact,
  type Qwen35VisionArtifactArgument
} from './qwen35-vision-runner.mts';
import type {Qwen35VisionWeightBinding} from './qwen35-vision-weights.mts';

export type {Qwen35VisionArtifacts} from './qwen35-vision-contract.mts';

interface PinnedBindingFile {
  readonly format: 'qwen35-vision-pinned-source-weight-bindings-v1';
  readonly bindingsSha256: string;
  readonly argumentNamesSha256: string;
  readonly bindings: readonly Qwen35VisionWeightBinding[];
}

export interface OpenQwen35VisionArtifactOptions {
  readonly cache?: AssetCache | null | undefined;
  readonly offline?: boolean | undefined;
  readonly signal?: AbortSignal | null | undefined;
  readonly onBytes?: ((bytes: number) => void) | undefined;
  readonly onCacheStatus?: ((cached: boolean) => void) | undefined;
}

function invalid(message: string): WebDecisionError {
  return new WebDecisionError('INVALID_ASSET', message);
}

function assetLocation(assetUrl: string): Readonly<{base: URL; name: string}> {
  let url: URL;
  try {
    url = new URL(assetUrl, import.meta.url);
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'Invalid packaged Qwen3.5 vision artifact URL.', {cause});
  }
  const name = decodeURIComponent(url.pathname.slice(url.pathname.lastIndexOf('/') + 1));
  if (!name) throw invalid('Packaged Qwen3.5 vision artifact URL has no filename.');
  if ([...url.searchParams.keys()].some((key) => !['import', 'url', 'no-inline'].includes(key)))
    throw invalid(`Packaged Qwen3.5 vision artifact URL has an unsupported query: ${url.search}.`);
  if (url.hash) throw invalid('Packaged Qwen3.5 vision artifact URL has an unexpected fragment.');
  return Object.freeze({base: new URL('.', url), name});
}

async function readAsset(
  assetUrl: string,
  name: string,
  metadata: AssetFileMetadata,
  options: OpenQwen35VisionArtifactOptions
): Promise<Uint8Array> {
  const {base} = assetLocation(assetUrl);
  const buffer = await readVerifiedAsset(base, {files: {[name]: metadata}}, name, options);
  return new Uint8Array(buffer);
}

function artifact(value: unknown): Qwen35VisionArtifact {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw invalid('Invalid Qwen3.5 vision fixture.');
  const parsed = value as Qwen35VisionArtifact;
  if (
    parsed.format !== 'qwen35-dynamic-per-image-vision-tvmjs-fixture-v1' ||
    !isPinnedQwen35VisionRopeInitializationEvidence(parsed.visionRopeInitialization) ||
    parsed.shapeContract?.oneImagePerInvocation !== true ||
    parsed.shapeContract.patchRows?.formula !== '4*S' ||
    parsed.shapeContract.patchRows.minimum !== 256 ||
    parsed.shapeContract.patchRows.maximum !== 65_536 ||
    !Array.isArray(parsed.mainArgumentOrder) ||
    parsed.mainArgumentOrder.length !== 303
  )
    throw invalid('Qwen3.5 vision fixture does not match the pinned graph contract.');
  verifyQwen35VisionFixtureWasm(parsed);

  const names = new Set<string>();
  for (const argument of parsed.mainArgumentOrder) {
    if (
      !argument ||
      typeof argument.name !== 'string' ||
      !argument.name ||
      names.has(argument.name) ||
      (argument.sourceKind !== 'USER_INPUT' && argument.sourceKind !== 'PARAMETER') ||
      typeof argument.dtype !== 'string'
    )
      throw invalid('Qwen3.5 vision fixture contains an invalid graph argument.');
    names.add(argument.name);
  }
  if (parsed.mainArgumentOrder.filter(({sourceKind}) => sourceKind === 'USER_INPUT').length !== 6)
    throw invalid('Qwen3.5 vision fixture must declare six runtime inputs.');
  if (parsed.mainArgumentOrder.filter(({sourceKind}) => sourceKind === 'PARAMETER').length !== 297)
    throw invalid('Qwen3.5 vision fixture must declare 297 source parameters.');
  return Object.freeze({
    ...parsed,
    wasm: Object.freeze({...parsed.wasm!}),
    visionRopeInitialization: Object.freeze({
      ...parsed.visionRopeInitialization,
      restoredBuffers: Object.freeze([...parsed.visionRopeInitialization.restoredBuffers]) as readonly [
        'inv_freq',
        'original_inv_freq'
      ],
      bufferSha256: Object.freeze({...parsed.visionRopeInitialization.bufferSha256})
    }),
    shapeContract: Object.freeze({
      ...parsed.shapeContract,
      patchRows: Object.freeze({...parsed.shapeContract.patchRows})
    }),
    mainArgumentOrder: Object.freeze(
      parsed.mainArgumentOrder.map((argument: Qwen35VisionArtifactArgument) =>
        Object.freeze({
          ...argument,
          ...(argument.shape ? {shape: Object.freeze([...argument.shape])} : {}),
          ...(argument.relaxShape ? {relaxShape: Object.freeze([...argument.relaxShape])} : {})
        })
      )
    )
  });
}

function pinnedBindings(bytes: Uint8Array, graph: Qwen35VisionArtifact): readonly Qwen35VisionWeightBinding[] {
  const file = parseJson(bytes, 'qwen35-vision-weight-bindings.json') as PinnedBindingFile;
  if (
    file?.format !== 'qwen35-vision-pinned-source-weight-bindings-v1' ||
    file.bindingsSha256 !== QWEN35_VISION_BINDINGS_SHA256 ||
    file.argumentNamesSha256 !== QWEN35_VISION_ARGUMENT_NAMES_SHA256 ||
    !Array.isArray(file.bindings) ||
    file.bindings.length !== 297
  )
    throw invalid('Packaged Qwen3.5 vision weight bindings differ from their pin.');

  const graphParameters = graph.mainArgumentOrder.filter(({sourceKind}) => sourceKind === 'PARAMETER');
  const bindings = file.bindings.map((binding, index) => {
    const argument = graphParameters[index];
    if (
      !binding ||
      binding.index !== index ||
      !argument ||
      binding.targetName !== argument.name ||
      typeof binding.sourceName !== 'string' ||
      !Array.isArray(binding.shape) ||
      !Array.isArray(argument.shape) ||
      binding.shape.length !== argument.shape.length ||
      binding.shape.some((dimension: number, axis: number) => dimension !== argument.shape![axis]) ||
      binding.elementCount !== binding.shape.reduce((count: number, dimension: number) => count * dimension, 1) ||
      binding.sourceByteCount !== binding.elementCount * 2 ||
      binding.sourceDtype !== 'BF16' ||
      binding.targetDtype !== 'float32' ||
      binding.conversion !== 'exact-bf16-to-float32' ||
      !/^[0-9a-f]{64}$/.test(binding.sourcePayloadSha256)
    )
      throw invalid('Packaged Qwen3.5 vision weight binding is malformed at index ' + index + '.');
    return Object.freeze({...binding, shape: Object.freeze([...binding.shape])});
  });
  return Object.freeze(bindings);
}

/** Load the hash-pinned dynamic vision graph and source-weight map. */
export async function openPinnedQwen35VisionArtifacts({
  cache,
  offline = false,
  signal,
  onBytes,
  onCacheStatus
}: OpenQwen35VisionArtifactOptions = {}): Promise<Qwen35VisionArtifacts> {
  if (typeof offline !== 'boolean')
    throw new WebDecisionError('INVALID_INPUT', 'Invalid Qwen3.5 vision artifact options.');
  signal?.throwIfAborted();
  const options = {cache, offline, signal, onBytes, onCacheStatus};
  const [fixtureBytes, wasmBytes, bindingBytes] = await Promise.all([
    readAsset(fixtureUrl, 'fixture.json', QWEN35_VISION_FIXTURE_PIN, options),
    readAsset(wasmUrl, 'qwen35-dynamic-vision.webgpu.wasm', QWEN35_VISION_WASM_PIN, options),
    readAsset(bindingUrl, 'weight-bindings.json', QWEN35_VISION_BINDINGS_FILE_PIN, options)
  ]);
  signal?.throwIfAborted();
  const parsedArtifact = artifact(parseJson(fixtureBytes, 'qwen35-dynamic-vision-fixture.json'));
  const bindings = pinnedBindings(bindingBytes, parsedArtifact);
  return Object.freeze({fixtureBytes, wasmBytes, artifact: parsedArtifact, bindings});
}
