import {parseJson, sha256} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import type {Qwen35VisionArtifact} from './qwen35-vision-runner.mts';
import type {Qwen35VisionWeightBinding} from './qwen35-vision-weights.mts';

export const QWEN35_VISION_FIXTURE_PIN = Object.freeze({
  bytes: 336938,
  sha256: 'aea3326f3d06b1e813b93e662a28b72c44f02b7030dd5328f4e71c4164418ede'
});
export const QWEN35_VISION_WASM_PIN = Object.freeze({
  activationPrecision: 'float32',
  bytes: 4316202,
  sha256: '4fa14d07c05c881ef3134fbdf16925e856401fb9608fcee2054d2b6d7aa10ec0'
});
export const QWEN35_VISION_BINDINGS_FILE_PIN = Object.freeze({
  bytes: 104_081,
  sha256: '340b604c8db09219cfb2aa79b2e930da1c5d1a0f07972263787f26a2c63dd60a'
});
export const QWEN35_VISION_BINDINGS_SHA256 = 'db9bc5cad76fe05237dec5673fa761ebc8ccdd92ebe59757c6e5f0a123c510be';
export const QWEN35_VISION_ARGUMENT_NAMES_SHA256 = '7dbb6ce3d4e1f102736f80def6c8f2d792802f215b13905c37538e0fab40f66a';

export interface Qwen35VisionArtifacts {
  readonly fixtureBytes: Uint8Array;
  readonly wasmBytes: Uint8Array;
  readonly artifact: Qwen35VisionArtifact;
  readonly bindings: readonly Qwen35VisionWeightBinding[];
}

function invalid(message: string): WebDecisionError {
  return new WebDecisionError('INVALID_ASSET', message);
}

export function verifyQwen35VisionFixtureWasm(artifact: Qwen35VisionArtifact): void {
  const precision = artifact.activationPrecision === undefined ? 'bfloat16' : artifact.activationPrecision;
  if (
    precision !== QWEN35_VISION_WASM_PIN.activationPrecision ||
    artifact.wasm?.bytes !== QWEN35_VISION_WASM_PIN.bytes ||
    artifact.wasm.sha256 !== QWEN35_VISION_WASM_PIN.sha256
  )
    throw invalid('Qwen3.5 vision fixture precision or executable differs from its pinned WebGPU graph.');
}

/** Recheck mutable artifact bytes immediately before TVM instantiation. */
export async function verifyPinnedQwen35VisionArtifacts(artifacts: Qwen35VisionArtifacts): Promise<void> {
  if (
    !artifacts ||
    !(artifacts.fixtureBytes instanceof Uint8Array) ||
    !(artifacts.wasmBytes instanceof Uint8Array) ||
    artifacts.fixtureBytes.byteLength !== QWEN35_VISION_FIXTURE_PIN.bytes ||
    artifacts.wasmBytes.byteLength !== QWEN35_VISION_WASM_PIN.bytes ||
    (await sha256(artifacts.fixtureBytes)) !== QWEN35_VISION_FIXTURE_PIN.sha256 ||
    (await sha256(artifacts.wasmBytes)) !== QWEN35_VISION_WASM_PIN.sha256
  )
    throw invalid('Qwen3.5 vision graph bytes differ from the pinned artifacts.');

  const parsedFixture = parseJson(artifacts.fixtureBytes, 'qwen35-dynamic-vision-fixture.json');
  if (JSON.stringify(parsedFixture) !== JSON.stringify(artifacts.artifact))
    throw invalid('Qwen3.5 vision fixture metadata differs from its pinned bytes.');
  verifyQwen35VisionFixtureWasm(artifacts.artifact);
}
