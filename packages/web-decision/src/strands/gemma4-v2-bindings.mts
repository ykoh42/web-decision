import bindingInventory from '../../assets/strands-gemma4-v2/decoder-bindings.json' with {type: 'json'};
import sourceInventory from '../../assets/strands-gemma4-v2/source-inventory.json' with {type: 'json'};
import {sha256} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {GEMMA4_V2_REPOSITORY, GEMMA4_V2_REVISION} from './gemma4-v2-source.mjs';

const BINDING_SHA256 = 'd559f57f95338b9b90429b61baf367f5e8a18b186013ec377c6e9aeb253d951c';
type Parameter = Readonly<{name: string; shape: readonly number[]; dtype: string; bytes: number; packedWords: number}>;
type Layer = Readonly<{layer: number; family: number; scalar: string; parameters: readonly Parameter[]}>;
type Bindings = Readonly<{
  format: string;
  repository: string;
  revision: string;
  sourceInventorySha256: string;
  sourceGraphArtifacts: Readonly<Record<string, {bytes: number; sha256: string}>>;
  layers: readonly Layer[];
}>;
type Source = Readonly<{tensors: Readonly<Record<string, {shape: readonly number[]; dtype: string; length: number}>>}>;
const invalid = (message: string) => new WebDecisionError('INVALID_ASSET', message);
const digest = (value: unknown) => sha256(new TextEncoder().encode(JSON.stringify(value) + '\n'));

function validateLayer(layer: Layer, index: number, source: Source): void {
  const family = index < 15 ? (index % 5 === 4 ? 4 : 0) : index % 5 === 4 ? 19 : 15;
  const prefix = `model.language_model.layers.${index}.`;
  const scalar = source.tensors[layer.scalar];
  if (
    layer.layer !== index ||
    layer.family !== family ||
    layer.scalar !== `${prefix}layer_scalar` ||
    !scalar ||
    scalar.dtype !== 'BF16' ||
    scalar.length !== 2 ||
    scalar.shape.length !== 1 ||
    scalar.shape[0] !== 1 ||
    !Array.isArray(layer.parameters) ||
    layer.parameters.length !== (index < 15 ? 16 : 13)
  )
    throw invalid('Invalid Gemma 4 v2 decoder layer binding.');
  const seen = new Set<string>();
  for (const binding of layer.parameters) {
    const tensor = source.tensors[binding.name];
    if (
      !binding.name.startsWith(prefix) ||
      seen.has(binding.name) ||
      !tensor ||
      binding.dtype !== 'BF16' ||
      tensor.dtype !== binding.dtype ||
      tensor.length !== binding.bytes ||
      binding.bytes !== binding.packedWords * 4 ||
      tensor.shape.length !== binding.shape.length ||
      binding.shape.some((dimension: number, offset: number) => dimension !== tensor.shape[offset])
    )
      throw invalid('Gemma 4 v2 decoder parameter does not match the pinned source.');
    seen.add(binding.name);
  }
}

/** Validate graph argument metadata against the exact ordered source binding contract. */
export function assertGemma4V2GraphBindings(
  layer: Layer,
  parameters: readonly Pick<Parameter, 'name' | 'packedWords'>[]
): void {
  if (
    !Array.isArray(parameters) ||
    parameters.length !== layer.parameters.length ||
    parameters.some(
      (parameter, index) =>
        parameter.name !== layer.parameters[index]!.name ||
        parameter.packedWords !== layer.parameters[index]!.packedWords
    )
  )
    throw invalid('Gemma 4 v2 graph arguments do not match the pinned decoder bindings.');
}

/** Internal test seam; normal loading uses both pinned manifests below. */
export async function loadGemma4V2BindingsFromManifests(value: Bindings, expectedHash: string, source: Source) {
  if ((await digest(value)) !== expectedHash || (await digest(source)) !== value.sourceInventorySha256)
    throw invalid('Gemma 4 v2 binding or source inventory digest differs.');
  const manifest = JSON.parse(JSON.stringify(value)) as Bindings;
  if (
    manifest.format !== 'web-decision-gemma4-v2-decoder-bindings-v1' ||
    manifest.repository !== GEMMA4_V2_REPOSITORY ||
    manifest.revision !== GEMMA4_V2_REVISION ||
    !Array.isArray(manifest.layers) ||
    manifest.layers.length !== 35
  )
    throw invalid('Invalid Gemma 4 v2 binding manifest.');
  for (const family of [0, 4, 15, 19]) {
    const graph = manifest.sourceGraphArtifacts[String(family)];
    if (!graph || !Number.isSafeInteger(graph.bytes) || graph.bytes < 1 || !/^[0-9a-f]{64}$/u.test(graph.sha256))
      throw invalid('Invalid Gemma 4 v2 source graph evidence.');
  }
  for (const [index, layer] of manifest.layers.entries()) validateLayer(layer, index, source);
  return Object.freeze({
    layer(index: number): Layer {
      if (!Number.isInteger(index) || index < 0 || index >= manifest.layers.length)
        throw new WebDecisionError('INVALID_INPUT', 'Gemma 4 v2 decoder layer is invalid.');
      const layer = manifest.layers[index]!;
      return {
        layer: layer.layer,
        family: layer.family,
        scalar: layer.scalar,
        parameters: layer.parameters.map((entry) => ({...entry, shape: [...entry.shape]}))
      };
    },
    sourceGraph(family: number) {
      const item = manifest.sourceGraphArtifacts[String(family)];
      if (!item) throw invalid('Unknown Gemma 4 v2 source graph.');
      return {...item};
    }
  });
}

export function loadPinnedGemma4V2Bindings() {
  return loadGemma4V2BindingsFromManifests(
    bindingInventory as unknown as Bindings,
    BINDING_SHA256,
    sourceInventory as unknown as Source
  );
}
