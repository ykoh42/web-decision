import type {AssetSourceOptions, ManifestSourceSpec} from '../asset-source-types.js';
import type {LoadedLayaTypedDecisionsModel, LayaTypedDecisionsLoadOptions} from '../index.js';
import {loadLayaModel} from '../laya/loader.mjs';
import {openLayaSource} from '../laya/source.mjs';
import {LAYA_TYPED_MODEL, LAYA_TYPED_SPEC} from './contract.mjs';

type LayaSource = Awaited<ReturnType<typeof openLayaSource>>;
type LayaSourceOptions = AssetSourceOptions & {readonly spec?: ManifestSourceSpec};
type LayaTypedLoadedModel = Omit<LoadedLayaTypedDecisionsModel, 'description'>;

interface LayaTypedRuntime {
  readonly [dependency: string]: unknown;
  readonly openSource?: (sourceOptions: LayaSourceOptions) => LayaSource | Promise<LayaSource>;
}

interface LayaTypedLoaderRuntime extends LayaTypedRuntime {
  readonly model: typeof LAYA_TYPED_MODEL;
  readonly modelName: string;
  readonly spec: typeof LAYA_TYPED_SPEC;
  readonly openSource: (sourceOptions: LayaSourceOptions) => LayaSource | Promise<LayaSource>;
}

type LayaModelLoader = (
  options: LayaTypedDecisionsLoadOptions,
  runtime: LayaTypedLoaderRuntime
) => Promise<LayaTypedLoadedModel>;

// The JavaScript loader infers its defaults as base-Laya literals and widens
// shared session output fields. This boundary reflects the generic model/spec
// parameters and the typed session it actually returns before registry wrapping.
const loadTypedLayaModel = loadLayaModel as unknown as LayaModelLoader;

export function loadLayaTypedDecisionsModel(
  options: LayaTypedDecisionsLoadOptions,
  runtime: LayaTypedRuntime = {}
): Promise<LayaTypedLoadedModel> {
  const openSource = (sourceOptions: LayaSourceOptions): LayaSource | Promise<LayaSource> => {
    if (runtime.openSource) return runtime.openSource({...sourceOptions, spec: LAYA_TYPED_SPEC});
    return openLayaSource({...sourceOptions, spec: LAYA_TYPED_SPEC});
  };

  return loadTypedLayaModel(options, {
    ...runtime,
    model: LAYA_TYPED_MODEL,
    modelName: 'Laya Typed-Decisions',
    spec: LAYA_TYPED_SPEC,
    openSource
  });
}
