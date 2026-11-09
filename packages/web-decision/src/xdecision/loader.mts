import type {AssetSourceOptions, ManifestSourceSpec} from '../asset-source-types.js';
import type {LoadedXDecisionModel, XDecisionLoadOptions} from '../index.js';
import {loadLayaModel} from '../laya/loader.mjs';
import {openLayaSource} from '../laya/source.mjs';
import {XDECISION_MODEL, XDECISION_SPEC} from './contract.mjs';

type XDecisionSource = Awaited<ReturnType<typeof openLayaSource>>;
type XDecisionSourceOptions = AssetSourceOptions & {readonly spec?: ManifestSourceSpec};
type XDecisionSession = Omit<LoadedXDecisionModel, 'description'>;

interface XDecisionRuntime {
  readonly [dependency: string]: unknown;
  readonly openSource?: (options: XDecisionSourceOptions) => XDecisionSource | Promise<XDecisionSource>;
}

interface XDecisionLoaderRuntime extends XDecisionRuntime {
  readonly model: typeof XDECISION_MODEL;
  readonly modelName: string;
  readonly spec: typeof XDECISION_SPEC;
  readonly openSource: (options: XDecisionSourceOptions) => XDecisionSource | Promise<XDecisionSource>;
}

type LayaModelLoader = (options: XDecisionLoadOptions, runtime: XDecisionLoaderRuntime) => Promise<XDecisionSession>;

// The JavaScript loader infers its defaults as base-Laya literals and widens
// the shared session shape. This boundary narrows its supported model/spec
// overrides and the xDecision session returned before registry wrapping.
const loadXDecisionLayaModel = loadLayaModel as unknown as LayaModelLoader;

const openSource = (options: XDecisionSourceOptions): XDecisionSource | Promise<XDecisionSource> =>
  openLayaSource({...options, spec: XDECISION_SPEC});

export function loadXDecisionModel(
  options: XDecisionLoadOptions,
  runtime: XDecisionRuntime = {}
): Promise<XDecisionSession> {
  return loadXDecisionLayaModel(options, {
    ...runtime,
    model: XDECISION_MODEL,
    modelName: 'xDecision',
    spec: XDECISION_SPEC,
    openSource: runtime.openSource ?? openSource
  });
}
