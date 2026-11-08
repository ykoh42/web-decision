import {loadLayaModel} from '../laya/loader.mjs';
import {openLayaSource} from '../laya/source.mjs';
import {LAYA_TYPED_MODEL, LAYA_TYPED_SPEC} from './contract.mjs';

export function loadLayaTypedDecisionsModel(options, runtime = {}) {
  const openSource = (sourceOptions) =>
    runtime.openSource
      ? runtime.openSource({...sourceOptions, spec: LAYA_TYPED_SPEC})
      : openLayaSource({...sourceOptions, spec: LAYA_TYPED_SPEC});

  return loadLayaModel(options, {
    ...runtime,
    model: LAYA_TYPED_MODEL,
    modelName: 'Laya Typed-Decisions',
    spec: LAYA_TYPED_SPEC,
    openSource
  });
}
