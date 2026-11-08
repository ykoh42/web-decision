import {loadLayaModel} from '../laya/loader.mjs';
import {openLayaSource} from '../laya/source.mjs';
import {XDECISION_MODEL, XDECISION_SPEC} from './contract.mjs';

const openSource = (options) => openLayaSource({...options, spec: XDECISION_SPEC});

export function loadXDecisionModel(options, runtime = {}) {
  return loadLayaModel(options, {
    ...runtime,
    model: XDECISION_MODEL,
    modelName: 'xDecision',
    openSource: runtime.openSource ?? openSource
  });
}
