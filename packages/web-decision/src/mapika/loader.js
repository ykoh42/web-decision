import {WebDecisionError} from '../kev.js';
import {buildMapikaLabels} from '../mapika.js';
import {loadPinnedTextModel} from '../pinned-text-loader.js';
import {readMapikaManifestResponse, validateMapikaWeightIndex} from './contract.js';
import {requestMapikaDevice} from './device.js';
import {ExperimentalMapikaSession} from './session.js';
import {uploadMapikaWeights} from './upload.js';

/** Load the verified FP16 text-only Mapika prototype from user-hosted assets. */
export function loadExperimentalMapika(options) {
  return loadPinnedTextModel(options, {
    name: 'Mapika',
    offlineMissingMessage: 'The Mapika manifest is not available offline. Load the model online first.',
    readManifest: readMapikaManifestResponse,
    validateIndex: validateMapikaWeightIndex,
    requestDevice: requestMapikaDevice,
    prepare(data, tokenizer, manifest) {
      if (!Array.isArray(data.added_tokens) || !data.added_tokens.length)
        throw new WebDecisionError('INVALID_ASSET', 'The Mapika tokenizer has no control-token metadata.');
      return {
        labels: buildMapikaLabels(tokenizer),
        limits: {
          maxInputTokens: manifest.maxInputTokens,
          maxStateTokens: manifest.maxStateTokens,
          controlIds: data.added_tokens.map((entry) => entry.id),
          temperature: manifest.temperature
        }
      };
    },
    upload({extras, ...args}) {
      return uploadMapikaWeights({...args, labels: extras.labels});
    },
    createSession: (args) => new ExperimentalMapikaSession({...args, modelId: 'mapika-0.8b-f16'})
  });
}
