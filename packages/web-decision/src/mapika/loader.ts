import {WebDecisionError} from '../kev.js';
import {buildMapikaLabels} from '../mapika.js';
import {loadPinnedTextModel} from '../pinned-text-loader.js';
import {readMapikaManifestResponse, validateMapikaWeightIndex} from './contract.js';
import {requestMapikaDevice} from './device.js';
import {ExperimentalMapikaSession} from './session.js';
import {uploadMapikaWeights} from './upload.js';

/** Load the verified FP16 text-only Mapika prototype from user-hosted assets. */
export function loadExperimentalMapika(options: Parameters<typeof loadPinnedTextModel>[0]) {
  return loadPinnedTextModel(options, {
    name: 'Mapika',
    offlineMissingMessage: 'The Mapika manifest is not available offline. Load the model online first.',
    readManifest: readMapikaManifestResponse,
    validateIndex: validateMapikaWeightIndex,
    requestDevice: requestMapikaDevice,
    prepare(data, tokenizer, manifest) {
      if (
        !Array.isArray((data as {added_tokens: {id: number}[]}).added_tokens) ||
        !(data as {added_tokens: {id: number}[]}).added_tokens.length
      )
        throw new WebDecisionError('INVALID_ASSET', 'The Mapika tokenizer has no control-token metadata.');
      return {
        labels: buildMapikaLabels(tokenizer),
        limits: {
          maxInputTokens: (manifest as {maxInputTokens: number}).maxInputTokens,
          maxStateTokens: (manifest as {maxStateTokens: number}).maxStateTokens,
          controlIds: (data as {added_tokens: {id: number}[]}).added_tokens.map((entry) => entry.id),
          temperature: (manifest as {temperature: number}).temperature
        }
      };
    },
    upload({extras, ...args}) {
      return uploadMapikaWeights({...args, labels: extras.labels} as Parameters<typeof uploadMapikaWeights>[0]);
    },
    createSession: (args) =>
      new ExperimentalMapikaSession({
        ...(args as Record<string, unknown>),
        modelId: 'mapika-0.8b-f16'
      } as ConstructorParameters<typeof ExperimentalMapikaSession>[0])
  });
}
