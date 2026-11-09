import {loadPinnedTextModel} from '../pinned-text-loader.js';
import {uploadTensorShards} from '../tensor-shard-upload.js';
import {readKev08ManifestResponse, validateKev08TokenizerData, validateKev08WeightIndex} from './contract.js';
import {requestKev08Device} from './device.js';
import {ExperimentalKev08Session} from './session.js';

/** Load the verified FP16 text-only Kev 0.8B prototype. */
export function loadExperimentalKev08(options: Parameters<typeof loadPinnedTextModel>[0]) {
  return loadPinnedTextModel(options, {
    name: 'Kev 0.8B',
    offlineMissingMessage: 'The Kev 0.8B manifest is not available offline. Load the model online first.',
    readManifest: readKev08ManifestResponse,
    validateIndex: validateKev08WeightIndex,
    requestDevice: requestKev08Device,
    prepare(data, _tokenizer, manifest) {
      validateKev08TokenizerData(data);
      return {
        limits: {
          maxInputTokens: (manifest as {maxInputTokens: number}).maxInputTokens,
          maxStateTokens: (manifest as {maxStateTokens: number}).maxStateTokens
        }
      };
    },
    upload(args) {
      return uploadTensorShards(args as Parameters<typeof uploadTensorShards>[0]);
    },
    createSession: (args) =>
      new ExperimentalKev08Session(args as ConstructorParameters<typeof ExperimentalKev08Session>[0])
  });
}
