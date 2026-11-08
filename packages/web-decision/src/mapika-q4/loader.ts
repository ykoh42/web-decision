import {WebDecisionError} from '../kev.js';
import {buildMapikaLabels} from '../mapika.js';
import {ExperimentalMapikaSession} from '../mapika/session.js';
import {loadPinnedTextModel} from '../pinned-text-loader.js';
import {readMapikaQ4ManifestResponse, validateMapikaQ4WeightIndex} from './contract.js';
import {requestMapikaQ4Device} from './device.js';
import {uploadMapikaQ4Weights} from './upload.js';
import type {
  MapikaQ4Extras,
  MapikaQ4LoadOptions,
  MapikaQ4Manifest,
  MapikaQ4ModelUploadArgs,
  MapikaQ4SessionArgs,
  MapikaTokenizer
} from './types.js';

/** Load the pinned, weight-only Q4 Mapika prototype from user-hosted assets. */
export function loadExperimentalMapikaQ4(options: MapikaQ4LoadOptions): Promise<ExperimentalMapikaSession> {
  return loadPinnedTextModel(options, {
    name: 'Mapika Q4',
    offlineMissingMessage: 'The Mapika Q4 manifest is not available offline. Load the model online first.',
    readManifest: readMapikaQ4ManifestResponse,
    validateIndex: validateMapikaQ4WeightIndex,
    requestDevice: requestMapikaQ4Device,
    prepare(data: unknown, tokenizer: MapikaTokenizer, manifest: MapikaQ4Manifest): MapikaQ4Extras {
      const tokenizerData = data as {added_tokens?: unknown};
      if (!Array.isArray(tokenizerData.added_tokens) || !tokenizerData.added_tokens.length)
        throw new WebDecisionError('INVALID_ASSET', 'The Mapika Q4 tokenizer has no control-token metadata.');
      const addedTokens = tokenizerData.added_tokens as {id: number}[];
      return {
        labels: buildMapikaLabels(tokenizer),
        limits: {
          maxInputTokens: manifest.maxInputTokens,
          maxStateTokens: manifest.maxStateTokens,
          controlIds: addedTokens.map((entry) => entry.id),
          temperature: manifest.temperature
        }
      };
    },
    upload({extras, ...args}: MapikaQ4ModelUploadArgs): Promise<void> {
      return uploadMapikaQ4Weights({...args, labels: extras.labels});
    },
    createSession: (args: MapikaQ4SessionArgs) => new ExperimentalMapikaSession({...args, modelId: 'mapika-0.8b-q4f16'})
  });
}
