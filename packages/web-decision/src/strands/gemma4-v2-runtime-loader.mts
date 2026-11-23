import {WebDecisionError} from '../kev.js';
import {createGemma4V2ImageInserter} from './gemma4-v2-image-runner.mjs';
import {loadGemma4V2Model, type Gemma4V2LoaderOptions} from './gemma4-v2-loader.mjs';
import {
  createGemma4V2RunnerFactory,
  type Gemma4V2GraphAsset,
  type Gemma4V2GraphAssets
} from './gemma4-v2-runner-factory.mjs';
import {transposedSplitGraphs} from './gemma4-v2-split-adapter.mjs';

interface Gemma4V2RuntimeAssets extends Gemma4V2GraphAssets {
  readonly scoring: Gemma4V2GraphAsset;
}

/** Internal until the complete graph has been verified and its artifact metadata pinned. */
export function loadGemma4V2Runtime(options: Gemma4V2LoaderOptions, assets: Gemma4V2RuntimeAssets) {
  if (!assets?.scoring) throw new WebDecisionError('INVALID_ASSET', 'A pinned Gemma 4 v2 scoring asset is required.');
  const createRunner = createGemma4V2RunnerFactory({
    assets: {...assets, hostReadout: true},
    splitExecution: {graphs: transposedSplitGraphs(), chunkTokens: 256},
    createImageInserter: ({runtime, device, vm, source, signal}) =>
      createGemma4V2ImageInserter(runtime, device, vm, source, signal)
  });
  return loadGemma4V2Model(options, {
    createRunner,
    scoringAsset: assets.scoring,
    graphAssets: [assets.wasm, assets.localFrequency, assets.globalFrequency]
  });
}
