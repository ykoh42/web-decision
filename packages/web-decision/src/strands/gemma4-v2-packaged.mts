import graphUrl from '../../assets/strands-gemma4-v2/graph.wasm?url&no-inline';
import localUrl from '../../assets/strands-gemma4-v2/local.bin?url&no-inline';
import globalUrl from '../../assets/strands-gemma4-v2/global.bin?url&no-inline';
import scoringUrl from '../../assets/strands-gemma4-v2/scoring.bin?url&no-inline';
import manifest from '../../assets/strands-gemma4-v2/manifest.json' with {type: 'json'};
import {loadGemma4V2Runtime} from './gemma4-v2-runtime-loader.mjs';
import type {Gemma4V2LoaderOptions} from './gemma4-v2-loader.mjs';

function packagedAsset(url: string, metadata: {bytes: number; sha256: string}) {
  const asset = new URL(url, import.meta.url);
  const name = decodeURIComponent(asset.pathname.slice(asset.pathname.lastIndexOf('/') + 1));
  return {base: new URL('.', asset), name, metadata};
}

/** Internal artifact binding; the model remains excluded from the public catalog. */
export function gemma4V2PackagedAssets() {
  return {
    wasm: packagedAsset(graphUrl, manifest.files['graph.wasm']),
    localFrequency: packagedAsset(localUrl, manifest.files['local.bin']),
    globalFrequency: packagedAsset(globalUrl, manifest.files['global.bin']),
    scoring: packagedAsset(scoringUrl, manifest.files['scoring.bin'])
  };
}

export function loadPackagedGemma4V2(options: Gemma4V2LoaderOptions) {
  return loadGemma4V2Runtime(options, gemma4V2PackagedAssets());
}
