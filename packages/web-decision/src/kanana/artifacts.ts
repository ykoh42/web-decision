/// <reference types="vite/client" />

import artifactManifest from '../../assets/kanana/manifest.json?raw';
import textWasmUrl from '../../assets/kanana/kanana-text-model.wasm?url&no-inline';
import textWeights from '../../assets/kanana/text-weight-manifest.json?raw';
import visionWasmUrl from '../../assets/kanana/kanana-vision-model.wasm?url&no-inline';
import visionWeights from '../../assets/kanana/vision-weight-manifest.json?raw';
import rangeManifest from '../../assets/kanana/safetensors-range-manifest.json?raw';

export const KANANA_ARTIFACT_SOURCES = Object.freeze({
  artifactManifest,
  textWasmUrl,
  textWeights,
  visionWasmUrl,
  visionWeights,
  rangeManifest
});
