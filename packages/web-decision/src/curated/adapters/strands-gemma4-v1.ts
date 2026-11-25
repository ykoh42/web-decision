import graphWasm from '../../../assets/strands-gemma4-v1/graph.wasm?url&no-inline';
import graph from '../../../assets/strands-gemma4-v1/graph.json?url&no-inline';
import inventory from '../../../assets/strands-gemma4-v1/source-inventory.json?url&no-inline';
import tokenizer from '../../../assets/strands-gemma4-v1/tokenizer.json?url&no-inline';
import tokenizerConfig from '../../../assets/strands-gemma4-v1/tokenizer_config.json?url&no-inline';
import manifest from '../../../assets/strands-gemma4-v1/manifest.json' with {type: 'json'};
import {createGemma4V1CuratedAdapter} from './strands-gemma4-v1.mjs';

const asset = (url: string) => new URL(url, import.meta.url);
export const {loadModel} = createGemma4V1CuratedAdapter(
  {
    'graph.wasm': asset(graphWasm),
    'graph.json': asset(graph),
    'source-inventory.json': asset(inventory),
    'tokenizer.json': asset(tokenizer),
    'tokenizer_config.json': asset(tokenizerConfig)
  },
  manifest
);
