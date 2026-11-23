import {loadPackagedGemma4V2} from '../../strands/gemma4-v2-packaged.mjs';
import {createGemma4V2CuratedAdapter} from './strands-gemma4-v2.mjs';

// The generator only imports this adapter after provenance is verified.
export const {loadModel} = createGemma4V2CuratedAdapter(loadPackagedGemma4V2);
