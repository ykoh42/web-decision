// Internal test seam only: caller-injected fake runtime is not a model loader.
import {createImajevResultProcessor} from './result-processor.mjs';
import {ImajevSession} from './session.mjs';

export function createFixtureImajevSession({tokenizer, manifest, calibration, maxInputTokens, ...runtime}) {
  const processor = createImajevResultProcessor({tokenizer, manifest, calibration, maxInputTokens});
  return new ImajevSession({...runtime, processor, maxInputTokens});
}
