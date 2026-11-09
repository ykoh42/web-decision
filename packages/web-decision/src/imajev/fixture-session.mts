// Internal test seam only: caller-injected fake runtime is not a model loader.
import {createImajevResultProcessor} from './result-processor.mjs';
import {ImajevSession} from './session.mjs';

interface FixtureImajevSessionOptions {
  readonly tokenizer?: unknown;
  readonly manifest?: unknown;
  readonly calibration?: unknown;
  readonly maxInputTokens?: unknown;
  readonly [key: string]: unknown;
}

export function createFixtureImajevSession({
  tokenizer,
  manifest,
  calibration,
  maxInputTokens,
  ...runtime
}: FixtureImajevSessionOptions) {
  const processor = createImajevResultProcessor({tokenizer, manifest, calibration, maxInputTokens});
  return new ImajevSession({...runtime, processor, maxInputTokens} as ConstructorParameters<typeof ImajevSession>[0]);
}
