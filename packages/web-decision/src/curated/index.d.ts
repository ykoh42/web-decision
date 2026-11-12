import type {
  CommonLoadOptions,
  KananaDecisionModel,
  KananaLoadOptions,
  LoadOptions,
  ModelDescription,
  ModelId
} from './types.js';

export {WebDecisionError} from './types.js';
export type {
  CommonLoadOptions,
  ErrorCode,
  KananaAnswer,
  KananaDecisionModel,
  KananaDecisionRequest,
  KananaDecisionResult,
  KananaImageSource,
  KananaLoadOptions,
  KananaMessage,
  KananaMessagePart,
  KananaQuestion,
  KananaRequestOptions,
  KananaValue,
  LoadOptions,
  LoadedDecisionModel,
  ModelDescription,
  ModelId,
  ModelProgress
} from './types.js';

export {KananaFloat, kananaFloat} from './types.js';

export function listModels(): readonly ModelDescription[];
export function describeModel(modelId: ModelId): ModelDescription;
export function loadDecisionModel(options: KananaLoadOptions): Promise<KananaDecisionModel>;
export function loadDecisionModel(options: LoadOptions): Promise<KananaDecisionModel>;
export function loadDecisionModel(options: CommonLoadOptions): Promise<KananaDecisionModel>;
export function clearModelCache(): Promise<boolean>;
