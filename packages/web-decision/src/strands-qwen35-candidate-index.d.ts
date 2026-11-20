import type {
  StrandsQwen35CandidateLoadOptions,
  StrandsQwen35LoadedCandidateModel as CandidateModel
} from './curated/strands-qwen35-candidate-types.js';

export {KananaFloat, kananaFloat} from './kanana/json-float.js';
export {WebDecisionError} from './curated/public-types.js';
export type {CatalogModelLimit, ErrorCode} from './curated/public-types.js';
export type {StrandsQwen35ModelProgress as ModelProgress} from './curated/strands-qwen35-candidate-types.js';
export type {
  StrandsQwen35CandidateLoadOptions,
  StrandsQwen35CandidateModel,
  StrandsQwen35Content,
  StrandsQwen35Dictionary,
  StrandsQwen35ModelProgress,
  StrandsQwen35Object,
  StrandsQwen35Question,
  StrandsQwen35SystemOneAnswer,
  StrandsQwen35SystemOneOptions,
  StrandsQwen35SystemOneRequest,
  StrandsQwen35SystemOneResult,
  StrandsQwen35Value
} from './curated/strands-qwen35-candidate-types.js';

export type ModelId = 'strands-decider-2b-qwen3.5-v1-2610';
export type LoadOptions = StrandsQwen35CandidateLoadOptions & Readonly<{modelId: ModelId}>;
export type StrandsQwen35LoadedCandidateModel = CandidateModel<ModelId>;
export type LoadedDecisionModel = StrandsQwen35LoadedCandidateModel;
export type ModelDescription = LoadedDecisionModel['description'];

export function listModels(): readonly ModelDescription[];
export function describeModel(modelId: ModelId): ModelDescription;
export function loadDecisionModel(options: LoadOptions): Promise<LoadedDecisionModel>;
export function clearModelCache(): Promise<boolean>;
