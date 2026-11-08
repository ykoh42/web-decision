import type {DecisionModel, ModelDescription} from './curated/types.js';

/** Internal boundary while the existing Kev loader is migrated to TypeScript. */
export function loadCuratedKevModel(
  options: object,
  description: ModelDescription & {readonly model: ModelDescription['modelId']}
): Promise<Omit<DecisionModel, 'description'>>;
