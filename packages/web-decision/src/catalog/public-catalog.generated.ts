import type {CatalogModelDescription, ModelId} from './types.generated.js';

type PublicCatalogEntry = Readonly<{modelId: ModelId; description: CatalogModelDescription}>;

function deepFreeze<T>(value: T): Readonly<T> {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) deepFreeze(child);
  return Object.freeze(value);
}

const catalog = [] as unknown as readonly PublicCatalogEntry[];
export const APPROVED_MODEL_IDS = Object.freeze(catalog.map(({modelId}) => modelId));
export const MODEL_CATALOG = deepFreeze(catalog);
export const MODEL_CATALOG_BY_ID = Object.freeze(
  Object.fromEntries(MODEL_CATALOG.map((entry) => [entry.modelId, entry])) as Record<ModelId, PublicCatalogEntry>
);
