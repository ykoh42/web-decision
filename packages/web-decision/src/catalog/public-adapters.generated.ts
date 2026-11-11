import type {ModelDescription} from '../curated/types.js';

type ModelAdapterModule = Readonly<{
  loadModel(options: object, description: ModelDescription): Promise<Readonly<{dispose?: () => Promise<void>}>>;
}>;

export const APPROVED_ADAPTER_LOADERS: Readonly<Record<string, () => Promise<ModelAdapterModule>>> = Object.freeze({});
