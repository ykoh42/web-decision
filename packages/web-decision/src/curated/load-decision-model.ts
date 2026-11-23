import {WebDecisionError} from './errors.ts';
import {readLoadOptions} from './load-options.ts';

type DisposableModel = Readonly<{dispose?: () => void | Promise<void>}>;
type Adapter<Description, Model extends DisposableModel> = Readonly<{
  loadModel(options: object, description: Description): Promise<Model>;
}>;

/** Internal dependency boundary for the same curated loader used by the public API. */
export function createDecisionModelLoader<Id extends string, Description, Model extends DisposableModel>(dependencies: {
  readonly isModelId: (value: unknown) => value is Id;
  readonly describe: (modelId: Id) => Description;
  readonly loaderDescription: (modelId: Id) => Description;
  readonly loadAdapter: (modelId: Id) => (() => Promise<Adapter<Description, Model>>) | undefined;
  readonly approvedLoadOptions?: Readonly<Record<string, readonly string[]>>;
}) {
  return async (input: unknown): Promise<Model> => {
    const {modelId, options} = readLoadOptions(input, dependencies.isModelId, dependencies.approvedLoadOptions);
    (options as {signal?: AbortSignal}).signal?.throwIfAborted();
    const description = dependencies.describe(modelId);
    const adapter = dependencies.loadAdapter(modelId);
    if (!adapter) throw new WebDecisionError('INVALID_ASSET', 'The model adapter is unavailable.', undefined);
    const {loadModel} = await adapter();
    const model = await loadModel(options, dependencies.loaderDescription(modelId));
    try {
      Object.defineProperty(model, 'description', {value: description});
      return model;
    } catch (error) {
      await Promise.allSettled([Promise.resolve().then(() => model.dispose?.())]);
      throw error;
    }
  };
}
