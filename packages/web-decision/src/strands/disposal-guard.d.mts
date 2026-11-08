export function retainStrandsResources(runtime: unknown, gpu: unknown): void;
export function guardStrandsDisposal(
  runtime: {dispose(...args: unknown[]): unknown},
  gpu: {lost: PromiseLike<unknown>},
  lifecycle: {disposed: boolean; loss: unknown}
): {retain(): void; readonly blocked: boolean};
