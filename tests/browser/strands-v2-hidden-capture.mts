import type {DLDevice, Instance, Tensor} from '../../packages/web-decision/src/runtime/tvm-types.js';

type OwnedTensor = Tensor & {dispose(): void};
type Runtime = Instance & {detachFromCurrentScope<T extends OwnedTensor>(value: T): T};

export function captureLastHiddenRows(runtime: Runtime, device: DLDevice) {
  const staged = new Map<string, OwnedTensor>();

  return {
    queue(name: string, tensor: Tensor) {
      if (tensor.dtype !== 'float32' || tensor.shape.at(-1) !== 1536 || staged.has(name))
        throw new Error(`Unexpected diagnostic hidden tensor: ${name}`);
      const copy = runtime.empty(tensor.shape, 'float32', runtime.cpu()).copyFrom(tensor) as OwnedTensor;
      staged.set(name, runtime.detachFromCurrentScope(copy));
    },
    async finish() {
      try {
        await device.sync();
        return Object.fromEntries(Array.from(staged, ([name, row]) => [name, Array.from(row.toArray().slice(-1536))]));
      } finally {
        for (const copy of staged.values()) copy.dispose();
        staged.clear();
      }
    }
  };
}
