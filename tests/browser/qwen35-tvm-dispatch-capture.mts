export type Qwen35TvmDispatchGroup = 'text' | 'vision' | 'accepted-boundary' | 'batch-3-context-4096';

export interface Qwen35TvmDispatchCapture {
  activeGroup: Qwen35TvmDispatchGroup | null;
  dispatchCounts: Record<string, number>;
  tvmRuntimeDispatchCounts: Record<string, number>;
  tvmRuntimeCallers: Record<string, string>;
}

type DispatchWorkgroupsMethod = (this: unknown, ...args: [number, number?, number?]) => unknown;
type DispatchWorkgroupsPrototype = {dispatchWorkgroups: DispatchWorkgroupsMethod};

export function installQwen35TvmDispatchCapture(
  prototype: DispatchWorkgroupsPrototype,
  stackProvider: () => string = () => new Error().stack ?? ''
): Qwen35TvmDispatchCapture {
  const descriptor = Object.getOwnPropertyDescriptor(prototype, 'dispatchWorkgroups');
  if (
    !descriptor ||
    typeof descriptor.value !== 'function' ||
    (descriptor.configurable !== true && descriptor.writable !== true)
  )
    throw new Error('Cannot instrument WebGPU compute dispatches for Qwen graph invocation evidence.');

  const capture: Qwen35TvmDispatchCapture = {
    activeGroup: null,
    dispatchCounts: {},
    tvmRuntimeDispatchCounts: {},
    tvmRuntimeCallers: {}
  };
  const original = descriptor.value as DispatchWorkgroupsMethod;
  const wrapped: DispatchWorkgroupsMethod = function (this: unknown, ...args: [number, number?, number?]) {
    const group = capture.activeGroup;
    if (group) {
      capture.dispatchCounts[group] = (capture.dispatchCounts[group] ?? 0) + 1;
      const tvmRuntimeCaller = stackProvider()
        .split('\n')
        .find((line) => line.includes('tvm.generated'));
      if (tvmRuntimeCaller) {
        capture.tvmRuntimeDispatchCounts[group] = (capture.tvmRuntimeDispatchCounts[group] ?? 0) + 1;
        capture.tvmRuntimeCallers[group] ??= tvmRuntimeCaller;
      }
    }
    return Reflect.apply(original, this, args);
  };
  Object.defineProperty(prototype, 'dispatchWorkgroups', {...descriptor, value: wrapped});
  return capture;
}
