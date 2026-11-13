export interface FakeKevLoaderState {
  assetFetches: string[];
  bundledUrls: Readonly<Record<string, string>> | undefined;
  cacheManifest: unknown;
  candidateSourceCreates: number;
  events: string[];
  externalOptions: unknown;
  externalPrepares: number;
  gpu: unknown;
  gpuAvailable: boolean;
  helperLoads: number;
  instantiations: number;
  manifest: unknown;
  modelConfig: unknown;
  pending: Promise<unknown>;
  selectedFamily: string | undefined;
  selectedOptions: unknown;
  stableBundledUrls: Readonly<Record<string, string>> | undefined;
  stableFetchUrls: string[];
  stableHelperLoads: number;
  stablePrepareOptions: unknown;
  stableSourceCreates: number;
  stableSourceDisposals: number;
  started: () => void;
  uploads: number;
  warmCache: boolean;
}

export function getFakeKevLoaderState(): FakeKevLoaderState {
  const state = (globalThis as typeof globalThis & {__fakeKevLoader?: FakeKevLoaderState}).__fakeKevLoader;
  if (!state) throw new Error('Kev loader test fixture state has not been initialized.');
  return state;
}
