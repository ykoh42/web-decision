import {getFakeKevLoaderState} from './state.ts';

export {waitForAbort} from '../../../src/abort.js';

interface SourceOverrides {
  bundledUrls?: Readonly<Record<string, string>> | undefined;
}

interface SourceOptions {
  signal?: AbortSignal;
  offline?: boolean;
  maxPrefetchBytes?: number;
}

interface AssetReaderOptions {
  base: string;
  fetchImpl?: (url: string, init?: RequestInit) => Promise<Response>;
  signal?: AbortSignal;
}

interface FakeModel {
  family: string;
  dispose?: unknown;
  systemOne?: unknown;
}

const fixture = getFakeKevLoaderState;

export class Tokenizer {}

export function createPolyfillWASI(): object {
  return {};
}

export function instantiate(): Promise<unknown> {
  fixture().instantiations++;
  fixture().started();
  return fixture().pending;
}

export async function detectKevGPUDevice(): Promise<{device: unknown; adapter: {features: Set<string>}}> {
  return {device: fixture().gpu, adapter: {features: new Set(['shader-f16'])}};
}

export const CACHE_NAME = 'fake';

export function parseJson<T>(value: T): T {
  return value;
}

export async function readManifestResponse(): Promise<unknown> {
  return fixture().manifest;
}

export async function readKev06ManifestResponse(): Promise<unknown> {
  return fixture().manifest;
}

export const EXPERIMENTAL_KEV_MODEL = 'experimental-kev-0.6b-q4f16-upstream';
export const KEV06_STABLE_MODEL = 'kev-0.6b-q4f16';

export function createKevReconstructedSource(overrides: SourceOverrides = {}) {
  fixture().candidateSourceCreates++;
  fixture().bundledUrls = overrides.bundledUrls;
  return {
    async prepare(options: SourceOptions) {
      fixture().events.push('prepare');
      options.signal?.throwIfAborted();
      if (!fixture().gpuAvailable) {
        throw Object.assign(new Error('WebGPU unavailable'), {code: 'UNSUPPORTED_ENVIRONMENT'});
      }
      return {
        base: 'https://kev-assets.web-decision.invalid/experimental-kev-upstream/',
        signal: options.signal,
        cache: null,
        response: {},
        maxPrefetchBytes: 0
      };
    },
    async fetch(_url: string, {signal}: {signal?: AbortSignal} = {}) {
      if (!fixture().bundledUrls) {
        const module = await import('./kev06/bundled-assets.js');
        fixture().bundledUrls = module.experimentalKevBundledUrls;
      }
      signal?.throwIfAborted();
      return new Response(new Uint8Array([1]));
    },
    setProgress() {},
    dispose() {}
  };
}

export function createKev06StableReconstructedSource() {
  fixture().stableSourceCreates++;
  return {
    async prepare(options: SourceOptions) {
      fixture().events.push('stable-prepare');
      fixture().stablePrepareOptions = options;
      options.signal?.throwIfAborted();
      if (!fixture().gpuAvailable) {
        throw Object.assign(new Error('WebGPU unavailable'), {code: 'UNSUPPORTED_ENVIRONMENT'});
      }
      if (options.offline && !fixture().warmCache) {
        throw Object.assign(new Error('The stable model is not cached.'), {code: 'ASSET_LOAD_FAILED'});
      }
      return {
        base: 'https://kev-assets.web-decision.invalid/kev-0.6b-q4f16/',
        signal: options.signal,
        cache: null,
        response: {},
        maxPrefetchBytes: options.maxPrefetchBytes ?? 0
      };
    },
    async fetch(url: string, {signal}: {signal?: AbortSignal} = {}) {
      fixture().stableFetchUrls.push(url);
      if (!fixture().stableBundledUrls) {
        const module = await import('./kev06/stable-bundled-assets.js');
        fixture().stableBundledUrls = module.stableKevBundledUrls;
      }
      signal?.throwIfAborted();
      return new Response(new Uint8Array([4]));
    },
    dispose() {
      fixture().stableSourceDisposals++;
    }
  };
}

export async function readExperimentalKevManifest(): Promise<unknown> {
  return fixture().manifest;
}

export async function commitCachedManifest(): Promise<void> {}

export async function prepareModelSource(options: {assetsUrl?: string; signal?: AbortSignal}) {
  fixture().externalPrepares++;
  fixture().externalOptions = options;
  return {
    base: options.assetsUrl ?? 'https://legacy-assets.example.test/',
    signal: options.signal,
    response: {},
    cache: null
  };
}

export function createModelAssetReader({base, fetchImpl, signal}: AssetReaderOptions) {
  return {
    async read(name: string, readSignal = signal): Promise<ArrayBuffer> {
      if (fetchImpl && !fixture().warmCache) {
        const url = new URL(name, base).href;
        fixture().assetFetches.push(url);
        await fetchImpl(url, readSignal ? {signal: readSignal} : {});
      }
      return new ArrayBuffer(1);
    },
    isCacheComplete() {
      return fixture().warmCache;
    },
    setProgressPhase() {}
  };
}

export async function readAssetBatch(
  names: string[],
  read: (name: string, signal?: AbortSignal) => Promise<unknown>,
  signal?: AbortSignal
) {
  await Promise.all(names.map((name) => read(name, signal)));
  return [{}, {}, fixture().modelConfig, {}, fixture().cacheManifest];
}

export async function uploadModelShards(): Promise<void> {
  fixture().uploads++;
}

export class DecisionModel {
  constructor(
    readonly runtime: {dispose(): void},
    _device: unknown,
    readonly gpu: {destroy(): void},
    _fn: unknown,
    _weights: unknown,
    _tokenizer: unknown,
    readonly limits: unknown
  ) {}

  dispose(): void {
    this.runtime.dispose();
    this.gpu.destroy();
  }
}

function loadFake(family: string, options: unknown): FakeModel {
  fixture().selectedFamily = family;
  fixture().selectedOptions = options;
  return {family};
}

export async function loadExperimentalDeberta(options: unknown): Promise<FakeModel> {
  return loadFake('deberta', options);
}

export async function loadExperimentalMapika(options: unknown): Promise<FakeModel> {
  return loadFake('mapika', options);
}

export async function loadExperimentalMapikaQ4(options: unknown): Promise<FakeModel> {
  return loadFake('mapika-q4', options);
}

export async function loadExperimentalKev08(options: unknown): Promise<FakeModel> {
  return loadFake('kev08', options);
}

export async function loadExperimentalStrandsFromSource(options: unknown): Promise<FakeModel> {
  return loadFake('strands', options);
}

export async function loadTamev(options: {model: string}): Promise<FakeModel> {
  const model = loadFake('tamev', options);
  model.dispose = async () => {
    fixture().tamevDisposals++;
    if (fixture().tamevDisposeFails) throw new Error('fake TAMEV disposal failed');
  };
  const mode = fixture().tamevSystemOneMode;
  if (mode === 'accessor') {
    Object.defineProperty(model, 'systemOne', {get: () => () => ({model: options.model, answers: {}})});
  } else if (mode === 'nonconfigurable') {
    Object.defineProperty(model, 'systemOne', {
      configurable: false,
      enumerable: true,
      value: async () => ({model: options.model, answers: {}}),
      writable: false
    });
  } else if (mode === 'nonfunction') {
    model.systemOne = 'not a function';
  } else if (mode !== 'missing') {
    model.systemOne = async () => ({model: options.model, answers: {}});
  }
  return model;
}
