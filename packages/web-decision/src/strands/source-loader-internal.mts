import type {ModelProgress} from '../curated/types.js';
import {WebDecisionError} from '../kev.js';
import {validatePrefetchBytes} from '../shard-prefetch.js';
import {requestStrandsAdapter} from './preflight.mjs';

export interface StrandsSourceOptions {
  readonly assetsUrl?: string;
  readonly cache?: boolean;
  readonly offline?: boolean;
  readonly signal?: AbortSignal;
  readonly createTokenizer?: (data: unknown, config: unknown) => unknown;
  readonly createRuntime?: (wasm: Uint8Array) => unknown;
  readonly gpuApi?: {requestAdapter(): PromiseLike<unknown>};
  readonly onProgress?: (progress: ModelProgress) => void;
  readonly maxPrefetchBytes?: number;
}

interface RangeSpec {
  readonly path: string;
  readonly totalBytes: number;
  readonly offset: number;
  readonly length: number;
  readonly sha256: string;
}

interface RangeReadOptions {
  readonly signal: AbortSignal | undefined;
}

interface VerifiedRange {
  readonly bytes: ArrayBuffer;
  readonly cachedByHash: boolean;
}

interface SourceBytes {
  readonly manifest?: Uint8Array;
  readonly wasm?: Uint8Array;
  readonly tokenizer?: Uint8Array;
  readonly tokenizerConfig?: Uint8Array;
}

interface SourceTransaction {
  read(base: URL, spec: RangeSpec, options: RangeReadOptions): PromiseLike<VerifiedRange>;
  commit(options: RangeReadOptions): PromiseLike<unknown>;
}

interface OpenSourceOptions {
  readonly base: URL;
  readonly cache: Cache | null;
  readonly offline: boolean;
  readonly signal: AbortSignal | undefined;
}

interface OpenSourceResult {
  readonly bytes: SourceBytes;
  readonly transaction: SourceTransaction;
}

interface LoadBytesOptions {
  readonly base: URL;
  readonly manifestBytes: SourceBytes['manifest'];
  readonly wasmBytes: SourceBytes['wasm'];
  readonly tokenizerBytes: SourceBytes['tokenizer'];
  readonly tokenizerConfigBytes: SourceBytes['tokenizerConfig'];
  readonly createTokenizer: StrandsSourceOptions['createTokenizer'];
  readonly createRuntime: StrandsSourceOptions['createRuntime'];
  readonly gpuApi: StrandsSourceOptions['gpuApi'];
  readonly signal: AbortSignal | undefined;
  readonly onProgress: StrandsSourceOptions['onProgress'];
  readonly cache: false;
  readonly offline: false;
}

interface LoadBytesDependencies {
  readonly preparedAdapter: unknown;
  readonly readRange: (base: URL, spec: RangeSpec, options: RangeReadOptions) => PromiseLike<VerifiedRange>;
  readonly finalize: (signal: AbortSignal | undefined) => PromiseLike<unknown>;
  /** Prevent the source loader from overriding the byte loader's test-only seams. */
  readonly verifyAssets?: never;
  readonly uploadWeights?: never;
  readonly retainLateDevice?: never;
}

type MaybePromise<Value> = Value | PromiseLike<Value>;

interface SourceLoaderDependencies<Session> {
  readonly openSource: (options: OpenSourceOptions) => MaybePromise<OpenSourceResult>;
  readonly openCache: (enabled: boolean, signal: AbortSignal | undefined) => MaybePromise<Cache | null>;
  readonly loadBytes: (options: LoadBytesOptions, dependencies: LoadBytesDependencies) => MaybePromise<Session>;
}

/** Internal factory; dependencies are selected by the module, never by user options. */
export function createSourceLoader<Session>({
  openSource,
  openCache,
  loadBytes
}: SourceLoaderDependencies<Session>): (options?: StrandsSourceOptions) => Promise<Session> {
  return async function loadSource({
    assetsUrl,
    cache = true,
    offline = false,
    signal,
    createTokenizer,
    createRuntime,
    gpuApi,
    onProgress,
    maxPrefetchBytes
  }: StrandsSourceOptions = {}) {
    if (
      typeof assetsUrl !== 'string' ||
      !assetsUrl ||
      typeof cache !== 'boolean' ||
      typeof offline !== 'boolean' ||
      (offline && !cache)
    )
      throw new WebDecisionError('INVALID_INPUT', 'Invalid Strands asset or cache options.');
    if (validatePrefetchBytes(maxPrefetchBytes) !== 0)
      throw new WebDecisionError('INVALID_INPUT', 'Strands does not support prefetching.');
    let base: URL;
    try {
      base = new URL(assetsUrl, globalThis.location?.href);
    } catch {
      throw new WebDecisionError('INVALID_INPUT', 'Invalid asset URL.');
    }
    if (!['http:', 'https:'].includes(base.protocol) || base.search || base.hash || base.username || base.password)
      throw new WebDecisionError(
        'INVALID_INPUT',
        'The asset URL must be an HTTP(S) directory without credentials, query, or fragment.'
      );
    if (!base.pathname.endsWith('/')) base.pathname += '/';
    signal?.throwIfAborted();
    const preparedAdapter = await requestStrandsAdapter(gpuApi, signal);
    signal?.throwIfAborted();
    const storage = await openCache(cache, signal);
    signal?.throwIfAborted();
    const {bytes, transaction} = await openSource({base, cache: storage, offline, signal});
    return loadBytes(
      {
        base,
        manifestBytes: bytes.manifest,
        wasmBytes: bytes.wasm,
        tokenizerBytes: bytes.tokenizer,
        tokenizerConfigBytes: bytes.tokenizerConfig,
        createTokenizer,
        createRuntime,
        gpuApi,
        signal,
        onProgress,
        cache: false,
        offline: false
      },
      {
        preparedAdapter,
        readRange: (rangeBase, spec, options) => transaction.read(rangeBase, spec, options),
        finalize: (loadSignal) => transaction.commit({signal: loadSignal})
      }
    );
  };
}
