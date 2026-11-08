import type {TinyDecideMetadata} from './engine.js';
import type {TinyDecideSession} from './session.mjs';
import type {TinyDecideAssetSource, TinyDecideSourceOptions} from './source.mjs';

export interface TinyDecideLoadProgress {
  readonly loadedBytes: number;
  readonly totalBytes: number;
  readonly fraction: number;
}

export interface TinyDecideLoadOptions {
  readonly model: string;
  readonly assetsUrl?: string | undefined;
  readonly cache?: boolean | undefined;
  readonly offline?: boolean | undefined;
  readonly signal?: AbortSignal | undefined;
  readonly onProgress?: ((progress: TinyDecideLoadProgress) => void) | undefined;
}

export interface TinyDecideLoaderEngine {
  readonly meta: TinyDecideMetadata;
  readonly cfg: TinyDecideMetadata['cfg'];
  readonly tok: import('./engine.js').WordPiece;
  answer: import('./engine.js').TinyDecide['answer'];
}

export interface TinyDecideLoadDependencies {
  readonly TinyDecideClass?:
    | (new (metadata: TinyDecideMetadata, bytes: ArrayBuffer) => TinyDecideLoaderEngine)
    | undefined;
  readonly openSource?: ((options: TinyDecideSourceOptions) => Promise<TinyDecideAssetSource>) | undefined;
}

export function validateTinyDecideLoadOptions(options: unknown): {
  readonly base: URL;
  readonly signal: AbortSignal | undefined;
  readonly cacheEnabled: boolean;
  readonly offline: boolean;
  readonly onProgress: ((progress: TinyDecideLoadProgress) => void) | undefined;
};

export function loadTinyDecide(options: unknown, dependencies?: TinyDecideLoadDependencies): Promise<TinyDecideSession>;
