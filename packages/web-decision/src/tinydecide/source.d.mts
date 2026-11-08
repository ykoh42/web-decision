export type TinyDecideAssetName = 'meta.json' | 'model.bin';

export interface TinyDecideAssetFile {
  readonly bytes: number;
  readonly sha256: string;
}

export interface TinyDecideAssetManifestFile {
  readonly bytes?: number;
  readonly sha256: string;
}

export interface TinyDecideAssetSpec {
  readonly files: Readonly<Partial<Record<TinyDecideAssetName, TinyDecideAssetFile>>>;
  readonly manifest: {
    readonly files: Readonly<Partial<Record<TinyDecideAssetName, TinyDecideAssetManifestFile>>>;
  };
  readonly totalBytes: number;
}

export interface TinyDecideProgress {
  readonly loadedBytes: number;
  readonly totalBytes: number;
  readonly fraction: number;
}

export interface TinyDecideSourceOptions {
  readonly base: URL;
  readonly signal?: AbortSignal | undefined;
  readonly cacheEnabled: boolean;
  readonly offline: boolean;
  readonly onProgress?: ((progress: TinyDecideProgress) => void) | undefined;
  readonly spec?: TinyDecideAssetSpec | undefined;
}

export interface TinyDecideAssetSource {
  read(name: TinyDecideAssetName): Promise<ArrayBuffer>;
}

export function openTinyDecideSource(options: TinyDecideSourceOptions): Promise<TinyDecideAssetSource>;
