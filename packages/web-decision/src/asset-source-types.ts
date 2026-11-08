import type {ModelProgress} from './curated/types.js';

/** Inputs shared by verified, model-specific asset readers. */
export interface AssetSourceOptions {
  readonly base: string | URL;
  readonly signal?: AbortSignal | undefined;
  readonly cacheEnabled?: boolean | undefined;
  readonly offline?: boolean | undefined;
  readonly onProgress?: ((progress: ModelProgress) => void) | undefined;
}

export interface AssetSourceFile {
  readonly bytes: number;
  readonly sha256: string;
}

export interface AssetSourceSpec {
  readonly files: Readonly<Record<string, AssetSourceFile>>;
  readonly totalBytes: number;
}

export interface ManifestSourceSpec {
  readonly manifest: {readonly files: Readonly<Record<string, AssetSourceFile>>};
  readonly maxAssetBytes: number;
  readonly totalBytes: number;
}
