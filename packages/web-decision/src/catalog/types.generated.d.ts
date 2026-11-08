export type ModelId = 'kev-0.6b-q4f16';
export type ModelIdTuple = readonly ['kev-0.6b-q4f16'];
export type ModelAdapterId = 'kev06';
export interface CatalogModelLimit {
  readonly resource: 'tokens' | 'options' | 'questions';
  readonly scope: 'state' | 'question' | 'request';
  readonly min?: number;
  readonly max?: number;
  readonly description?: string;
}
export interface CatalogModelDescription {
  readonly adapter: ModelAdapterId;
  readonly precision: string;
  readonly checkpoint: Readonly<{repository: string; revision: string; weightSha256?: string}>;
  readonly assets: Readonly<{
    integrity: 'pinned-manifest';
    manifestSha256: string;
    location: string;
    requiredFiles: readonly string[];
    additionalFiles: string;
  }>;
  readonly licenses: Readonly<Record<string, string>>;
  readonly backend: Readonly<{
    name: 'tvm-webgpu';
    requiredFeatures: readonly string[];
    requiredLimits: Readonly<Record<string, number>>;
    requiresNonFallbackAdapter: boolean;
  }>;
  readonly features: Readonly<{
    textChoice: Readonly<{
      status: 'implemented';
      abstention: string;
      limits: readonly CatalogModelLimit[];
      handling: string;
      tokenCount: string;
      score: Readonly<Record<string, unknown>>;
    }>;
    typedTextQuestionTypes?: readonly string[];
    imageChoice?: Readonly<{status: 'not-implemented'}>;
    rating?: Readonly<{status: 'not-implemented'}>;
  }>;
  readonly verification: Readonly<Record<string, string>>;
}
export interface ModelArtifactLock {
  readonly schemaVersion: 1;
  readonly manifest: Readonly<{
    sourcePath: string;
    rawSha256: string;
    canonicalSha256: string;
    canonicalBytes: number;
    experimental: boolean;
    files: Readonly<Record<string, Readonly<{sha256: string; bytes: number}>>>;
  }>;
  readonly bundledRuntimeFiles: readonly Readonly<{
    manifestName: string;
    sourcePath: string;
    bytes: number;
    sha256: string;
  }>[];
  readonly upstream: Readonly<{
    repository: string;
    revision: string;
    sourcePath: string;
    sourceBytes: number;
    sourceSha256: string;
    shardMapPath: string;
    shardMapSha256: string;
  }>;
}
export interface ModelCatalogEntry {
  readonly modelId: ModelId;
  readonly description: CatalogModelDescription;
  readonly artifacts: ModelArtifactLock;
}
export declare const MODEL_IDS: ModelIdTuple;
export declare const MODEL_CATALOG: readonly ModelCatalogEntry[];
export declare const MODEL_CATALOG_BY_ID: Readonly<Record<ModelId, ModelCatalogEntry>>;
