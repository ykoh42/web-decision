export type ModelId =
  | 'kanana-2-v-decide-3b-2611'
  | 'strands-decider-2b-qwen3.5-v1-2610'
  | 'mapika-decider-2b-v12'
  | 'strands-decider-e2b-gemma4-v2-2611'
  | 'strands-decider-e2b-gemma4-v1-2610';
export type ModelIdTuple = readonly [
  'kanana-2-v-decide-3b-2611',
  'strands-decider-2b-qwen3.5-v1-2610',
  'mapika-decider-2b-v12',
  'strands-decider-e2b-gemma4-v2-2611',
  'strands-decider-e2b-gemma4-v1-2610'
];
export type ModelAdapterId = 'kanana' | 'strands-qwen35' | 'mapika2b' | 'strands-gemma4-v2' | 'strands-gemma4-v1';
export interface CatalogModelLimit {
  readonly resource: 'tokens' | 'options' | 'questions' | 'images';
  readonly scope: 'state' | 'question' | 'request';
  readonly min?: number;
  readonly max?: number;
  /** Default maximum the caller may override. */
  readonly defaultMax?: number;
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
    directArtifactRoot?: string;
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
    imageChoice?: Readonly<{status: 'implemented' | 'not-implemented'; limits?: readonly CatalogModelLimit[]}>;
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
  readonly upstream?: Readonly<{
    repository: string;
    revision: string;
    sourcePath: string;
    sourceBytes: number;
    sourceSha256: string;
    shardMapPath?: string;
    shardMapSha256?: string;
    rangeManifestPath?: string;
    rangeManifestSha256?: string;
  }>;
  readonly upstreams?: readonly Readonly<{
    role: string;
    repository: string;
    revision: string;
    sourcePath: string;
    sourceBytes: number;
    sourceSha256: string;
  }>[];
}
export interface ModelCatalogEntry {
  readonly modelId: ModelId;
  readonly description: CatalogModelDescription;
  readonly artifacts: ModelArtifactLock;
}
export declare const MODEL_IDS: ModelIdTuple;
export declare const MODEL_CATALOG: readonly ModelCatalogEntry[];
export declare const MODEL_CATALOG_BY_ID: Readonly<Record<ModelId, ModelCatalogEntry>>;
