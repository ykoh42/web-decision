export const MAPIKA2B_MODEL_ID = 'mapika-decider-2b-v12';
export const MAPIKA2B_REPOSITORY = 'Mapika/decider-2b';
export const MAPIKA2B_REVISION = '7614a6c7ea5a047aa7d7bcfb1a0015469381c3e3';
export const MAPIKA2B_DECIDER_REVISION = 'ab45f9147de7a2da974f68c77cc8ed60908b379c';

export const MAPIKA2B_CONFIG_SHA256 = 'bbe0f382362b23a5ab9115e23d01a1b0511bf334c9355092a3b4d2ea0ddb5e1d';
export const MAPIKA2B_TOKENIZER_SHA256 = '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523';
export const MAPIKA2B_TOKENIZER_CONFIG_SHA256 = '171ecbe7ddae98d11840698f7df2b8d5b4722139db0f0620d3bbf429bd656250';

export const MAPIKA2B_VOCAB_SIZE = 248_320;
export const MAPIKA2B_MAX_OPTIONS = 255;
export const MAPIKA2B_MAX_SCORE_LEVELS = 10;
export const MAPIKA2B_BASE_TEMPERATURE = 2.515;
export const MAPIKA2B_TEMPERATURES = Object.freeze({choice: 2.465, noul: 0.05, score: 2.833});
export const MAPIKA2B_LIMITS = Object.freeze({
  maxStateTokens: 32_768,
  maxRowTokens: 36_864,
  maxRequestTokens: 1_048_576,
  maxRows: 1_024,
  maxImages: 4
});

export const MAPIKA2B_VISION = Object.freeze({
  startToken: 248_053,
  endToken: 248_054,
  imageToken: 248_056,
  patchSize: 16,
  mergeSize: 2,
  maxPixels: 768 * 768
});

export type Mapika2bQuestionKind = keyof typeof MAPIKA2B_TEMPERATURES;
export type Mapika2bTemperatures = Readonly<Record<Mapika2bQuestionKind, number>>;

export interface Mapika2bTemperatureOptions {
  readonly temperature?: number | undefined;
  readonly temperatureByType?: Partial<Mapika2bTemperatures> | undefined;
}

export interface Mapika2bLimits {
  readonly maxStateTokens: number;
  readonly maxRowTokens: number;
  readonly maxRequestTokens: number;
  readonly maxRows: number;
  readonly maxImages: number;
}

export interface Mapika2bTokenizer {
  encode(text: string, options: Readonly<{add_special_tokens: false}>): {readonly ids: Iterable<number>};
}

export interface Mapika2bLabels {
  readonly names: readonly string[];
  readonly ids: readonly number[];
  readonly open: readonly number[];
}

export interface Mapika2bImageAsset {
  readonly reference: string;
  readonly format: 'png' | 'jpg' | 'jpeg';
  readonly data: Uint8Array;
}

export interface Mapika2bSlot {
  readonly tokenIndex: number;
  readonly optionCount: number;
  readonly temperature: number;
}

export interface Mapika2bRow {
  readonly inputIds: readonly number[];
  readonly slots: readonly Mapika2bSlot[];
}

export type Mapika2bQuestion = Readonly<{name: string; firstSlot: number}> &
  (
    | Readonly<{kind: 'choice'; names: readonly string[]}>
    | Readonly<{kind: 'noul'}>
    | Readonly<{kind: 'score'; legend: readonly string[]; isolated: boolean}>
  );

export interface Mapika2bPlan {
  readonly rows: readonly Mapika2bRow[];
  readonly questions: readonly Mapika2bQuestion[];
  readonly images: readonly Mapika2bImageAsset[];
  readonly stateTokenCount: number;
  readonly inputTokenCount: number;
  readonly imagePrefixLength: number;
  readonly limits: Mapika2bLimits;
}
