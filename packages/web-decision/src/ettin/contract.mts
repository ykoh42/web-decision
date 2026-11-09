import type {AssetSourceFile, AssetSourceSpec} from '../asset-source-types.js';

export const ETTIN_MODEL = 'experimental-ettin-68m-int8';
const ETTIN_REPOSITORY = 'aravind1706/decider-ettin';
export const ETTIN_REVISION = 'df8600e1f6e220b302cdd49897c2b1ff76e021ae';
export const ETTIN_MAX_SEQUENCE_TOKENS = 2048;
export const ETTIN_MAX_OPTIONS = 10;
export const ETTIN_MAX_SCORE_LEVELS = 7;

export interface EttinMarkerIds {
  readonly decision: number;
  readonly state: number;
  readonly question: number;
  readonly options: readonly number[];
}

export interface EttinProfile extends AssetSourceSpec {
  readonly model: typeof ETTIN_MODEL;
  readonly repository: string;
  readonly revision: string;
  readonly markerIds: EttinMarkerIds;
  readonly maxSequenceTokens: number;
  readonly maxOptions: number;
  readonly maxScoreLevels: number;
}

export const ETTIN_MARKER_IDS: EttinMarkerIds = Object.freeze({
  decision: 50368,
  state: 50369,
  question: 50370,
  options: Object.freeze(Array.from({length: ETTIN_MAX_OPTIONS}, (_, index) => 50371 + index))
});

export const ETTIN_FILES: Readonly<Record<string, AssetSourceFile>> = Object.freeze({
  'config.json': Object.freeze({
    bytes: 1081,
    sha256: 'a397404e5866e1bf20d213df4b14ef3082f883528a0bb078066c8e8093db758f'
  }),
  'manifest.json': Object.freeze({
    bytes: 293,
    sha256: 'b7072ab7aeb1c934b4098ed5974eae73270279f0c160f22acb18a5373e5d2a82'
  }),
  'model_int8.onnx': Object.freeze({
    bytes: 70_612_289,
    sha256: 'cfd0184cf450c7ceeaab10ef1a7069bd8924ba46b0d9f73fd13a6a51d9616bd1'
  }),
  'tokenizer.json': Object.freeze({
    bytes: 3_585_871,
    sha256: '3d64ed9d2fb01d2bfaa6aed63d07863a3af61882ccc6ef1a962550c20406bccd'
  }),
  'tokenizer_config.json': Object.freeze({
    bytes: 613,
    sha256: 'd12e0a1ecc8ba4d17aa0878f16347974662d163a7d92983dc4963332da686304'
  })
});

export const ETTIN_TOTAL_BYTES = Object.values(ETTIN_FILES).reduce((total, file) => total + file.bytes, 0);

export const ETTIN_PROFILE: EttinProfile = Object.freeze({
  model: ETTIN_MODEL,
  repository: ETTIN_REPOSITORY,
  revision: ETTIN_REVISION,
  files: ETTIN_FILES,
  markerIds: ETTIN_MARKER_IDS,
  maxSequenceTokens: ETTIN_MAX_SEQUENCE_TOKENS,
  maxOptions: ETTIN_MAX_OPTIONS,
  maxScoreLevels: ETTIN_MAX_SCORE_LEVELS,
  totalBytes: ETTIN_TOTAL_BYTES
});
