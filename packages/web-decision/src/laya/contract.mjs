export const LAYA_MODEL = 'experimental-laya-multilingual-fp16';
export const LAYA_REPOSITORY = 'onnx-community/laya-multilingual-ONNX';
export const LAYA_REVISION = '46b77bbf5642fec5f14e540570228a8cbe8ab81f';
const LAYA_SOURCE_REPOSITORY = 'convaiinnovations/laya-multilingual';
const LAYA_SOURCE_REVISION = '1720e3e3357cfe1e281542e223f8273b0890ca34';
export const LAYA_MAX_SEQUENCE_TOKENS = 1024;
export const LAYA_MAX_HEAD_TOKENS = 256;
export const LAYA_MAX_OPTION_TOKENS = 48;
// Each option needs a mask and at least one label token, plus eight instruction
// tokens and three special tokens in the 256-token head.
export const LAYA_MAX_OPTIONS = Math.floor((LAYA_MAX_HEAD_TOKENS - 11) / 2);
export const LAYA_MODEL_CONFIG = Object.freeze({
  modelType: 'modernbert',
  vocabSize: 256000,
  specialTokenIds: Object.freeze({sepId: 1, padId: 0, maskId: 4})
});

export const LAYA_FILES = Object.freeze({
  'config.json': Object.freeze({
    bytes: 2859,
    sha256: 'cc32109b1fded91ec6734127ffb1827f80cdcf8ffdd4eeda75a1b0f4d3687dcc'
  }),
  'tokenizer_config.json': Object.freeze({
    bytes: 502,
    sha256: '424b69444bf7b5809dc2cd2e36d0bd71b8055124dd24274d6db3c655d38205e7'
  }),
  'tokenizer.json': Object.freeze({
    bytes: 34363188,
    sha256: '609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f'
  }),
  'onnx/model_fp16.onnx': Object.freeze({
    bytes: 3572287,
    sha256: '81f641e21289ddb2cd8ac774c1cddebfdb0fa7e4b183e49ad476d2faee46d98a'
  }),
  'onnx/model_fp16.onnx_data': Object.freeze({
    bytes: 643817472,
    sha256: '3c9eb52f5f795f76d7a681197228131fac7e92c3e5a4c90b68ec4065699ee4cf'
  })
});

const LAYA_TOTAL_BYTES = Object.values(LAYA_FILES).reduce((sum, file) => sum + file.bytes, 0);
export const LAYA_SPEC = Object.freeze({
  model: LAYA_MODEL,
  repository: LAYA_REPOSITORY,
  revision: LAYA_REVISION,
  sourceRepository: LAYA_SOURCE_REPOSITORY,
  sourceRevision: LAYA_SOURCE_REVISION,
  modelConfig: LAYA_MODEL_CONFIG,
  files: LAYA_FILES,
  manifest: Object.freeze({files: LAYA_FILES}),
  totalBytes: LAYA_TOTAL_BYTES,
  maxAssetBytes: Math.max(...Object.values(LAYA_FILES).map((file) => file.bytes)),
  maxSequenceTokens: LAYA_MAX_SEQUENCE_TOKENS,
  maxHeadTokens: LAYA_MAX_HEAD_TOKENS,
  maxOptionTokens: LAYA_MAX_OPTION_TOKENS,
  maxOptions: LAYA_MAX_OPTIONS,
  typedQuestionTypes: Object.freeze(['choice', 'score', 'noul'])
});
