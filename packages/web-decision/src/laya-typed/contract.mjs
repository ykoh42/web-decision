export const LAYA_TYPED_MODEL = 'experimental-laya-typed-decisions-fp16';
export const LAYA_TYPED_REPOSITORY = 'onnx-community/laya-typed-decisions-ONNX';
export const LAYA_TYPED_REVISION = '79c441b22da79eb3727bb01e8452e29a32cfa548';
const LAYA_TYPED_SOURCE_REPOSITORY = 'convaiinnovations/laya-typed-decisions';
const LAYA_TYPED_SOURCE_REVISION = 'e929ae5cf69bc34259cd2f95c9e91145b818b1f0';
const LAYA_TYPED_MODEL_CONFIG = Object.freeze({
  modelType: 'modernbert',
  vocabSize: 50368,
  specialTokenIds: Object.freeze({clsId: 50281, sepId: 50282, padId: 50283, maskId: 50284})
});
export const LAYA_TYPED_MAX_SEQUENCE_TOKENS = 1024;
const LAYA_TYPED_MAX_HEAD_TOKENS = 256;
const LAYA_TYPED_MAX_OPTION_TOKENS = 48;
export const LAYA_TYPED_MAX_OPTIONS = Math.floor((LAYA_TYPED_MAX_HEAD_TOKENS - 11) / 2);

export const LAYA_TYPED_FILES = Object.freeze({
  'config.json': Object.freeze({
    bytes: 3300,
    sha256: '17276af4594edd2528024d1a23de7905e789f8befe659a47e337aa4ce144a226'
  }),
  'tokenizer_config.json': Object.freeze({
    bytes: 337,
    sha256: '08d4cf3ac4dca381759441b85b91a6d40e688471dcd33d15d6649eb0a9a854d1'
  }),
  'tokenizer.json': Object.freeze({
    bytes: 3583228,
    sha256: '6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30'
  }),
  'onnx/model_fp16.onnx': Object.freeze({
    bytes: 4466054,
    sha256: '94c25ed9a1ebe3f13b85bdbd54cd20ed407fb1993e3993f3deda1bc5662b8ae1'
  }),
  'onnx/model_fp16.onnx_data': Object.freeze({
    bytes: 842587136,
    sha256: '2565dd3a00465c5c615502894176cdb96a528ad66256dd8d08eb0af3a0f73582'
  })
});

const LAYA_TYPED_TOTAL_BYTES = Object.values(LAYA_TYPED_FILES).reduce((sum, file) => sum + file.bytes, 0);
export const LAYA_TYPED_SPEC = Object.freeze({
  model: LAYA_TYPED_MODEL,
  repository: LAYA_TYPED_REPOSITORY,
  revision: LAYA_TYPED_REVISION,
  sourceRepository: LAYA_TYPED_SOURCE_REPOSITORY,
  sourceRevision: LAYA_TYPED_SOURCE_REVISION,
  modelConfig: LAYA_TYPED_MODEL_CONFIG,
  files: LAYA_TYPED_FILES,
  manifest: Object.freeze({files: LAYA_TYPED_FILES}),
  totalBytes: LAYA_TYPED_TOTAL_BYTES,
  maxAssetBytes: Math.max(...Object.values(LAYA_TYPED_FILES).map((file) => file.bytes)),
  maxSequenceTokens: LAYA_TYPED_MAX_SEQUENCE_TOKENS,
  maxHeadTokens: LAYA_TYPED_MAX_HEAD_TOKENS,
  maxOptionTokens: LAYA_TYPED_MAX_OPTION_TOKENS,
  maxOptions: LAYA_TYPED_MAX_OPTIONS,
  typedQuestionTypes: Object.freeze(['choice', 'score', 'noul'])
});
