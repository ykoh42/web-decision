export const GLINER_MOBILE_MODEL = 'experimental-gliner2.5-decide-mobile-q4f16';
export const GLINER_MOBILE_REPOSITORY = 'onnx-community/GLiNER2.5-Decide-mobile-ONNX';
export const GLINER_MOBILE_REVISION = '880d1921d315b44648c9f6c6fb6ea0b87dea2b62';
const GLINER_Q4_REPOSITORY = 'onnx-community/GLiNER2.5-Decide-ONNX';
const GLINER_Q4_REVISION = '18271032588167fe9a3fbb8c5a4900b1231f8119';
export const GLINER_SOURCE_REPOSITORY = 'fastino/GLiNER2.5-Decide';
export const GLINER_SOURCE_REVISION = '9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90';
export const GLINER_MAX_TOKENS = 512;
export const GLINER_MAX_OPTIONS = 64;

export const GLINER_MOBILE_FILES = Object.freeze({
  'config.json': Object.freeze({
    bytes: 1617,
    sha256: 'fe7207771be1163b9837be6dab8799ceaa9cfd8bd39357f435c6ad813c757fcb'
  }),
  'special_tokens_map.json': Object.freeze({
    bytes: 2414,
    sha256: '84ea70143f533d7e99b393d87f20010887a9ac2cba955828ef313886e4e83f4f'
  }),
  'tokenizer.json': Object.freeze({
    bytes: 8333952,
    sha256: '3ad87d9ffe669147063e70850927dd2da90249e2acc5c8527f1eb65df467bcc8'
  }),
  'tokenizer_config.json': Object.freeze({
    bytes: 3360,
    sha256: 'ae050dc100bd157db8995b9604e4d5a9bef535f3170fdba43f167dd3126d2bf1'
  }),
  'onnx/model_q4f16.onnx': Object.freeze({
    bytes: 342105,
    sha256: '99a4e4c57fb15095a7e0aada14303d3d4f4ae0c3ce09c7b456a444bc2aff0dc2'
  }),
  'onnx/model_q4f16.onnx_data': Object.freeze({
    bytes: 344936496,
    sha256: '602b28f32ad602d84355de99deb3dcd549386a7ef3b8ab9c3bc4e9911e918974'
  })
});

export const GLINER_MOBILE_TOTAL_BYTES = Object.values(GLINER_MOBILE_FILES).reduce((sum, file) => sum + file.bytes, 0);

export const GLINER_MOBILE_SPEC = Object.freeze({
  model: GLINER_MOBILE_MODEL,
  repository: GLINER_MOBILE_REPOSITORY,
  revision: GLINER_MOBILE_REVISION,
  q4Repository: GLINER_Q4_REPOSITORY,
  q4Revision: GLINER_Q4_REVISION,
  sourceRepository: GLINER_SOURCE_REPOSITORY,
  sourceRevision: GLINER_SOURCE_REVISION,
  files: GLINER_MOBILE_FILES,
  manifest: Object.freeze({files: GLINER_MOBILE_FILES}),
  totalBytes: GLINER_MOBILE_TOTAL_BYTES,
  maxAssetBytes: Math.max(...Object.values(GLINER_MOBILE_FILES).map((file) => file.bytes)),
  maxTokens: GLINER_MAX_TOKENS,
  maxOptions: GLINER_MAX_OPTIONS,
  typedQuestionTypes: Object.freeze(['choice', 'score', 'noul'])
});
