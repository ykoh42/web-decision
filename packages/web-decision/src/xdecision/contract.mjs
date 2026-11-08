export const XDECISION_MODEL = 'experimental-xdecision-fp16';
export const XDECISION_REPOSITORY = 'mccoysc/xDecision';
export const XDECISION_REVISION = '356461a7b9f163a179906ad0f3caff7fc160ce8d';
const XDECISION_SOURCE_REPOSITORY = 'mccoysc/xDecision';
export const XDECISION_MAX_SEQUENCE_TOKENS = 1024;
export const XDECISION_MAX_HEAD_TOKENS = 256;
export const XDECISION_MAX_OPTION_TOKENS = 48;
// Each option uses one marker and at least one label token; the head reserves 11 tokens.
export const XDECISION_MAX_OPTIONS = Math.floor((XDECISION_MAX_HEAD_TOKENS - 11) / 2);

export const XDECISION_FILES = Object.freeze({
  'config.json': Object.freeze({
    bytes: 2397,
    sha256: '49d5ab19a573590ff906029727de8590d7e7c0a08cabedda04407f724e4f92d6'
  }),
  'tokenizer_config.json': Object.freeze({
    bytes: 525,
    sha256: '30d70c8bf5bd2891248089fb1be43477e09a3a3deeb9f26e8dafb4d16ced64e0'
  }),
  'tokenizer.json': Object.freeze({
    bytes: 34363189,
    sha256: '0be5487f39c5ddc334beb659aa180e4d8a347293bc21a5d85df7ec1674cc4c8c'
  }),
  'onnx/model_fp16.onnx': Object.freeze({
    bytes: 2733734,
    sha256: 'b279f2a8217f03582ce7dd5fe047f652810091d96bad0504240aaa13ca468413'
  }),
  'onnx/model_fp16.onnx_data': Object.freeze({
    bytes: 643817472,
    sha256: 'cdf0f991a7473e6964d643729edb629fc130f800c1bba1eedf6b9dafad78a1ac'
  })
});

const XDECISION_TOTAL_BYTES = Object.values(XDECISION_FILES).reduce((sum, file) => sum + file.bytes, 0);
export const XDECISION_SPEC = Object.freeze({
  model: XDECISION_MODEL,
  repository: XDECISION_REPOSITORY,
  revision: XDECISION_REVISION,
  sourceRepository: XDECISION_SOURCE_REPOSITORY,
  sourceRevision: XDECISION_REVISION,
  files: XDECISION_FILES,
  manifest: Object.freeze({files: XDECISION_FILES}),
  totalBytes: XDECISION_TOTAL_BYTES,
  maxAssetBytes: Math.max(...Object.values(XDECISION_FILES).map((file) => file.bytes)),
  maxSequenceTokens: XDECISION_MAX_SEQUENCE_TOKENS,
  maxHeadTokens: XDECISION_MAX_HEAD_TOKENS,
  maxOptionTokens: XDECISION_MAX_OPTION_TOKENS,
  maxOptions: XDECISION_MAX_OPTIONS,
  typedQuestionTypes: Object.freeze(['choice', 'score', 'noul'])
});
