export const SAFEDECIDE_MODEL = 'experimental-safedecide-140m-fp32';
export const SAFEDECIDE_REPOSITORY = 'yotambr/safedecide-140m';
export const SAFEDECIDE_REVISION = '1934a00e6ba5242668c3f1b06fa39fb438741207';
export const SAFEDECIDE_SOURCE_REVISION = '2dea76fcdf626367e5c5d87480886576d9f87afd';
export const SAFEDECIDE_MAX_SEQUENCE_TOKENS = 640;
export const SAFEDECIDE_BATCH_SIZE = 8;
const SAFEDECIDE_MAX_ASSET_BYTES = 563_099_019;

export const SAFEDECIDE_MODEL_CONFIG = Object.freeze({
  modelType: 'modernbert',
  vocabSize: 256000,
  hiddenSize: 384,
  layers: 22,
  specialTokenIds: Object.freeze({bosId: 2, eosId: 1, padId: 0, maskId: 4})
});

export const SAFEDECIDE_FILES = Object.freeze({
  'config.json': Object.freeze({
    bytes: 1995,
    sha256: '7332fae3daf6e9b850f896f418c4fbafa2a97cd10c5c51f3b5d180508d639dfd'
  }),
  'safedecide_config.json': Object.freeze({
    bytes: 2520,
    sha256: 'b28efc28f7bd96933f7dfb67d798d63c573ac36a502acc704b7126d9d88a1855'
  }),
  'tokenizer.json': Object.freeze({
    bytes: 34363441,
    sha256: 'e6b5466e8c0b706f6fed81edcd7df56f01fa2f423a59306552962c1dd1b392cd'
  }),
  'tokenizer_config.json': Object.freeze({
    bytes: 737,
    sha256: 'f0a2c5ac1e8d92a5d1afa18fa66ceae6a8464976b5c83bb15423b17211f0cdb1'
  }),
  'onnx/model.onnx': Object.freeze({
    bytes: 563099019,
    sha256: 'e987d2e4c8da29b8366d2c914b93483ffe661f30593b966202051ee86811eacc'
  })
});

export const SAFEDECIDE_TOTAL_BYTES = Object.values(SAFEDECIDE_FILES).reduce((sum, file) => sum + file.bytes, 0);
export const SAFEDECIDE_SPEC = Object.freeze({
  model: SAFEDECIDE_MODEL,
  repository: SAFEDECIDE_REPOSITORY,
  revision: SAFEDECIDE_REVISION,
  sourceRevision: SAFEDECIDE_SOURCE_REVISION,
  modelConfig: SAFEDECIDE_MODEL_CONFIG,
  files: SAFEDECIDE_FILES,
  manifest: Object.freeze({files: SAFEDECIDE_FILES}),
  totalBytes: SAFEDECIDE_TOTAL_BYTES,
  maxAssetBytes: SAFEDECIDE_MAX_ASSET_BYTES,
  maxSequenceTokens: SAFEDECIDE_MAX_SEQUENCE_TOKENS,
  typedQuestionTypes: Object.freeze(['choice', 'score', 'noul'])
});
