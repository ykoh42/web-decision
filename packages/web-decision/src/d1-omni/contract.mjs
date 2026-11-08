export const D1_OMNI_MODEL = 'experimental-d1-omni-600m-q8';
export const D1_OMNI_REPOSITORY = 'onnx-community/d1-omni-600M-ONNX';
export const D1_OMNI_REVISION = '4ebc1b97bf1477485371c79d1cf8d5e4e8eebfa0';
export const D1_OMNI_SOURCE_REPOSITORY = 'LiquidAI/d1-omni-600M';
export const D1_OMNI_SOURCE_REVISION = '02b55d7076f15129e59ab3f94783f32c4b088674';
export const D1_OMNI_MODEL_MAX_TOKENS = 16384;
// Keep browser inputs below the ONNX Runtime WebGPU shape overflow observed on M4 Pro.
export const D1_OMNI_MAX_TOKENS = 4096;

export const D1_OMNI_FILES = Object.freeze({
  'config.json': Object.freeze({
    bytes: 2433,
    sha256: '6402e50bd64e2d51c5f8dead2067d7873f30ec850f0cf1e732e4264b1576d313'
  }),
  'tokenizer_config.json': Object.freeze({
    bytes: 259,
    sha256: '03efc74b752b899b6292ff7e607738be59e6f206fec9aca32bbc05e9184090fc'
  }),
  'tokenizer.json': Object.freeze({
    bytes: 4733371,
    sha256: '1efc3a6609abf6b63b1f47188d139f3b59973a6a434dffe970a7261a51ed2711'
  }),
  'onnx/model_quantized.onnx': Object.freeze({
    bytes: 445782,
    sha256: 'e7b2d045b2efcdc38b9595aa5f00ec2c08098f997f136837c97ecad129834d74'
  }),
  'onnx/model_quantized.onnx_data': Object.freeze({
    bytes: 405883392,
    sha256: '9b84acb248daac2299df481240f5e6d2cd2f763aa1b8ccf73e7200848fab60da'
  })
});

export const D1_OMNI_TOTAL_BYTES = Object.values(D1_OMNI_FILES).reduce((sum, file) => sum + file.bytes, 0);
export const D1_OMNI_TEMPERATURES = Object.freeze({
  'choice:2': 1.7465145587921143,
  'choice:3-5': 1.3998981714248657,
  'choice:6-10': 1.1751071214675903,
  'choice:11+': 1.372515082359314,
  'noul:2': 1.6663223505020142,
  'score:3-5': 1.7301132678985596,
  'score:6-10': 1,
  choice: 1,
  score: 1,
  noul: 1
});

export const D1_OMNI_SPEC = Object.freeze({
  model: D1_OMNI_MODEL,
  repository: D1_OMNI_REPOSITORY,
  revision: D1_OMNI_REVISION,
  sourceRepository: D1_OMNI_SOURCE_REPOSITORY,
  sourceRevision: D1_OMNI_SOURCE_REVISION,
  files: D1_OMNI_FILES,
  manifest: Object.freeze({files: D1_OMNI_FILES}),
  totalBytes: D1_OMNI_TOTAL_BYTES,
  maxAssetBytes: Math.max(...Object.values(D1_OMNI_FILES).map((file) => file.bytes)),
  maxTokens: D1_OMNI_MODEL_MAX_TOKENS,
  temperatures: D1_OMNI_TEMPERATURES
});
