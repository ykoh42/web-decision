export const TAMEV_NANO_MODEL = 'experimental-tamev-nano-tinybert-int8';
export const TAMEV_NANO_REPOSITORY = 'Tamkimd/tamev-nano-tinybert';
export const TAMEV_NANO_REVISION = 'ad15b45f35526070d48fb91f6597d2805a2d80af';
export const TAMEV_NANO_MAX_CONTEXT_TOKENS = 128;
export const TAMEV_NANO_MAX_OPTION_TOKENS = 48;
export const TAMEV_NANO_TEMPERATURE = 1.5;

export const TAMEV_NANO_FILES = Object.freeze({
  'model_int8.onnx': Object.freeze({
    bytes: 14872474,
    sha256: 'a83a4604a803df9efed89ed2e419184c2a1de995174ae6e8b3682de985c47484'
  }),
  'tokenizer.json': Object.freeze({
    bytes: 711396,
    sha256: 'd241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66'
  }),
  'tokenizer_config.json': Object.freeze({
    bytes: 379,
    sha256: '9f0c2c65a70ea18113ffa2e717103e7210ea826e7acc3442b69b346686b55a48'
  }),
  'config.json': Object.freeze({
    bytes: 796,
    sha256: '05fb476ed9eda56e1de767c33b067e70ed5b5c66d42437b1b8d3c3cdfe4c2307'
  })
});

export const TAMEV_NANO_TOTAL_BYTES = Object.values(TAMEV_NANO_FILES).reduce((sum, file) => sum + file.bytes, 0);
export const TAMEV_NANO_CALIBRATION_BY_K = Object.freeze({
  2: 0.8208,
  3: 1.094,
  4: 0.86,
  5: 1.2479,
  14: 1.7086,
  77: 1.5835
});

export const TAMEV_NANO_PROFILE = Object.freeze({
  name: 'TAMEV Nano TinyBERT',
  model: TAMEV_NANO_MODEL,
  repository: TAMEV_NANO_REPOSITORY,
  revision: TAMEV_NANO_REVISION,
  maxContextTokens: TAMEV_NANO_MAX_CONTEXT_TOKENS,
  maxOptionTokens: TAMEV_NANO_MAX_OPTION_TOKENS,
  temperature: TAMEV_NANO_TEMPERATURE,
  files: TAMEV_NANO_FILES,
  calibrationByK: TAMEV_NANO_CALIBRATION_BY_K,
  totalBytes: TAMEV_NANO_TOTAL_BYTES,
  expectedConfig: Object.freeze({
    model_type: 'tamev',
    backbone: 'huawei-noah/TinyBERT_General_4L_312D',
    architecture_type: 'encoder',
    hidden_dim: 312,
    projection_dim: 64,
    temperature: TAMEV_NANO_TEMPERATURE,
    pooling: 'cls',
    tier: 'nano',
    max_state_len: TAMEV_NANO_MAX_CONTEXT_TOKENS,
    max_opt_len: TAMEV_NANO_MAX_OPTION_TOKENS,
    parameters_m: 14.390185,
    released_parameters: 14390185,
    released_artifact: 'Fused model (backbone + cross-fusion pointer head)',
    option_fusion: 'cross'
  })
});

export const TAMEV_MICRO_MODEL = 'experimental-tamev-micro-minilm-int8';
export const TAMEV_MICRO_REPOSITORY = 'Tamkimd/tamev-micro-minilm';
export const TAMEV_MICRO_REVISION = 'c1b95679390911748aad57f634bed70e5f87a084';
export const TAMEV_MICRO_TEMPERATURE = 1.66;
export const TAMEV_MICRO_FILES = Object.freeze({
  'model_int8.onnx': Object.freeze({
    bytes: 23447976,
    sha256: '7f3ef06cf6c7eb677d9232ddd52decdb617616b6561413c375bbc1ae972240f0'
  }),
  'tokenizer.json': Object.freeze({
    bytes: 711661,
    sha256: 'da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0'
  }),
  'tokenizer_config.json': Object.freeze({
    bytes: 594,
    sha256: '872b6936be955bc3aea75ed599264d865626d68feede7e58b01e378e6332bd74'
  }),
  'config.json': Object.freeze({
    bytes: 772,
    sha256: 'cbd3c2c982c02cff3801c81cc84978f93cf757845aa918a10969d1e7b0f884d4'
  })
});
export const TAMEV_MICRO_TOTAL_BYTES = Object.values(TAMEV_MICRO_FILES).reduce((sum, file) => sum + file.bytes, 0);
export const TAMEV_MICRO_CALIBRATION_BY_K = Object.freeze({
  2: 0.9241,
  3: 1.1722,
  4: 0.9025,
  5: 1.3209,
  14: 1.1383,
  77: 2.7375
});
export const TAMEV_MICRO_PROFILE = Object.freeze({
  name: 'TAMEV Micro MiniLM',
  model: TAMEV_MICRO_MODEL,
  repository: TAMEV_MICRO_REPOSITORY,
  revision: TAMEV_MICRO_REVISION,
  maxContextTokens: TAMEV_NANO_MAX_CONTEXT_TOKENS,
  maxOptionTokens: TAMEV_NANO_MAX_OPTION_TOKENS,
  temperature: TAMEV_MICRO_TEMPERATURE,
  files: TAMEV_MICRO_FILES,
  calibrationByK: TAMEV_MICRO_CALIBRATION_BY_K,
  totalBytes: TAMEV_MICRO_TOTAL_BYTES,
  expectedConfig: Object.freeze({
    model_type: 'tamev',
    backbone: 'sentence-transformers/all-MiniLM-L6-v2',
    architecture_type: 'encoder',
    hidden_dim: 384,
    projection_dim: 64,
    temperature: TAMEV_MICRO_TEMPERATURE,
    pooling: 'mean',
    tier: 'micro',
    max_state_len: TAMEV_NANO_MAX_CONTEXT_TOKENS,
    max_opt_len: TAMEV_NANO_MAX_OPTION_TOKENS,
    parameters_m: 22.762369,
    released_parameters: 22762369,
    released_artifact: 'Fused model (backbone + bilinear pointer head)'
  })
});
