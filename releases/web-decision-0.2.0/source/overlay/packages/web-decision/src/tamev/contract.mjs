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
