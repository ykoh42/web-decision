export const VERDICT_MODEL = 'experimental-openjev-verdict-151m-fp16';
export const VERDICT_REPOSITORY = 'heman10x/rlcd-modernbert-151m';
export const VERDICT_REVISION = '8af2496eb63c7fa66d7d234e1f62629380030eb4';
export const VERDICT_ASSETS_URL = `https://huggingface.co/${VERDICT_REPOSITORY}/resolve/${VERDICT_REVISION}/`;
export const VERDICT_SOURCE_REPOSITORY = 'Heman10x-NGU/Verdict-open-jev';
export const VERDICT_SOURCE_REVISION = '30f15564821626ca5c1ad5b2638c4eb7078787dd';

export const VERDICT_MAX_CANDIDATES = 24;
export const VERDICT_MAX_LABELS = 25;
export const VERDICT_MAX_TOKENS = 512;
export const VERDICT_ABSTENTION_ID = '__insufficient_evidence__';
export const VERDICT_ABSTENTION_DESCRIPTION = 'insufficient evidence';
export const VERDICT_GLOBAL_TEMPERATURE = 2.8039;
export const VERDICT_TEMPERATURES = Object.freeze({
  2: 5.0069,
  3: 5.0069,
  4: 4.0314,
  5: 3.056,
  6: 2.3898,
  7: 2.3898,
  9: 1.6668,
  11: 3.3919,
  17: 1.72,
  25: 1.5144
});

export const VERDICT_FILES = Object.freeze({
  'bundle_manifest.json': Object.freeze({
    bytes: 464,
    sha256: '3bc1c779f09b069b48f8f2179bb4f7478320ec7cab45c14c66597d6082e5a179'
  }),
  'calibrator.json': Object.freeze({
    bytes: 1259,
    sha256: 'af2a876993148efa0726b6ccf710fe2303897d20c0ce8c7c9036eb50f64d23de'
  }),
  'config.json': Object.freeze({
    bytes: 3836,
    sha256: '303f8eef1009cfdcb0cfba3e653247e625f16a4501e2351bad1a633f1f644695'
  }),
  'model_fp16.onnx': Object.freeze({
    bytes: 303785047,
    sha256: '4db28305590c714e33c2cacb75a57ec941c72bb210e1266ba2c7d77e85ddc526'
  }),
  'tokenizer.json': Object.freeze({
    bytes: 3583596,
    sha256: '8bb449eb0c037aae44115b65905bb339b8f3f74eb37067c19127feb3c0755723'
  }),
  'tokenizer_config.json': Object.freeze({
    bytes: 380,
    sha256: 'fb54f027372062b2ca52282efb04d178a8b57167a00cd8f4e816515823a2c016'
  })
});

export const VERDICT_TOTAL_BYTES = Object.values(VERDICT_FILES).reduce((total, file) => total + file.bytes, 0);

export const VERDICT_SPEC = Object.freeze({
  model: VERDICT_MODEL,
  repository: VERDICT_REPOSITORY,
  revision: VERDICT_REVISION,
  sourceRepository: VERDICT_SOURCE_REPOSITORY,
  sourceRevision: VERDICT_SOURCE_REVISION,
  files: VERDICT_FILES,
  manifest: Object.freeze({files: VERDICT_FILES}),
  totalBytes: VERDICT_TOTAL_BYTES,
  maxAssetBytes: Math.max(...Object.values(VERDICT_FILES).map((file) => file.bytes)),
  maxCandidates: VERDICT_MAX_CANDIDATES,
  maxLabels: VERDICT_MAX_LABELS,
  maxTokens: VERDICT_MAX_TOKENS,
  temperatures: VERDICT_TEMPERATURES,
  globalTemperature: VERDICT_GLOBAL_TEMPERATURE
});
