export const SKEJJA_INT8_MODEL = 'experimental-skejja-edge32k-ft6-int8';
export const SKEJJA_INT8DYN_MODEL = 'experimental-skejja-edge32k-ft6-int8dyn';
export const SKEJJA_REPOSITORY = 'vincyber/skejja-edge32k-ft6';
export const SKEJJA_REVISION = 'a076ad7d698b11b15dc70fe067d15c3d1140761f';
export const SKEJJA_ARCHIVE = Object.freeze({
  filename: 'skejja-edge32k-ft6_onnx-int8.tar.gz',
  bytes: 208260729,
  sha256: '751e243f1fba39421d36d44a832fc42a34bb74c6bec68f15d956e219e1131f42',
  root: 'skejja-edge32k-ft6_onnx-int8',
  maxExpandedBytes: 229000000,
  files: Object.freeze({
    LICENSE: 11357,
    NOTICE: 3636,
    'README.md': 1007,
    'calibration.json': 1329,
    'edge-v32k-ft6.int8.onnx': 571804,
    'edge-v32k-ft6.int8.onnx.data': 100674628,
    'edge-v32k-ft6.int8dyn.onnx': 1308215,
    'edge-v32k-ft6.int8dyn.onnx.data': 124288004,
    'export_check.json': 643,
    'skejja_infer.py': 9468,
    'special_tokens_map.json': 1051,
    'tokenizer.json': 1891156,
    'tokenizer_config.json': 2235
  })
});
export const SKEJJA_MAX_TOKENS = 512;
export const SKEJJA_MIN_BODY_TOKENS = 32;
export const SKEJJA_MAX_CHOICES = 255;
export const SKEJJA_MAX_SCORE_LEVELS = 10;
export const SKEJJA_MIN_CALIBRATION_EXAMPLES = 8;
export const SKEJJA_MAX_CALIBRATION_EXAMPLES = 256;
export const SKEJJA_FEWSHOT_SIGMA = 0.5;

const sharedFiles = Object.freeze({
  'calibration.json': Object.freeze({
    bytes: 1329,
    sha256: 'ec037b84efa566f11021a0720933cc71ef3ece6ae92e3cd74e3af9fa85116017'
  }),
  'tokenizer.json': Object.freeze({
    bytes: 1891156,
    sha256: '93d257dc6b334864a7cf1269c8663035697bd6e94fbd6800135c6d4e8a7b60e0'
  }),
  'tokenizer_config.json': Object.freeze({
    bytes: 2235,
    sha256: '5ae827176f62d4f8b9cd4fa0e69bc65ae1998e478aad8c8dff0b5d5b7c79ad62'
  })
});

const int8Files = Object.freeze({
  ...sharedFiles,
  'edge-v32k-ft6.int8.onnx': Object.freeze({
    bytes: 571804,
    sha256: '53b77f87f063310bf8dc19b224f5ed0d537ee26d91063e80b19913e0c6ed84a3'
  }),
  'edge-v32k-ft6.int8.onnx.data': Object.freeze({
    bytes: 100674628,
    sha256: '5dd704befc7a557f8d8e6a529fbd196925f4bcf7197e4317340f8fbe63ee29e1'
  })
});

const int8dynFiles = Object.freeze({
  ...sharedFiles,
  'edge-v32k-ft6.int8dyn.onnx': Object.freeze({
    bytes: 1308215,
    sha256: '8e6f805ed89c99d5aeea1978e12640587ceb50405a86118caa25c96d04a8c5bd'
  }),
  'edge-v32k-ft6.int8dyn.onnx.data': Object.freeze({
    bytes: 124288004,
    sha256: '7f51d86944b37ace626a93bab81dc8134e49ca1556a3057535bcc0d85b126108'
  })
});

export const SKEJJA_INT8_PROFILE = Object.freeze({
  name: 'Skejja edge32k-ft6 weight-only int8',
  model: SKEJJA_INT8_MODEL,
  repository: SKEJJA_REPOSITORY,
  revision: SKEJJA_REVISION,
  onnxFile: 'edge-v32k-ft6.int8.onnx',
  dataFile: 'edge-v32k-ft6.int8.onnx.data',
  temperature: 1.061926311530685,
  plattA: 0.8450836171439912,
  plattB: 0.40176501739057835,
  maxChoices: SKEJJA_MAX_CHOICES,
  maxScoreLevels: SKEJJA_MAX_SCORE_LEVELS,
  files: int8Files,
  totalBytes: Object.values(int8Files).reduce((sum, file) => sum + file.bytes, 0)
});

export const SKEJJA_INT8DYN_PROFILE = Object.freeze({
  name: 'Skejja edge32k-ft6 dynamic int8',
  model: SKEJJA_INT8DYN_MODEL,
  repository: SKEJJA_REPOSITORY,
  revision: SKEJJA_REVISION,
  onnxFile: 'edge-v32k-ft6.int8dyn.onnx',
  dataFile: 'edge-v32k-ft6.int8dyn.onnx.data',
  temperature: 1.0416030369517624,
  plattA: 0.841808613755462,
  plattB: 0.4707840185537924,
  maxChoices: SKEJJA_MAX_CHOICES,
  maxScoreLevels: SKEJJA_MAX_SCORE_LEVELS,
  files: int8dynFiles,
  totalBytes: Object.values(int8dynFiles).reduce((sum, file) => sum + file.bytes, 0)
});

export const SKEJJA_PROFILES = Object.freeze({
  [SKEJJA_INT8_MODEL]: SKEJJA_INT8_PROFILE,
  [SKEJJA_INT8DYN_MODEL]: SKEJJA_INT8DYN_PROFILE
});
