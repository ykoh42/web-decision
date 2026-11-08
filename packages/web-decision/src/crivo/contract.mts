export const CRIVO_MODEL_FP32 = 'experimental-crivo-32m-v2-fp32';
export const CRIVO_MODEL_INT8 = 'experimental-crivo-32m-v2-int8';
export const CRIVO_REPOSITORY = 'dizyann/crivo-web';
export const CRIVO_REVISION = 'fb3cf4bc66c06e51e08950399653e54a4ddeb154';
export const CRIVO_SOURCE_REPOSITORY = 'tardellirs/crivo-32m';
export const CRIVO_SOURCE_REVISION = '77d61fdb6421838320097d9bcb3e4e42a120323e';
export const CRIVO_MAX_SEQUENCE_TOKENS = 2048;
export const CRIVO_BATCH_SIZE = 32;
export const CRIVO_BATCH_TOKEN_BUDGET = 65_536;
export const CRIVO_PAD_MULTIPLE = 16;
export const CRIVO_OPT_TOKEN_ID = 50368;

interface CrivoFileMetadata {
  readonly bytes: number;
  readonly sha256: string;
}

type CrivoBackend = 'onnxruntime-web-webgpu' | 'onnxruntime-web-wasm';

export interface CrivoSpec {
  readonly model: string;
  readonly precision: 'f32' | 'int8';
  readonly backend: CrivoBackend;
  readonly graphFile: string;
  readonly files: Readonly<Record<string, CrivoFileMetadata>>;
  readonly totalBytes: number;
}

const COMMON_FILES: Readonly<Record<string, CrivoFileMetadata>> = Object.freeze({
  'config.json': Object.freeze({
    bytes: 4360,
    sha256: 'dbd358b1345c10df6411c11953971322b6bdf3fce8e87aeb8e3e3c40183865e7'
  }),
  'tokenizer.json': Object.freeze({
    bytes: 3583508,
    sha256: '858fe1cd418807d6c71c01e24b6d3239a224a1e3c995472fef20841e09eab056'
  }),
  'tokenizer_config.json': Object.freeze({
    bytes: 535,
    sha256: '3243894718a6c7ca8b779837cdf2aaae5ddbf96082e31cc949bd9182a04b6d94'
  })
});

function spec(
  model: string,
  precision: 'f32' | 'int8',
  backend: CrivoBackend,
  graphFile: string,
  graphBytes: number,
  graphSha256: string
): CrivoSpec {
  const files = Object.freeze({
    ...COMMON_FILES,
    [graphFile]: Object.freeze({bytes: graphBytes, sha256: graphSha256})
  });
  return Object.freeze({
    model,
    precision,
    backend,
    graphFile,
    files,
    totalBytes: Object.values(files).reduce((total, file) => total + file.bytes, 0)
  });
}

export const CRIVO_FP32_SPEC = spec(
  CRIVO_MODEL_FP32,
  'f32',
  'onnxruntime-web-webgpu',
  'onnx/model.onnx',
  128385184,
  '6e69d12727d4d566fc70d423b44f9295f0dc7da3512f32e0c71561aacae31b5a'
);

export const CRIVO_INT8_SPEC = spec(
  CRIVO_MODEL_INT8,
  'int8',
  'onnxruntime-web-wasm',
  'onnx/model_quantized.onnx',
  32729616,
  '2bd89b7d5a59ad6920971188862f360d1f2b80362f0539b0c62b4af7f67e4fa4'
);

export function getCrivoSpec(model: unknown): CrivoSpec | undefined {
  if (model === CRIVO_MODEL_FP32) return CRIVO_FP32_SPEC;
  if (model === CRIVO_MODEL_INT8) return CRIVO_INT8_SPEC;
  return undefined;
}
