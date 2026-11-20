import type {CatalogModelDescription, ModelId} from './types.generated.js';

type PublicCatalogEntry = Readonly<{modelId: ModelId; description: CatalogModelDescription}>;

function deepFreeze<T>(value: T): Readonly<T> {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) deepFreeze(child);
  return Object.freeze(value);
}

const catalog = [
  {
    modelId: 'kanana-2-v-decide-3b-2611',
    description: {
      adapter: 'kanana',
      precision: 'BF16 source, FP16 stored text weights, FP32 text compute, FP32 vision weights',
      checkpoint: {
        repository: 'kakaocorp/kanana-2-v-decide-3b-2611',
        revision: '12820d654ed9789a0f7f48daeee1703c3483c464',
        weightSha256: '528930fa261b31ed1f47eecbe7d08efd2e226207508124ae186e1a65dbbbda3b'
      },
      assets: {
        integrity: 'pinned-manifest',
        location: 'package-and-pinned-upstream',
        additionalFiles:
          'The full native TVM text and vision graphs and their manifests ship in this package. Tokenizer/configuration files and only graph-required checkpoint ranges load from the revision-pinned Hugging Face repository.',
        requiredFiles: [
          'manifest.json',
          'kanana-text-model.wasm',
          'text-weight-manifest.json',
          'kanana-vision-model.wasm',
          'vision-weight-manifest.json',
          'safetensors-range-manifest.json'
        ],
        manifestSha256: 'd4e6d1c735b1e61c8e8c941730886e0914f6820d5cc9025fd4ef3de087689eb5'
      },
      licenses: {
        project: 'Apache-2.0',
        modelTerms: 'KananaOpenLicense',
        modelCard:
          'https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611/tree/12820d654ed9789a0f7f48daeee1703c3483c464'
      },
      backend: {
        name: 'tvm-webgpu',
        requiredFeatures: [],
        requiredLimits: {
          maxBufferSize: 656670720,
          maxStorageBufferBindingSize: 656670720,
          maxComputeWorkgroupStorageSize: 32768,
          maxStorageBuffersPerShaderStage: 6
        },
        requiresNonFallbackAdapter: false
      },
      features: {
        textChoice: {
          status: 'implemented',
          abstention: 'explicit-noul',
          limits: [
            {
              resource: 'tokens',
              scope: 'question',
              max: 20480,
              description: 'Pinned model configuration max_length; applies to each query and candidate prompt.'
            },
            {
              resource: 'questions',
              scope: 'request',
              min: 1
            }
          ],
          handling: 'query-and-candidate-embedding-similarity',
          tokenCount: 'sum-of-query-and-candidate-tokens',
          score: {
            basis: 'cosine-similarity',
            decisionTemperature: 0.056139365295102696,
            requestTemperature: 1,
            calibration: 'not-guaranteed'
          }
        },
        typedTextQuestionTypes: ['choice', 'noul', 'score'],
        imageChoice: {
          status: 'implemented',
          limits: [
            {
              resource: 'images',
              scope: 'request',
              max: 10,
              description: 'Pinned model-card maximum across images in state messages and the images field.'
            }
          ]
        },
        rating: {
          status: 'not-implemented'
        }
      },
      verification: {
        currentBundleGpu: 'verified-m4-pro-chromium-157',
        currentBundleCpuWasm: 'not-applicable',
        currentBundleJavascript: 'not-applicable',
        priorEvidence:
          'The latest exact-package candidate passed public text and image inference, offline cache reuse with zero network attempts, and the 20,480 maxLength boundary on Apple M4 Pro / Chromium 157.0.8103.52 using the non-fallback Apple Metal 3 WebGPU adapter. Full 32-layer CPU parity passed for sequence lengths 1 and 8, including two image tokens. The 20,480 boundary run completed in 5.2 minutes total test time and used 11,088,503,984 live WebGPU buffer bytes. Text score, probability, and confidence deltas against pinned BF16 were 0.004926, 0.001932, and 0.007389, within their recorded tolerances.'
      }
    }
  },
  {
    modelId: 'strands-decider-2b-qwen3.5-v1-2610',
    description: {
      adapter: 'strands-qwen35',
      precision: 'BF16 base checkpoint with FP32 Strands adapter, pointer head, and vision weights',
      checkpoint: {
        repository: 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610',
        revision: '216d1ccb46a9e9b036eb9187b248db98480f77d4'
      },
      assets: {
        integrity: 'pinned-manifest',
        location: 'package-and-pinned-upstream',
        additionalFiles:
          'Pinned native TVM text and vision WebGPU graphs and their manifests are packaged. Base, LoRA, and pointer-head weights are loaded from immutable upstream revisions by verified ranges.',
        requiredFiles: [
          'manifest.json',
          'pinned-word-assets-manifest.json',
          'source-range-inventory.json',
          'r19/manifest.json',
          'r19/fixture.json',
          'r19/weight-bindings.json',
          'r19/qwen35-text-pointer.webgpu.wasm',
          'vision/fixture.json',
          'vision/weight-bindings.json',
          'vision/qwen35-dynamic-vision.webgpu.wasm'
        ],
        manifestSha256: '6fba62427036712893a527264ed0c5f21ca16bf17644e30d827d2163320eb20e'
      },
      licenses: {
        project: 'Apache-2.0',
        modelTerms: 'Apache-2.0 as declared by the pinned model card',
        modelCard:
          'https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610/tree/216d1ccb46a9e9b036eb9187b248db98480f77d4'
      },
      backend: {
        name: 'tvm-webgpu',
        requiredFeatures: [],
        requiredLimits: {
          maxBufferSize: 1073741824,
          maxStorageBufferBindingSize: 1073741824,
          maxComputeWorkgroupStorageSize: 8192
        },
        requiresNonFallbackAdapter: false
      },
      features: {
        textChoice: {
          status: 'implemented',
          abstention: 'not-implemented',
          limits: [
            {
              resource: 'tokens',
              scope: 'question',
              max: 4096,
              description: 'Pinned Qwen3.5 graph context length; the Strands request fitter applies source behavior.'
            },
            {
              resource: 'options',
              scope: 'question',
              min: 2,
              max: 255
            },
            {
              resource: 'options',
              scope: 'question',
              min: 2,
              max: 10
            },
            {
              resource: 'questions',
              scope: 'request',
              min: 1
            }
          ],
          handling: 'source-faithful-packed-requests',
          tokenCount: 'source-input-token-usage',
          score: {
            basis: 'ordered-levels',
            temperatureAppliedBy: 'decoder',
            calibration: 'not-guaranteed'
          }
        },
        typedTextQuestionTypes: ['choice', 'noul', 'score'],
        imageChoice: {
          status: 'implemented',
          limits: [
            {
              resource: 'images',
              scope: 'request',
              defaultMax: 4,
              description:
                "Defaults to 4 images per request; systemOne's maxImages option can set a different limit. The PIL processor has no separate fixed image-count ceiling."
            }
          ]
        },
        rating: {
          status: 'not-implemented'
        }
      },
      verification: {
        currentBundleGpu: 'see-release-evidence',
        priorBundleGpu: 'verified-apple-m4-pro-chromium-158.0.8127.18',
        currentBundleCpuWasm: 'not-applicable',
        currentBundleJavascript: 'not-applicable',
        priorEvidence:
          'The private 0.0.0-qwen35-candidate npm archive passed TVM WebGPU text and image parity, six accepted and three rejected boundary cases, a three-row batch at 4,096 tokens, and six offline cases with zero network attempts. Online model assets were served from a local hash-verified Hugging Face snapshot.'
      }
    }
  }
] as unknown as readonly PublicCatalogEntry[];
export const APPROVED_MODEL_IDS = Object.freeze(catalog.map(({modelId}) => modelId));
export const MODEL_CATALOG = deepFreeze(catalog);
export const MODEL_CATALOG_BY_ID = Object.freeze(
  Object.fromEntries(MODEL_CATALOG.map((entry) => [entry.modelId, entry])) as Record<ModelId, PublicCatalogEntry>
);
