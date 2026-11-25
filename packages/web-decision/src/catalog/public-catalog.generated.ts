import type {CatalogModelDescription} from './types.generated.js';
import type {PublicModelId as ModelId} from '../curated/public-model-id.generated.js';

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
  },
  {
    modelId: 'mapika-decider-2b-v12',
    description: {
      adapter: 'mapika2b',
      precision: 'BF16 source weights and FP32 text and vision compute',
      checkpoint: {
        repository: 'Mapika/decider-2b',
        revision: '7614a6c7ea5a047aa7d7bcfb1a0015469381c3e3'
      },
      assets: {
        integrity: 'pinned-manifest',
        location: 'package-and-pinned-upstream',
        additionalFiles:
          'The pinned TVM graphs are packaged. Native text and vision checkpoint ranges load from the immutable upstream revision.',
        requiredFiles: [
          'manifest.json',
          'text.wasm',
          'text-bindings.json',
          'source-inventory.json',
          'vision/fixture.json',
          'vision/weight-bindings.json',
          'vision/qwen35-dynamic-vision.webgpu.wasm'
        ],
        manifestSha256: '7f9d47b14426d7f7dea6cf9955e1d40f9a8a89fb6f564da1b39403a6a831d93a'
      },
      licenses: {
        project: 'Apache-2.0',
        modelTerms: 'Apache-2.0 as declared by the pinned Mapika model card and Qwen base license',
        modelCard: 'https://huggingface.co/Mapika/decider-2b/tree/7614a6c7ea5a047aa7d7bcfb1a0015469381c3e3'
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
              scope: 'state',
              defaultMax: 32768,
              description: 'Default state-token limit can be configured.'
            },
            {
              resource: 'options',
              scope: 'question',
              min: 2,
              max: 255
            }
          ],
          handling: 'source-faithful-system-one',
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
              description: 'Default image limit can be configured.'
            }
          ]
        },
        rating: {
          status: 'not-implemented'
        }
      },
      verification: {
        currentBundleGpu: 'see-release-evidence',
        currentBundleCpuWasm: 'not-applicable',
        currentBundleJavascript: 'not-applicable',
        priorEvidence:
          'On M4 Pro Chromium WebGPU, seven text and image cases matched the pinned FP32 reference. A 32,904-token smoke test completed. Long-input latency and device requirements vary.'
      }
    }
  },
  {
    modelId: 'strands-decider-e2b-gemma4-v2-2611',
    description: {
      adapter: 'strands-gemma4-v2',
      precision: 'BF16 source checkpoint with FP32 TVM WebGPU computation and FP32 readout',
      checkpoint: {
        repository: 'StrandsAgents/strands-decider-E2B-gemma4-v2-2611',
        revision: 'a5f32ea1bffe2e91c25c0c641d444e8115e28212',
        weightSha256: '593a28a67754653c7ffffff80a5186f18966cc6262c2f4f097414c187e5fd48c'
      },
      assets: {
        integrity: 'pinned-manifest',
        location: 'package-and-pinned-upstream',
        additionalFiles:
          'Direct TVM WebGPU graphs, scoring data, and binding inventories are bundled. Verified ranges of the BF16 checkpoint, readout, tokenizer, and configuration load from the pinned upstream revision.',
        requiredFiles: [
          'manifest.json',
          'graph.wasm',
          'local.bin',
          'global.bin',
          'scoring.bin',
          'source-inventory.json',
          'decoder-bindings.json',
          'vision-bindings.json'
        ],
        manifestSha256: 'f55c9fe5d530e591e10dec16301b4b2c724605b49543dc688c7bc0ca8ca28b7d'
      },
      licenses: {
        project: 'Apache-2.0',
        modelTerms: 'Apache-2.0',
        modelCard:
          'https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v2-2611/tree/a5f32ea1bffe2e91c25c0c641d444e8115e28212'
      },
      backend: {
        name: 'tvm-webgpu',
        requiredFeatures: [],
        requiredLimits: {
          maxBufferSize: 201326592,
          maxStorageBufferBindingSize: 201326592
        },
        requiresNonFallbackAdapter: true
      },
      features: {
        textChoice: {
          status: 'implemented',
          abstention: 'explicit-noul',
          limits: [
            {
              resource: 'tokens',
              scope: 'question',
              defaultMax: 32768,
              description:
                'State, image, and question tokens count together for each question. Subject to the compiled graph and device allocation limits.'
            },
            {
              resource: 'questions',
              scope: 'request',
              min: 1
            }
          ],
          handling: 'state-and-question-decoder-readout',
          tokenCount: 'state-and-image-tokens-plus-sum-of-question-tokens',
          score: {
            basis: 'strands-readout',
            calibration: 'not-guaranteed'
          }
        },
        typedTextQuestionTypes: ['choice', 'noul', 'score'],
        imageChoice: {
          status: 'implemented'
        }
      },
      verification: {
        currentBundleGpu: 'see-release-evidence',
        currentBundleCpuWasm: 'not-applicable',
        currentBundleJavascript: 'not-applicable'
      }
    }
  },
  {
    modelId: 'strands-decider-e2b-gemma4-v1-2610',
    description: {
      adapter: 'strands-gemma4-v1',
      precision: 'BF16 Gemma 4 E2B base with FP32 Strands adapter, pointer head, and TVM WebGPU computation',
      checkpoint: {
        repository: 'StrandsAgents/strands-decider-E2B-gemma4-v1-2610',
        revision: 'd383ecf8534b13d49b3b567a1b66f88fca6d50ba',
        weightSha256: 'c7786a59354aa190082b1c3aedb7ff3311fe260f070f360c4d3518762691c551'
      },
      assets: {
        integrity: 'pinned-manifest',
        location: 'package-and-pinned-upstream',
        additionalFiles:
          'The compiled graph, range inventory and tokenizer are bundled. The base checkpoint and adapter load from their pinned upstream revisions.',
        requiredFiles: [
          'manifest.json',
          'graph.wasm',
          'graph.json',
          'source-inventory.json',
          'tokenizer.json',
          'tokenizer_config.json'
        ],
        manifestSha256: '49f4a24431d8a9a0f317308c4e97268bc0703387a7e7a291e948dbfe7950feb9'
      },
      licenses: {
        project: 'Apache-2.0',
        modelTerms: 'Apache-2.0',
        modelCard:
          'https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v1-2610/tree/d383ecf8534b13d49b3b567a1b66f88fca6d50ba',
        baseModelCard: 'https://huggingface.co/google/gemma-4-E2B-it/tree/3e22461f65e89153144f8adb70e3b8c2cc9845a7'
      },
      backend: {
        name: 'tvm-webgpu',
        requiredFeatures: [],
        requiredLimits: {
          maxBufferSize: 75497472,
          maxStorageBufferBindingSize: 75497472
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
              defaultMax: 4096,
              max: 131072,
              description: 'Includes state and question tokens. Longer configured windows are subject to device limits.'
            },
            {
              resource: 'options',
              scope: 'question',
              max: 255,
              description: 'Maximum choices on a choice question.'
            },
            {
              resource: 'questions',
              scope: 'request',
              min: 1
            }
          ],
          handling: 'causal-state-and-question-pointer',
          tokenCount: 'state-tokens-per-prefix-evaluation-plus-sum-of-question-tokens',
          score: {
            basis: 'strands-pointer',
            calibration: 'not-guaranteed'
          }
        },
        typedTextQuestionTypes: ['choice', 'noul', 'score'],
        imageChoice: {
          status: 'not-implemented'
        }
      },
      verification: {
        currentBundleGpu: 'see-release-evidence',
        currentBundleCpuWasm: 'not-applicable',
        currentBundleJavascript: 'not-applicable'
      }
    }
  }
] as unknown as readonly PublicCatalogEntry[];
export const APPROVED_MODEL_IDS = Object.freeze(catalog.map(({modelId}) => modelId));
export const MODEL_CATALOG = deepFreeze(catalog);
export const MODEL_CATALOG_BY_ID = Object.freeze(
  Object.fromEntries(MODEL_CATALOG.map((entry) => [entry.modelId, entry])) as Record<ModelId, PublicCatalogEntry>
);
