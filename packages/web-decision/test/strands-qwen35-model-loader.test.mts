import assert from 'node:assert/strict';
import test from 'node:test';
import {WebDecisionError} from '../src/kev.js';
import {loadPinnedQwen35Model, requestQwen35GpuDevicePair} from '../src/strands/qwen35-model-loader.mts';
import {createQwen35PreEmbedSourceResolver} from '../src/strands/qwen35-pre-embed-sources.mts';
import {createQwen35Session} from '../src/strands/qwen35-session.mts';
import type {Qwen35GraphInputChunk, Qwen35GraphInputRecord} from '../src/strands/qwen35-graph-input.mts';
import type {
  Qwen35PreEmbedSourceResolver,
  Qwen35PreEmbedSourceResolverOptions
} from '../src/strands/qwen35-pre-embed-sources.mts';

type LoadDependencies = NonNullable<Parameters<typeof loadPinnedQwen35Model>[1]>;

interface LoaderFixtureOptions {
  readonly prefetch?:
    | ((options: {
        readonly signal?: AbortSignal | null;
        readonly onProgress?: (bytes: number) => void;
      }) => Promise<void>)
    | null;
  readonly commitSource?: () => Promise<boolean>;
  readonly commitTokenizer?: () => Promise<boolean>;
  readonly openSource?: (options: {readonly offline: boolean}) => Promise<unknown>;
}

function createLoaderFixture({
  prefetch,
  commitSource = async () => true,
  commitTokenizer = async () => true,
  openSource
}: LoaderFixtureOptions = {}) {
  const calls: string[] = [];
  let sessionCreations = 0;
  const source = {
    requiredRangeBytes: 1,
    ...(prefetch === null
      ? {}
      : {
          async prefetchRequiredRanges(options: {
            readonly signal?: AbortSignal | null;
            readonly onProgress?: (bytes: number) => void;
          }) {
            calls.push('prefetch-source');
            await (prefetch ?? (async () => {}))(options);
          }
        }),
    async commit() {
      calls.push('commit-source');
      return commitSource();
    },
    dispose() {
      calls.push('dispose-source');
    }
  };
  const tokenizer = {
    tokenizer: {encode: () => []},
    async commit() {
      calls.push('commit-tokenizer');
      return commitTokenizer();
    },
    dispose() {
      calls.push('dispose-tokenizer');
    }
  };
  const embeddings = {
    async lookupTokenEmbeddings() {
      return new Float32Array();
    },
    releaseCachedMemory() {},
    dispose() {
      calls.push('dispose-embeddings');
    }
  };
  const languageDevice = {
    destroy() {
      calls.push('destroy-language');
    }
  } as unknown as GPUDevice;
  const visionDevice = {
    destroy() {
      calls.push('destroy-vision-device');
    }
  } as unknown as GPUDevice;
  const graph = {
    async run() {
      return [];
    },
    async runInputs() {
      return [];
    },
    async dispose() {
      calls.push('dispose-r19');
    }
  };
  const vision = {
    async run() {
      calls.push('run-vision');
      return [];
    },
    async dispose() {
      calls.push('dispose-vision');
    }
  };
  const artifacts = {fixtureBytes: new Uint8Array(), wasmBytes: new Uint8Array(), bindings: [{shape: [1]}]};
  const dependencies = {
    async openSource(options: {readonly offline: boolean}) {
      calls.push(`open-source-${options.offline ? 'offline' : 'online'}`);
      return openSource ? openSource(options) : source;
    },
    async openTokenizer(options: {readonly offline: boolean}) {
      calls.push(`open-tokenizer-${options.offline ? 'offline' : 'online'}`);
      return tokenizer;
    },
    async openR19Artifacts() {
      calls.push('open-r19-artifacts');
      return artifacts;
    },
    async openVisionArtifacts() {
      calls.push('open-vision-artifacts');
      return {...artifacts, artifact: {activationPrecision: 'float32'}};
    },
    async createEmbeddingLookup() {
      calls.push('create-embeddings');
      return embeddings;
    },
    async requestDevices() {
      calls.push('request-devices');
      return {languageModel: languageDevice, vision: visionDevice};
    },
    async instantiateR19() {
      calls.push('instantiate-r19');
      return {runtime: {}, device: {}, main() {}, async dispose() {}};
    },
    async createR19Runner(options: {readonly onProgress?: (bytes: number) => void}) {
      calls.push('load-r19-weights');
      options.onProgress?.(1);
      return graph;
    },
    async createVisionRunner(options: {readonly onProgress?: (bytes: number) => void}) {
      calls.push('load-vision-weights');
      options.onProgress?.(1);
      return vision;
    },
    createSession(options: never) {
      sessionCreations += 1;
      calls.push('create-session');
      return createQwen35Session(options);
    },
    tvmjs: {instantiate: async () => ({}), createPolyfillWASI: () => ({})},
    resolveInputSources:
      ({embeddings: rows}: {readonly embeddings: unknown}) =>
      async () => ({
        imageFeatures: [],
        lookupTokenEmbeddings: async () => rows
      })
  } as unknown as LoadDependencies;

  return {
    calls,
    dependencies,
    get sessionCreations() {
      return sessionCreations;
    }
  };
}

test('image source resolver retains the vision runner after session ownership transfers', async () => {
  const fixture = createLoaderFixture();
  let capturedResolver: Qwen35PreEmbedSourceResolver | undefined;
  const dependencies: LoadDependencies = {
    ...fixture.dependencies,
    resolveInputSources(options: Qwen35PreEmbedSourceResolverOptions) {
      capturedResolver = createQwen35PreEmbedSourceResolver(options);
      return capturedResolver;
    }
  };

  const model = await loadPinnedQwen35Model({}, dependencies);
  assert.ok(capturedResolver);
  const sources = await capturedResolver({} as Qwen35GraphInputChunk, {imageCount: 1} as Qwen35GraphInputRecord);
  assert.equal(sources.visionActivationPrecision, 'float32');
  assert.equal(fixture.calls.includes('run-vision'), true);
  await model.dispose();
});

test('requests one device from each adapter for the two TVM instances', async () => {
  const adapters: GPUAdapter[] = [];
  const devices: GPUDevice[] = [];
  const descriptors: GPUDeviceDescriptor[] = [];
  const deviceRequestsByAdapter: number[] = [];
  let adapterRequests = 0;
  const gpu = {
    async requestAdapter(options: GPURequestAdapterOptions) {
      const adapterIndex = ++adapterRequests;
      const workspaceBytes = adapterIndex === 1 ? 301_989_888 : 1_073_741_824;
      let deviceRequests = 0;
      assert.equal(options.powerPreference, 'high-performance');
      const adapter = {
        limits: {
          maxComputeWorkgroupStorageSize: adapterIndex === 1 ? 16_384 : 8_192,
          maxComputeInvocationsPerWorkgroup: 1_024,
          maxBufferSize: workspaceBytes,
          maxStorageBufferBindingSize: workspaceBytes,
          maxStorageBuffersPerShaderStage: 8
        },
        features: new Set<GPUFeatureName>(['shader-f16', 'subgroups']),
        async requestDevice(descriptor: GPUDeviceDescriptor) {
          deviceRequestsByAdapter[adapterIndex - 1] = ++deviceRequests;
          descriptors.push(descriptor);
          const device = {destroy() {}} as unknown as GPUDevice;
          devices.push(device);
          return device;
        }
      } as unknown as GPUAdapter;
      adapters.push(adapter);
      return adapter;
    }
  } as unknown as Pick<GPU, 'requestAdapter'>;

  const pair = await requestQwen35GpuDevicePair(gpu);
  assert.equal(adapterRequests, 2);
  assert.deepEqual(deviceRequestsByAdapter, [1, 1]);
  assert.equal(descriptors.length, 2);
  assert.notEqual(adapters[0], adapters[1]);
  assert.deepEqual(devices, [pair.languageModel, pair.vision]);
  assert.notEqual(pair.languageModel, pair.vision);
  for (const [i, workspaceBytes] of [301_989_888, 1_073_741_824].entries()) {
    assert.deepEqual(descriptors[i]?.requiredLimits, {
      maxBufferSize: workspaceBytes,
      maxStorageBufferBindingSize: workspaceBytes,
      maxComputeWorkgroupStorageSize: 8_192,
      maxStorageBuffersPerShaderStage: 8,
      maxComputeInvocationsPerWorkgroup: 1_024
    });
    assert.deepEqual(descriptors[i]?.requiredFeatures, ['shader-f16', 'subgroups']);
  }
});

test('rejects adapters below the workgroup or graph workspace requirements', async (t) => {
  const cases = [
    [0, 'maxComputeWorkgroupStorageSize', 8_192, /8192 bytes of workgroup storage/],
    [0, 'maxBufferSize', 301_989_888, /text requires.*301989888 bytes/],
    [0, 'maxStorageBufferBindingSize', 301_989_888, /text requires.*301989888 bytes/],
    [1, 'maxBufferSize', 1_073_741_824, /vision requires.*1073741824 bytes/],
    [1, 'maxStorageBufferBindingSize', 1_073_741_824, /vision requires.*1073741824 bytes/]
  ] as const;
  for (const [deviceIndex, limit, minimum, message] of cases) {
    await t.test(`${deviceIndex === 0 ? 'text' : 'vision'} ${limit}`, async () => {
      let adapters = 0;
      let created = 0;
      let destroyed = 0;
      const gpu = {
        async requestAdapter() {
          const limits = {
            maxComputeWorkgroupStorageSize: 8_192,
            maxComputeInvocationsPerWorkgroup: 256,
            maxBufferSize: 1_073_741_824,
            maxStorageBufferBindingSize: 1_073_741_824,
            maxStorageBuffersPerShaderStage: 8
          };
          if (adapters++ === deviceIndex) limits[limit] = minimum - 1;
          return {
            limits,
            features: new Set<GPUFeatureName>(),
            async requestDevice() {
              created++;
              return {destroy: () => destroyed++};
            }
          } as unknown as GPUAdapter;
        }
      };
      await assert.rejects(requestQwen35GpuDevicePair(gpu), {code: 'DEVICE_RESOURCE_LIMIT', message});
      assert.equal(created, deviceIndex);
      assert.equal(destroyed, created);
    });
  }
});

test('destroys the first device when requesting the second device fails', async () => {
  const descriptors: GPUDeviceDescriptor[] = [];
  const deviceRequestsByAdapter: number[] = [];
  const destroyed: string[] = [];
  let adapterRequests = 0;
  const gpu = {
    async requestAdapter(options: GPURequestAdapterOptions) {
      const adapterIndex = ++adapterRequests;
      let deviceRequests = 0;
      assert.equal(options.powerPreference, 'high-performance');
      deviceRequestsByAdapter.push(0);
      return {
        limits: {
          maxComputeWorkgroupStorageSize: 32_768,
          maxComputeInvocationsPerWorkgroup: 1_024,
          maxBufferSize: 1_073_741_824,
          maxStorageBufferBindingSize: 1_073_741_824,
          maxStorageBuffersPerShaderStage: 8
        },
        features: new Set<GPUFeatureName>(['shader-f16', 'subgroups']),
        async requestDevice(descriptor: GPUDeviceDescriptor) {
          deviceRequestsByAdapter[adapterIndex - 1] = ++deviceRequests;
          descriptors.push(descriptor);
          if (adapterIndex === 2) throw new Error('second device failed');
          return {
            destroy() {
              destroyed.push('language');
            }
          } as unknown as GPUDevice;
        }
      } as unknown as GPUAdapter;
    }
  } as unknown as Pick<GPU, 'requestAdapter'>;

  await assert.rejects(requestQwen35GpuDevicePair(gpu), /second device failed/);
  assert.equal(adapterRequests, 2);
  assert.deepEqual(deviceRequestsByAdapter, [1, 1]);
  assert.equal(descriptors.length, 2);
  assert.deepEqual(descriptors[0]?.requiredFeatures, ['shader-f16', 'subgroups']);
  assert.deepEqual(destroyed, ['language']);
});

test('aborting while requesting the second adapter destroys the first device', async () => {
  const controller = new AbortController();
  const reason = Error('cancel second adapter');
  const destroyed: string[] = [];
  let adapterRequests = 0;
  let deviceRequests = 0;
  const firstAdapter = {
    limits: {
      maxComputeWorkgroupStorageSize: 32_768,
      maxComputeInvocationsPerWorkgroup: 1_024,
      maxBufferSize: 1_073_741_824,
      maxStorageBufferBindingSize: 1_073_741_824,
      maxStorageBuffersPerShaderStage: 8
    },
    features: new Set<GPUFeatureName>(['shader-f16', 'subgroups']),
    async requestDevice() {
      deviceRequests += 1;
      return {
        destroy() {
          destroyed.push('language');
        }
      } as unknown as GPUDevice;
    }
  } as unknown as GPUAdapter;
  const gpu = {
    async requestAdapter(options: GPURequestAdapterOptions) {
      assert.equal(options.powerPreference, 'high-performance');
      adapterRequests += 1;
      if (adapterRequests === 1) return firstAdapter;
      controller.abort(reason);
      return new Promise<GPUAdapter | null>(() => {});
    }
  } as unknown as Pick<GPU, 'requestAdapter'>;

  await assert.rejects(requestQwen35GpuDevicePair(gpu, controller.signal), (error) => error === reason);
  assert.equal(adapterRequests, 2);
  assert.equal(deviceRequests, 1);
  assert.deepEqual(destroyed, ['language']);
});

test('composes pinned source, R19, vision and tokenizer lifetimes into one model session', async () => {
  const calls: string[] = [];
  const progress: {loadedBytes: number; totalBytes: number; fraction: number; phase: string}[] = [];
  const languageDevice = {destroy: () => calls.push('destroy-language')} as unknown as GPUDevice;
  const visionDevice = {destroy: () => calls.push('destroy-vision')} as unknown as GPUDevice;
  const tokenizer = {
    tokenizer: {encode: () => []},
    async commit() {
      calls.push('commit-tokenizer');
      return true;
    },
    dispose() {
      calls.push('dispose-tokenizer');
    }
  };
  const source = {
    requiredRangeBytes: 4,
    async prefetchRequiredRanges(options: {onProgress?: (bytes: number) => void}) {
      calls.push('prefetch-source');
      options.onProgress?.(4);
    },
    async commit() {
      calls.push('commit-source');
      return true;
    },
    finishLoading() {
      calls.push('finish-source-loading');
    },
    dispose() {
      calls.push('dispose-source');
    }
  };
  const embeddings = {
    lookupTokenEmbeddings: async () => new Float32Array(),
    releaseCachedMemory() {
      calls.push('release-embedding-cache');
    },
    dispose() {
      calls.push('dispose-embeddings');
    }
  };
  const graph = {
    async run() {
      return [];
    },
    async runInputs() {
      return [];
    },
    async dispose() {
      calls.push('dispose-r19');
      languageDevice.destroy();
    }
  };
  const vision = {
    async run() {
      return [];
    },
    async dispose() {
      calls.push('dispose-vision');
      visionDevice.destroy();
    }
  };
  const handle = {
    runtime: {},
    device: {},
    main() {},
    async dispose() {
      calls.push('dispose-r19-handle');
      languageDevice.destroy();
    }
  };
  const dependencies = {
    async openSource() {
      calls.push('open-source');
      return source;
    },
    async openTokenizer() {
      calls.push('open-tokenizer');
      return tokenizer;
    },
    async openR19Artifacts() {
      calls.push('open-r19-artifacts');
      return {fixtureBytes: new Uint8Array(), wasmBytes: new Uint8Array(), bindings: [{shape: [1]}]};
    },
    async openVisionArtifacts() {
      calls.push('open-vision-artifacts');
      return {fixtureBytes: new Uint8Array(), wasmBytes: new Uint8Array(), bindings: [{shape: [1]}], artifact: {}};
    },
    async createEmbeddingLookup() {
      calls.push('create-embeddings');
      return embeddings;
    },
    async requestDevices() {
      calls.push('request-devices');
      return {languageModel: languageDevice, vision: visionDevice};
    },
    async instantiateR19() {
      calls.push('instantiate-r19');
      return handle;
    },
    async createR19Runner(options: {onProgress?: (bytes: number) => void}) {
      calls.push('load-r19-weights');
      options.onProgress?.(4);
      return graph;
    },
    async createVisionRunner(options: {onProgress?: (bytes: number) => void}) {
      calls.push('load-vision-weights');
      options.onProgress?.(4);
      return vision;
    },
    createSession: createQwen35Session,
    tvmjs: {instantiate: async () => ({}), createPolyfillWASI: () => ({})},
    resolveInputSources: ({embeddings: rows, vision: imageRunner}: never) => {
      assert.equal(rows, embeddings);
      assert.ok(imageRunner);
      return async () => ({
        imageFeatures: [],
        lookupTokenEmbeddings: async () => new Float32Array()
      });
    }
  } as unknown as NonNullable<Parameters<typeof loadPinnedQwen35Model>[1]>;

  const model = await loadPinnedQwen35Model(
    {cache: {} as never, onProgress: (value) => progress.push({...value})},
    dependencies
  );
  assert.ok(calls.indexOf('create-embeddings') < calls.indexOf('request-devices'));
  assert.ok(calls.indexOf('request-devices') < calls.indexOf('instantiate-r19'));
  assert.ok(calls.indexOf('load-r19-weights') < calls.indexOf('load-vision-weights'));
  assert.ok(calls.indexOf('load-vision-weights') < calls.indexOf('prefetch-source'));
  assert.ok(calls.indexOf('create-session') < calls.indexOf('finish-source-loading'));
  assert.deepEqual(progress, [
    {loadedBytes: 4, totalBytes: 12, fraction: 1 / 3, phase: 'model-assets'},
    {loadedBytes: 8, totalBytes: 12, fraction: 2 / 3, phase: 'vision-weights'},
    {loadedBytes: 12, totalBytes: 12, fraction: 1, phase: 'upstream-source'}
  ]);
  assert.equal(calls.includes('dispose-r19-handle'), false);

  await model.releaseCachedMemory();
  await model.dispose();
  await model.dispose();
  assert.deepEqual(calls.slice(-8), [
    'release-embedding-cache',
    'dispose-vision',
    'destroy-vision',
    'dispose-r19',
    'destroy-language',
    'dispose-embeddings',
    'dispose-source',
    'dispose-tokenizer'
  ]);
  assert.equal(calls.filter((call) => call === 'dispose-embeddings').length, 1);
  assert.equal(calls.filter((call) => call === 'dispose-tokenizer').length, 1);
  assert.equal(calls.filter((call) => call === 'commit-source').length, 1);
  assert.equal(calls.filter((call) => call === 'commit-tokenizer').length, 1);
  assert.ok(calls.indexOf('prefetch-source') < calls.indexOf('commit-source'));
});

test('returns a usable online session when optional cache prefetch or commit fails', async (t) => {
  const quotaError = Object.assign(new Error('browser storage quota reached'), {name: 'QuotaExceededError'});
  const cases: readonly {
    readonly name: string;
    readonly fixture: LoaderFixtureOptions;
  }[] = [
    {
      name: 'cache prefetch fails',
      fixture: {prefetch: async () => Promise.reject(quotaError), commitSource: async () => false}
    },
    {name: 'source commit returns false', fixture: {commitSource: async () => false}},
    {name: 'source commit rejects', fixture: {commitSource: async () => Promise.reject(quotaError)}},
    {name: 'tokenizer commit rejects', fixture: {commitTokenizer: async () => Promise.reject(quotaError)}}
  ];

  for (const {name, fixture: options} of cases) {
    await t.test(name, async () => {
      const fixture = createLoaderFixture(options);
      const model = await loadPinnedQwen35Model({cache: {} as never}, fixture.dependencies);

      assert.equal(typeof model.systemOne, 'function');
      assert.equal(fixture.sessionCreations, 1);
      assert.equal(
        fixture.calls.some((call) => call.startsWith('dispose-')),
        false
      );

      await model.dispose();
      assert.equal(fixture.calls.includes('dispose-r19'), true);
      assert.equal(fixture.calls.includes('dispose-vision'), true);
    });
  }
});

test('propagates source integrity failures and cancellation during optional cache prefetch', async (t) => {
  const integrityError = new WebDecisionError('INVALID_ASSET', 'Qwen3.5 source range integrity mismatch.');
  const controller = new AbortController();
  const abortReason = new Error('load cancelled');
  const cases = [
    {
      name: 'source integrity failure',
      prefetch: async () => Promise.reject(integrityError),
      signal: undefined,
      expected: integrityError
    },
    {
      name: 'abort during prefetch',
      prefetch: async () => {
        controller.abort(abortReason);
        throw abortReason;
      },
      signal: controller.signal,
      expected: abortReason
    }
  ] as const;

  for (const {name, prefetch, signal, expected} of cases) {
    await t.test(name, async () => {
      const fixture = createLoaderFixture({prefetch});
      await assert.rejects(
        loadPinnedQwen35Model({cache: {} as never, ...(signal ? {signal} : {})}, fixture.dependencies),
        (cause: unknown) => cause === expected
      );
      assert.equal(fixture.sessionCreations, 0);
      assert.equal(fixture.calls.includes('prefetch-source'), true);
      assert.equal(fixture.calls.includes('create-embeddings'), true);
      assert.equal(fixture.calls.includes('request-devices'), true);
      assert.equal(fixture.calls.includes('instantiate-r19'), true);
      assert.equal(fixture.calls.includes('load-r19-weights'), true);
      assert.equal(fixture.calls.includes('load-vision-weights'), true);
      assert.equal(fixture.calls.includes('commit-source'), false);
      assert.equal(fixture.calls.includes('commit-tokenizer'), false);
      assert.equal(fixture.calls.includes('dispose-source'), true);
      assert.equal(fixture.calls.includes('dispose-tokenizer'), true);
    });
  }
});

test('fails closed offline when the verified source marker is missing without fetching', async () => {
  const missingMarker = new WebDecisionError(
    'ASSET_LOAD_FAILED',
    'The complete verified model is not available offline.'
  );
  let fetches = 0;
  const fixture = createLoaderFixture({
    async openSource({offline}) {
      assert.equal(offline, true);
      if (!offline) fetches += 1;
      throw missingMarker;
    }
  });

  await assert.rejects(
    loadPinnedQwen35Model({cache: {} as never, offline: true}, fixture.dependencies),
    (cause: unknown) => cause === missingMarker
  );
  assert.equal(fetches, 0);
  assert.equal(fixture.calls.includes('open-source-offline'), true);
  assert.equal(fixture.calls.includes('open-tokenizer-offline'), true);
  assert.equal(fixture.calls.includes('request-devices'), false);
  assert.equal(fixture.calls.includes('prefetch-source'), false);
  assert.equal(fixture.sessionCreations, 0);
});

test('cleans loaded R19 resources and embeddings if vision weight loading fails', async () => {
  const calls: string[] = [];
  const source = {
    async commit() {
      return false;
    }
  };
  const tokenizer = {
    tokenizer: {encode: () => []},
    async commit() {
      return false;
    },
    dispose() {
      calls.push('dispose-tokenizer');
    }
  };
  const embeddings = {
    lookupTokenEmbeddings: async () => new Float32Array(),
    releaseCachedMemory() {},
    dispose() {
      calls.push('dispose-embeddings');
    }
  };
  const device = {
    destroy() {
      calls.push('destroy-language');
    }
  } as unknown as GPUDevice;
  const handle = {
    runtime: {},
    device: {},
    main() {},
    async dispose() {
      calls.push('dispose-handle');
    }
  };
  const graph = {
    async run() {
      return [];
    },
    async runInputs() {
      return [];
    },
    async dispose() {
      calls.push('dispose-r19-runner');
    }
  };
  const dependencies = {
    async openSource() {
      return source;
    },
    async openTokenizer() {
      return tokenizer;
    },
    async openR19Artifacts() {
      return {fixtureBytes: new Uint8Array(), wasmBytes: new Uint8Array(), bindings: []};
    },
    async openVisionArtifacts() {
      return {fixtureBytes: new Uint8Array(), wasmBytes: new Uint8Array(), bindings: [], artifact: {}};
    },
    async createEmbeddingLookup() {
      return embeddings;
    },
    async requestDevices() {
      return {
        languageModel: device,
        vision: {
          destroy() {
            calls.push('destroy-vision');
          }
        } as unknown as GPUDevice
      };
    },
    async instantiateR19() {
      return handle;
    },
    async createR19Runner() {
      return graph;
    },
    async createVisionRunner(options: {gpuDevice: GPUDevice}) {
      options.gpuDevice.destroy();
      throw new Error('vision weights failed');
    },
    createSession: createQwen35Session,
    tvmjs: {instantiate: async () => ({}), createPolyfillWASI: () => ({})},
    resolveInputSources: () => async () => ({imageFeatures: [], lookupTokenEmbeddings: async () => new Float32Array()})
  } as unknown as NonNullable<Parameters<typeof loadPinnedQwen35Model>[1]>;

  await assert.rejects(loadPinnedQwen35Model({}, dependencies), /vision weights failed/);
  assert.deepEqual(calls.sort(), ['destroy-vision', 'dispose-embeddings', 'dispose-r19-runner', 'dispose-tokenizer']);
});
