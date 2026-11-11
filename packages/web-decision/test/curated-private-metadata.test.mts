import assert from 'node:assert/strict';
import * as nodeModule from 'node:module';
import {test} from 'node:test';

type ResolveContext = {readonly parentURL?: string};
type ResolveResult = {readonly url: string; readonly shortCircuit?: boolean};
type ResolveHook = (
  specifier: string,
  context: ResolveContext,
  nextResolve: (specifier: string, context: ResolveContext) => ResolveResult
) => ResolveResult;
type HookHandle = {deregister(): void};
type RegisterHooks = (hooks: {resolve: ResolveHook}) => HookHandle;
type RuntimeDescription = {readonly modelId: string; readonly assets: Readonly<Record<string, unknown>>};
type RuntimeLibrary = {
  listModels(): readonly RuntimeDescription[];
  describeModel(modelId: 'kev-0.6b-q4f16'): RuntimeDescription;
  loadDecisionModel(input: unknown): Promise<{readonly description: RuntimeDescription}>;
};
type LoaderCapture = {readonly model: string; readonly directArtifactRoot: string};
type TestGlobal = typeof globalThis & {__curatedPrivateMetadataCapture?: LoaderCapture};

const registerHooks = (nodeModule as unknown as {readonly registerHooks?: RegisterHooks}).registerHooks;
const modelId = 'kev-0.6b-q4f16';
const directArtifactRoot = 'https://private.example.test/kev-package/';

test(
  'curated API hides the private Kev artifact root and keeps it on the loader path',
  {skip: typeof registerHooks !== 'function'},
  async () => {
    if (!registerHooks) return;

    const model = {
      modelId,
      description: {
        adapter: 'kev06',
        precision: 'q4f16',
        checkpoint: {repository: 'example/kev', revision: 'abc123'},
        assets: {
          integrity: 'pinned-manifest',
          manifestSha256: 'a'.repeat(64),
          location: 'package',
          directArtifactRoot,
          requiredFiles: ['manifest.json'],
          additionalFiles: 'none'
        },
        licenses: {},
        backend: {
          name: 'tvm-webgpu',
          requiredFeatures: [],
          requiredLimits: {},
          requiresNonFallbackAdapter: false
        },
        features: {
          textChoice: {
            status: 'implemented',
            abstention: 'none',
            limits: [],
            handling: 'native',
            tokenCount: 'model',
            score: {}
          }
        },
        verification: {}
      }
    };
    const catalogSource = `
      const entry = ${JSON.stringify(model)};
      export const MODEL_CATALOG = [entry];
      export const MODEL_CATALOG_BY_ID = {[entry.modelId]: entry};
    `;
    const fakeKevLoaderSource = `
      export async function loadCuratedKevModel(options, description) {
        globalThis.__curatedPrivateMetadataCapture = {
          model: options.model,
          directArtifactRoot: description.assets.directArtifactRoot
        };
        return {};
      }
    `;
    const dataUrl = (source: string) => `data:text/javascript,${encodeURIComponent(source)}`;
    const entryUrl = new URL('../src/curated/index.ts?private-metadata-test', import.meta.url);
    const indexUrl = new URL('../src/curated/index.ts', import.meta.url).href;
    const adapterUrl = new URL('../src/curated/adapters/kev06.ts', import.meta.url).href;
    const testGlobal = globalThis as TestGlobal;
    const previousCapture = Object.getOwnPropertyDescriptor(testGlobal, '__curatedPrivateMetadataCapture');
    const hooks = registerHooks({
      resolve(specifier, context, nextResolve) {
        if (context.parentURL === entryUrl.href && specifier === '../catalog/index.generated.ts')
          return {url: dataUrl(catalogSource), shortCircuit: true};
        if (context.parentURL === adapterUrl && specifier === '../../kev-loader.js')
          return {url: dataUrl(fakeKevLoaderSource), shortCircuit: true};
        if (context.parentURL === indexUrl && specifier === '../catalog/index.generated.ts')
          return {url: dataUrl(catalogSource), shortCircuit: true};
        return nextResolve(specifier, context);
      }
    });

    try {
      const library = (await import(entryUrl.href)) as RuntimeLibrary;
      const listed = library.listModels()[0]!;
      const described = library.describeModel(modelId);
      assert.equal(Object.hasOwn(listed.assets, 'directArtifactRoot'), false);
      assert.equal(Object.hasOwn(described.assets, 'directArtifactRoot'), false);

      const loaded = await library.loadDecisionModel({modelId});
      assert.equal(loaded.description, described);
      assert.equal(Object.hasOwn(loaded.description.assets, 'directArtifactRoot'), false);
      assert.deepEqual(testGlobal.__curatedPrivateMetadataCapture, {model: modelId, directArtifactRoot});
    } finally {
      hooks.deregister();
      if (previousCapture) Object.defineProperty(testGlobal, '__curatedPrivateMetadataCapture', previousCapture);
      else delete testGlobal.__curatedPrivateMetadataCapture;
    }
  }
);
