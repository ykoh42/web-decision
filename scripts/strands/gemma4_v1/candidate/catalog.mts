import type {Plugin} from 'vite';
import {APPROVED_MODEL_IDS} from '../../../../packages/web-decision/src/catalog/public-catalog.generated.ts';

export const modelId = 'strands-decider-e2b-gemma4-v1-2610';
type VerifiedFile = Readonly<{bytes: number; sha256: string}>;
type Upstream = Readonly<{
  repository: string;
  revision: string;
  file: string;
  bytes: number;
  sha256: string;
}>;

export function catalogEntry(
  manifest: {base: Upstream; adapter: Upstream; files: Record<string, VerifiedFile>},
  sha256: string
) {
  const {repository, revision, sha256: weightSha256} = manifest.adapter;
  return {
    modelId,
    description: {
      adapter: 'strands-gemma4-v1',
      precision: 'BF16 Gemma 4 E2B base with FP32 Strands adapter, pointer head, and TVM WebGPU computation',
      checkpoint: {repository, revision, weightSha256},
      assets: {
        integrity: 'pinned-manifest',
        location: 'package-and-pinned-upstream',
        requiredFiles: ['manifest.json', ...Object.keys(manifest.files)],
        manifestSha256: sha256,
        additionalFiles:
          'The compiled graph, range inventory and tokenizer are bundled. The base checkpoint and adapter load from their pinned upstream revisions.'
      },
      licenses: {
        project: 'Apache-2.0',
        modelTerms: 'Apache-2.0',
        modelCard: `https://huggingface.co/${repository}/tree/${revision}`,
        baseModelCard: `https://huggingface.co/${manifest.base.repository}/tree/${manifest.base.revision}`
      },
      backend: {
        name: 'tvm-webgpu',
        requiredFeatures: [],
        requiredLimits: {maxBufferSize: 75_497_472, maxStorageBufferBindingSize: 75_497_472},
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
              defaultMax: 4_096,
              max: 131_072,
              description: 'Includes state and question tokens. Longer configured windows are subject to device limits.'
            },
            {resource: 'options', scope: 'question', max: 255, description: 'Maximum choices on a choice question.'},
            {resource: 'questions', scope: 'request', min: 1}
          ],
          handling: 'causal-state-and-question-pointer',
          tokenCount: 'state-tokens-per-prefix-evaluation-plus-sum-of-question-tokens',
          score: {basis: 'strands-pointer', calibration: 'not-guaranteed'}
        },
        typedTextQuestionTypes: ['choice', 'noul', 'score'],
        imageChoice: {status: 'not-implemented'}
      },
      verification: {
        currentBundleGpu: 'pending',
        currentBundleCpuWasm: 'not-applicable',
        currentBundleJavascript: 'not-applicable'
      }
    }
  };
}

export function curatedCandidate(
  root: string,
  entry: ReturnType<typeof catalogEntry>,
  files: Record<string, Uint8Array>,
  manifest: {files: Record<string, VerifiedFile>}
): Plugin {
  if ((APPROVED_MODEL_IDS as readonly string[]).includes(modelId)) throw new Error('Gemma 4 v1 is already public.');
  const catalog = 'public-catalog.generated.ts';
  const adapters = 'public-adapters.generated.ts';
  const options = 'public-load-options.generated.ts';
  return {
    name: 'private-gemma4-v1-curated-catalog',
    enforce: 'pre',
    resolveId(source, importer) {
      if (source === 'gemma4-v1-private-adapter') return '\0' + source;
      if (importer === root + '/src/curated/index.ts' && (source.endsWith(catalog) || source.endsWith(adapters)))
        return '\0' + source.slice('../catalog/'.length);
      if (importer === root + '/src/curated/load-options.ts' && source === '../catalog/' + options)
        return '\0' + options;
      return null;
    },
    // Keep runtime fetch URLs when a consumer's bundler would inline a small JSON asset.
    resolveFileUrl({fileName, relativePath}) {
      if (!fileName.startsWith('gemma4-v1/')) return null;
      return `new URL(${JSON.stringify('./' + relativePath + '?no-inline')}, import.meta.url).href`;
    },
    load(id) {
      if (id === '\0' + catalog)
        return `import {MODEL_CATALOG as approved} from ${JSON.stringify(root + '/src/catalog/' + catalog)};
          const entry = ${JSON.stringify(entry)};
          function deepFreeze(value) {
            for (const child of Object.values(value)) if (child && typeof child === 'object') deepFreeze(child);
            return Object.freeze(value);
          }
          export const MODEL_CATALOG = Object.freeze([...approved, deepFreeze(entry)]);
          export const MODEL_CATALOG_BY_ID = Object.freeze(Object.fromEntries(MODEL_CATALOG.map(item => [item.modelId, item])));`;
      if (id === '\0' + adapters)
        return `import {APPROVED_ADAPTER_LOADERS as approved} from ${JSON.stringify(root + '/src/catalog/' + adapters)};
          export const APPROVED_ADAPTER_LOADERS = Object.freeze({...approved, 'strands-gemma4-v1': () => import('gemma4-v1-private-adapter')});`;
      if (id === '\0' + options)
        return `import {APPROVED_MODEL_LOAD_OPTIONS as approved} from ${JSON.stringify(root + '/src/catalog/' + options)};
          export const APPROVED_MODEL_LOAD_OPTIONS = Object.freeze({...approved, '${modelId}': ['maxLength', 'maxBatch', 'maxBatchTokens', 'usePrefixCache']});`;
      if (id !== '\0gemma4-v1-private-adapter') return null;
      const urls = Object.entries(manifest.files).map(([name]) => {
        const ref = this.emitFile({type: 'asset', fileName: 'gemma4-v1/' + name, source: files[name]!});
        return `${JSON.stringify(name)}: new URL(import.meta.ROLLUP_FILE_URL_${ref})`;
      });
      this.emitFile({type: 'asset', fileName: 'gemma4-v1/manifest.json', source: files['manifest.json']!});
      return `import {createGemma4V1CuratedAdapter} from ${JSON.stringify(root + '/src/curated/adapters/strands-gemma4-v1.mts')};
        export const {loadModel} = createGemma4V1CuratedAdapter({${urls.join(',')}}, ${JSON.stringify({files: manifest.files})});`;
    }
  };
}
