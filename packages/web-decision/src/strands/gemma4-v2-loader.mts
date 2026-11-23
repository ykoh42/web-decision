import {Tokenizer} from '@huggingface/tokenizers';
import {parseJson, readVerifiedAsset, waitForAbort, type AssetCache, type AssetFileManifest} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {parseSafetensorsHeader} from '../kanana/safetensors.mjs';
import {openPinnedGemma4V2Source} from './gemma4-v2-source.mjs';
import {createGemma4V2ImagePreparer} from './gemma4-v2-image.mjs';
import {createGemma4V2OfflinePreparation, pinnedGemma4V2OfflineAssets} from './gemma4-v2-offline.mjs';
import {createGemma4V2Session, type Gemma4V2OwnedRunner, type Gemma4V2PreparedImage} from './gemma4-v2-session.mjs';
import {createGemma4V2ScoringKernel, GEMMA4_V2_SCORING_BYTES, GEMMA4_V2_SCORING_SHA256} from './gemma4-v2-scoring.mjs';

export const GEMMA4_V2_MODEL = 'strands-decider-e2b-gemma4-v2-2611';
const PINNED = pinnedGemma4V2OfflineAssets([]);
const SOURCE_ASSETS = [
  PINNED.modelConfig,
  PINNED.decisionConfig,
  PINNED.processorConfig,
  PINNED.tokenizer,
  PINNED.tokenizerConfig,
  PINNED.readout
];
const BASE = PINNED.modelConfig.base;
const ASSETS: AssetFileManifest = {
  files: Object.fromEntries(SOURCE_ASSETS.map((asset) => [asset.name, asset.metadata]))
};

type Source = Awaited<ReturnType<typeof openPinnedGemma4V2Source>>;
type TokenizerLike = {encode(text: string, options: {add_special_tokens: false}): {ids: number[]}};
export type Gemma4V2RunnerFactory = (
  input: Readonly<{
    source: Source;
    readout: Uint8Array;
    signal?: AbortSignal | null;
    cache?: AssetCache | null;
    offline: boolean;
  }>
) => Promise<Gemma4V2OwnedRunner>;

export interface Gemma4V2LoaderOptions {
  readonly cache?: AssetCache | null;
  readonly offline?: boolean;
  readonly signal?: AbortSignal | null;
  readonly maxLength?: number;
  readonly onBytes?: (bytes: number) => void;
}

interface Dependencies {
  readonly createRunner: Gemma4V2RunnerFactory;
  readonly graphAssets?: Parameters<typeof pinnedGemma4V2OfflineAssets>[0];
  readonly scoringAsset?: Parameters<typeof pinnedGemma4V2OfflineAssets>[0][number];
  readonly imagePreparer?: ReturnType<typeof createGemma4V2ImagePreparer>;
  readonly prepareImage?: (image: string, signal?: AbortSignal | null) => Promise<Gemma4V2PreparedImage>;
  readonly readAsset?: typeof readVerifiedAsset;
  readonly openSource?: typeof openPinnedGemma4V2Source;
  readonly createTokenizer?: (model: object, config: object) => TokenizerLike;
}

const invalid = (message: string) => new WebDecisionError('INVALID_ASSET', message);
function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw invalid(`Invalid ${label}.`);
  return value as Record<string, unknown>;
}

function settings(
  decision: Record<string, unknown>,
  model: Record<string, unknown>,
  processor: Record<string, unknown>
) {
  const text = object(model.text_config, 'text configuration');
  const image = object(processor.image_processor, 'image processor configuration');
  if (
    decision.prompt !== 'strands-chat' ||
    decision.attention_mode !== 'prefix_bidir' ||
    decision.pooling !== 'last' ||
    text.num_hidden_layers !== 35 ||
    text.hidden_size !== 1536 ||
    image.patch_size !== 16 ||
    image.pooling_kernel_size !== 3 ||
    image.max_soft_tokens !== 280 ||
    !Number.isSafeInteger(decision.max_length_validated) ||
    (decision.max_length_validated as number) < 1 ||
    typeof decision.temperature !== 'number' ||
    !Number.isFinite(decision.temperature) ||
    decision.temperature <= 0 ||
    typeof decision.final_logit_softcapping !== 'number' ||
    !Number.isFinite(decision.final_logit_softcapping) ||
    decision.final_logit_softcapping <= 0
  )
    throw invalid('Unexpected Gemma 4 v2 model configuration.');
  return {
    maxLength: decision.max_length_validated as number,
    temperature: decision.temperature,
    softcap: decision.final_logit_softcapping
  };
}

function readoutTensor(bytes: ArrayBuffer): Uint8Array {
  if (bytes.byteLength < 8) throw invalid('Gemma 4 v2 readout is truncated.');
  const size = new DataView(bytes).getBigUint64(0, true);
  if (size > BigInt(bytes.byteLength - 8)) throw invalid('Invalid Gemma 4 v2 readout header.');
  const header = parseSafetensorsHeader(new Uint8Array(bytes, 0, Number(size) + 8), bytes.byteLength);
  const tensor = header.tensors[0];
  if (
    header.tensorCount !== 1 ||
    !tensor ||
    tensor.name !== 'weight' ||
    tensor.dtype !== 'BF16' ||
    tensor.shape.length !== 2 ||
    tensor.shape[0] !== 255 ||
    tensor.shape[1] !== 1536
  )
    throw invalid('Unexpected Gemma 4 v2 readout tensor.');
  return new Uint8Array(bytes.slice(tensor.offset, tensor.offset + tensor.length));
}

function validateLoad(options: Gemma4V2LoaderOptions, dependencies: Dependencies): void {
  if (!dependencies || typeof dependencies.createRunner !== 'function')
    throw new TypeError('A Gemma 4 v2 TVM runner factory is required.');
  if (dependencies.prepareImage && dependencies.imagePreparer)
    throw new TypeError('Use one Gemma 4 v2 image preparer for inference and offline preparation.');
  if (
    !options ||
    (options.offline !== undefined && typeof options.offline !== 'boolean') ||
    (options.offline && !options.cache)
  )
    throw new WebDecisionError('INVALID_INPUT', 'Invalid Gemma 4 v2 load options.');
  if (options.maxLength !== undefined && (!Number.isSafeInteger(options.maxLength) || options.maxLength < 1))
    throw new WebDecisionError('INVALID_INPUT', 'maxLength must be a positive integer.');
}

async function readScoringKernel(
  options: Gemma4V2LoaderOptions,
  dependencies: Dependencies,
  read: typeof readVerifiedAsset
) {
  const asset = dependencies.scoringAsset;
  if (!asset) return undefined;
  if (asset.metadata?.bytes !== GEMMA4_V2_SCORING_BYTES || asset.metadata?.sha256 !== GEMMA4_V2_SCORING_SHA256)
    throw invalid('Gemma 4 v2 scoring asset pin differs.');
  const bytes = await read(asset.base, {files: {[asset.name]: asset.metadata}}, asset.name, {
    cache: options.cache,
    offline: options.offline ?? false,
    signal: options.signal
  });
  return createGemma4V2ScoringKernel(bytes);
}

/** Build an internal session using the pinned source and a supplied TVM runner. */
export async function loadGemma4V2Model(options: Gemma4V2LoaderOptions, dependencies: Dependencies) {
  validateLoad(options, dependencies);
  options.signal?.throwIfAborted();
  const read = dependencies.readAsset ?? readVerifiedAsset;
  const scoringAsset = dependencies.scoringAsset;
  const readOptions = {
    cache: options.cache,
    offline: options.offline ?? false,
    signal: options.signal,
    onBytes: options.onBytes
  };
  const names = [
    'config.json',
    'decision_config.json',
    'processor_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'readout.safetensors'
  ] as const;
  const [buffers, scoringKernel] = await Promise.all([
    Promise.all(names.map((name) => read(BASE, ASSETS, name, readOptions))),
    readScoringKernel(options, dependencies, read)
  ]);
  options.signal?.throwIfAborted();
  const decision = object(parseJson(buffers[1]!, names[1]), 'decision configuration');
  const config = settings(
    decision,
    object(parseJson(buffers[0]!, names[0]), 'model configuration'),
    object(parseJson(buffers[2]!, names[2]), 'processor configuration')
  );
  if (scoringKernel && (config.temperature !== 1.5456 || config.softcap !== 30))
    throw invalid('Gemma 4 v2 scoring asset does not match its decision configuration.');
  const tokenizerData = object(parseJson(buffers[3]!, names[3]), 'tokenizer');
  if (!Array.isArray(tokenizerData.added_tokens)) throw invalid('Invalid Gemma 4 v2 tokenizer tokens.');
  const safeTokenizer = {
    ...tokenizerData,
    added_tokens: tokenizerData.added_tokens.filter((token: unknown) => !object(token, 'tokenizer entry').special)
  };
  const tokenizerConfig = object(parseJson(buffers[4]!, names[4]), 'tokenizer configuration');
  const tokenizer = (dependencies.createTokenizer ?? ((data, configuration) => new Tokenizer(data, configuration)))(
    safeTokenizer,
    tokenizerConfig
  );
  const readout = readoutTensor(buffers[5]!);
  const openSource = dependencies.openSource ?? openPinnedGemma4V2Source;
  const source = await waitForAbort(
    () =>
      openSource({...(options.cache === undefined ? {} : {cache: options.cache}), offline: options.offline ?? false}),
    options.signal,
    (late) => late.dispose()
  );
  let runner: Gemma4V2OwnedRunner | undefined;
  let offlinePreparation: ReturnType<typeof createGemma4V2OfflinePreparation> | undefined;
  try {
    const imagePreparer =
      dependencies.imagePreparer ??
      (dependencies.prepareImage
        ? undefined
        : createGemma4V2ImagePreparer({cache: options.cache, offline: options.offline}));
    if (options.cache && dependencies.graphAssets?.length && imagePreparer) {
      offlinePreparation = createGemma4V2OfflinePreparation(
        source,
        options.cache,
        pinnedGemma4V2OfflineAssets([...dependencies.graphAssets, ...(scoringAsset ? [scoringAsset] : [])]),
        imagePreparer
      );
    }
    runner = await waitForAbort(
      () =>
        dependencies.createRunner({
          source,
          readout,
          ...(options.signal === undefined ? {} : {signal: options.signal}),
          ...(options.cache === undefined ? {} : {cache: options.cache}),
          offline: options.offline ?? false
        }),
      options.signal,
      (late) => {
        void Promise.resolve()
          .then(() => late.dispose())
          .catch(() => {});
      }
    );
    options.signal?.throwIfAborted();
    const owned = runner;
    const session = createGemma4V2Session({
      tokenizer,
      prepareImage: dependencies.prepareImage ?? imagePreparer!.prepareImage,
      modelName: GEMMA4_V2_MODEL,
      ...config,
      ...(scoringKernel === undefined ? {} : {scoringKernel}),
      ...(options.maxLength === undefined ? {} : {maxLength: options.maxLength}),
      runner: {
        run: owned.run.bind(owned),
        releaseCachedMemory: owned.releaseCachedMemory.bind(owned),
        async dispose() {
          try {
            offlinePreparation?.dispose();
            offlinePreparation = undefined;
            await owned.dispose();
          } finally {
            source.dispose();
          }
        }
      }
    });
    let closed = false;
    return {
      modelId: GEMMA4_V2_MODEL,
      ...session,
      dispose() {
        closed = true;
        return session.dispose();
      },
      async prepareOffline(options?: Parameters<NonNullable<typeof offlinePreparation>['prepareOffline']>[0]) {
        if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The Gemma 4 v2 session is closed.');
        if (!offlinePreparation)
          throw new WebDecisionError(
            'ASSET_LOAD_FAILED',
            'A cache and pinned graph assets are required for offline preparation.'
          );
        await offlinePreparation.prepareOffline(options);
      },
      async isOfflineReady(signal?: AbortSignal | null) {
        if (closed) throw new WebDecisionError('SESSION_CLOSED', 'The Gemma 4 v2 session is closed.');
        if (!offlinePreparation) return false;
        return offlinePreparation.isOfflineReady(signal);
      }
    };
  } catch (error) {
    offlinePreparation?.dispose();
    if (runner) await Promise.allSettled([Promise.resolve().then(() => runner!.dispose())]);
    source.dispose();
    throw error;
  }
}
