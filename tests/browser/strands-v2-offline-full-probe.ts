// Bootstrap the consumer's JavaScript while online; model assets are tested after disconnecting.
import {jpegUrl, magickUrl} from '../../packages/web-decision/test/strands-v2-offline-codecs-bootstrap.mts';
import '../../packages/web-decision/src/strands/qwen35-image-input.mts';
import {createGemma4V2ImagePreparer} from '../../packages/web-decision/src/strands/gemma4-v2-image.mts';
import {createGemma4V2ImageInserter} from '../../packages/web-decision/src/strands/gemma4-v2-image-runner.mts';
import {loadGemma4V2Model} from '../../packages/web-decision/src/strands/gemma4-v2-loader.mts';
import {
  createGemma4V2RunnerFactory,
  type Gemma4V2GraphAssets
} from '../../packages/web-decision/src/strands/gemma4-v2-runner-factory.mts';
import {openGemma4V2SourceFromInventory} from '../../packages/web-decision/src/strands/gemma4-v2-source.mts';
import inventory from '../../packages/web-decision/assets/strands-gemma4-v2/source-inventory.json' with {type: 'json'};

type Metadata = {bytes: number; sha256: string};
type Params = {
  base: string;
  cache: string;
  pins: Record<string, Metadata>;
  requiredBytes: number;
  checkCapacity: boolean;
};
type CodecFixture = {encoded: string; expected: {tokens: number; pixelsSha256: string; positionsSha256: string}};
type State = {base: URL; cache: Cache; graph: Gemma4V2GraphAssets; attempts: string[]};
let state: State | undefined;
const sourceHash = 'c4c1f9ca91d272d88f2559dd186fa0d37e878fa70d378c44a20de978179366aa';
const mirrorPrefix =
  'https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v2-2611/resolve/a5f32ea1bffe2e91c25c0c641d444e8115e28212/';

declare global {
  interface Window {
    gemmaOfflineBootstrap: (params: Params) => Promise<{quota?: number; usage?: number; persisted: boolean}>;
    gemmaOfflinePrepare: () => Promise<{ready: boolean; progressBytes: number}>;
    gemmaOfflineRun: (verifyAll: boolean, codecs: CodecFixture[]) => Promise<unknown>;
    gemmaOfflineFetchAttempts: () => string[];
  }
}

function environment(): State {
  if (!state) throw new Error('Offline harness is not initialized');
  return state;
}

function dependencies(current: State) {
  const openSource = (options: Parameters<typeof openGemma4V2SourceFromInventory>[3]) =>
    openGemma4V2SourceFromInventory(inventory as never, sourceHash, new URL('checkpoint/', current.base), options);
  const createRunner = createGemma4V2RunnerFactory({
    assets: current.graph,
    createImageInserter: ({runtime, device, vm, source, signal}) =>
      createGemma4V2ImageInserter(runtime, device, vm, source, signal)
  });
  return {openSource, createRunner, graphAssets: Object.values(current.graph)};
}

async function hash(bytes: Uint8Array): Promise<string> {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new Uint8Array(bytes))), (value) =>
    value.toString(16).padStart(2, '0')
  ).join('');
}

function image() {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas is unavailable');
  context.fillStyle = 'rgb(220,20,30)';
  context.fillRect(0, 0, 32, 64);
  context.fillStyle = 'rgb(15,40,210)';
  context.fillRect(32, 0, 32, 64);
  return canvas.toDataURL('image/png').split(',')[1]!;
}

window.gemmaOfflineBootstrap = async (params) => {
  if (state) throw new Error('Offline harness is already initialized');
  if (!jpegUrl || !magickUrl) throw new Error('Image codec bootstrap is incomplete');
  const base = new URL(params.base);
  const asset = (name: string) => ({base: new URL('graph/', base), name, metadata: params.pins[name]!});
  const graph = {wasm: asset('linked.wasm'), localFrequency: asset('local.bin'), globalFrequency: asset('global.bin')};
  const cache = await caches.open(params.cache);
  const attempts: string[] = [];
  const originalFetch = globalThis.fetch.bind(globalThis);
  globalThis.fetch = (input, init) => {
    const request = new Request(input, init);
    attempts.push(request.url);
    if (!request.url.startsWith(mirrorPrefix)) return originalFetch(request);
    const path = request.url.slice(mirrorPrefix.length);
    return originalFetch(new Request(new URL('checkpoint/' + path, base), request));
  };
  state = {base, cache, graph, attempts};
  const estimate = await navigator.storage.estimate();
  const available = estimate.quota === undefined ? undefined : estimate.quota - (estimate.usage ?? 0);
  if (params.checkCapacity && available !== undefined && available < params.requiredBytes)
    throw new Error(`Browser quota has ${available} bytes free; at least ${params.requiredBytes} bytes are needed.`);
  const persisted = await navigator.storage.persist();
  return {...estimate, persisted};
};

window.gemmaOfflinePrepare = async () => {
  const current = environment();
  const session = await loadGemma4V2Model({cache: current.cache, offline: false}, dependencies(current));
  let progressBytes = 0;
  let reportedBytes = 0;
  try {
    await session.prepareOffline({
      onProgress(progress) {
        if (progress.loadedBytes - reportedBytes >= 1024 * 1024 * 1024) {
          console.log(`Verified source ${progress.loadedBytes}/${progress.totalBytes} bytes`);
          reportedBytes = progress.loadedBytes;
        }
        progressBytes = progress.loadedBytes;
      }
    });
    return {ready: await session.isOfflineReady(), progressBytes};
  } finally {
    await session.dispose();
  }
};

window.gemmaOfflineRun = async (verifyAll, codecs) => {
  const current = environment();
  current.attempts.length = 0;
  const source = await openGemma4V2SourceFromInventory(
    inventory as never,
    sourceHash,
    new URL('checkpoint/', current.base),
    {cache: current.cache, offline: true}
  );
  let verified = false;
  try {
    verified = verifyAll ? await source.verifyOfflineWeights() : await source.hasCompleteOfflineWeights();
  } finally {
    source.dispose();
  }
  if (!verified) throw new Error('Offline checkpoint verification failed');
  const preparer = createGemma4V2ImagePreparer({cache: current.cache, offline: true});
  const codecResults = [];
  for (const fixture of codecs) {
    const result = await preparer.prepareImage(fixture.encoded);
    codecResults.push({
      tokens: result.tokens,
      pixelsSha256: await hash(new Uint8Array(result.value.pixels.buffer)),
      positionsSha256: await hash(new Uint8Array(result.value.positions.buffer))
    });
  }
  const session = await loadGemma4V2Model({cache: current.cache, offline: true}, dependencies(current));
  try {
    const ready = await session.isOfflineReady();
    const answer = await session.systemOne({
      state: 'The image contains two color regions.',
      images: [image()],
      questions: {
        color: {type: 'choice', instructions: 'Which color is on the left?', criteria: {red: 'red', blue: 'blue'}}
      }
    });
    return {ready, verified, codecResults, answer, attempts: [...current.attempts]};
  } finally {
    await session.dispose();
  }
};
window.gemmaOfflineFetchAttempts = () => [...environment().attempts];
