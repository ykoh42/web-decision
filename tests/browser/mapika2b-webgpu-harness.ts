import {
  loadMapika2bCandidate,
  loadMapika2bCuratedForTest,
  mapika2bPackagedAsset,
  warmMapika2bPackagedCode
} from 'mapika2b-qa-candidate';
import {MAPIKA2B_MAX_OPTIONS} from '../../packages/web-decision/src/mapika2b/contract.mts';
import type {Mapika2bTextBinding} from '../../packages/web-decision/src/mapika2b/text-runtime.mts';
import {
  MAPIKA2B_TEST_CACHE,
  requestFromMapikaFixture,
  type MapikaBrowserHarness,
  type MapikaCacheMode
} from './mapika2b-evidence.mts';

interface CompiledTextManifest {
  readonly wasmSha256: string;
  readonly bindings: readonly Mapika2bTextBinding[];
}

type CandidateArtifacts = Parameters<typeof loadMapika2bCandidate>[0]['artifacts'];
let session: Awaited<ReturnType<typeof loadMapika2bCandidate>> | undefined;

async function fetchArtifact(path: string, cache: Cache | null, mode: MapikaCacheMode): Promise<Response> {
  const url = new URL(path, location.href);
  if (mode === 'offline') {
    const response = await cache?.match(url);
    if (!response) throw new Error(`Mapika QA cache is missing packaged artifact ${url.pathname}.`);
    return response;
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Mapika candidate artifact ${path} returned HTTP ${response.status}.`);
  if (cache) await cache.put(url, response.clone());
  return response;
}

async function loadArtifacts(
  installedArchive: boolean,
  cache: Cache | null,
  mode: MapikaCacheMode
): Promise<CandidateArtifacts> {
  const artifactUrl = installedArchive
    ? mapika2bPackagedAsset
    : (name: 'text.wasm' | 'text-bindings.json') => '/mapika2b/' + name;
  const [wasmResponse, manifestResponse] = await Promise.all([
    fetchArtifact(artifactUrl('text.wasm'), cache, mode),
    fetchArtifact(artifactUrl('text-bindings.json'), cache, mode)
  ]);
  const textWasm = new Uint8Array(await wasmResponse.arrayBuffer());
  const {wasmSha256, bindings} = (await manifestResponse.json()) as CompiledTextManifest;
  return {textWasm, textWasmSha256: wasmSha256, textBindings: bindings};
}

const harness: MapikaBrowserHarness = {
  stage: warmMapika2bPackagedCode,

  async curated(item, cacheMode = 'none') {
    const progress: {loadedBytes: number; totalBytes: number; fraction: number}[] = [];
    const model = await loadMapika2bCuratedForTest({
      cache: cacheMode !== 'none',
      offline: cacheMode === 'offline',
      onProgress: (event: (typeof progress)[number]) => progress.push(event)
    });
    try {
      if (cacheMode === 'fill') await model.prepareOffline();
      return {result: await model.systemOne(requestFromMapikaFixture(item)), progress};
    } finally {
      await model.dispose();
    }
  },

  async initialize(installedArchive, cacheMode = 'none') {
    if (session) throw new Error('Mapika candidate is already loaded.');
    const cache = cacheMode === 'none' ? null : await caches.open(MAPIKA2B_TEST_CACHE);
    const artifacts = await loadArtifacts(installedArchive, cache, cacheMode);
    const loaded = await loadMapika2bCandidate({artifacts, cache, offline: cacheMode === 'offline'});
    try {
      if (cacheMode === 'fill') await loaded.prepareOffline();
    } catch (cause) {
      await loaded.dispose();
      throw cause;
    }
    session = loaded;
    return {wasmSha256: artifacts.textWasmSha256};
  },

  async evaluate(item) {
    if (!session) throw new Error('Mapika candidate is not loaded.');
    try {
      const {result, logitsByRow, rows} = await session.evaluate(requestFromMapikaFixture(item));
      return {
        status: 'accepted',
        result: JSON.parse(JSON.stringify(result)) as typeof result,
        questionNames: Object.keys(result.answers),
        rows: rows.map(({inputIds, slots}) => ({
          inputIds,
          slots: slots.map(({tokenIndex}) => tokenIndex),
          optionCounts: slots.map(({optionCount}) => optionCount),
          temperatures: slots.map(({temperature}) => temperature)
        })),
        logits: rows.map(({slots}, rowIndex) =>
          slots.map(({optionCount}, index) => {
            const start = index * MAPIKA2B_MAX_OPTIONS;
            return Array.from(logitsByRow[rowIndex]!.subarray(start, start + optionCount));
          })
        )
      };
    } catch (cause) {
      if (!(cause instanceof Error)) throw cause;
      const code = 'code' in cause && typeof cause.code === 'string' ? cause.code : undefined;
      const sources: string[] = [];
      for (let error: unknown = cause; error instanceof Error; error = error.cause) sources.push(error.message);
      return {status: 'rejected', message: sources.join(' Caused by: '), ...(code ? {code} : {})};
    }
  },

  async dispose() {
    await session?.dispose();
    session = undefined;
  }
};

window.mapika2bGpuHarness = harness;
