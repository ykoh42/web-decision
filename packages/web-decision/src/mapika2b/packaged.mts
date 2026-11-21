import bindingsUrl from '../../assets/mapika2b/text-bindings.json?url&no-inline';
import wasmUrl from '../../assets/mapika2b/text.wasm?url&no-inline';
import {parseJson, readVerifiedAsset, type AssetCache, type AssetFileMetadata} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import type {Mapika2bTextBinding} from './text-runtime.mjs';

const wasmPin = {bytes: 5_053_519, sha256: '95e6bea454ee9d5dbbe40e2e84523d53077f7f1fa285f609c0e5422146b40bb2'};
const bindingsPin = {bytes: 26_768, sha256: 'af19172e2396b40141804a2414c44172caa46b12e464640e5731428f9c93ba1c'};

interface AssetOptions {
  readonly cache?: AssetCache | null | undefined;
  readonly offline?: boolean | undefined;
  readonly signal?: AbortSignal | null | undefined;
}

function readPackagedAsset(url: string, pin: AssetFileMetadata, options: AssetOptions): Promise<ArrayBuffer> {
  const asset = new URL(url, import.meta.url);
  const name = decodeURIComponent(asset.pathname.slice(asset.pathname.lastIndexOf('/') + 1));
  return readVerifiedAsset(new URL('.', asset), {files: {[name]: pin}}, name, options);
}

/** Read the immutable, independently compiled text graph and its argument mapping. */
export async function openPackagedMapika2bText(options: AssetOptions = {}) {
  const [wasm, manifest] = await Promise.all([
    readPackagedAsset(wasmUrl, wasmPin, options),
    readPackagedAsset(bindingsUrl, bindingsPin, options)
  ]);
  const data = parseJson(manifest, 'Mapika text bindings') as {
    readonly wasmSha256?: unknown;
    readonly bindings?: unknown;
  };
  if (data.wasmSha256 !== wasmPin.sha256 || !Array.isArray(data.bindings) || data.bindings.length !== 319)
    throw new WebDecisionError('INVALID_ASSET', 'The packaged Mapika text binding manifest differs from its graph.');
  return {
    textWasm: new Uint8Array(wasm),
    textWasmSha256: wasmPin.sha256,
    textBindings: data.bindings as readonly Mapika2bTextBinding[]
  };
}
