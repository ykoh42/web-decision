import bindingsUrl from '../../assets/mapika2b/text-bindings.json?url&no-inline';
import wasmUrl from '../../assets/mapika2b/text.wasm?url&no-inline';
import {parseJson, readVerifiedAsset, type AssetCache, type AssetFileMetadata} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import type {Mapika2bTextBinding} from './text-runtime.mjs';

const wasmPin = {bytes: 4_915_210, sha256: 'da20aa80a0dd0cb8b4e1c58ce6a0f313ffb5f82675e610be18dcc5805cfe70d4'};
const bindingsPin = {bytes: 26_768, sha256: '0e9c2076ea327a49f78be0c6098531c007cf414839920b80897a5e419f4ce6f7'};

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
