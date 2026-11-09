// Internal source binding; this is not a public package entry or model loader.
import {createH2oPublicSourceFactory} from './source.mjs';
import type {H2oHashConstructor, H2oOpenedSource} from './source.mjs';
import {Hash} from './sha256.mjs';
import {validateH2oSignal} from './signal.mjs';
import {readH2oInventory} from './source-evidence.mjs';
import {H2O_BASELINE_MODEL_ID, h2oGraphIdentity} from './plan.mjs';

/** Open the pinned inventory and unsigned path-free projection; no model execution. */
type OpenH2oAssetSourceOptions = {
  assetsUrl?: string | URL;
  inventoryBytes?: Uint8Array;
  signal?: AbortSignal | undefined;
  model?: string;
};

export async function openH2oAssetSource({
  assetsUrl,
  inventoryBytes,
  signal,
  model = H2O_BASELINE_MODEL_ID
}: OpenH2oAssetSourceOptions = {}): Promise<H2oOpenedSource> {
  h2oGraphIdentity(model);
  validateH2oSignal(signal);
  signal?.throwIfAborted();
  const selected =
    inventoryBytes === undefined ? await readH2oInventory(assetsUrl as string | URL, signal) : inventoryBytes;
  const openSource = createH2oPublicSourceFactory({
    inventoryBytes: selected,
    HashClass: Hash as unknown as H2oHashConstructor,
    model
  });
  signal?.throwIfAborted();
  return openSource({base: assetsUrl as string | URL, signal});
}
