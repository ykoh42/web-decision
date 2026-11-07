import {validateManifest} from '../assets.js';
import {readPinnedManifest} from '../pinned-manifest.js';

export const KEV06_MODEL = 'kev-0.6b-q4f16';
export const KEV06_MANIFEST_SHA256 = 'bc79c92a7011d2aa1aff4ede66c06cc365de81e744507256afb5c0449870716a';

/** Accept only the verified default Kev 0.6B asset manifest. */
export async function readKev06ManifestResponse(response, signal) {
  const manifest = await readPinnedManifest(response, signal, KEV06_MANIFEST_SHA256, 'Kev 0.6B');
  return validateManifest(manifest);
}
