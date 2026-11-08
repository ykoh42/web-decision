import {validateManifest} from '../assets.js';
import {readPinnedManifest} from '../pinned-manifest.js';

export const KEV06_MANIFEST_SHA256 = '8577804e638051cb056af51fee89eb18ea5de85ccec4f196594d830c16fdc767';

/** Accept only the verified default Kev 0.6B asset manifest. */
export async function readKev06ManifestResponse(response, signal) {
  const manifest = await readPinnedManifest(response, signal, KEV06_MANIFEST_SHA256, 'Kev 0.6B');
  return validateManifest(manifest);
}
