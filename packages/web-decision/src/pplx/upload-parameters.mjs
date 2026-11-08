import {uploadParameterEntries} from '../parameter-upload.mjs';

/** Read, verify and upload the pinned Perplexity graph parameters. */
export async function uploadFlashParameters(manifest, callbacks) {
  if (!manifest || manifest.state !== 'pplx-q4-joint-assets-pending') throw new Error('Invalid asset manifest');
  return uploadParameterEntries(manifest.entries, callbacks);
}
