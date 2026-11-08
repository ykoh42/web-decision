import {uploadParameterEntries} from '../parameter-upload.mjs';
import type {FlashManifest, FlashManifestEntry, FlashQ4CandidateEntry, FlashRange, FlashTensor} from './types.mjs';

interface FlashUploadCallbacks {
  signal?: AbortSignal | undefined;
  createTensor(entry: FlashManifestEntry): FlashTensor | PromiseLike<FlashTensor>;
  readRange(entry: FlashQ4CandidateEntry, index: number, part: FlashRange): ArrayBuffer | PromiseLike<ArrayBuffer>;
  writeRange(tensor: FlashTensor, offset: number, bytes: Uint8Array): unknown | PromiseLike<unknown>;
}

const uploadParameterEntriesWithFlashCallbacks = uploadParameterEntries as unknown as (
  entries: FlashManifestEntry[],
  callbacks: FlashUploadCallbacks
) => Promise<unknown[]>;

/** Read, verify and upload the pinned Perplexity graph parameters. */
export async function uploadFlashParameters(
  manifest: FlashManifest,
  callbacks: FlashUploadCallbacks
): Promise<unknown[]> {
  if (!manifest || manifest.state !== 'pplx-q4-joint-assets-pending') throw new Error('Invalid asset manifest');
  return uploadParameterEntriesWithFlashCallbacks(manifest.entries, callbacks);
}
