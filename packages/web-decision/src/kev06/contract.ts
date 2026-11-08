import {validateManifest} from '../assets.js';
import {readPinnedManifest} from '../pinned-manifest.js';

type Kev06Manifest = {
  readonly model: 'kev-0.6b-q4f16';
  readonly experimental: boolean;
  readonly files: Readonly<Record<string, {readonly bytes: number; readonly sha256: string}>>;
  readonly parameterCount: number;
  readonly maxInputTokens?: number;
  readonly maxStateTokens: number;
  readonly maxBranchTokens: number;
};

export const KEV06_MANIFEST_SHA256 = '57c554e757a96093669c1555e9e2564780c7eaba5e102b4fcd9ad0c535878fe2';

/** Accept only the verified default Kev 0.6B asset manifest. */
export async function readKev06ManifestResponse(response: Response, signal?: AbortSignal): Promise<Kev06Manifest> {
  const manifest = await readPinnedManifest(response, signal, KEV06_MANIFEST_SHA256, 'Kev 0.6B');
  return validateManifest(manifest) as Kev06Manifest;
}
