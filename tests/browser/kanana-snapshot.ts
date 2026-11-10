import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {stat} from 'node:fs/promises';
import {homedir} from 'node:os';
import {resolve} from 'node:path';
import {
  KANANA_FILES,
  KANANA_MODEL_ID,
  KANANA_REPOSITORY,
  KANANA_REVISION
} from '../../packages/web-decision/src/kanana/contract.mts';
import {KANANA_SAFETENSORS_SPEC} from '../../packages/web-decision/src/kanana/safetensors-spec.mts';

const defaultSnapshotRoot = resolve(
  homedir(),
  '.cache/huggingface/hub',
  `models--${KANANA_REPOSITORY.replace('/', '--')}`,
  'snapshots'
);

export const kananaModelId = KANANA_MODEL_ID;
export const kananaRevision = KANANA_REVISION;
export const kananaCheckpointSha256 = KANANA_SAFETENSORS_SPEC.fileSha256;
export const kananaSnapshotDirectory = resolve(
  process.env.WEB_DECISION_KANANA_SNAPSHOT ?? resolve(defaultSnapshotRoot, KANANA_REVISION)
);
export const kananaUpstreamBase = `https://huggingface.co/${KANANA_REPOSITORY}/resolve/${KANANA_REVISION}/`;

async function sha256File(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

async function verifyFile(path: string, name: string, expected: {bytes: number; sha256: string}): Promise<void> {
  const metadata = await stat(path);
  if (metadata.size !== expected.bytes)
    throw new Error(`Pinned Kanana ${name} has ${metadata.size} bytes; expected ${expected.bytes}.`);
  const actualSha256 = await sha256File(path);
  if (actualSha256 !== expected.sha256)
    throw new Error(`Pinned Kanana ${name} has SHA-256 ${actualSha256}; expected ${expected.sha256}.`);
}

let verification: Promise<void> | undefined;

/** Accept a cache directory from another commit only when its runtime inputs match this package's pin. */
export function verifyPinnedKananaSnapshot(): Promise<void> {
  verification ??= (async () => {
    for (const [name, expected] of Object.entries(KANANA_FILES)) {
      await verifyFile(resolve(kananaSnapshotDirectory, name), name, expected);
    }
    await verifyFile(resolve(kananaSnapshotDirectory, KANANA_SAFETENSORS_SPEC.path), KANANA_SAFETENSORS_SPEC.path, {
      bytes: KANANA_SAFETENSORS_SPEC.totalBytes,
      sha256: KANANA_SAFETENSORS_SPEC.fileSha256
    });
  })();
  return verification;
}
