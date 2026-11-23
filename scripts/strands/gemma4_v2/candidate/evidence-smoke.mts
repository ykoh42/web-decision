import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {verifyGemmaV2Candidate} from './gemma4-v2-candidate-evidence.mts';

const packageRoot = fileURLToPath(new URL('../../../../packages/web-decision/', import.meta.url));
const assetDirectory = 'assets/strands-gemma4-v2';
const modelId = 'strands-decider-e2b-gemma4-v2-2611';
const sha256 = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');

type AssetPin = {bytes: number; sha256: string};
type SourceSnapshot = {file: string; sha256: string};

async function assertAssetPin(path: string, expected: AssetPin): Promise<Buffer> {
  const bytes = await readFile(path);
  assert.deepEqual({bytes: bytes.length, sha256: sha256(bytes)}, expected, path);
  return bytes;
}

async function verifyPendingManifest(installed: string, sourceSnapshot: SourceSnapshot[], candidatePins: unknown) {
  const manifestPath = join(assetDirectory, 'manifest.json');
  const [manifestBytes, pending] = await Promise.all([
    readFile(join(packageRoot, manifestPath)),
    readFile(join(packageRoot, 'src/catalog/provenance', modelId + '.json'), 'utf8').then((text) => JSON.parse(text))
  ]);
  const manifest = JSON.parse(manifestBytes.toString('utf8')) as {
    modelId: string;
    experimental: boolean;
    files: Record<string, AssetPin>;
  };
  const manifestSha256 = sha256(manifestBytes);
  assert.equal(pending.status, 'pending');
  assert.equal(pending.modelId, modelId);
  assert.equal(pending.candidateArtifacts.manifest.sha256, manifestSha256);
  assert.equal(manifest.modelId, modelId);
  assert.equal(pending.candidateArtifacts.status, 'pending');
  assert.equal(manifest.experimental, false);
  const runtimePins = Object.fromEntries(Object.entries(manifest.files).filter(([name]) => !name.endsWith('.json')));
  assert.deepEqual(Object.keys(runtimePins).sort(), ['global.bin', 'graph.wasm', 'local.bin', 'scoring.bin']);
  assert.deepEqual(candidatePins, runtimePins, 'Private candidate pins differ from the pending manifest.');
  for (const [name, pin] of Object.entries(runtimePins)) {
    const sourceBytes = await assertAssetPin(join(packageRoot, assetDirectory, name), pin);
    assert.deepEqual(await readFile(join(installed, 'dist/gemma4-v2', name)), sourceBytes, name);
  }
  for (const name of ['source-inventory.json', 'decoder-bindings.json', 'vision-bindings.json']) {
    const pin = manifest.files[name];
    assert.ok(pin, 'Missing Gemma v2 JSON asset pin: ' + name + '.');
    const path = join(assetDirectory, name);
    await assertAssetPin(join(packageRoot, path), pin);
    assert.ok(
      sourceSnapshot.some(({file, sha256}) => file === path && sha256 === pin.sha256),
      'The candidate did not bind the pending Gemma v2 JSON asset: ' + name + '.'
    );
  }
  const licenseSha256 = sha256(await readFile(join(packageRoot, 'STRANDS-GEMMA4-V2-LICENSE')));
  const noticeSha256 = sha256(await readFile(join(packageRoot, 'STRANDS-GEMMA4-V2-NOTICE')));
  assert.equal(licenseSha256, pending.referenceFiles.license.sha256);
  assert.equal(noticeSha256, pending.referenceFiles.notice.sha256);
  return {manifestSha256, licenseSha256, noticeSha256};
}

const input = process.argv[2];
if (!input) throw new Error('Usage: evidence-smoke.mts <private-candidate-report>');
const reportPath = resolve(input);
const report = JSON.parse(await readFile(reportPath, 'utf8'));
const candidate = await verifyGemmaV2Candidate(reportPath, report.installedBundle);
const evidence = await verifyPendingManifest(candidate.installed, candidate.sourceSnapshot, report.pins);
const directory = await mkdtemp(join(tmpdir(), 'gemma-v2-evidence-test-'));
try {
  for (const key of ['archiveSha256', 'sourceSnapshotSha256', 'tokenizerSha256', 'consumerLockSha256']) {
    const changed = {...report, [key]: '0'.repeat(64)};
    const path = join(directory, `${key}.json`);
    await writeFile(path, JSON.stringify(changed));
    await assert.rejects(verifyGemmaV2Candidate(path, report.installedBundle));
  }
} finally {
  await rm(directory, {recursive: true, force: true});
}
console.log('Private candidate evidence accepts the exact install and rejects four altered digests.');
console.log(JSON.stringify({status: 'pending', archiveSha256: candidate.archiveSha256, ...evidence}));
