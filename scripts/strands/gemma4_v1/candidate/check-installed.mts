import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {dirname, extname, join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {verifyArchiveInstall} from '../../../release/verify-release-evidence.ts';

const reportPath = resolve(process.argv[2]!);
const report = JSON.parse(await readFile(reportPath, 'utf8'));
const archive = await verifyArchiveInstall(report.archive.archivePath, report.archive.installedRoot);
assert.deepEqual(archive, report.archive);
const {loadDecisionModel, listModels} = await import(pathToFileURL(join(archive.installedRoot, 'dist/index.js')).href);
assert.deepEqual(listModels(), [{modelId: report.modelId, status: 'private-test-candidate'}]);
await assert.rejects(async () => loadDecisionModel({modelId: 'strands-decider-e2b-gemma4-v2-2611'}), {
  code: 'INVALID_INPUT'
});
const requests: string[] = [];
const prior = globalThis.fetch;
const root = pathToFileURL(join(archive.installedRoot, 'dist/gemma4-v1/')).href;
let bad = false;
globalThis.fetch = async (url, options) => {
  const path = String(url);
  assert.equal(extname(dirname(path)), '');
  assert.ok(path.startsWith(root));
  options?.signal?.throwIfAborted();
  const name = path.slice(root.length);
  assert.ok(Object.hasOwn(report.assets, name));
  requests.push(name);
  const bytes = await readFile(new URL(path));
  if (bad && name === 'graph.wasm') bytes[0] = bytes[0]! ^ 1;
  return new Response(new Uint8Array(bytes), {headers: {'Content-Length': String(bytes.length)}});
};
try {
  await assert.rejects(loadDecisionModel({modelId: report.modelId, cache: false}), {code: 'DEVICE_RESOURCE_LIMIT'});
  assert.deepEqual(requests.sort(), Object.keys(report.assets).sort());
  bad = true;
  await assert.rejects(loadDecisionModel({modelId: report.modelId, cache: false}), /integrity/i);
} finally {
  globalThis.fetch = prior;
}
console.log(
  JSON.stringify({
    archiveSha256: archive.archiveSha256,
    pinnedAssets: 5,
    wrongModelRejected: true,
    tamperedGraphRejected: true,
    noGpuReported: true,
    gpuExecuted: false
  })
);
