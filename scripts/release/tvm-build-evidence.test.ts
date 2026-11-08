import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {link, mkdir, mkdtemp, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import test from 'node:test';
import {verifyTvmBuildEvidence, type TvmBuildModel} from './tvm-build-evidence.ts';

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), 'web-decision-tvm-evidence-'));
  const bytes = Buffer.from('independent TVM build');
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const models: TvmBuildModel[] = [
    {modelId: 'model-a', tvmVersion: '0.27.0', manifestFiles: {'model.wasm': {bytes: bytes.length, sha256}}}
  ];
  const report = {
    evidenceSchema: 'web-decision-tvm-build-v1',
    modelId: 'model-a',
    artifact: 'model.wasm',
    wasm: 'rebuilt.wasm',
    status: 'success',
    target: 'webgpu / wasm32',
    tvmVersion: '0.27.0',
    embeddedBuildRoot: false,
    wasmBytes: bytes.length,
    wasmSha256: sha256
  };
  const record = {
    modelId: 'model-a',
    artifact: 'model.wasm',
    buildRoot: '.',
    wasm: 'rebuilt.wasm',
    report: 'build.json'
  };
  await writeFile(join(root, 'rebuilt.wasm'), bytes);
  await writeFile(join(root, 'build.json'), JSON.stringify(report));
  await writeFile(join(root, 'evidence.json'), JSON.stringify({schemaVersion: 1, builds: [record]}));
  return {root, bytes, sha256, models, report, record, path: join(root, 'evidence.json')};
}

test('matches independently supplied WASM and successful build report to the catalog pin', async (t) => {
  const f = await fixture();
  t.after(() => rm(f.root, {recursive: true, force: true}));
  const [result] = await verifyTvmBuildEvidence(f.path, f.models);
  assert.equal(result?.modelId, 'model-a');
  assert.equal(result?.artifact, 'model.wasm');
  assert.equal(result?.sha256, f.sha256);
  assert.match(result?.evidenceSha256 ?? '', /^[a-f0-9]{64}$/u);
  assert.match(result?.reportSha256 ?? '', /^[a-f0-9]{64}$/u);
});

test('rejects missing, duplicated and unrecognized build records', async (t) => {
  const f = await fixture();
  t.after(() => rm(f.root, {recursive: true, force: true}));
  for (const builds of [[], [f.record, f.record], [{...f.record, modelId: 'other'}]]) {
    await writeFile(f.path, JSON.stringify({schemaVersion: 1, builds}));
    await assert.rejects(verifyTvmBuildEvidence(f.path, f.models), /TVM build|TVM WASM/u);
  }
});

test('requires each WASM for every model, including multiple artifacts', async (t) => {
  const f = await fixture();
  t.after(() => rm(f.root, {recursive: true, force: true}));
  const models: TvmBuildModel[] = [
    {
      ...f.models[0]!,
      manifestFiles: {
        'model.wasm': {bytes: f.bytes.length, sha256: f.sha256},
        'aux.wasm': {bytes: f.bytes.length, sha256: f.sha256}
      }
    },
    {
      modelId: 'model-b',
      tvmVersion: f.report.tvmVersion,
      manifestFiles: {'model.wasm': {bytes: f.bytes.length, sha256: f.sha256}}
    }
  ];
  await writeFile(join(f.root, 'aux.json'), JSON.stringify({...f.report, artifact: 'aux.wasm'}));
  await writeFile(join(f.root, 'b.json'), JSON.stringify({...f.report, modelId: 'model-b'}));
  const builds = [
    f.record,
    {...f.record, artifact: 'aux.wasm', report: 'aux.json'},
    {...f.record, modelId: 'model-b', report: 'b.json'}
  ];
  await writeFile(f.path, JSON.stringify({schemaVersion: 1, builds}));
  assert.equal((await verifyTvmBuildEvidence(f.path, models)).length, 3);
  await writeFile(f.path, JSON.stringify({schemaVersion: 1, builds: builds.slice(0, 2)}));
  await assert.rejects(verifyTvmBuildEvidence(f.path, models), /Every curated TVM WASM/u);
});

test('rejects a different rebuilt WASM or an inconsistent build report', async (t) => {
  const f = await fixture();
  t.after(() => rm(f.root, {recursive: true, force: true}));
  await writeFile(join(f.root, 'rebuilt.wasm'), Buffer.from(f.bytes).fill(0));
  await assert.rejects(verifyTvmBuildEvidence(f.path, f.models), /digest differs/u);
  await writeFile(join(f.root, 'rebuilt.wasm'), f.bytes);
  await writeFile(join(f.root, 'other.wasm'), f.bytes);
  for (const report of [
    {...f.report, evidenceSchema: undefined},
    {...f.report, modelId: 'other'},
    {...f.report, artifact: 'other.wasm'},
    {...f.report, wasm: 'other.wasm'},
    {...f.report, status: 'failed'},
    {...f.report, wasmSha256: '0'.repeat(64)},
    {...f.report, embeddedBuildRoot: true},
    {...f.report, tvmVersion: '0.26.0'}
  ]) {
    await writeFile(join(f.root, 'build.json'), JSON.stringify(report));
    await assert.rejects(verifyTvmBuildEvidence(f.path, f.models), /TVM build|TVM WASM/u);
  }
});

test('rejects a package file or hardlink as the independent build', async (t) => {
  const f = await fixture();
  t.after(() => rm(f.root, {recursive: true, force: true}));
  const source = join(f.root, 'source');
  const packaged = join(source, 'model.wasm');
  await mkdir(source);
  await writeFile(packaged, f.bytes);
  const models = [{...f.models[0]!, sourceArtifacts: {'model.wasm': packaged}}];
  await writeFile(f.path, JSON.stringify({schemaVersion: 1, builds: [{...f.record, wasm: 'source/model.wasm'}]}));
  await writeFile(join(f.root, 'build.json'), JSON.stringify({...f.report, wasm: 'source/model.wasm'}));
  await assert.rejects(verifyTvmBuildEvidence(f.path, models, [source]), /outside the source/u);
  await rm(join(f.root, 'rebuilt.wasm'));
  await link(packaged, join(f.root, 'rebuilt.wasm'));
  await writeFile(f.path, JSON.stringify({schemaVersion: 1, builds: [f.record]}));
  await writeFile(join(f.root, 'build.json'), JSON.stringify(f.report));
  await assert.rejects(verifyTvmBuildEvidence(f.path, models, [source]), /packaged source file/u);
});

test('includes evidence and report digests for detecting changes between checks', async (t) => {
  const f = await fixture();
  t.after(() => rm(f.root, {recursive: true, force: true}));
  const [before] = await verifyTvmBuildEvidence(f.path, f.models);
  await writeFile(join(f.root, 'build.json'), JSON.stringify({...f.report, additional: true}));
  const [afterReport] = await verifyTvmBuildEvidence(f.path, f.models);
  assert.notEqual(before?.reportSha256, afterReport?.reportSha256);
  await writeFile(f.path, JSON.stringify({schemaVersion: 1, builds: [f.record], additional: true}));
  const [afterEvidence] = await verifyTvmBuildEvidence(f.path, f.models);
  assert.notEqual(afterReport?.evidenceSha256, afterEvidence?.evidenceSha256);
});
