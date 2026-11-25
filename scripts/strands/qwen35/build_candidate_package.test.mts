import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {copyFile, cp, mkdtemp, mkdir, readFile, readdir, realpath, rm, symlink, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import test from 'node:test';
import {fileURLToPath} from 'node:url';
import {assertQwen35ProductionArchive} from '../../release/pin-qwen35-provenance.ts';
import {verifyQwen35ExactPackage} from '../../release/strands-qwen35-release-evidence.ts';
import {
  buildCandidateAdapterModule,
  buildCandidateCatalogModule,
  buildQwen35CandidatePackage,
  candidatePackageMetadata,
  resolvePackageFilePath,
  validateCandidateOutputDirectory
} from './build_candidate_package.mts';

test('candidate catalog contains only the pending Qwen public entry', () => {
  const entry = {
    modelId: 'strands-decider-2b-qwen3.5-v1-2610',
    description: {
      adapter: 'strands-qwen35',
      assets: {manifestSha256: 'a'.repeat(64), requiredFiles: ['manifest.json', 'text.wasm']}
    }
  };
  const moduleSource = buildCandidateCatalogModule(entry);
  const serializedEntry = /^const entry = (.+);$/mu.exec(moduleSource)?.[1];

  assert.ok(serializedEntry);
  assert.deepEqual(JSON.parse(serializedEntry), entry);
  assert.match(moduleSource, /deepFreeze\(\[entry\]\)/u);
  assert.doesNotMatch(moduleSource, /kanana/u);
  assert.match(buildCandidateAdapterModule(), /strands-qwen35/u);
  assert.match(buildCandidateAdapterModule(), /@web-decision\/qwen35-candidate-adapter/u);
});

test('candidate package metadata is explicitly unpublishable', () => {
  const metadata = candidatePackageMetadata({'@huggingface/tokenizers': '0.2.0'});

  assert.equal(metadata.name, 'web-decision');
  assert.equal(metadata.version, '0.0.0-qwen35-candidate');
  assert.equal(metadata.private, true);
  assert.deepEqual(metadata.files, ['dist']);
  assert.equal(metadata.types, './dist/index.d.ts');
  assert.deepEqual(metadata.exports, {'.': {types: './dist/index.d.ts', import: './dist/index.js'}});
  assert.equal(Object.hasOwn(metadata, 'scripts'), false);
});

test('the pending-only builder rejects the verified public source without creating outputs', async (t) => {
  const output = await mkdtemp(join(tmpdir(), 'web-decision-qwen35-verified-source-'));
  t.after(() => rm(output, {recursive: true, force: true}));
  await assert.rejects(buildQwen35CandidatePackage(output), /only for pending provenance/u);
  assert.deepEqual(await readdir(output), []);
});

async function stagePendingSource(workspace: string): Promise<void> {
  const checkout = new URL('../../../', import.meta.url);
  const packageRoot = 'packages/web-decision';
  const sourceFiles = [
    `${packageRoot}/src`,
    `${packageRoot}/assets/qwen35`,
    `${packageRoot}/package.json`,
    `${packageRoot}/vite.config.ts`,
    '.toolchains/apache-tvm-src/web/src',
    'scripts/strands/qwen35/build_candidate_package.mts'
  ];
  await Promise.all(sourceFiles.map((path) => cp(new URL(path, checkout), join(workspace, path), {recursive: true})));
  for (const path of ['node_modules', `${packageRoot}/node_modules`, 'scripts/release']) {
    await symlink(fileURLToPath(new URL(path, checkout)), join(workspace, path), 'junction');
  }

  // Only the isolated fixture's provenance changes. The builder uses the real locked assets.
  const stagedPackage = join(workspace, packageRoot);
  const evidencePath = 'src/catalog/provenance/test-pending-qwen35.json';
  const evidence = JSON.stringify({status: 'pending', fixture: 'private package consumer test'});
  await Promise.all([
    writeFile(join(stagedPackage, evidencePath), evidence),
    writeFile(
      join(stagedPackage, 'src/catalog/model-provenance.json'),
      JSON.stringify({
        models: [
          {
            modelId: 'strands-decider-2b-qwen3.5-v1-2610',
            status: 'pending',
            evidence: {path: evidencePath, sha256: createHash('sha256').update(evidence).digest('hex')}
          }
        ]
      })
    )
  ]);
}

test('a strict TypeScript consumer imports the installed private candidate archive', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'web-decision-qwen35-packed-types-'));
  t.after(() => rm(root, {recursive: true, force: true}));
  const workspace = join(root, 'source');
  const output = join(root, 'output');
  await Promise.all([stagePendingSource(workspace), mkdir(output)]);
  execFileSync(
    process.execPath,
    ['--import', 'tsx', 'scripts/strands/qwen35/build_candidate_package.mts', '--output-dir', output],
    {cwd: workspace, stdio: 'inherit'}
  );
  const consumer = join(output, 'consumer');
  const installedPackagePath = await realpath(join(consumer, 'node_modules/web-decision'));
  const report = JSON.parse(await readFile(join(output, 'candidate-package.json'), 'utf8')) as {
    readonly status: string;
    readonly provenanceStatus: string;
    readonly package: {readonly exactInstallMatch: boolean; readonly private: boolean; readonly bundlePath: string};
  };
  assert.equal(report.status, 'unpublishable-pending-qwen-candidate');
  assert.equal(report.provenanceStatus, 'pending');
  assert.equal(report.package.private, true);
  assert.equal(report.package.exactInstallMatch, true);
  assert.equal(await realpath(join(output, report.package.bundlePath)), join(installedPackagePath, 'dist/index.js'));
  const archivePath = join(output, 'pack/web-decision-0.0.0-qwen35-candidate.tgz');
  const {archive} = await verifyQwen35ExactPackage(archivePath, installedPackagePath);
  assert.throws(() => assertQwen35ProductionArchive(archive), /stable npm version/u);
  const fixture = new URL('../../../tests/types/', import.meta.url);
  await Promise.all([
    copyFile(new URL('qwen35-candidate-consumer.mts', fixture), join(consumer, 'consumer.mts')),
    copyFile(new URL('tsconfig.base.json', fixture), join(consumer, 'tsconfig.base.json')),
    writeFile(
      join(consumer, 'tsconfig.json'),
      JSON.stringify({extends: './tsconfig.base.json', files: ['./consumer.mts']})
    )
  ]);
  const compiler = fileURLToPath(new URL('../../../node_modules/typescript/bin/tsc', import.meta.url));
  for (const [module, resolution] of [
    ['ESNext', 'Bundler'],
    ['NodeNext', 'NodeNext']
  ] as const) {
    execFileSync(
      process.execPath,
      [
        compiler,
        '--project',
        'tsconfig.json',
        '--module',
        module,
        '--moduleResolution',
        resolution,
        '--noEmit',
        'false',
        '--outDir',
        'compiled'
      ],
      {
        cwd: consumer,
        stdio: 'inherit'
      }
    );
  }
  execFileSync(process.execPath, ['compiled/consumer.mjs'], {cwd: consumer, stdio: 'inherit'});
});

test('candidate output validation requires a fresh real directory outside the workspace', async (t) => {
  const parent = await mkdtemp(join(tmpdir(), 'web-decision-qwen35-candidate-guard-'));
  t.after(() => rm(parent, {recursive: true, force: true}));
  const workspace = join(parent, 'workspace');
  const output = join(parent, 'output');
  const occupied = join(parent, 'occupied');
  const file = join(parent, 'not-a-directory');
  const link = join(parent, 'directory-link');
  await Promise.all([mkdir(workspace), mkdir(output), mkdir(occupied), writeFile(file, '')]);
  await writeFile(join(occupied, 'keep.txt'), 'keep');
  await symlink(output, link, 'dir');

  assert.equal(await validateCandidateOutputDirectory(output, workspace), await realpath(output));
  await assert.rejects(validateCandidateOutputDirectory(occupied, workspace), /must be empty/u);
  await assert.rejects(validateCandidateOutputDirectory(file, workspace), /real directory/u);
  await assert.rejects(validateCandidateOutputDirectory(link, workspace), /real directory/u);
  await assert.rejects(
    validateCandidateOutputDirectory(join(parent, 'missing'), workspace),
    /existing empty directory/u
  );
  const nested = join(workspace, 'output');
  await mkdir(nested);
  await assert.rejects(validateCandidateOutputDirectory(nested, workspace), /outside the repository/u);
  assert.equal(await readFile(join(occupied, 'keep.txt'), 'utf8'), 'keep');
});

test('candidate source paths reject traversal and symlinks escaping the package root', async (t) => {
  const parent = await mkdtemp(join(tmpdir(), 'web-decision-qwen35-source-path-'));
  t.after(() => rm(parent, {recursive: true, force: true}));
  const packageRoot = join(parent, 'package');
  const outsideRoot = join(parent, 'outside');
  const internalAsset = join(packageRoot, 'assets/qwen35/vision/graph.wasm');
  const externalAsset = join(outsideRoot, 'secret.json');

  await Promise.all([mkdir(join(packageRoot, 'assets/qwen35/vision'), {recursive: true}), mkdir(outsideRoot)]);
  await Promise.all([writeFile(internalAsset, 'graph'), writeFile(externalAsset, 'secret')]);
  await symlink(externalAsset, join(packageRoot, 'assets/qwen35/escape.json'));

  assert.equal(
    await resolvePackageFilePath(packageRoot, 'assets/qwen35/vision/graph.wasm', 'Qwen graph path'),
    await realpath(internalAsset)
  );
  await assert.rejects(
    resolvePackageFilePath(packageRoot, '../outside/secret.json', 'Qwen manifest path'),
    /safe relative path/u
  );
  await assert.rejects(
    resolvePackageFilePath(packageRoot, 'C:/outside/secret.json', 'Qwen manifest path'),
    /safe relative path/u
  );
  await assert.rejects(
    resolvePackageFilePath(packageRoot, 'assets/qwen35/escape.json', 'Qwen asset path'),
    /resolves outside the package root/u
  );
});
