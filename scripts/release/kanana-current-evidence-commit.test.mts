import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdir, mkdtemp, readFile, realpath, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import test, {type TestContext} from 'node:test';
import {gitEnvironmentFor} from '../git-worktree-environment.mjs';
import {
  assertKananaCurrentEvidenceRun,
  KANANA_CONVERSION_CONFIG_PATH,
  KANANA_MODEL_PROVENANCE_PATH,
  readKananaCurrentEvidenceCommit
} from './kanana-current-evidence-commit.mjs';
import {sha256} from './curated-gpu-evidence.mjs';
import {verifyKananaCurrentReleaseEvidence} from './verify-release-evidence.ts';

const packageMetadata = '{"name":"web-decision","version":"0.7.0","type":"module","exports":"./dist/index.js"}\n';
const browserReport = Buffer.from('{"browser":"test fixture"}\n');
const maxContextReport = Buffer.from('{"context":"test fixture"}\n');
const verifiedArchive = {
  packageVersion: '0.7.0',
  archiveSha256: 'a'.repeat(64),
  packageTreeSha256: 'b'.repeat(64),
  bundleSha256: 'c'.repeat(64)
};
const verifiedRun = {
  archive: verifiedArchive,
  playwrightReportBytes: browserReport,
  maxContextReportBytes: maxContextReport,
  maxContextTokens: 32768
};
const evidencePath = (sha: string) => `scripts/release/evidence/kanana-current/${sha}.json`;

async function makeFixture(t: TestContext, {includePolicy = false, includeOldEvidence = false} = {}) {
  const root = await mkdtemp(join(tmpdir(), 'kanana-evidence-commit-test-'));
  t.after(() => rm(root, {recursive: true, force: true}));
  const git = (...args: string[]) =>
    execFileSync('git', args, {
      cwd: root,
      env: gitEnvironmentFor(root),
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe']
    }).trim();
  const write = async (path: string, contents: string | Buffer) => {
    const target = join(root, path);
    await mkdir(dirname(target), {recursive: true});
    await writeFile(target, contents);
  };
  git('init', '-q');
  git('config', 'user.name', 'Evidence Test');
  git('config', 'user.email', 'evidence-test@example.invalid');
  git('config', 'commit.gpgsign', 'false');
  await write('source.ts', 'export const source = true;\n');
  const model = '{"review":"original model provenance"}\n';
  const config = '{"recipe":"original conversion config"}\n';
  await write(KANANA_MODEL_PROVENANCE_PATH, model);
  await write(KANANA_CONVERSION_CONFIG_PATH, config);
  if (includePolicy) {
    const policy = await readFile(new URL('./release-evidence-policy.json', import.meta.url));
    await write('scripts/release/release-evidence-policy.json', policy);
  }
  if (includeOldEvidence) await write(evidencePath(verifiedArchive.archiveSha256), '{"historic":true}\n');
  git('add', '.');
  git('commit', '-qm', 'tested source');
  const sourceCommit = git('rev-parse', 'HEAD');
  const record = {
    schemaVersion: 1,
    sourceCommit,
    ...verifiedArchive,
    playwrightReportSha256: sha256(browserReport),
    maxContextReportSha256: sha256(maxContextReport),
    maxContextTokens: verifiedRun.maxContextTokens,
    maxContextPassed: true,
    modelProvenanceSha256: sha256(Buffer.from(model)),
    conversionConfigSha256: sha256(Buffer.from(config))
  };
  const commit = (message: string) => {
    git('add', '-A');
    git('commit', '-qm', message);
    const child = git('rev-parse', 'HEAD');
    git('checkout', '--detach', sourceCommit);
    return child;
  };
  const writeEvidence = async (overrides: Record<string, unknown> = {}, path?: string) => {
    const data = {...record, ...overrides};
    await write(path ?? evidencePath(String(data.archiveSha256)), JSON.stringify(data) + '\n');
  };
  return {root, git, write, record, commit, writeEvidence};
}

test('committed Kanana evidence binds a later report without moving the tested clean HEAD', async (t) => {
  const fixture = await makeFixture(t);
  await fixture.writeEvidence();
  const evidenceCommit = fixture.commit('release evidence only');
  const pin = readKananaCurrentEvidenceCommit(fixture.root, evidenceCommit);
  assert.deepEqual(pin, fixture.record);
  assert.equal(fixture.git('rev-parse', 'HEAD'), pin.sourceCommit);
  assert.equal(fixture.git('status', '--porcelain'), '');
  assert.doesNotThrow(() => assertKananaCurrentEvidenceRun(pin, verifiedRun));
  await fixture.write('untracked.txt', 'dirty\n');
  assert.throws(() => readKananaCurrentEvidenceCommit(fixture.root, evidenceCommit), /clean tested source/u);
});

test('the evidence child scope permits only one new regular archive-specific JSON file', async (t) => {
  const mixed = await makeFixture(t);
  await mixed.writeEvidence();
  await mixed.write(KANANA_CONVERSION_CONFIG_PATH, '{"recipe":"replaced"}\n');
  assert.throws(
    () => readKananaCurrentEvidenceCommit(mixed.root, mixed.commit('mixed provenance change')),
    /add exactly one/u
  );
  const prior = await makeFixture(t, {includeOldEvidence: true});
  await prior.writeEvidence();
  assert.throws(
    () => readKananaCurrentEvidenceCommit(prior.root, prior.commit('overwrite earlier evidence')),
    /add exactly one/u
  );
  const mode = await makeFixture(t);
  await mode.writeEvidence();
  const path = evidencePath(verifiedArchive.archiveSha256);
  mode.git('add', path);
  mode.git('update-index', '--chmod=+x', path);
  mode.git('commit', '-qm', 'executable evidence');
  const executable = mode.git('rev-parse', 'HEAD');
  mode.git('checkout', '--force', '--detach', mode.record.sourceCommit);
  assert.throws(() => readKananaCurrentEvidenceCommit(mode.root, executable), /regular data file/u);
  assert.throws(() => readKananaCurrentEvidenceCommit(mode.root, 'HEAD'), /full Git commit ID/u);
});

test('the evidence child has one direct source parent and cannot repin another source', async (t) => {
  const fixture = await makeFixture(t);
  await fixture.writeEvidence({sourceCommit: 'f'.repeat(40)});
  const wrongSource = fixture.commit('wrong recorded source');
  assert.throws(() => readKananaCurrentEvidenceCommit(fixture.root, wrongSource), /source commit differs/u);
  fixture.git('checkout', '--detach', wrongSource);
  await fixture.write('second.txt', 'second commit\n');
  const descendant = fixture.commit('not a direct child');
  assert.throws(() => readKananaCurrentEvidenceCommit(fixture.root, descendant), /exactly the tested source/u);
  const tree = fixture.git('rev-parse', `${wrongSource}^{tree}`);
  const merge = fixture.git('commit-tree', tree, '-p', fixture.record.sourceCommit, '-p', wrongSource, '-m', 'merge');
  assert.throws(() => readKananaCurrentEvidenceCommit(fixture.root, merge), /exactly the tested source/u);
});

test('archive paths, the provenance pair and evidence schema are immutable pins', async (t) => {
  const fixture = await makeFixture(t);
  const cases = [
    [{}, evidencePath('d'.repeat(64)), /path differs from its archive digest/u],
    [{modelProvenanceSha256: 'd'.repeat(64)}, undefined, /model provenance differs/u],
    [{conversionConfigSha256: 'd'.repeat(64)}, undefined, /conversion provenance differs/u],
    [{allowAnyArchive: true}, undefined, /Unexpected fields/u],
    [{maxContextPassed: false}, undefined, /maximum context must pass/u]
  ] as const;
  for (const [overrides, path, message] of cases) {
    await fixture.writeEvidence(overrides, path);
    assert.throws(() => readKananaCurrentEvidenceCommit(fixture.root, fixture.commit('unexpected pin')), message);
  }
});

test('archive, bundle, both report byte strings and context length must match the committed outcome', async (t) => {
  const fixture = await makeFixture(t);
  await fixture.writeEvidence();
  const pin = readKananaCurrentEvidenceCommit(fixture.root, fixture.commit('release evidence only'));
  for (const key of ['archiveSha256', 'packageTreeSha256', 'bundleSha256', 'packageVersion'] as const) {
    const archive = {...verifiedArchive, [key]: 'd'.repeat(64)};
    assert.throws(() => assertKananaCurrentEvidenceRun(pin, {...verifiedRun, archive}), new RegExp(key, 'u'));
  }
  assert.throws(
    () =>
      assertKananaCurrentEvidenceRun(pin, {
        ...verifiedRun,
        playwrightReportBytes: Buffer.concat([browserReport, Buffer.from(' ')])
      }),
    /browser report differs/u
  );
  assert.throws(
    () => assertKananaCurrentEvidenceRun(pin, {...verifiedRun, maxContextReportBytes: browserReport}),
    /max-context report differs/u
  );
  assert.throws(
    () => assertKananaCurrentEvidenceRun(pin, {...verifiedRun, maxContextTokens: 8192}),
    /maximum context differs/u
  );
});

test('the actual Kanana verifier reaches exact archive checks from the clean source with a valid child pin', async (t) => {
  const fixture = await makeFixture(t, {includePolicy: true});
  const packageRoot = await realpath(await mkdtemp(join(tmpdir(), 'kanana-evidence-npm-')));
  t.after(() => rm(packageRoot, {recursive: true, force: true}));
  const installedPackagePath = join(packageRoot, 'package');
  await mkdir(join(installedPackagePath, 'dist'), {recursive: true});
  await writeFile(join(installedPackagePath, 'package.json'), packageMetadata);
  await writeFile(join(installedPackagePath, 'dist/index.js'), 'export const bundled = true;\n');
  const archivePath = join(packageRoot, 'web-decision-0.7.0.tgz');
  execFileSync(process.platform === 'win32' ? 'tar.exe' : 'tar', ['-czf', archivePath, '-C', packageRoot, 'package']);
  const options = {
    workspaceRoot: fixture.root,
    archivePath,
    installedPackagePath,
    playwrightReportPath: join(packageRoot, 'browser.json'),
    maxContextReportPath: join(packageRoot, 'max-context.json'),
    buildEvidencePath: join(packageRoot, 'builds.json')
  };
  await fixture.writeEvidence();
  const wrongArchive = fixture.commit('pin a different archive');
  await assert.rejects(
    verifyKananaCurrentReleaseEvidence({...options, evidenceCommit: wrongArchive, candidateConfigPath: 'candidate'}),
    /cannot be combined/u
  );
  await assert.rejects(
    verifyKananaCurrentReleaseEvidence({...options, evidenceCommit: wrongArchive}),
    /require the pinned npm archive/u
  );
  await fixture.writeEvidence({archiveSha256: sha256(await readFile(archivePath))});
  const rightArchive = fixture.commit('pin this archive without a valid model');
  await assert.rejects(
    verifyKananaCurrentReleaseEvidence({...options, evidenceCommit: rightArchive}),
    /missing its pinned manifest/u
  );
  assert.equal(fixture.git('rev-parse', 'HEAD'), fixture.record.sourceCommit);
  assert.equal(fixture.git('status', '--porcelain'), '');
});
