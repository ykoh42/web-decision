import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {gitEnvironmentFor} from '../git-worktree-environment.mjs';
import {record, sha256} from './curated-gpu-evidence.mjs';

const commitPattern = /^[a-f0-9]{40}$/u;
const digestPattern = /^[a-f0-9]{64}$/u;
const evidencePathPattern = /^scripts\/release\/evidence\/kanana-current\/[a-f0-9]{64}\.json$/u;
export const KANANA_MODEL_PROVENANCE_PATH =
  'packages/web-decision/src/catalog/provenance/kanana-2-v-decide-3b-2611.json';
export const KANANA_CONVERSION_CONFIG_PATH =
  'packages/web-decision/src/catalog/provenance/kanana-conversion-config.json';
const digestKeys = [
  'archiveSha256',
  'packageTreeSha256',
  'bundleSha256',
  'playwrightReportSha256',
  'maxContextReportSha256',
  'modelProvenanceSha256',
  'conversionConfigSha256'
] as const;

type EvidenceCommit = Record<(typeof digestKeys)[number], string> & {
  schemaVersion: 1;
  sourceCommit: string;
  packageVersion: string;
  maxContextTokens: number;
  maxContextPassed: true;
};

function parseEvidence(bytes: Buffer): EvidenceCommit {
  const data = record(JSON.parse(bytes.toString('utf8')), 'Kanana evidence commit must contain a JSON object.');
  const keys = [
    'schemaVersion',
    'sourceCommit',
    'packageVersion',
    'maxContextTokens',
    'maxContextPassed',
    ...digestKeys
  ];
  assert.deepEqual(Object.keys(data).sort(), keys.sort(), 'Unexpected fields in Kanana evidence commit.');
  assert.equal(data.schemaVersion, 1, 'Unsupported Kanana evidence commit schema.');
  assert.ok(
    typeof data.sourceCommit === 'string' && commitPattern.test(data.sourceCommit),
    'Invalid tested source commit.'
  );
  assert.equal(typeof data.packageVersion, 'string', 'Missing Kanana evidence package version.');
  assert.ok(
    typeof data.maxContextTokens === 'number' &&
      Number.isSafeInteger(data.maxContextTokens) &&
      data.maxContextTokens > 0,
    'Invalid Kanana evidence maximum context length.'
  );
  assert.equal(data.maxContextPassed, true, 'Kanana evidence maximum context must pass.');
  for (const key of digestKeys) {
    const value = data[key];
    assert.ok(typeof value === 'string' && digestPattern.test(value), `Invalid Kanana evidence ${key}.`);
  }
  return data as EvidenceCommit;
}

/** Read one append-only evidence pin from a direct child of the clean tested source. */
export function readKananaCurrentEvidenceCommit(workspaceRoot: string, evidenceCommit: string): EvidenceCommit {
  assert.match(evidenceCommit, commitPattern, 'Kanana evidence requires a full Git commit ID.');
  const git = (...args: string[]) =>
    execFileSync('git', ['--no-replace-objects', ...args], {
      cwd: workspaceRoot,
      env: gitEnvironmentFor(workspaceRoot),
      maxBuffer: 1024 * 1024
    });
  assert.equal(
    git('status', '--porcelain', '--untracked-files=all', '--ignore-submodules=none').toString(),
    '',
    'Kanana evidence requires a clean tested source checkout.'
  );
  const sourceCommit = git('rev-parse', 'HEAD').toString().trim();
  assert.equal(
    git('rev-list', '--parents', '-n', '1', evidenceCommit).toString().trim(),
    `${evidenceCommit} ${sourceCommit}`,
    'Kanana evidence commit must have exactly the tested source HEAD as parent.'
  );
  const changes = git(
    'diff-tree',
    '--no-commit-id',
    '--no-renames',
    '--name-status',
    '-r',
    '-z',
    sourceCommit,
    evidenceCommit
  );
  const [status, path, terminator, ...other] = changes.toString().split('\0');
  assert.ok(
    status === 'A' && path && evidencePathPattern.test(path) && terminator === '' && other.length === 0,
    'Kanana evidence commit must add exactly one archive-specific evidence file.'
  );
  const mode = git('ls-tree', '-z', evidenceCommit, '--', path).toString().split('\t', 1)[0];
  assert.match(mode ?? '', /^100644 blob [a-f0-9]{40}$/u, 'Kanana evidence must be a regular data file.');
  const evidence = parseEvidence(git('cat-file', 'blob', `${evidenceCommit}:${path}`));
  assert.equal(
    path,
    `scripts/release/evidence/kanana-current/${evidence.archiveSha256}.json`,
    'Evidence path differs from its archive digest.'
  );
  assert.equal(evidence.sourceCommit, sourceCommit, 'Kanana evidence source commit differs from the tested HEAD.');
  assert.equal(
    evidence.modelProvenanceSha256,
    sha256(git('cat-file', 'blob', `${sourceCommit}:${KANANA_MODEL_PROVENANCE_PATH}`)),
    'Kanana model provenance differs from the tested source.'
  );
  assert.equal(
    evidence.conversionConfigSha256,
    sha256(git('cat-file', 'blob', `${sourceCommit}:${KANANA_CONVERSION_CONFIG_PATH}`)),
    'Kanana conversion provenance differs from the tested source.'
  );
  return evidence;
}

type VerifiedKananaRun = {
  archive: Pick<EvidenceCommit, 'packageVersion' | 'archiveSha256' | 'packageTreeSha256' | 'bundleSha256'>;
  playwrightReportBytes: Uint8Array;
  maxContextReportBytes: Uint8Array;
  maxContextTokens: number;
};

/** Bind the validated Kanana run and exact installed archive to their post-build pins. */
export function assertKananaCurrentEvidenceRun(pin: EvidenceCommit, run: VerifiedKananaRun): void {
  const {archive, playwrightReportBytes, maxContextReportBytes, maxContextTokens} = run;
  for (const key of ['packageVersion', 'archiveSha256', 'packageTreeSha256', 'bundleSha256'] as const)
    assert.equal(archive[key], pin[key], `Current Kanana ${key} differs from the evidence commit.`);
  assert.equal(
    sha256(playwrightReportBytes),
    pin.playwrightReportSha256,
    'Current Kanana browser report differs from its evidence commit.'
  );
  assert.equal(
    sha256(maxContextReportBytes),
    pin.maxContextReportSha256,
    'Current Kanana max-context report differs from its evidence commit.'
  );
  assert.equal(
    maxContextTokens,
    pin.maxContextTokens,
    'Current Kanana maximum context differs from its evidence commit.'
  );
}
