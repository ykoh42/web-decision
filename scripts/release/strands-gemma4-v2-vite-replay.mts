import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile} from 'node:fs/promises';
import {dirname, join} from 'node:path';
import {record, sha256} from './curated-gpu-evidence.mjs';
import {gitEnvironmentFor} from '../git-worktree-environment.mjs';
import {gemmaV2ArchiveTokenizerSha256} from '../../tests/browser/gemma-v2-exact-archive-page.mjs';
import type {ArchiveInfo} from './verify-release-evidence.ts';

// This reviewed replay is evidence for this candidate only. A different candidate needs a new reviewed pin.
export const VITE_REPLAY_ARCHIVE_SHA256 = '7c3bb65bdd8a4f3195a613435f6d2a23c8699ff0c435db4614e6d2a88af78edd';
export const VITE_REPLAY_SHA256 = '75733072d30f51ea8175be8430a2b4204249db6c77d725ae80be00fc73ae1adb';
export const VITE_REPLAY_SCRIPT_SHA256 = 'fd751409831f39dbf935ee76d9d26fe84cf71a853c152bbce61dfe097b9d03b0';
const sourceFiles = [
  'tests/browser/vite.config.ts',
  'tests/browser/index.html',
  'tests/browser/harness.ts',
  'playwright.config.ts',
  'scripts/strands/gemma4_v2/candidate/playwright.config.mts',
  'scripts/strands/gemma4_v2/candidate/gemma4-v2-installed-reference.spec.ts'
];
const scope = 'CPU Vite replay; no browser response bytes from the live run';

export function assertViteReplayRecord(
  data: Record<string, unknown>,
  expected: {
    sourceCommit: string;
    archiveSha256: string;
    packageTreeSha256: string;
    startSha256: string;
    requestSha256: string;
    referenceSha256: string;
  }
) {
  assert.equal(data.scope, scope);
  assert.equal(data.gpuExecuted, false);
  assert.equal(data.sourceCommit, expected.sourceCommit);
  assert.equal(data.archiveSha256, expected.archiveSha256);
  assert.equal(data.packageTreeSha256, expected.packageTreeSha256);
  assert.equal(data.recordedStartSha256, expected.startSha256);
  assert.deepEqual(data.fixturePins, {
    requestSha256: expected.requestSha256,
    referenceSha256: expected.referenceSha256
  });
  assert.equal(data.viteVersion, '8.3.6');
  assert.equal(data.parserVersion, '8.19.0');
  assert.equal(data.tokenizerSha256, gemmaV2ArchiveTokenizerSha256);
  assert.deepEqual(data.issues, []);
  assert.equal(data.archiveModules, 27);
  assert.equal(data.matchedModules, 28);
  const shell = record(data.devShell, 'Missing Vite replay shell.');
  assert.equal(shell.entryImportsExactInstalledBundle, true);
  assert.deepEqual(shell.pageModuleScripts, ['/@vite/client', '/harness.ts']);
  assert.ok(Array.isArray(data.modules) && data.modules.length === 28);
  assert.ok(Array.isArray(data.assets) && data.assets.length === 17);
}

const reviewedReplays = new Map([
  [VITE_REPLAY_ARCHIVE_SHA256, {reportSha256: VITE_REPLAY_SHA256, scriptSha256: VITE_REPLAY_SCRIPT_SHA256}],
  [
    'e729bb940f5361e56ebc367f26b515f0c36a904fd5d1544064751f9dce4c698f',
    {
      reportSha256: 'e32d01bbf24866260d98d94c84a2150fd88a24bb949f371226b8e7cee28ac143',
      scriptSha256: '2b3e26df7a4597c5de69a1439b5754879c41665760fd37462669f044e28fa529'
    }
  ]
]);

export function assertViteReplayPins(
  {reportSha256, scriptSha256}: {reportSha256: string; scriptSha256: string},
  archiveSha256: string
) {
  const pin = reviewedReplays.get(archiveSha256);
  assert.ok(pin, 'No reviewed Vite replay is pinned for archive ' + archiveSha256 + '.');
  assert.equal(reportSha256, pin.reportSha256, 'Unreviewed Vite replay report.');
  assert.equal(scriptSha256, pin.scriptSha256, 'Unreviewed Vite replay script.');
}

export async function verifyViteReplay(options: {
  reportPath: string;
  scriptPath: string;
  workspace: string;
  sourceCommit: string;
  archive: ArchiveInfo;
  startSha256: string;
  requestSha256: string;
  referenceSha256: string;
}) {
  const [bytes, script] = await Promise.all([readFile(options.reportPath), readFile(options.scriptPath)]);
  const digests = {reportSha256: sha256(bytes), scriptSha256: sha256(script)};
  assertViteReplayPins(digests, options.archive.archiveSha256);
  const data = record(JSON.parse(bytes.toString('utf8')), 'Invalid Vite replay.');
  assertViteReplayRecord(data, {
    ...options,
    archiveSha256: options.archive.archiveSha256,
    packageTreeSha256: options.archive.packageTreeSha256
  });
  const pins = record(data.sourcePins, 'Missing Vite replay source pins.');
  assert.deepEqual(Object.keys(pins).sort(), [...sourceFiles].sort());
  for (const path of sourceFiles) {
    const blob = execFileSync(
      'git',
      ['--no-optional-locks', '--no-replace-objects', 'show', options.sourceCommit + ':' + path],
      {
        cwd: options.workspace,
        env: gitEnvironmentFor(options.workspace)
      }
    );
    assert.equal(pins[path], sha256(blob), 'Vite replay source differs: ' + path);
  }
  const expectedModules = options.archive.archiveFiles.filter(
    (path) => path.startsWith('dist/') && /\.(?:mjs|js|cjs)$/u.test(path)
  );
  assert.equal(expectedModules.length, 27);
  const tokenizer = '@huggingface/tokenizers/dist/tokenizers.mjs';
  const moduleEntries = (data.modules as unknown[]).map((value) => record(value, 'Invalid Vite replay module.'));
  assert.deepEqual(moduleEntries.map((entry) => entry.label).sort(), [...expectedModules, tokenizer].sort());
  for (const entry of moduleEntries) {
    assert.equal(entry.revertedToOriginalByteForByte, true);
    assert.ok(Array.isArray(entry.runtimeConstructs) && entry.runtimeConstructs.length === 0);
    assert.ok(typeof entry.label === 'string');
    const path =
      entry.label === tokenizer
        ? join(dirname(options.archive.installedRoot), tokenizer)
        : join(options.archive.installedRoot, entry.label);
    assert.equal(entry.diskSha256, sha256(await readFile(path)), 'Vite replay module differs: ' + entry.label);
  }
  const assets = (data.assets as unknown[]).map((value) => record(value, 'Invalid Vite replay asset.'));
  const names = assets.map((entry) => entry.file);
  assert.equal(new Set(names).size, 17);
  for (const entry of assets) {
    assert.ok(typeof entry.file === 'string' && options.archive.archiveFiles.includes(entry.file));
    assert.equal(entry.status, 200);
    assert.equal(entry.exact, true);
    const asset = await readFile(join(options.archive.installedRoot, entry.file));
    assert.equal(entry.bytes, asset.byteLength);
    assert.equal(entry.sha256, sha256(asset));
  }
  return {...digests, scope, liveResponseCaptured: false};
}
