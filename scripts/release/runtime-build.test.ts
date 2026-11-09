import {execFileSync} from 'node:child_process';
import {mkdtemp, rm} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import test from 'node:test';
import {fileURLToPath} from 'node:url';

const workspaceRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const require = createRequire(import.meta.url);
const compiler = join(dirname(require.resolve('typescript/package.json')), 'bin/tsc');

test('release evidence compiles and its emitted tests run outside the workspace', async () => {
  const output = await mkdtemp(join(tmpdir(), 'web-decision-release-build-'));
  try {
    execFileSync(
      process.execPath,
      [compiler, '--project', 'scripts/release/tsconfig.runtime.json', '--outDir', output],
      {
        cwd: workspaceRoot,
        stdio: 'pipe'
      }
    );
    const releaseDirectory = join(output, 'scripts/release');
    execFileSync(
      process.execPath,
      [
        '--test',
        join(releaseDirectory, 'verify-release-evidence.test.js'),
        join(releaseDirectory, 'tvm-build-evidence.test.js')
      ],
      {cwd: workspaceRoot, env: {...process.env, WEB_DECISION_WORKSPACE_ROOT: workspaceRoot}, stdio: 'pipe'}
    );
  } finally {
    await rm(output, {recursive: true, force: true});
  }
});
