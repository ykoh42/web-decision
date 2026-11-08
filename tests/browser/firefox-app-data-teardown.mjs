import {execFileSync} from 'node:child_process';
import {lstat, rm} from 'node:fs/promises';
import {basename, dirname, resolve} from 'node:path';
import {tmpdir} from 'node:os';

export default async function globalTeardown() {
  const home = process.env.WEB_DECISION_PLAYWRIGHT_FIREFOX_HOME;
  if (!home) return;

  const target = resolve(home);
  if (dirname(target) !== resolve(tmpdir()) || !/^web-decision-firefox-home-\d+$/u.test(basename(target)))
    throw new Error('Refusing to remove an unexpected Playwright Firefox app-data directory.');

  const info = await lstat(target);
  if (!info.isDirectory() || info.isSymbolicLink())
    throw new Error('Refusing to remove an unexpected Playwright Firefox app-data directory.');

  if (process.platform === 'darwin') execFileSync('chmod', ['-RN', target]);
  await rm(target, {recursive: true, force: true});
}
