import {existsSync, readFileSync} from 'node:fs';
import path from 'node:path';

function gitDirectoryForHost(gitDirectory: string): string | undefined {
  if (process.platform === 'win32') {
    const wslPath = /^\/mnt\/([a-z])\/(.+)$/iu.exec(gitDirectory);
    return wslPath ? `${wslPath[1]?.toUpperCase()}:/${wslPath[2]}` : undefined;
  }
  const windowsPath = /^([a-z]):[\\/](.+)$/iu.exec(gitDirectory);
  if (!windowsPath) return undefined;
  const linuxPath = `/mnt/${windowsPath[1]?.toLowerCase()}/${windowsPath[2]?.replaceAll('\\', '/')}`;
  return existsSync(linuxPath) ? linuxPath : undefined;
}

export function gitEnvironmentFor(root: string): NodeJS.ProcessEnv {
  const environment = Object.fromEntries(Object.entries(process.env).filter(([name]) => !name.startsWith('GIT_')));
  let marker: string;
  try {
    marker = readFileSync(path.join(root, '.git'), 'utf8');
  } catch {
    return environment;
  }
  const gitDirectory = /^gitdir:\s*(.+)\s*$/mu.exec(marker)?.[1] ?? '';
  const hostGitDirectory = gitDirectoryForHost(gitDirectory);
  if (!hostGitDirectory) return environment;
  environment.GIT_DIR = hostGitDirectory;
  environment.GIT_WORK_TREE = root;
  return environment;
}
