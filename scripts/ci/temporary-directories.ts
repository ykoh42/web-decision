import {mkdtempSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

export type TemporaryDirectoryFactory = (prefix: string) => string;
export type TemporaryDirectoryCallback<Result> = (makeTempDirectory: TemporaryDirectoryFactory) => Result;

/** Create temporary directories for a synchronous task and remove them afterward. */
export function withTemporaryDirectories<Result>(callback: TemporaryDirectoryCallback<Result>): Result {
  const directories: string[] = [];
  const makeTempDirectory: TemporaryDirectoryFactory = (prefix) => {
    const directory = mkdtempSync(join(tmpdir(), prefix));
    directories.push(directory);
    return directory;
  };

  try {
    return callback(makeTempDirectory);
  } finally {
    for (const directory of directories.reverse()) {
      rmSync(directory, {recursive: true, force: true});
    }
  }
}
