import {mkdtempSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

export function withTemporaryDirectories(callback) {
  const directories = [];
  const makeTempDirectory = (prefix) => {
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
