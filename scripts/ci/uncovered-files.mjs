import {isAbsolute, relative, sep} from 'node:path';

export function findUncoveredFiles(summary, sourceRoot) {
  const files = [];
  for (const [path, coverage] of Object.entries(summary)) {
    if (path === 'total') continue;
    const name = relative(sourceRoot, path);
    if (name.startsWith('..' + sep) || name === '..' || isAbsolute(name)) {
      throw new Error('Coverage contains a file outside the source directory.');
    }
    if (!Number.isInteger(coverage.lines?.total) || !Number.isInteger(coverage.lines?.covered)) {
      throw new Error(`Invalid line coverage for ${name}.`);
    }
    if (coverage.lines.total > 0 && coverage.lines.covered === 0) files.push(name.split(sep).join('/'));
  }
  return files.sort();
}

export function compareUncoveredFiles(files, baseline) {
  const accepted = new Set(baseline);
  const current = new Set(files);
  return {
    added: files.filter((file) => !accepted.has(file)),
    resolved: baseline.filter((file) => !current.has(file))
  };
}
