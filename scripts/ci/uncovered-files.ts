import {isAbsolute, relative, sep} from 'node:path';

interface LineCoverage {
  total: number;
  covered: number;
}

interface FileSummary {
  lines: LineCoverage;
}

export interface UncoveredFileChanges {
  added: string[];
  resolved: string[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function validateLineCoverage(name: string, coverage: unknown): asserts coverage is FileSummary {
  const lines = isRecord(coverage) ? coverage.lines : undefined;
  if (
    !isRecord(lines) ||
    typeof lines.total !== 'number' ||
    !Number.isInteger(lines.total) ||
    typeof lines.covered !== 'number' ||
    !Number.isInteger(lines.covered)
  ) {
    throw new Error(`Invalid line coverage for ${name}.`);
  }
}

export function findUncoveredFiles(summary: unknown, sourceRoot: string): string[] {
  if (!isRecord(summary)) {
    throw new TypeError('Coverage summary must be an object.');
  }

  const files: string[] = [];
  for (const [path, coverage] of Object.entries(summary)) {
    if (path === 'total') continue;
    const name = relative(sourceRoot, path);
    if (name.startsWith('..' + sep) || name === '..' || isAbsolute(name)) {
      throw new Error('Coverage contains a file outside the source directory.');
    }
    validateLineCoverage(name, coverage);
    if (coverage.lines.total > 0 && coverage.lines.covered === 0) files.push(name.split(sep).join('/'));
  }
  return files.sort();
}

export function compareUncoveredFiles(files: readonly string[], baseline: readonly string[]): UncoveredFileChanges {
  const accepted = new Set(baseline);
  const current = new Set(files);
  return {
    added: files.filter((file) => !accepted.has(file)),
    resolved: baseline.filter((file) => !current.has(file))
  };
}
