import {isAbsolute, relative, sep} from 'node:path';
import {readdirSync} from 'node:fs';

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

export interface CoverageInventoryMismatches {
  missing: string[];
  unexpected: string[];
}

export const RUNTIME_SOURCE_EXTENSIONS = ['.js', '.cjs', '.mjs', '.ts', '.tsx', '.jsx', '.mts', '.cts'] as const;

const ignoredSourceFiles = new Set(['tinydecide/corrections.js']);

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

function sourceFileNames(directory: string, sourceRoot: string, result: string[]): void {
  for (const entry of readdirSync(directory, {withFileTypes: true})) {
    const absolutePath = `${directory}/${entry.name}`;
    if (entry.isSymbolicLink()) {
      const name = relative(sourceRoot, absolutePath).split(sep).join('/');
      throw new Error(`Source directory contains a symbolic link: ${name}.`);
    }
    if (entry.isDirectory()) {
      sourceFileNames(absolutePath, sourceRoot, result);
      continue;
    }
    if (!entry.isFile()) continue;

    const name = relative(sourceRoot, absolutePath).split(sep).join('/');
    if (
      ignoredSourceFiles.has(name) ||
      /\.d\.(?:ts|mts|cts)$/.test(name) ||
      !RUNTIME_SOURCE_EXTENSIONS.some((extension) => name.endsWith(extension))
    ) {
      continue;
    }
    result.push(name);
  }
}

export function listRuntimeSourceFiles(sourceRoot: string): string[] {
  const files: string[] = [];
  sourceFileNames(sourceRoot, sourceRoot, files);
  const sortedFiles = files.sort();
  if (sortedFiles.length === 0) {
    throw new Error(`Source directory contains no executable JavaScript or TypeScript files: ${sourceRoot}.`);
  }
  return sortedFiles;
}

function relativeReportPath(sourceRoot: string, path: string): string {
  const name = relative(sourceRoot, path);
  if (name.startsWith('..' + sep) || name === '..' || isAbsolute(name)) {
    throw new Error('Coverage contains a file outside the source directory.');
  }
  return name.split(sep).join('/');
}

function reportedFileNames(summary: unknown, sourceRoot: string): string[] {
  if (!isRecord(summary)) {
    throw new TypeError('Coverage summary must be an object.');
  }
  return Object.keys(summary)
    .filter((path) => path !== 'total')
    .map((path) => relativeReportPath(sourceRoot, path));
}

export function findCoverageInventoryMismatches(
  summary: unknown,
  sourceRoot: string,
  sourceFiles: readonly string[]
): CoverageInventoryMismatches {
  const reported = new Set(reportedFileNames(summary, sourceRoot));
  const sources = new Set(sourceFiles);
  return {
    missing: [...sources].filter((file) => !reported.has(file)).sort(),
    unexpected: [...reported].filter((file) => !sources.has(file)).sort()
  };
}

export function findUncoveredFiles(summary: unknown, sourceRoot: string): string[] {
  if (!isRecord(summary)) {
    throw new TypeError('Coverage summary must be an object.');
  }
  const files: string[] = [];
  for (const [path, coverage] of Object.entries(summary)) {
    if (path === 'total') continue;
    const name = relativeReportPath(sourceRoot, path);
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
