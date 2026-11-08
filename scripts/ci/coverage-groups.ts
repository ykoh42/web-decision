import {isAbsolute, relative, sep} from 'node:path';

export interface CoverageCounts {
  total: number;
  covered: number;
}

export interface CoverageFile {
  lines: CoverageCounts;
  branches: CoverageCounts;
}

export interface CoverageGroup {
  name: string;
  prefixes: string[];
  lines: number;
  branches: number;
}

export interface CoverageGroupConfiguration {
  groups: CoverageGroup[];
}

interface SourceFile {
  name: string;
  coverage: CoverageFile;
}

interface GroupCoverageCounts {
  linesCovered: number;
  linesTotal: number;
  branchesCovered: number;
  branchesTotal: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function validateCounts(name: string, coverage: unknown): asserts coverage is CoverageFile {
  if (!isRecord(coverage)) {
    throw new Error(`Invalid line coverage for ${name}.`);
  }

  for (const kind of ['lines', 'branches'] as const) {
    const counts = coverage[kind];
    if (
      !isRecord(counts) ||
      typeof counts.total !== 'number' ||
      !Number.isInteger(counts.total) ||
      typeof counts.covered !== 'number' ||
      !Number.isInteger(counts.covered) ||
      counts.total < 0 ||
      counts.covered < 0 ||
      counts.covered > counts.total
    ) {
      throw new Error(`Invalid ${kind} coverage for ${name}.`);
    }
  }
}

function sourceFiles(report: unknown, sourceRoot: string): SourceFile[] {
  if (!isRecord(report)) {
    throw new TypeError('Coverage report must be an object.');
  }

  const files: SourceFile[] = [];
  for (const [path, rawCoverage] of Object.entries(report)) {
    if (path === 'total') continue;
    const name = relative(sourceRoot, path);
    if (name.startsWith('..' + sep) || name === '..' || isAbsolute(name)) {
      throw new Error('Coverage contains a file outside the source directory.');
    }
    validateCounts(name, rawCoverage);
    files.push({name: name.split(sep).join('/'), coverage: rawCoverage});
  }
  return files;
}

function validThreshold(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 100;
}

function validPrefix(prefix: unknown): prefix is string {
  return (
    typeof prefix === 'string' &&
    !prefix.startsWith('/') &&
    !prefix.includes('\\') &&
    !prefix.split('/').some((part) => part === '..' || part === '.')
  );
}

function validateGroup(candidate: unknown, names: Set<string>): asserts candidate is CoverageGroup {
  if (!isRecord(candidate)) {
    throw new TypeError('Coverage groups must be objects.');
  }

  const {name, prefixes, lines, branches} = candidate;
  if (typeof name !== 'string' || !name.trim() || names.has(name)) {
    throw new TypeError('Coverage groups must have unique, non-empty names.');
  }
  names.add(name);

  if (!Array.isArray(prefixes) || prefixes.length === 0) {
    throw new TypeError(`Coverage group ${name} must define path prefixes.`);
  }
  for (const prefix of prefixes) {
    if (!validPrefix(prefix)) {
      throw new TypeError(`Coverage group ${name} has an invalid path prefix.`);
    }
  }

  if (!validThreshold(lines) || !validThreshold(branches)) {
    throw new TypeError(`Coverage group ${name} must define line and branch percentages from 0 to 100.`);
  }
}

function validateGroups(config: unknown): asserts config is CoverageGroupConfiguration {
  if (!isRecord(config) || !Array.isArray(config.groups) || config.groups.length === 0) {
    throw new TypeError('Coverage group configuration must contain a non-empty groups array.');
  }

  const names = new Set<string>();
  for (const group of config.groups) {
    validateGroup(group, names);
  }
}

function matchesGroup(file: SourceFile, group: CoverageGroup): boolean {
  return group.prefixes.some((prefix) => {
    if (prefix === '' || file.name === prefix) return true;
    return file.name.startsWith(prefix.endsWith('/') ? prefix : `${prefix}/`);
  });
}

function percentage(covered: number, total: number): number {
  return total === 0 ? 100 : (covered * 100) / total;
}

function groupCoverage(files: SourceFile[]): GroupCoverageCounts {
  return files.reduce<GroupCoverageCounts>(
    (total, file) => ({
      linesCovered: total.linesCovered + file.coverage.lines.covered,
      linesTotal: total.linesTotal + file.coverage.lines.total,
      branchesCovered: total.branchesCovered + file.coverage.branches.covered,
      branchesTotal: total.branchesTotal + file.coverage.branches.total
    }),
    {linesCovered: 0, linesTotal: 0, branchesCovered: 0, branchesTotal: 0}
  );
}

export function checkCoverageGroups(report: unknown, sourceRoot: string, config: unknown): string[] {
  validateGroups(config);
  const files = sourceFiles(report, sourceRoot);
  const errors: string[] = [];

  for (const group of config.groups) {
    const members = files.filter((file) => matchesGroup(file, group));
    if (members.length === 0) {
      errors.push(`Coverage group ${group.name} matched no source files.`);
      continue;
    }

    const counts = groupCoverage(members);
    const lines = percentage(counts.linesCovered, counts.linesTotal);
    const branches = percentage(counts.branchesCovered, counts.branchesTotal);
    if (lines < group.lines) {
      errors.push(
        `Coverage group ${group.name} has ${lines.toFixed(2)}% line coverage; ` +
          `${group.lines}% required across ${members.length} files.`
      );
    }
    if (branches < group.branches) {
      errors.push(
        `Coverage group ${group.name} has ${branches.toFixed(2)}% branch coverage; ` +
          `${group.branches}% required across ${members.length} files.`
      );
    }
  }
  return errors;
}
