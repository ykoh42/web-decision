import {isAbsolute, relative, sep} from 'node:path';

function sourceFiles(report, sourceRoot) {
  const files = [];
  for (const [path, coverage] of Object.entries(report)) {
    if (path === 'total') continue;
    const name = relative(sourceRoot, path);
    if (name.startsWith('..' + sep) || name === '..' || isAbsolute(name)) {
      throw new Error('Coverage contains a file outside the source directory.');
    }
    validateCounts(name, coverage);
    files.push({name: name.split(sep).join('/'), coverage});
  }
  return files;
}

function validateCounts(name, coverage) {
  for (const kind of ['lines', 'branches']) {
    const counts = coverage?.[kind];
    if (
      !Number.isInteger(counts?.total) ||
      !Number.isInteger(counts?.covered) ||
      counts.total < 0 ||
      counts.covered < 0 ||
      counts.covered > counts.total
    ) {
      throw new Error(`Invalid ${kind} coverage for ${name}.`);
    }
  }
}

function validateGroups(config) {
  if (!Array.isArray(config?.groups) || config.groups.length === 0) {
    throw new TypeError('Coverage group configuration must contain a non-empty groups array.');
  }

  const names = new Set();
  for (const group of config.groups) {
    if (!group || typeof group !== 'object' || Array.isArray(group)) {
      throw new TypeError('Coverage groups must be objects.');
    }
    if (typeof group.name !== 'string' || !group.name.trim() || names.has(group.name)) {
      throw new TypeError('Coverage groups must have unique, non-empty names.');
    }
    names.add(group.name);
    if (!Array.isArray(group.prefixes) || group.prefixes.length === 0) {
      throw new TypeError(`Coverage group ${group.name} must define path prefixes.`);
    }
    if (!validThreshold(group.lines) || !validThreshold(group.branches)) {
      throw new TypeError(`Coverage group ${group.name} must define line and branch percentages from 0 to 100.`);
    }
    for (const prefix of group.prefixes) {
      if (
        typeof prefix !== 'string' ||
        prefix.startsWith('/') ||
        prefix.includes('\\') ||
        prefix.split('/').some((part) => part === '..' || part === '.')
      ) {
        throw new TypeError(`Coverage group ${group.name} has an invalid path prefix.`);
      }
    }
  }
}

function validThreshold(value) {
  return Number.isFinite(value) && value >= 0 && value <= 100;
}

function matchesGroup(file, group) {
  return group.prefixes.some((prefix) => {
    if (prefix === '' || file.name === prefix) return true;
    return file.name.startsWith(prefix.endsWith('/') ? prefix : `${prefix}/`);
  });
}

function percentage(covered, total) {
  return total === 0 ? 100 : (covered * 100) / total;
}

function groupCoverage(files) {
  return files.reduce(
    (total, file) => ({
      linesCovered: total.linesCovered + file.coverage.lines.covered,
      linesTotal: total.linesTotal + file.coverage.lines.total,
      branchesCovered: total.branchesCovered + file.coverage.branches.covered,
      branchesTotal: total.branchesTotal + file.coverage.branches.total
    }),
    {linesCovered: 0, linesTotal: 0, branchesCovered: 0, branchesTotal: 0}
  );
}

export function checkCoverageGroups(report, sourceRoot, config) {
  validateGroups(config);
  const files = sourceFiles(report, sourceRoot);
  const errors = [];

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
