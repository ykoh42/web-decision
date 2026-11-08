#!/usr/bin/env node

// Run only after the full measured SonarJS ratchet has validated this baseline.

import {spawnSync} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import {TextDecoder} from 'node:util';
import {fileURLToPath} from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const THRESHOLD = 15;
const SOURCE_EXTENSIONS = new Set(['.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx', '.mts', '.cts']);
const HISTORICAL_RELEASE_DIRECTORIES = new Set([
  'releases/web-decision-0.1.0',
  'releases/web-decision-0.2.0',
  'releases/web-decision-0.3.0'
]);

interface Finding {
  file: string;
  identity: string;
  score: number;
}

interface SonarBaseline {
  schemaVersion: 1;
  threshold: 15;
  versions: {
    oxlint: string;
    sonarjs: string;
    typescriptParser: string;
  };
  generatedExclusions: Array<{path: string; reason: string}>;
  findings: Finding[];
}

function fail(message: string): never {
  throw new Error(message);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, expected: string[]): boolean {
  const actual = Object.keys(value).sort();
  return JSON.stringify(actual) === JSON.stringify([...expected].sort());
}

function normalizedSourcePath(value: unknown, label: string): string {
  if (
    typeof value !== 'string' ||
    value.length === 0 ||
    value.includes('\\') ||
    value.includes('\0') ||
    value.startsWith('/') ||
    /^[A-Za-z]:/u.test(value) ||
    path.posix.normalize(value) !== value ||
    value.split('/').some((part) => part === '' || part === '.' || part === '..') ||
    !SOURCE_EXTENSIONS.has(path.posix.extname(value))
  ) {
    fail(`Invalid ${label} path: ${String(value)}`);
  }
  return value;
}

function isHistoricalReleasePath(file: string): boolean {
  for (const directory of HISTORICAL_RELEASE_DIRECTORIES) {
    if (file.startsWith(`${directory}/`)) return true;
  }
  return false;
}

export function parseSonarBaseline(text: string): SonarBaseline {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (error) {
    fail(`Cannot parse SonarJS complexity baseline: ${String(error)}`);
  }

  if (
    !isRecord(value) ||
    !hasExactKeys(value, ['schemaVersion', 'threshold', 'versions', 'generatedExclusions', 'findings']) ||
    value.schemaVersion !== 1 ||
    value.threshold !== THRESHOLD ||
    !isRecord(value.versions) ||
    !hasExactKeys(value.versions, ['oxlint', 'sonarjs', 'typescriptParser']) ||
    !Array.isArray(value.generatedExclusions) ||
    !Array.isArray(value.findings)
  ) {
    fail('SonarJS complexity baseline has an invalid schema or threshold.');
  }

  const versions = value.versions;
  for (const name of ['oxlint', 'sonarjs', 'typescriptParser'] as const) {
    if (typeof versions[name] !== 'string' || versions[name].trim() === '') {
      fail(`SonarJS complexity baseline has an invalid ${name} version.`);
    }
  }

  const oxlint = versions.oxlint as string;
  const sonarjs = versions.sonarjs as string;
  const typescriptParser = versions.typescriptParser as string;

  const exclusions = value.generatedExclusions.map((entry, index) => {
    if (!isRecord(entry) || !hasExactKeys(entry, ['path', 'reason'])) {
      fail(`SonarJS generated exclusion ${index} is malformed.`);
    }
    const exclusionPath = normalizedSourcePath(entry.path, 'generated exclusion');
    if (typeof entry.reason !== 'string' || entry.reason.trim() === '') {
      fail(`SonarJS generated exclusion has no reason: ${exclusionPath}`);
    }
    return {path: exclusionPath, reason: entry.reason};
  });
  if (new Set(exclusions.map(({path: file}) => file)).size !== exclusions.length) {
    fail('SonarJS complexity baseline contains duplicate generated exclusions.');
  }

  const keys = new Set<string>();
  const findings = value.findings.map((entry, index): Finding => {
    if (!isRecord(entry) || !hasExactKeys(entry, ['file', 'identity', 'score'])) {
      fail(`SonarJS complexity finding ${index} is malformed.`);
    }
    const file = normalizedSourcePath(entry.file, 'SonarJS baseline');
    if (typeof entry.identity !== 'string' || entry.identity.trim() === '') {
      fail(`SonarJS complexity finding has an empty function identity in ${file}.`);
    }
    if (typeof entry.score !== 'number' || !Number.isSafeInteger(entry.score) || entry.score <= THRESHOLD) {
      fail(`SonarJS complexity finding has an invalid score in ${file}.`);
    }
    const key = `${file}\0${entry.identity}`;
    if (keys.has(key)) fail(`SonarJS complexity baseline has a duplicate function identity in ${file}.`);
    keys.add(key);
    return {file, identity: entry.identity, score: entry.score};
  });

  return {
    schemaVersion: 1,
    threshold: THRESHOLD,
    versions: {oxlint, sonarjs, typescriptParser},
    generatedExclusions: exclusions,
    findings
  };
}

export function completionViolations(baseline: SonarBaseline, sourcePaths: ReadonlySet<string>): Finding[] {
  const missingFiles = baseline.findings.filter(({file}) => !sourcePaths.has(file));
  if (missingFiles.length > 0) {
    fail(`SonarJS baseline references a missing or unscanned source: ${missingFiles[0]?.file ?? ''}`);
  }
  return baseline.findings.filter(({file, score}) => !isHistoricalReleasePath(file) && score > THRESHOLD);
}

export function requireComplete(baseline: SonarBaseline, sourcePaths: ReadonlySet<string>): void {
  const violations = completionViolations(baseline, sourcePaths);
  if (violations.length === 0) return;
  const details = violations.map(({file, identity, score}) => `  ${file} :: ${identity} (${score})`).join('\n');
  fail(`Managed JavaScript/TypeScript source remains above cognitive complexity ${THRESHOLD}:\n${details}`);
}

function workingTreeSources(root: string): Set<string> {
  const result = spawnSync('git', ['ls-files', '--cached', '--others', '--exclude-standard', '-z'], {
    cwd: root,
    encoding: 'buffer',
    maxBuffer: 128 * 1024 * 1024
  });
  if (result.error) fail(`Could not list JavaScript/TypeScript sources: ${result.error.message}`);
  if (result.status !== 0) fail(`Could not list JavaScript/TypeScript sources: ${result.stderr.toString('utf8')}`);

  const sources = new Set<string>();
  for (const file of result.stdout.toString('utf8').split('\0')) {
    if (file === '' || !SOURCE_EXTENSIONS.has(path.posix.extname(file))) continue;
    const normalized = normalizedSourcePath(file, 'working tree source');
    const absolute = path.join(root, ...normalized.split('/'));
    let stats: fs.Stats;
    try {
      stats = fs.lstatSync(absolute);
    } catch {
      fail(`JavaScript/TypeScript source is missing: ${normalized}`);
    }
    if (!stats.isFile() || stats.isSymbolicLink())
      fail(`JavaScript/TypeScript source is not a regular file: ${normalized}`);
    sources.add(normalized);
  }
  if (sources.size === 0) fail('No JavaScript/TypeScript sources were found.');
  return sources;
}

export function loadSonarBaseline(baselinePath: string): SonarBaseline {
  let source: string;
  try {
    source = new TextDecoder('utf-8', {fatal: true}).decode(fs.readFileSync(baselinePath));
  } catch (error) {
    fail(`SonarJS complexity baseline is missing or unreadable: ${String(error)}`);
  }
  return parseSonarBaseline(source);
}

function run(root = ROOT): void {
  const baseline = loadSonarBaseline(path.join(root, 'sonarjs-cognitive-complexity-baseline.json'));
  requireComplete(baseline, workingTreeSources(root));
  process.stdout.write(`SonarJS complexity completion gate passed: no managed baseline findings above ${THRESHOLD}.\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    run();
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
