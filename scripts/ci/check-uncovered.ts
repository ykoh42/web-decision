import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {checkCoverageGroups} from './coverage-groups.ts';
import {compareUncoveredFiles, findUncoveredFiles} from './uncovered-files.ts';

function readJson(url: URL): unknown {
  return JSON.parse(readFileSync(url, 'utf8')) as unknown;
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value) || !value.every((item: unknown) => typeof item === 'string')) {
    throw new TypeError('Uncovered-file baseline must be an array of paths.');
  }
  return value;
}

const source = fileURLToPath(new URL('../../packages/web-decision/src/', import.meta.url));
const report = readJson(new URL('../../packages/web-decision/coverage/coverage-summary.json', import.meta.url));
const baseline = stringArray(readJson(new URL('./uncovered-files-baseline.json', import.meta.url)));
const result = compareUncoveredFiles(findUncoveredFiles(report, source), baseline);
const thresholds = readJson(new URL('./coverage-groups.json', import.meta.url));
const groupErrors = checkCoverageGroups(report, source, thresholds);
if (result.resolved.length) console.log(`Covered since baseline; remove entries: ${result.resolved.join(', ')}`);
if (result.added.length) {
  console.error(`New source files without line coverage: ${result.added.join(', ')}`);
  process.exitCode = 1;
} else {
  console.log('No new source files without line coverage.');
}
if (groupErrors.length) {
  for (const error of groupErrors) console.error(error);
  process.exitCode = 1;
} else {
  console.log('All source groups meet their line and branch coverage thresholds.');
}
