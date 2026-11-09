import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {existsSync, readFileSync} from 'node:fs';
import {join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {defaultTreeAdapter, parse, type DefaultTreeAdapterTypes} from 'parse5';

const policyPath = 'scripts/ci/typescript-migration-policy.json';
const initialInventoryCommit = 'baa3c144d23e1882ba29084c9f95897b8c56accb';

const sharedReleaseInputHashes = new Map([
  [
    'scripts/release/kev-current-source-preview-package-policy.mjs',
    '219f7a59d67f2e5fe91af021812e19708421cbaec81e2eeef0b950ff22cb9540'
  ],
  [
    'scripts/release/kev-current-source-preview-safety.mjs',
    'a80d1930d2de2e41a799df87302efbf040e0082e3a62251ca89033c473f57492'
  ]
]);
const sharedReleaseVersions = ['0.1.0', '0.2.0', '0.3.0'];
const sharedReleaseReason = (hash: string): string =>
  `Immutable shared historical release input; releases 0.1.0, 0.2.0 and 0.3.0 pin SHA-256 ${hash}.`;

const javascriptExtension = /\.(?:js|mjs|cjs|jsx)$/u;
const releasePath = /^releases\/web-decision-(\d+\.\d+\.\d+)\//u;
const acceptedReasonPrefixes = [
  'Immutable historical release ',
  'Generated from ',
  'Upstream vendored from ',
  'Browser deployment build output from '
];

export type JavaScriptException = {
  path: string;
  reason: string;
};

export type TypeScriptMigrationPolicy = {
  schemaVersion: 1;
  baselineCommit: string;
  managedJavaScript: string[];
  exceptions: JavaScriptException[];
};

export type PolicyCheckContext = {
  currentJavaScriptPaths: readonly string[];
  seedJavaScriptPaths: ReadonlySet<string>;
  previousPolicy: TypeScriptMigrationPolicy | null;
  changedJavaScriptPaths: ReadonlySet<string>;
  requireComplete: boolean;
};

export type PolicyCheckResult = {
  managedCount: number;
  exceptionCount: number;
  problems: string[];
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isExactJavaScriptPath(path: string): boolean {
  return (
    path.length > 0 &&
    !path.startsWith('/') &&
    !path.includes('\\') &&
    !path.includes('\0') &&
    !/[?*[\]{}]/u.test(path) &&
    path.split('/').every((part) => part !== '' && part !== '.' && part !== '..') &&
    javascriptExtension.test(path)
  );
}

function validateExceptionReason(exception: JavaScriptException): string | null {
  const {path, reason} = exception;
  const sharedHash = sharedReleaseInputHashes.get(path);
  if (sharedHash)
    return reason === sharedReleaseReason(sharedHash)
      ? null
      : `Exception ${path} must retain its exact shared release pin.`;
  if (reason.startsWith('Immutable shared historical release input;'))
    return `Exception ${path} is not a pinned shared release input.`;
  const prefix = acceptedReasonPrefixes.find((candidate) => reason.startsWith(candidate));
  if (!prefix || reason.trim().length < 32) {
    return `Exception ${path} needs an exact path and a specific historical, generated, upstream, or browser-build reason.`;
  }

  const releaseMatch = path.match(releasePath);
  if (releaseMatch) {
    const expected = `Immutable historical release ${releaseMatch[1]};`;
    if (!reason.startsWith(expected)) return `Exception ${path} must name its exact historical release version.`;
    return null;
  }

  if (prefix.startsWith('Immutable historical release ')) {
    return `Exception ${path} names a historical release but is outside a versioned releases directory.`;
  }

  if (
    prefix.startsWith('Browser deployment build output ') &&
    !path.split('/').some((part) => ['build', 'dist', 'public'].includes(part))
  ) {
    return `Exception ${path} is marked as a browser build output but is outside a build, dist, or public directory.`;
  }

  if (prefix === 'Generated from ' && !/\.generated\.(?:js|mjs|cjs|jsx)$/u.test(path)) {
    return `Exception ${path} is marked as generated output but does not use the .generated JavaScript suffix.`;
  }

  if (
    prefix === 'Upstream vendored from ' &&
    !path.split('/').some((part) => ['vendor', 'vendors', 'vendored'].includes(part.toLowerCase()))
  ) {
    return `Exception ${path} is marked as upstream vendored code but is outside a vendor directory.`;
  }

  return null;
}

export function parseTypeScriptMigrationPolicy(value: unknown, source: string): TypeScriptMigrationPolicy {
  if (!isRecord(value)) throw new Error(`${source} must contain a JSON object.`);
  if (value.schemaVersion !== 1) throw new Error(`${source} must use schemaVersion 1.`);
  if (value.baselineCommit !== initialInventoryCommit) {
    throw new Error(`${source} must keep the original inventory commit ${initialInventoryCommit}.`);
  }
  if (!Array.isArray(value.managedJavaScript) || !Array.isArray(value.exceptions)) {
    throw new Error(`${source} must contain managedJavaScript and exceptions arrays.`);
  }

  const managedJavaScript = value.managedJavaScript.map((entry, index) => {
    if (typeof entry !== 'string' || !isExactJavaScriptPath(entry)) {
      throw new Error(`${source} managedJavaScript[${index}] must be an exact JavaScript file path.`);
    }
    return entry;
  });
  const exceptions = value.exceptions.map((entry, index): JavaScriptException => {
    if (!isRecord(entry) || typeof entry.path !== 'string' || typeof entry.reason !== 'string') {
      throw new Error(`${source} exceptions[${index}] must contain an exact path and a reason.`);
    }
    if (!isExactJavaScriptPath(entry.path)) {
      throw new Error(`${source} exceptions[${index}].path must be an exact JavaScript file path.`);
    }
    const exception = {path: entry.path, reason: entry.reason.trim()};
    const problem = validateExceptionReason(exception);
    if (problem) throw new Error(`${source}: ${problem}`);
    return exception;
  });

  const allPaths = [...managedJavaScript, ...exceptions.map(({path}) => path)];
  if (new Set(allPaths).size !== allPaths.length) throw new Error(`${source} contains duplicate JavaScript paths.`);
  if (managedJavaScript.some((path) => path.startsWith('releases/'))) {
    throw new Error(`${source} must record historical releases as exact exceptions.`);
  }
  if (!isSorted(managedJavaScript) || !isSorted(exceptions.map(({path}) => path))) {
    throw new Error(`${source} path entries must be sorted for reviewable changes.`);
  }

  return {
    schemaVersion: 1,
    baselineCommit: initialInventoryCommit,
    managedJavaScript,
    exceptions
  };
}

function isSorted(paths: readonly string[]): boolean {
  return paths.every((path, index) => index === 0 || paths[index - 1]! < path);
}

export function validateTypeScriptMigrationPolicy(
  policy: TypeScriptMigrationPolicy,
  context: PolicyCheckContext
): PolicyCheckResult {
  const currentPaths = new Set(context.currentJavaScriptPaths);
  const managedPaths = new Set(policy.managedJavaScript);
  const exceptionPaths = new Map(policy.exceptions.map(({path, reason}) => [path, reason]));
  const previousManaged = new Set(context.previousPolicy?.managedJavaScript ?? []);
  const previousExceptions = new Map(context.previousPolicy?.exceptions.map(({path, reason}) => [path, reason]) ?? []);
  const problems = [
    ...currentPathProblems(currentPaths, managedPaths, exceptionPaths),
    ...managedPathProblems(managedPaths, currentPaths, context, previousManaged),
    ...exceptionPathProblems(
      exceptionPaths,
      currentPaths,
      context.seedJavaScriptPaths,
      context.previousPolicy,
      previousExceptions
    ),
    ...changedPathProblems(
      context.changedJavaScriptPaths,
      currentPaths,
      context.previousPolicy ? previousManaged : context.seedJavaScriptPaths,
      new Set(
        policy.exceptions
          .filter(
            ({path, reason}) => reason.startsWith('Immutable historical release ') || sharedReleaseInputHashes.has(path)
          )
          .map(({path}) => path)
      )
    )
  ];

  if (context.requireComplete && managedPaths.size > 0) {
    problems.push(`TypeScript migration is incomplete: ${managedPaths.size} direct-managed JavaScript files remain.`);
  }

  return {
    managedCount: [...currentPaths].filter((path) => !exceptionPaths.has(path)).length,
    exceptionCount: [...currentPaths].filter((path) => exceptionPaths.has(path)).length,
    problems
  };
}

function currentPathProblems(
  currentPaths: ReadonlySet<string>,
  managedPaths: ReadonlySet<string>,
  exceptionPaths: ReadonlyMap<string, string>
): string[] {
  const problems: string[] = [];
  for (const path of currentPaths) {
    if (exceptionPaths.has(path)) continue;
    if (path.startsWith('releases/')) {
      problems.push(`${path} is a historical release and must have an exact exception entry.`);
    } else if (!managedPaths.has(path)) {
      problems.push(`${path} is direct-managed JavaScript but is missing from managedJavaScript.`);
    }
  }
  return problems;
}

function managedPathProblems(
  managedPaths: ReadonlySet<string>,
  currentPaths: ReadonlySet<string>,
  context: PolicyCheckContext,
  previousManaged: ReadonlySet<string>
): string[] {
  const problems: string[] = [];
  for (const path of managedPaths) {
    if (!currentPaths.has(path))
      problems.push(`${path} is stale in managedJavaScript; remove it after conversion or deletion.`);
    if (!context.seedJavaScriptPaths.has(path)) {
      problems.push(`${path} is new to the migration baseline; project JavaScript cannot be added.`);
    }
    if (context.previousPolicy && !previousManaged.has(path)) {
      problems.push(`${path} grows managedJavaScript; the migration baseline may only shrink.`);
    }
  }
  return problems;
}

function exceptionPathProblems(
  exceptionPaths: ReadonlyMap<string, string>,
  currentPaths: ReadonlySet<string>,
  seedJavaScriptPaths: ReadonlySet<string>,
  previousPolicy: TypeScriptMigrationPolicy | null,
  previousExceptions: ReadonlyMap<string, string>
): string[] {
  const problems: string[] = [];
  for (const [path, reason] of exceptionPaths) {
    if (!currentPaths.has(path)) problems.push(`${path} is stale in exceptions; remove the missing exception entry.`);
    if (
      !previousExceptions.has(path) &&
      seedJavaScriptPaths.has(path) &&
      !path.startsWith('releases/') &&
      !sharedReleaseInputHashes.has(path)
    ) {
      problems.push(`${path} was in the original inventory; it cannot be newly exempted from migration.`);
    }
    if (previousPolicy && previousExceptions.has(path) && previousExceptions.get(path) !== reason) {
      problems.push(`${path} changed its recorded exception reason; keep exception history stable.`);
    }
  }
  for (const [path] of previousExceptions) {
    if (!exceptionPaths.has(path))
      problems.push(`${path} was removed from exceptions; historical exception entries are immutable.`);
  }
  return problems;
}

function changedPathProblems(
  changedPaths: ReadonlySet<string>,
  currentPaths: ReadonlySet<string>,
  previouslyManaged: ReadonlySet<string>,
  immutableExceptionPaths: ReadonlySet<string>
): string[] {
  const problems: string[] = [];
  for (const path of changedPaths) {
    if (currentPaths.has(path) && (previouslyManaged.has(path) || immutableExceptionPaths.has(path))) {
      problems.push(`${path} was modified while it remains JavaScript; convert it to TypeScript instead.`);
    }
  }
  return problems;
}

function runGit(root: string, args: string[]): string {
  return execFileSync('git', args, {cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe']});
}

function tryRunGit(root: string, args: string[]): string | null {
  try {
    return runGit(root, args);
  } catch {
    return null;
  }
}

function resolveCommit(root: string, ref: string): string {
  if (!ref || ref.startsWith('-') || /[\s\0]/u.test(ref)) throw new Error(`Invalid base commit reference: ${ref}`);
  const commit = tryRunGit(root, ['rev-parse', '--verify', `${ref}^{commit}`]);
  if (!commit) throw new Error(`Could not resolve the base commit ${ref}.`);
  return commit.trim();
}

function parseGitPathList(output: string): string[] {
  return output.split('\0').filter(Boolean);
}

export function discoverJavaScriptPaths(root: string): string[] {
  const output = runGit(root, ['ls-files', '--cached', '--others', '--exclude-standard', '-z']);
  return [...new Set(parseGitPathList(output))]
    .filter((path) => javascriptExtension.test(path) && existsSync(join(root, path)))
    .sort();
}

function isInlineScript(node: DefaultTreeAdapterTypes.Node): boolean {
  if (!defaultTreeAdapter.isElementNode(node) || node.tagName !== 'script') return false;
  if (node.attrs.some(({name}) => name === 'src')) return false;
  const type =
    node.attrs
      .find(({name}) => name === 'type')
      ?.value.trim()
      .toLowerCase() ?? '';
  if (['application/json', 'application/ld+json', 'importmap', 'speculationrules'].includes(type)) return false;
  return node.childNodes.some((child) => defaultTreeAdapter.isTextNode(child) && child.value.trim().length > 0);
}

function hasInlineHandler(node: DefaultTreeAdapterTypes.Node): boolean {
  if (!defaultTreeAdapter.isElementNode(node)) return false;
  return node.attrs.some(({name, value}) => {
    if (/^on[a-z]+$/u.test(name) && value.trim()) return true;
    if (!['href', 'src', 'action', 'formaction'].includes(name)) return false;
    return value
      .split('')
      .filter((character) => character.charCodeAt(0) > 32)
      .join('')
      .toLowerCase()
      .startsWith('javascript:');
  });
}

function containsInlineJavaScript(node: DefaultTreeAdapterTypes.Node): boolean {
  if (isInlineScript(node) || hasInlineHandler(node)) return true;
  if (defaultTreeAdapter.isElementNode(node) && node.tagName === 'template' && 'content' in node) {
    if (containsInlineJavaScript(defaultTreeAdapter.getTemplateContent(node))) return true;
  }
  return 'childNodes' in node && node.childNodes.some(containsInlineJavaScript);
}

/** Find inline JavaScript in tracked or nonignored HTML. */
export function discoverInlineJavaScriptPaths(root: string): string[] {
  const output = runGit(root, ['ls-files', '--cached', '--others', '--exclude-standard', '-z']);
  const paths = [...new Set(parseGitPathList(output))].filter(
    (path) => /\.html?$/iu.test(path) && existsSync(join(root, path))
  );
  return paths.filter((path) => containsInlineJavaScript(parse(readFileSync(join(root, path), 'utf8')))).sort();
}

function discoverSeedJavaScriptPaths(root: string, baselineCommit: string): Set<string> {
  const output = runGit(root, ['ls-tree', '-r', '-z', '--name-only', baselineCommit]);
  return new Set(parseGitPathList(output).filter((path) => javascriptExtension.test(path)));
}

export function discoverChangedJavaScriptPaths(root: string, baseCommit: string): Set<string> {
  const output = runGit(root, ['diff', '--no-renames', '--name-only', '-z', baseCommit, '--']);
  return new Set(parseGitPathList(output).filter((path) => javascriptExtension.test(path)));
}

function readCurrentPolicy(root: string): TypeScriptMigrationPolicy {
  const source = readFileSync(join(root, policyPath), 'utf8');
  return parseTypeScriptMigrationPolicy(JSON.parse(source) as unknown, policyPath);
}

function readPolicyAtCommit(root: string, commit: string): TypeScriptMigrationPolicy | null {
  const revision = `${commit}:${policyPath}`;
  if (!tryRunGit(root, ['cat-file', '-e', revision])) return null;
  const source = runGit(root, ['show', revision]);
  return parseTypeScriptMigrationPolicy(JSON.parse(source) as unknown, `${revision}`);
}

function defaultBaseCommit(root: string, policy: TypeScriptMigrationPolicy): string {
  const environmentBase = process.env.BASE_SHA;
  if (environmentBase) return resolveCommit(root, environmentBase);

  const remoteMain = tryRunGit(root, ['rev-parse', '--verify', 'origin/main^{commit}']);
  if (remoteMain) return remoteMain.trim();
  return resolveCommit(root, policy.baselineCommit);
}

/** Verify immutable shared inputs and the release locks that reference them. */
export function checkSharedReleaseInputs(root: string): string[] {
  const problems: string[] = [];
  for (const [path, expectedHash] of sharedReleaseInputHashes) {
    try {
      const actualHash = createHash('sha256')
        .update(readFileSync(join(root, path)))
        .digest('hex');
      if (actualHash !== expectedHash) problems.push(`${path} differs from its historical release hash.`);
    } catch {
      problems.push(`${path} is missing from the historical release inputs.`);
    }
    for (const version of sharedReleaseVersions) checkSharedReleaseLock(root, version, path, expectedHash, problems);
  }
  return problems;
}

function checkSharedReleaseLock(
  root: string,
  version: string,
  path: string,
  expectedHash: string,
  problems: string[]
): void {
  const lockPath = `releases/web-decision-${version}/source-lock.json`;
  try {
    const lock: unknown = JSON.parse(readFileSync(join(root, lockPath), 'utf8'));
    const files = isRecord(lock) && Array.isArray(lock.files) ? lock.files.filter(isRecord) : [];
    const matching = files.filter((entry) => entry.path === path);
    if (matching.length !== 1 || matching[0]?.sha256 !== expectedHash)
      problems.push(`${lockPath} does not pin the expected ${path} hash.`);
  } catch {
    problems.push(`${lockPath} cannot be read as a historical release lock.`);
  }
}

export function checkTypeScriptMigration(
  root: string,
  options: {baseRef?: string; requireComplete?: boolean} = {}
): PolicyCheckResult {
  const policy = readCurrentPolicy(root);
  const baseCommit = resolveCommit(root, options.baseRef ?? defaultBaseCommit(root, policy));
  const previousPolicy = readPolicyAtCommit(root, baseCommit);
  const context: PolicyCheckContext = {
    currentJavaScriptPaths: discoverJavaScriptPaths(root),
    seedJavaScriptPaths: discoverSeedJavaScriptPaths(root, policy.baselineCommit),
    previousPolicy,
    changedJavaScriptPaths: discoverChangedJavaScriptPaths(root, baseCommit),
    requireComplete: options.requireComplete ?? false
  };
  const result = validateTypeScriptMigrationPolicy(policy, context);
  result.problems.push(...checkSharedReleaseInputs(root));
  if (context.requireComplete) {
    for (const path of discoverInlineJavaScriptPaths(root)) {
      result.problems.push(`${path} contains inline JavaScript; move it to a TypeScript module.`);
    }
  }
  return result;
}

function parseArguments(arguments_: string[]): {baseRef?: string; requireComplete: boolean} {
  const parsed: {baseRef?: string; requireComplete: boolean} = {requireComplete: false};
  for (let index = 0; index < arguments_.length; index += 1) {
    const argument = arguments_[index];
    const nextArgument = arguments_[index + 1];
    if (argument === '--require-complete') {
      parsed.requireComplete = true;
    } else if (argument === '--base-ref' && nextArgument) {
      parsed.baseRef = nextArgument;
      index += 1;
    } else {
      throw new Error(`Unknown or incomplete argument: ${argument ?? ''}`);
    }
  }
  return parsed;
}

function main(): number {
  try {
    const options = parseArguments(process.argv.slice(2));
    const root = fileURLToPath(new URL('../../', import.meta.url));
    const result = checkTypeScriptMigration(root, options);
    if (result.problems.length > 0) {
      console.error('TypeScript migration policy failed:');
      for (const problem of result.problems) console.error(`- ${problem}`);
      return 1;
    }

    console.log(
      `TypeScript migration policy passed: ${result.managedCount} direct-managed JavaScript files remain; ${result.exceptionCount} exact exceptions are recorded.`
    );
    return 0;
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    return 1;
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = main();
}
