import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  GENERATED_EXCLUSIONS,
  ROOT,
  TYPESCRIPT_PARSER_VERSION,
  checkBaseline,
  compareFindingsToTrusted,
  diagnosticIdentity,
  findComplexitySuppressions,
  runOxlint,
  selectTrackedSources,
  sourceNamesInWorkingTree,
  workingTreeSources,
  validateFindings
} from './check_cognitive_complexity.mjs';
import type {ComplexityBaseline, ComplexityFinding, OxlintDiagnostic} from './check_cognitive_complexity.mjs';

function diagnostic(source: string, marker: string, file = 'fixture.mjs', score = 16): OxlintDiagnostic {
  const index = source.indexOf(marker);
  assert.notEqual(index, -1, `Missing diagnostic marker ${marker}`);
  return {
    message: `Refactor this function to reduce its Cognitive Complexity from ${score} to the 15 allowed.`,
    labels: [{span: {offset: Buffer.byteLength(source.slice(0, index), 'utf8'), line: 1, column: 1}}],
    filename: file
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function readToolVersions(): {oxlint: string; 'eslint-plugin-sonarjs': string} {
  const packageJson: unknown = JSON.parse(fs.readFileSync(new URL('../../package.json', import.meta.url), 'utf8'));
  if (!isRecord(packageJson) || !isRecord(packageJson.devDependencies)) {
    throw new Error('package.json is missing devDependencies.');
  }
  const {oxlint, 'eslint-plugin-sonarjs': sonarjs} = packageJson.devDependencies;
  if (typeof oxlint !== 'string' || typeof sonarjs !== 'string') {
    throw new Error('package.json is missing pinned Oxlint or SonarJS versions.');
  }
  return {oxlint, 'eslint-plugin-sonarjs': sonarjs};
}

const versions = readToolVersions();

function baseline(findings: readonly ComplexityFinding[]): ComplexityBaseline {
  return {
    schemaVersion: 1,
    threshold: 15,
    versions: {
      oxlint: versions.oxlint,
      sonarjs: versions['eslint-plugin-sonarjs'],
      typescriptParser: TYPESCRIPT_PARSER_VERSION
    },
    generatedExclusions: GENERATED_EXCLUSIONS,
    findings
  };
}

function makeCacheFixture(prefix: string): string {
  const cacheDirectory = path.join(ROOT, 'node_modules/.cache');
  fs.mkdirSync(cacheDirectory, {recursive: true});
  return fs.mkdtempSync(path.join(cacheDirectory, prefix));
}

function makeGitSourceFixture(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'complexity-source-scan-'));
  fs.mkdirSync(path.join(root, 'src'), {recursive: true});
  fs.writeFileSync(path.join(root, 'src/legacy.mjs'), 'export const legacy = true;\n');
  fs.writeFileSync(path.join(root, 'src/retained.ts'), 'export const retained = true;\n');
  execFileSync('git', ['init', '--quiet'], {cwd: root});
  execFileSync('git', ['add', '--', 'src/legacy.mjs', 'src/retained.ts'], {cwd: root});
  fs.rmSync(path.join(root, 'src/legacy.mjs'));
  fs.writeFileSync(path.join(root, 'src/new-module.mts'), 'export const moduleSource = true;\n');
  fs.writeFileSync(path.join(root, 'src/new-browser.ts'), 'export const browserSource = true;\n');
  return root;
}

test('fails closed when the tracked JavaScript/TypeScript list is empty', () => {
  assert.throws(() => selectTrackedSources([], []), /No tracked JavaScript or TypeScript sources/);
});

test('includes ignored tracked source paths and exact release-source overlays', () => {
  const tracked = [
    '.ignored/hidden.ts',
    'releases/web-decision-0.1.0/source/overlay/packages/web-decision/src/main.js',
    ...GENERATED_EXCLUSIONS.map(({path: name}) => name)
  ];
  const selected = selectTrackedSources(tracked);
  assert.ok(selected.includes('.ignored/hidden.ts'));
  assert.ok(selected.includes('releases/web-decision-0.1.0/source/overlay/packages/web-decision/src/main.js'));
  for (const {path: name} of GENERATED_EXCLUSIONS) assert.ok(!selected.includes(name));
});

test('working-tree scan omits only git-confirmed deletions and includes new TS and MTS files', (context) => {
  const root = makeGitSourceFixture();
  context.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const selected = workingTreeSources(root, []);
  assert.deepEqual(selected, ['src/new-browser.ts', 'src/new-module.mts', 'src/retained.ts']);
});

test('maps diagnostics to stable function identity when lines move', () => {
  const original = `function route(value) {\n  if (value) return 1;\n  return 0;\n}\n`;
  const shifted = `\n\n${original}`;
  const first = diagnosticIdentity(diagnostic(original, 'if (value)'), original, 'fixture.mjs');
  const second = diagnosticIdentity(diagnostic(shifted, 'if (value)'), shifted, 'fixture.mjs');
  assert.equal(first.identity, second.identity);
  assert.equal(first.identity, 'module/function:route');
});

test('maps TypeScript nested functions to the innermost stable function', () => {
  const source = `class Router {\n  outer(value: number) {\n    const inner = (next: number): number => {\n      if (next > 0) return next;\n      return value;\n    };\n    return inner(value);\n  }\n}\n`;
  const finding = diagnosticIdentity(diagnostic(source, 'if (next', 'fixture.ts'), source, 'fixture.ts');
  assert.match(finding.identity, /member:MethodDeclaration:outer/u);
  assert.match(finding.identity, /binding:inner/u);
  assert.ok(!finding.identity.endsWith('member:MethodDeclaration:outer'));
});

test('anonymous callback identity is stable across line shifts and retains its lexical parent', () => {
  const original = `function outer(values) {\n  return values.map((value) => {\n    if (value > 0) return value;\n    return 0;\n  });\n}\n`;
  const shifted = `\n\n${original}`;
  const first = diagnosticIdentity(diagnostic(original, 'if (value > 0)'), original, 'fixture.mjs');
  const second = diagnosticIdentity(diagnostic(shifted, 'if (value > 0)'), shifted, 'fixture.mjs');
  assert.equal(first.identity, second.identity);
  assert.match(first.identity, /function:outer\/callback:/u);
});

test('converts Oxlint UTF-8 byte offsets before resolving Unicode TypeScript positions', () => {
  const source = `const label = "🎯";\nfunction route(value) {\n  if (value) return 1;\n  return 0;\n}\n`;
  const finding = diagnosticIdentity(diagnostic(source, 'if (value)'), source, 'fixture.mjs');
  assert.equal(finding.identity, 'module/function:route');
});

test('fails closed when the TypeScript identity parser reports syntax errors', () => {
  const source = `const invalid: number = ;\nfunction route(value: number) {\n  if (value) return 1;\n  return 0;\n}\n`;
  assert.throws(
    () => diagnosticIdentity(diagnostic(source, 'if (value)', 'fixture.ts'), source, 'fixture.ts'),
    /could not safely parse fixture\.ts/u
  );
});

test('rejects duplicate file/function identities', () => {
  const findings = [
    {file: 'fixture.ts', identity: 'module/function:route', score: 16},
    {file: 'fixture.ts', identity: 'module/function:route', score: 17}
  ];
  assert.throws(() => validateFindings(findings), /duplicate function identities/u);
});

test('rejects new and worsened violations relative to the trusted baseline', () => {
  const initial = baseline([{file: 'fixture.ts', identity: 'module/function:route', score: 16}]);
  assert.throws(
    () =>
      checkBaseline(
        baseline([...initial.findings, {file: 'new.ts', identity: 'module/function:newRoute', score: 16}]),
        initial
      ),
    /New SonarJS cognitive complexity violation/u
  );
  const initialFinding = initial.findings[0];
  assert.ok(initialFinding);
  assert.throws(() => checkBaseline(baseline([{...initialFinding, score: 17}]), initial), /increased/u);
  assert.throws(
    () => checkBaseline(initial, {...initial, versions: {...initial.versions, sonarjs: '5.0.0'}}),
    /tool versions differ/u
  );
});

test('rejects broad or complexity-specific suppressions but permits unrelated lint rules', () => {
  assert.equal(findComplexitySuppressions('// eslint-disable-next-line no-control-regex\n').length, 0);
  assert.equal(findComplexitySuppressions('/* eslint-disable sonarjs/cognitive-complexity */\n').length, 1);
  assert.equal(findComplexitySuppressions('// oxlint-disable-next-line\n').length, 1);
  assert.equal(findComplexitySuppressions('// eslint-disable sonarjs/*\n').length, 1);
});

test('the configured Oxlint invocation detects complexity in real JavaScript and TypeScript files', () => {
  const fixtureDirectory = makeCacheFixture('cognitive-fixture-');
  const jsPath = path.relative(ROOT, path.join(fixtureDirectory, 'high-complexity.js')).replaceAll('\\', '/');
  const tsPath = path.relative(ROOT, path.join(fixtureDirectory, 'high-complexity.ts')).replaceAll('\\', '/');
  const statements = Array.from({length: 16}, (_, index) => `  if (value === ${index}) value += 1;`).join('\n');
  const jsSource = `export function highComplexityJs(value) {\n${statements}\n  return value;\n}\n`;
  const tsSource = `export function highComplexityTs(value: number): number {\n${statements}\n  return value;\n}\n`;
  try {
    fs.writeFileSync(path.join(fixtureDirectory, 'high-complexity.js'), jsSource);
    fs.writeFileSync(path.join(fixtureDirectory, 'high-complexity.ts'), tsSource);
    const diagnostics = runOxlint([jsPath, tsPath]);
    assert.equal(diagnostics.length, 2, JSON.stringify(diagnostics));
    const findings = diagnostics.map((item) => {
      const file = item.filename.replaceAll('\\', '/');
      const source = fs.readFileSync(path.join(ROOT, ...file.split('/')), 'utf8');
      return diagnosticIdentity(item, source, file);
    });
    assert.deepEqual(findings.map(({file}) => file).sort(), [jsPath, tsPath].sort());
    assert.ok(findings.every(({score}) => score === 16));
    assert.ok(findings.some(({identity}) => identity.endsWith('function:highComplexityJs')));
    assert.ok(findings.some(({identity}) => identity.endsWith('function:highComplexityTs')));
  } finally {
    fs.rmSync(fixtureDirectory, {recursive: true, force: true});
  }
});

test('a new untracked non-ignored high-complexity file is discovered and rejected', () => {
  const fixtureDirectory = fs.mkdtempSync(path.join(ROOT, '.sonar-untracked-fixture-'));
  const filePath = path.join(fixtureDirectory, 'new-route.js');
  const relativePath = path.relative(ROOT, filePath).replaceAll('\\', '/');
  const statements = Array.from({length: 16}, (_, index) => `  if (value === ${index}) value += 1;`).join('\n');
  const source = `export function newRoute(value) {\n${statements}\n  return value;\n}\n`;
  try {
    fs.writeFileSync(filePath, source);
    const selected = selectTrackedSources(sourceNamesInWorkingTree());
    assert.ok(selected.includes(relativePath), `${relativePath} was not selected from git ls-files --others`);
    const diagnostics = runOxlint([relativePath]);
    assert.equal(diagnostics.length, 1);
    const findings = diagnostics.map((item) => diagnosticIdentity(item, source, relativePath));
    assert.throws(() => compareFindingsToTrusted(findings, []), /New SonarJS cognitive complexity violation/u);
  } finally {
    fs.rmSync(fixtureDirectory, {recursive: true, force: true});
  }
});

test('ambiguous anonymous callbacks produce duplicate identities and fail closed', () => {
  const fixtureDirectory = makeCacheFixture('cognitive-duplicate-');
  const filePath = path.join(fixtureDirectory, 'duplicate.js');
  const relativePath = path.relative(ROOT, filePath).replaceAll('\\', '/');
  const statements = Array.from({length: 16}, (_, index) => `    if (value === ${index}) value += 1;`).join('\n');
  const callback = `values.map((value) => {\n${statements}\n    return value;\n  });`;
  const source = `function outer(values) {\n  ${callback}\n  ${callback}\n  return values;\n}\n`;
  try {
    fs.writeFileSync(filePath, source);
    const diagnostics = runOxlint([relativePath]);
    assert.equal(diagnostics.length, 2);
    const findings = diagnostics.map((item) => diagnosticIdentity(item, source, relativePath));
    const firstFinding = findings[0];
    const secondFinding = findings[1];
    assert.ok(firstFinding);
    assert.ok(secondFinding);
    assert.equal(firstFinding.identity, secondFinding.identity);
    assert.throws(() => validateFindings(findings), /duplicate function identities/u);
  } finally {
    fs.rmSync(fixtureDirectory, {recursive: true, force: true});
  }
});
