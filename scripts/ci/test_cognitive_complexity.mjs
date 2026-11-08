import assert from 'node:assert/strict';
import fs from 'node:fs';
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
  validateFindings
} from './check_cognitive_complexity.mjs';

function diagnostic(source, marker, file = 'fixture.mjs', score = 16) {
  const index = source.indexOf(marker);
  assert.notEqual(index, -1, `Missing diagnostic marker ${marker}`);
  return {
    message: `Refactor this function to reduce its Cognitive Complexity from ${score} to the 15 allowed.`,
    labels: [{span: {offset: Buffer.byteLength(source.slice(0, index), 'utf8'), line: 1, column: 1}}],
    filename: file
  };
}

const versions = JSON.parse(fs.readFileSync(new URL('../../package.json', import.meta.url), 'utf8')).devDependencies;

function baseline(findings) {
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

function makeCacheFixture(prefix) {
  const cacheDirectory = path.join(ROOT, 'node_modules/.cache');
  fs.mkdirSync(cacheDirectory, {recursive: true});
  return fs.mkdtempSync(path.join(cacheDirectory, prefix));
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
  assert.throws(() => checkBaseline(baseline([{...initial.findings[0], score: 17}]), initial), /increased/u);
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
    assert.equal(findings[0].identity, findings[1].identity);
    assert.throws(() => validateFindings(findings), /duplicate function identities/u);
  } finally {
    fs.rmSync(fixtureDirectory, {recursive: true, force: true});
  }
});
