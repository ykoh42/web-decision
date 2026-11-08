import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  completionViolations,
  loadSonarBaseline,
  parseSonarBaseline,
  requireComplete
} from './check_cognitive_complexity_completion.mts';

function baseline(findings: unknown[]): string {
  return JSON.stringify({
    schemaVersion: 1,
    threshold: 15,
    versions: {oxlint: '1.92.0', sonarjs: '4.2.2', typescriptParser: '6.0.3'},
    generatedExclusions: [],
    findings
  });
}

function completionFindings(text: string, sourcePaths: string[]) {
  return completionViolations(parseSonarBaseline(text), new Set(sourcePaths));
}

test('excludes only the fixed historical semver release directories', () => {
  const findings = [
    {file: 'releases/web-decision-0.1.0/legacy.js', identity: 'module/function:old', score: 80},
    {file: 'releases/web-decision-0.2.0/legacy.ts', identity: 'module/function:old', score: 80},
    {file: 'releases/web-decision-0.3.0/legacy.mjs', identity: 'module/function:old', score: 80}
  ];
  assert.deepEqual(
    completionFindings(
      baseline(findings),
      findings.map(({file}) => file)
    ),
    []
  );
  assert.doesNotThrow(() =>
    requireComplete(parseSonarBaseline(baseline(findings)), new Set(findings.map(({file}) => file)))
  );
});

test('new release snapshots and vendor paths stay managed', () => {
  const findings = [
    {file: 'releases/web-decision-0.4.0/future.js', identity: 'module/function:future', score: 16},
    {file: 'vendor/tvm/runtime.ts', identity: 'module/function:runtime', score: 17}
  ];
  assert.throws(
    () => requireComplete(parseSonarBaseline(baseline(findings)), new Set(findings.map(({file}) => file))),
    /Managed JavaScript\/TypeScript source remains/u
  );
});

test('fails while any managed JavaScript or TypeScript baseline finding exceeds 15', () => {
  const findings = [{file: 'packages/app/src/route.mts', identity: 'module/function:route', score: 16}];
  assert.throws(
    () => requireComplete(parseSonarBaseline(baseline(findings)), new Set(['packages/app/src/route.mts'])),
    /Managed JavaScript\/TypeScript source remains/u
  );
});

test('rejects malformed baselines and invalid baseline paths', () => {
  assert.throws(() => parseSonarBaseline('{'), /Cannot parse SonarJS/u);
  assert.throws(() => parseSonarBaseline('[]'), /invalid schema or threshold/u);
  const extraField = JSON.parse(baseline([])) as Record<string, unknown>;
  extraField['unexpected'] = true;
  assert.throws(() => parseSonarBaseline(JSON.stringify(extraField)), /invalid schema or threshold/u);
  assert.throws(
    () => parseSonarBaseline(baseline([{file: '../outside.ts', identity: 'module/function:route', score: 16}])),
    /Invalid SonarJS baseline path/u
  );
  assert.throws(
    () => parseSonarBaseline(baseline([{file: 'packages/app/src/route.ts', identity: '', score: 16}])),
    /empty function identity/u
  );
  assert.throws(
    () =>
      parseSonarBaseline(baseline([{file: 'packages/app/src/route.ts', identity: 'module/function:route', score: 15}])),
    /invalid score/u
  );
});

test('rejects baseline findings without a current scanned source', () => {
  const text = baseline([{file: 'packages/app/src/missing.js', identity: 'module/function:old', score: 16}]);
  assert.throws(() => completionFindings(text, []), /missing or unscanned source/u);
});

test('fails closed when the SonarJS baseline file is missing', (context) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'complexity-completion-'));
  context.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  assert.throws(() => loadSonarBaseline(path.join(directory, 'missing.json')), /baseline is missing or unreadable/u);
  const malformedPath = path.join(directory, 'malformed.json');
  fs.writeFileSync(malformedPath, Buffer.from([0xff]));
  assert.throws(() => loadSonarBaseline(malformedPath), /baseline is missing or unreadable/u);
});
