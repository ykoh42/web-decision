#!/usr/bin/env node
// Ratchet SonarJS cognitive complexity across tracked JavaScript and TypeScript.

import {spawnSync} from 'node:child_process';
import type {SpawnSyncOptionsWithBufferEncoding, SpawnSyncReturns} from 'node:child_process';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import fs from 'node:fs';

interface ParserNode {
  readonly kind: number;
  readonly parent?: ParserNode;
  readonly end: number;
  readonly text?: string;
  readonly name?: ParserNode;
  readonly body?: ParserNode;
  readonly initializer?: ParserNode;
  readonly propertyName?: ParserNode;
  readonly elements?: readonly ParserBindingElement[];
  readonly expression?: ParserNode;
  readonly right?: ParserNode;
  readonly left?: ParserNode;
  readonly operatorToken?: ParserNode;
  readonly arguments?: readonly ParserNode[];
  getStart(sourceFile?: ParserSourceFile): number;
  forEachChild(callback: (child: ParserNode) => void): unknown;
}

interface ParserBindingElement extends ParserNode {
  readonly name: ParserNode;
}

interface ParserFunctionLike extends ParserNode {
  readonly parameters: readonly ParserNode[];
  readonly name?: ParserNode;
  readonly body?: ParserNode;
}

interface ParserCallLike extends ParserNode {
  readonly expression: ParserNode;
  readonly arguments?: readonly ParserNode[];
}

interface ParserSourceFile extends ParserNode {
  readonly parseDiagnostics?: readonly {readonly messageText: string | ParserDiagnosticChain}[];
}

interface ParserDiagnosticChain {
  readonly messageText: string;
  readonly next?: readonly ParserDiagnosticChain[];
}

interface ParserScanner {
  scan(): number;
  getTokenText(): string;
  getTokenPos(): number;
}

interface ParserSyntaxKind {
  readonly SingleLineCommentTrivia: number;
  readonly MultiLineCommentTrivia: number;
  readonly EndOfFileToken: number;
  readonly EqualsToken: number;
  readonly [name: string]: number | string;
}

interface SonarTypeScriptParser {
  readonly ScriptTarget: {readonly Latest: number};
  readonly LanguageVariant: {readonly Standard: number};
  readonly ScriptKind: {readonly TSX: number; readonly JSX: number};
  readonly SyntaxKind: ParserSyntaxKind;
  readonly version: string;
  createScanner(languageVersion: number, skipTrivia: boolean, languageVariant: number, text: string): ParserScanner;
  createSourceFile(
    fileName: string,
    sourceText: string,
    languageVersion: number,
    setParentNodes?: boolean,
    scriptKind?: number
  ): ParserSourceFile;
  flattenDiagnosticMessageText(message: string | ParserDiagnosticChain, newLine: string, indent?: number): string;
  isFunctionLike(node: ParserNode): boolean;
  isIdentifier(node: ParserNode): node is ParserNode & {readonly text: string};
  isPrivateIdentifier(node: ParserNode): node is ParserNode & {readonly text: string};
  isStringLiteralLike(node: ParserNode): node is ParserNode & {readonly text: string};
  isNumericLiteral(node: ParserNode): node is ParserNode & {readonly text: string};
  isComputedPropertyName(node: ParserNode): node is ParserNode & {readonly expression: ParserNode};
  isObjectBindingPattern(node: ParserNode): node is ParserNode & {readonly elements: readonly ParserBindingElement[]};
  isArrayBindingPattern(node: ParserNode): node is ParserNode & {readonly elements: readonly ParserBindingElement[]};
  isFunctionDeclaration(node: ParserNode): boolean;
  isFunctionExpression(node: ParserNode): boolean;
  isArrowFunction(node: ParserNode): boolean;
  isMethodDeclaration(node: ParserNode): boolean;
  isGetAccessorDeclaration(node: ParserNode): boolean;
  isSetAccessorDeclaration(node: ParserNode): boolean;
  isConstructorDeclaration(node: ParserNode): boolean;
  isVariableDeclaration(
    node: ParserNode
  ): node is ParserNode & {readonly name: ParserNode; readonly initializer?: ParserNode};
  isPropertyAssignment(
    node: ParserNode
  ): node is ParserNode & {readonly name: ParserNode; readonly initializer: ParserNode};
  isBinaryExpression(
    node: ParserNode
  ): node is ParserNode & {readonly left: ParserNode; readonly right: ParserNode; readonly operatorToken: ParserNode};
  isExportAssignment(node: ParserNode): node is ParserNode & {readonly expression: ParserNode};
  isCallExpression(node: ParserNode): node is ParserCallLike & {readonly arguments: readonly ParserNode[]};
  isNewExpression(node: ParserNode): node is ParserCallLike;
  isStatement(node: ParserNode): boolean;
  isClassDeclaration(node: ParserNode): boolean;
  isClassExpression(node: ParserNode): boolean;
}

export interface GeneratedExclusion {
  readonly path: string;
  readonly reason: string;
}

export interface ComplexityFinding {
  readonly file: string;
  readonly identity: string;
  readonly score: number;
}

export interface ToolVersions {
  readonly oxlint: string;
  readonly sonarjs: string;
  readonly typescriptParser: string;
}

export interface ComplexityBaseline {
  readonly schemaVersion: 1;
  readonly threshold: 15;
  readonly versions: ToolVersions;
  readonly generatedExclusions: readonly GeneratedExclusion[];
  readonly findings: readonly ComplexityFinding[];
}

export interface OxlintDiagnostic {
  readonly message: string;
  readonly filename: string;
  readonly labels?: readonly {
    readonly span?: {
      readonly offset?: number;
      readonly line?: number;
      readonly column?: number;
    };
  }[];
}

export interface ComplexitySuppression {
  readonly offset: number;
  readonly directive: string;
}

interface InternalOxlintDiagnostic extends OxlintDiagnostic {
  readonly code: string;
}

interface OxlintReport {
  readonly number_of_files: number;
  readonly diagnostics: readonly InternalOxlintDiagnostic[];
}

interface UnvalidatedBaseline {
  readonly schemaVersion?: unknown;
  readonly threshold?: unknown;
  readonly versions?: {
    readonly oxlint?: unknown;
    readonly sonarjs?: unknown;
    readonly typescriptParser?: unknown;
  } | null;
  readonly generatedExclusions?: unknown;
  readonly findings?: unknown;
}

interface PackageManifest {
  readonly devDependencies?: {
    readonly oxlint?: string;
    readonly 'eslint-plugin-sonarjs'?: string;
  };
}

interface PnpmLockSnapshot {
  readonly dependencies?: {
    readonly typescript?: string;
  };
}

interface PnpmLockValue {
  readonly snapshots?: Readonly<Record<string, PnpmLockSnapshot>>;
}

interface ValidatedBaseline extends ComplexityBaseline {
  readonly scores: Map<string, number>;
}

interface SuppressionDirective {
  readonly rules: string[];
  readonly broad: boolean;
}

interface TreeEntry {
  readonly mode: string;
  readonly type: string;
  readonly oid: string;
}

interface SourceBlobEntry {
  readonly path: string;
  readonly oid: string;
}

interface YamlParseError {
  readonly message: string;
}

interface YamlDocument {
  readonly errors: readonly YamlParseError[];
  toJSON(): unknown;
}

interface YamlParser {
  parseAllDocuments(source: string): YamlDocument[];
}

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const BASELINE_PATH = 'sonarjs-cognitive-complexity-baseline.json';
const BASELINE = path.join(ROOT, BASELINE_PATH);
const THRESHOLD = 15;
const OXLINT_BATCH_SIZE = 80;
const RULE_CODE = 'sonarjs(cognitive-complexity)';
const RULE_ID = 'sonarjs/cognitive-complexity';
const SOURCE_EXTENSIONS = new Set(['.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx', '.mts', '.cts']);

// These tracked files are generated artifacts, not maintainable source. Every
// release source overlay and vendored TypeScript source file remains in scope.
export const GENERATED_EXCLUSIONS: readonly GeneratedExclusion[] = Object.freeze([
  {
    path: 'releases/web-decision-0.3.0/package/dist/index.js',
    reason:
      'Published distribution bundle generated from the audited release source; minified/compiled output is not a maintainable source file.'
  },
  ...['0.1.0', '0.2.0', '0.3.0'].map((version) => ({
    path: `releases/web-decision-${version}/source/overlay/.toolchains/apache-tvm-src/web/src/tvmjs_runtime_wasi.js`,
    reason:
      'Four-line Emscripten/TVM-generated glue; the pinned upstream TypeScript/C++ source remains in scope and CI regenerates this artifact.'
  }))
]);

const require = createRequire(import.meta.url);
const sonarPluginEntry = require.resolve('eslint-plugin-sonarjs');
// Keep the parser loaded by the pinned SonarJS plugin; this facade covers only the compiler API used here.
const ts = createRequire(sonarPluginEntry)('typescript') as SonarTypeScriptParser;
const yaml = createRequire(sonarPluginEntry)('yaml') as YamlParser;

export {ROOT};
export const TYPESCRIPT_PARSER_VERSION = ts.version;

function fail(message: string): never {
  throw new Error(message);
}

function normalizedTrackedPath(name: string): string {
  if (
    typeof name !== 'string' ||
    name.length === 0 ||
    name.includes('\\') ||
    name.startsWith('/') ||
    path.posix.normalize(name) !== name ||
    name.split('/').includes('..')
  ) {
    fail(`Invalid tracked source path: ${String(name)}`);
  }
  return name;
}

export function selectTrackedSources(
  trackedNames: readonly string[],
  exclusions: readonly GeneratedExclusion[] = GENERATED_EXCLUSIONS
): string[] {
  const names = trackedNames.map(normalizedTrackedPath);
  if (new Set(names).size !== names.length) fail('Duplicate tracked paths were reported by git.');

  const tracked = new Set(names);
  for (const exclusion of exclusions) {
    normalizedTrackedPath(exclusion.path);
    if (!tracked.has(exclusion.path)) {
      fail(`Declared generated exclusion is no longer tracked: ${exclusion.path}`);
    }
    if (typeof exclusion.reason !== 'string' || exclusion.reason.trim() === '') {
      fail(`Generated exclusion has no reason: ${exclusion.path}`);
    }
  }

  const excluded = new Set(exclusions.map(({path: name}) => name));
  const selected = names.filter((name) => SOURCE_EXTENSIONS.has(path.posix.extname(name)) && !excluded.has(name));
  if (selected.length === 0) fail('No tracked JavaScript or TypeScript sources were selected.');
  return selected.sort();
}

function git(args: readonly string[], options: SpawnSyncOptionsWithBufferEncoding = {}): SpawnSyncReturns<Buffer> {
  const result = spawnSync('git', args, {
    cwd: ROOT,
    encoding: 'buffer',
    maxBuffer: 128 * 1024 * 1024,
    ...options
  });
  if (result.error) throw result.error;
  return result;
}

export function sourceNamesInWorkingTree(root = ROOT): string[] {
  const result = spawnSync('git', ['ls-files', '--cached', '--others', '--exclude-standard', '-z'], {
    cwd: root,
    encoding: 'buffer',
    maxBuffer: 128 * 1024 * 1024
  });
  if (result.error) throw result.error;
  if (result.status !== 0) fail(`git ls-files failed: ${result.stderr.toString('utf8')}`);
  return result.stdout
    .toString('utf8')
    .split('\0')
    .filter((name) => name.length > 0);
}

function workingTreeSources(): string[] {
  return selectTrackedSources(sourceNamesInWorkingTree());
}

function verifySourceFile(relativePath: string): string {
  const absolutePath = path.join(ROOT, ...relativePath.split('/'));
  const stats = fs.lstatSync(absolutePath);
  if (!stats.isFile() || stats.isSymbolicLink()) {
    fail(`Tracked source is not a regular file: ${relativePath}`);
  }
  return fs.readFileSync(absolutePath, 'utf8');
}

function suppressionDirective(comment: string): SuppressionDirective | null {
  const match = /\b(?:eslint|oxlint)-disable(?:-next-line|-line)?\b/i.exec(comment);
  if (!match) return null;
  const tail = comment
    .slice(match.index + match[0].length)
    .split(/\s+--\s+/u, 1)[0]!
    .trim();
  const rules = tail
    .split(/[,\s]+/u)
    .map((value) => value.trim())
    .filter(Boolean);
  return {rules, broad: rules.length === 0 || rules.includes('*')};
}

export function findComplexitySuppressions(source: string): ComplexitySuppression[] {
  const scanner = ts.createScanner(ts.ScriptTarget.Latest, false, ts.LanguageVariant.Standard, source);
  const violations: ComplexitySuppression[] = [];
  let token: number;
  do {
    token = scanner.scan();
    if (token !== ts.SyntaxKind.SingleLineCommentTrivia && token !== ts.SyntaxKind.MultiLineCommentTrivia) {
      continue;
    }
    const text = scanner.getTokenText();
    const directive = suppressionDirective(text);
    if (
      directive &&
      (directive.broad ||
        directive.rules.some(
          (rule) =>
            rule === RULE_ID ||
            rule === 'sonarjs' ||
            rule === 'sonarjs/*' ||
            rule.endsWith('/cognitive-complexity') ||
            rule === 'cognitive-complexity'
        ))
    ) {
      violations.push({offset: scanner.getTokenPos(), directive: text.trim()});
    }
  } while (token !== ts.SyntaxKind.EndOfFileToken);
  return violations;
}

function sourceFileFor(source: string, fileName: string): ParserSourceFile {
  const kind = /\.[cm]?tsx?$/u.test(fileName) ? ts.ScriptKind.TSX : ts.ScriptKind.JSX;
  const sourceFile = ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, true, kind);
  if (sourceFile.parseDiagnostics?.length) {
    const diagnostic = sourceFile.parseDiagnostics[0];
    if (!diagnostic) fail(`TypeScript ${ts.version} could not safely parse ${fileName}.`);
    const message = ts.flattenDiagnosticMessageText(diagnostic.messageText, ' ');
    fail(`TypeScript ${ts.version} could not safely parse ${fileName}: ${message}`);
  }
  return sourceFile;
}

function utf8ByteOffsetToUtf16(source: string, byteOffset: number): number {
  const buffer = Buffer.from(source, 'utf8');
  if (!Number.isInteger(byteOffset) || byteOffset < 0 || byteOffset > buffer.length) {
    fail(`Oxlint returned an invalid UTF-8 byte offset: ${byteOffset}`);
  }
  const prefix = buffer.subarray(0, byteOffset).toString('utf8');
  if (Buffer.byteLength(prefix, 'utf8') !== byteOffset) {
    fail(`Oxlint offset splits a UTF-8 code point: ${byteOffset}`);
  }
  return prefix.length;
}

function propertyNameText(name: ParserNode | undefined): string {
  if (!name) return '';
  if (ts.isIdentifier(name) || ts.isPrivateIdentifier(name)) return name.text;
  if (ts.isStringLiteralLike(name) || ts.isNumericLiteral(name)) return name.text;
  if (ts.isComputedPropertyName(name)) return `[${canonicalNode(name.expression)}]`;
  return (ts.SyntaxKind[name.kind] ?? 'unknown') as string;
}

function bindingNameText(name: ParserNode): string {
  if (ts.isIdentifier(name)) return name.text;
  if (ts.isObjectBindingPattern(name)) {
    return `{${name.elements.map((entry) => propertyNameText(entry.propertyName ?? entry.name)).join(',')}}`;
  }
  if (ts.isArrayBindingPattern(name)) {
    return `[${name.elements.map((entry) => propertyNameText(entry.name)).join(',')}]`;
  }
  return canonicalNode(name);
}

function canonicalNode(node: ParserNode, replacedNode: ParserNode | null = null): string {
  if (node === replacedNode) return '<function>';
  if (ts.isFunctionLike(node)) return '<nested-function>';
  let value = (ts.SyntaxKind[node.kind] ?? String(node.kind)) as string;
  if (ts.isIdentifier(node) || ts.isPrivateIdentifier(node)) value += `:${node.text}`;
  else if (ts.isStringLiteralLike(node) || ts.isNumericLiteral(node)) value += `:${node.text}`;
  else if (node.operatorToken) value += `:op-${ts.SyntaxKind[node.operatorToken.kind]}`;
  const children: string[] = [];
  node.forEachChild((child) => {
    children.push(canonicalNode(child, replacedNode));
  });
  return children.length ? `${value}(${children.join(';')})` : value;
}

function isFunctionLike(node: ParserNode): node is ParserFunctionLike {
  return (
    ts.isFunctionDeclaration(node) ||
    ts.isFunctionExpression(node) ||
    ts.isArrowFunction(node) ||
    ts.isMethodDeclaration(node) ||
    ts.isGetAccessorDeclaration(node) ||
    ts.isSetAccessorDeclaration(node) ||
    ts.isConstructorDeclaration(node)
  );
}

function parentFunction(node: ParserNode): ParserFunctionLike | null {
  let current = node.parent;
  while (current && !isFunctionLike(current)) current = current.parent;
  return current ?? null;
}

function nearestStatement(node: ParserNode): ParserNode {
  let current = node.parent;
  while (current && !ts.isStatement(current)) current = current.parent;
  return current ?? node;
}

function nearestCall(node: ParserNode): ParserCallLike | null {
  let current = node.parent;
  while (current && !ts.isCallExpression(current) && !ts.isNewExpression(current)) {
    if (isFunctionLike(current)) return null;
    current = current.parent;
  }
  return current ?? null;
}

function functionLocalName(node: ParserFunctionLike): string {
  if (ts.isConstructorDeclaration(node)) return 'member:constructor';
  if (node.name) {
    if (ts.isMethodDeclaration(node) || ts.isGetAccessorDeclaration(node) || ts.isSetAccessorDeclaration(node)) {
      return `member:${ts.SyntaxKind[node.kind]}:${propertyNameText(node.name)}`;
    }
    return `function:${propertyNameText(node.name)}`;
  }

  const parent = node.parent!;
  if (ts.isVariableDeclaration(parent) && parent.initializer === node) {
    return `binding:${bindingNameText(parent.name)}`;
  }
  if (ts.isPropertyAssignment(parent) && parent.initializer === node) {
    return `property:${propertyNameText(parent.name)}`;
  }
  if (
    ts.isBinaryExpression(parent) &&
    parent.right === node &&
    parent.operatorToken.kind === ts.SyntaxKind.EqualsToken
  ) {
    return `assignment:${canonicalNode(parent.left)}`;
  }
  if (ts.isExportAssignment(parent) && parent.expression === node) return 'default-export';

  const call = nearestCall(node);
  if (call) {
    const argumentIndex = call.arguments?.indexOf(node) ?? -1;
    const callee = ts.isCallExpression(call) ? canonicalNode(call.expression) : `new:${canonicalNode(call.expression)}`;
    return `callback:${callee}:arg${argumentIndex}`;
  }

  const statement = nearestStatement(node);
  return `anonymous:${ts.SyntaxKind[node.kind]}:${canonicalNode(statement, node)}`;
}

function classOwnerName(node: ParserNode): string {
  if (node.name) return propertyNameText(node.name);
  const parent = node.parent!;
  if (ts.isVariableDeclaration(parent)) return bindingNameText(parent.name);
  if (ts.isPropertyAssignment(parent)) return propertyNameText(parent.name);
  if (ts.isBinaryExpression(parent) && parent.right === node) return canonicalNode(parent.left);
  return 'anonymous';
}

function nonFunctionOwnerSegments(node: ParserNode): string[] {
  const segments: string[] = [];
  let current = node.parent;
  while (current && !isFunctionLike(current)) {
    if (ts.isClassDeclaration(current) || ts.isClassExpression(current)) {
      segments.unshift(`class:${classOwnerName(current)}`);
    } else if (ts.isVariableDeclaration(current) && current.initializer) {
      segments.unshift(`binding:${bindingNameText(current.name)}`);
    } else if (ts.isPropertyAssignment(current)) {
      segments.unshift(`property:${propertyNameText(current.name)}`);
    }
    current = current.parent;
  }
  return segments;
}

function functionIdentity(node: ParserFunctionLike): string {
  const parent = parentFunction(node);
  const outer = parent ? functionIdentity(parent) : 'module';
  const local = functionLocalName(node);
  const owners = nonFunctionOwnerSegments(node);
  if (owners.at(-1) === local) owners.pop();
  const segments = [outer, ...owners, local].filter(Boolean);
  const nameIsStructural = local.startsWith('callback:') || local.startsWith('anonymous:');
  if (nameIsStructural) {
    const statement = nearestStatement(node);
    segments.push(`site:${canonicalNode(statement, node)}`);
  }
  return segments.join('/');
}

function functionAtOffset(sourceFile: ParserSourceFile, offset: number): ParserFunctionLike | null {
  const candidates: ParserFunctionLike[] = [];
  const visit = (node: ParserNode): void => {
    if (isFunctionLike(node) && node.body) {
      const bodyStart = node.body.getStart(sourceFile);
      if (bodyStart <= offset && offset < node.body.end) candidates.push(node);
    }
    node.forEachChild(visit);
  };
  visit(sourceFile);
  if (candidates.length === 0) {
    const anyContaining: ParserFunctionLike[] = [];
    const visitAny = (node: ParserNode): void => {
      if (isFunctionLike(node) && node.getStart(sourceFile) <= offset && offset < node.end) {
        anyContaining.push(node);
      }
      node.forEachChild(visitAny);
    };
    visitAny(sourceFile);
    candidates.push(...anyContaining);
  }
  if (candidates.length === 0) return null;
  candidates.sort((left, right) => left.end - left.getStart(sourceFile) - (right.end - right.getStart(sourceFile)));
  return candidates[0]!;
}

export function diagnosticIdentity(diagnostic: OxlintDiagnostic, source: string, fileName: string): ComplexityFinding {
  const scoreMatch = /from (\d+) to the (\d+) allowed/u.exec(diagnostic.message);
  if (!scoreMatch || Number(scoreMatch[2]) !== THRESHOLD) {
    fail(`Cannot parse SonarJS complexity diagnostic: ${diagnostic.message}`);
  }
  const score = Number(scoreMatch[1]);
  if (!Number.isSafeInteger(score) || score <= THRESHOLD) {
    fail(`Oxlint reported an invalid complexity score: ${score}`);
  }
  const span = diagnostic.labels?.[0]?.span;
  if (!span || !Number.isInteger(span.offset)) fail('Oxlint diagnostic is missing a byte offset.');
  const offset = utf8ByteOffsetToUtf16(source, span.offset as number);
  const sourceFile = sourceFileFor(source, fileName);
  const node = functionAtOffset(sourceFile, offset);
  if (!node) {
    fail(`Could not map cognitive complexity diagnostic to a function: ${fileName}:${span.line}`);
  }
  return {
    file: fileName,
    identity: functionIdentity(node),
    score
  };
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function parseOxlintDiagnostics(stdout: string, batchLength: number): OxlintDiagnostic[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(stdout);
  } catch (error) {
    fail(`Oxlint returned malformed JSON: ${errorMessage(error)}`);
  }
  const report = parsed as OxlintReport;
  if (report.number_of_files !== batchLength || !Array.isArray(report.diagnostics)) {
    fail(`Oxlint scanned ${report.number_of_files} of ${batchLength} selected files.`);
  }
  // Oxlint is the pinned local tool; preserve its diagnostic objects after checking the report envelope.
  const diagnostics = report.diagnostics as InternalOxlintDiagnostic[];
  return diagnostics.filter((item) => item.code === RULE_CODE);
}

export function runOxlint(files: readonly string[]): OxlintDiagnostic[] {
  const oxlintPackage = require.resolve('oxlint/package.json');
  const oxlintEntry = path.join(path.dirname(oxlintPackage), 'bin', 'oxlint');
  const cacheDirectory = path.join(ROOT, 'node_modules', '.cache');
  fs.mkdirSync(cacheDirectory, {recursive: true});
  const configPath = path.join(cacheDirectory, `sonarjs-complexity-${process.pid}-${Date.now()}.json`);
  fs.writeFileSync(
    configPath,
    JSON.stringify({
      jsPlugins: ['eslint-plugin-sonarjs'],
      rules: {[RULE_ID]: ['warn', THRESHOLD]}
    }),
    {flag: 'wx'}
  );
  const all: OxlintDiagnostic[] = [];
  try {
    for (let start = 0; start < files.length; start += OXLINT_BATCH_SIZE) {
      const batch = files.slice(start, start + OXLINT_BATCH_SIZE);
      if (batch.length === 0) fail('Oxlint received an empty batch.');
      const result = spawnSync(
        process.execPath,
        [
          oxlintEntry,
          '--config',
          configPath,
          '--allow',
          'all',
          '--warn',
          RULE_ID,
          '--no-ignore',
          '--disable-nested-config',
          '--format',
          'json',
          ...batch
        ],
        {cwd: ROOT, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024}
      );
      if (result.error) throw result.error;
      if (result.status !== 0) {
        fail(`Oxlint failed while scanning tracked sources:\n${result.stderr || result.stdout}`);
      }
      all.push(...parseOxlintDiagnostics(result.stdout, batch.length));
    }
  } finally {
    fs.rmSync(configPath, {force: true});
  }
  return all;
}

function findingKey(entry: ComplexityFinding): string {
  return `${entry.file}\0${entry.identity}`;
}

export function validateFindings(
  findings: readonly ComplexityFinding[],
  label = 'Complexity findings'
): Map<string, number> {
  const keys = new Set();
  for (const finding of findings) {
    const file = normalizedTrackedPath(finding.file);
    if (!SOURCE_EXTENSIONS.has(path.posix.extname(file))) fail(`${label} has a non-source path: ${file}`);
    if (typeof finding.identity !== 'string' || !finding.identity.trim()) {
      fail(`${label} has an empty function identity in ${file}`);
    }
    if (typeof finding.score !== 'number' || !Number.isSafeInteger(finding.score) || finding.score <= THRESHOLD) {
      fail(`${label} has an invalid complexity score in ${file}`);
    }
    const key = findingKey(finding);
    if (keys.has(key)) fail(`${label} contains duplicate function identities: ${file} ${finding.identity}`);
    keys.add(key);
  }
  return new Map(findings.map((finding) => [findingKey(finding), finding.score]));
}

function validateBaseline(inputValue: unknown, label: string): ValidatedBaseline {
  const input = inputValue as UnvalidatedBaseline;
  if (
    !input ||
    input.schemaVersion !== 1 ||
    input.threshold !== THRESHOLD ||
    !Array.isArray(input.generatedExclusions) ||
    !Array.isArray(input.findings) ||
    !input.versions ||
    typeof input.versions.oxlint !== 'string' ||
    typeof input.versions.sonarjs !== 'string' ||
    typeof input.versions.typescriptParser !== 'string'
  ) {
    fail(`${label} has an invalid schema or threshold.`);
  }
  const generatedExclusions = input.generatedExclusions as GeneratedExclusion[];
  const findings = input.findings as ComplexityFinding[];
  const expectedExclusions = JSON.stringify(GENERATED_EXCLUSIONS);
  if (JSON.stringify(generatedExclusions) !== expectedExclusions) {
    fail(`${label} generated exclusions differ from the checked-in list.`);
  }
  const scores = validateFindings(findings, label);
  return {...(input as ComplexityBaseline), scores};
}

function versionsFromManifest(
  packageJson: unknown,
  lockText: string,
  label: string,
  expectedParserVersion: string | null = null
): ToolVersions {
  const manifest = packageJson as PackageManifest;
  const oxlintVersion = manifest.devDependencies?.oxlint;
  const sonarVersion = manifest.devDependencies?.['eslint-plugin-sonarjs'];
  if (!oxlintVersion || !sonarVersion) fail('Pinned Oxlint/SonarJS versions are missing from package.json.');
  const documents = yaml.parseAllDocuments(lockText);
  const locks = documents.map((document) => {
    if (document.errors.length) fail(`Cannot parse ${label} pnpm lockfile: ${document.errors[0]!.message}`);
    return document.toJSON() as PnpmLockValue | undefined;
  });
  const pluginPrefix = `eslint-plugin-sonarjs@${sonarVersion}`;
  const parserVersions = new Set(
    locks.flatMap((lock) =>
      Object.entries(lock?.snapshots ?? [])
        .filter(([name]) => name === pluginPrefix || name.startsWith(`${pluginPrefix}(`))
        .map(([, snapshot]) => snapshot.dependencies?.typescript)
    )
  );
  if (parserVersions.size !== 1 || (expectedParserVersion && !parserVersions.has(expectedParserVersion))) {
    fail(`Pinned TypeScript parser version does not match ${label} pnpm lockfile.`);
  }
  const typescriptParser = [...parserVersions][0];
  if (typeof typescriptParser !== 'string')
    fail(`Pinned TypeScript parser version is invalid in ${label} pnpm lockfile.`);
  return {oxlint: oxlintVersion, sonarjs: sonarVersion, typescriptParser};
}

function versionsFromPackage(): ToolVersions {
  const packageJson: unknown = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
  const lockText = fs.readFileSync(path.join(ROOT, 'pnpm-lock.yaml'), 'utf8');
  return versionsFromManifest(packageJson, lockText, 'current', ts.version);
}

function versionsAtCommit(commit: string): ToolVersions {
  const packageResult = git(['show', `${commit}:package.json`]);
  const lockResult = git(['show', `${commit}:pnpm-lock.yaml`]);
  if (packageResult.status !== 0 || lockResult.status !== 0) {
    fail(`Cannot read trusted package tool versions at ${commit}.`);
  }
  const packageJson = loadJsonBytes(packageResult.stdout, 'trusted package.json');
  return versionsFromManifest(packageJson, lockResult.stdout.toString('utf8'), `trusted ${commit}`);
}

function compareVersions(left: string, right: string): number {
  const leftParts = left.split('.').map(Number);
  const rightParts = right.split('.').map(Number);
  if (
    leftParts.length !== 3 ||
    rightParts.length !== 3 ||
    [...leftParts, ...rightParts].some((part) => !Number.isSafeInteger(part) || part < 0)
  ) {
    fail(`Unsupported pinned tool version format: ${left} / ${right}`);
  }
  for (let index = 0; index < 3; index += 1) {
    if (leftParts[index]! !== rightParts[index]!) return Math.sign(leftParts[index]! - rightParts[index]!);
  }
  return 0;
}

function ensureNoToolDowngrades(candidate: ToolVersions, trusted: ToolVersions): void {
  for (const name of ['oxlint', 'sonarjs', 'typescriptParser'] as const) {
    if (compareVersions(candidate[name], trusted[name]) < 0) {
      fail(`Cognitive complexity tool downgrade is not allowed: ${name} ${trusted[name]} -> ${candidate[name]}.`);
    }
  }
}

function sameToolVersions(left: ToolVersions, right: ToolVersions): boolean {
  return (['oxlint', 'sonarjs', 'typescriptParser'] as const).every((name) => left[name] === right[name]);
}

export function actualBaseline(findings: readonly ComplexityFinding[]): ComplexityBaseline {
  const compareText = (left: string, right: string): number => (left < right ? -1 : left > right ? 1 : 0);
  const sorted = [...findings].sort((left, right) => {
    const fileOrder = compareText(left.file, right.file);
    return fileOrder || compareText(left.identity, right.identity);
  });
  return {
    schemaVersion: 1,
    threshold: THRESHOLD,
    versions: versionsFromPackage(),
    generatedExclusions: GENERATED_EXCLUSIONS,
    findings: sorted
  };
}

function loadJsonBytes(buffer: Buffer, source: string): unknown {
  let value: unknown;
  try {
    value = JSON.parse(buffer.toString('utf8'));
  } catch (error) {
    fail(`Cannot parse ${source}: ${errorMessage(error)}`);
  }
  return value;
}

function trustedBaseline(baseSha: string): {readonly sha: string; readonly baseline: ValidatedBaseline | null} {
  if (!/^[0-9a-f]{40}$/u.test(baseSha)) fail('Missing or invalid trusted base SHA.');
  const exists = git(['cat-file', '-e', `${baseSha}^{commit}`]);
  if (exists.status !== 0) fail(`Trusted base commit is unavailable: ${baseSha}`);
  const blob = git(['show', `${baseSha}:${BASELINE_PATH}`]);
  if (blob.status === 0) {
    return {
      sha: baseSha,
      baseline: validateBaseline(loadJsonBytes(blob.stdout, 'trusted complexity baseline'), 'Trusted baseline')
    };
  }

  const changes = git(['diff', '--name-status', '--no-renames', baseSha, 'HEAD', '--', BASELINE_PATH]);
  if (changes.status !== 0 || changes.stdout.toString('utf8').trim() !== `A\t${BASELINE_PATH}`) {
    fail('Trusted cognitive complexity baseline is unavailable.');
  }
  return {sha: baseSha, baseline: null};
}

function compareToTrusted(candidate: ValidatedBaseline, trusted: ValidatedBaseline | null): void {
  if (!trusted) return;
  const current = candidate.scores;
  for (const [key, score] of current) {
    const prior = trusted.scores.get(key);
    if (prior === undefined) fail(`New SonarJS cognitive complexity violation: ${key.replace('\0', ' ')} (${score}).`);
    if (score > prior) fail(`SonarJS cognitive complexity increased: ${key.replace('\0', ' ')} ${prior} -> ${score}.`);
  }
  for (const [key, score] of candidate.scores) {
    const prior = trusted.scores.get(key);
    if (prior === undefined || score > prior) {
      fail(`Cognitive complexity baseline was loosened: ${key.replace('\0', ' ')} (${score}).`);
    }
  }
}

function sourceTreeAtCommit(commit: string): SourceBlobEntry[] {
  const result = git(['ls-tree', '-r', '-z', '--full-tree', commit]);
  if (result.status !== 0) fail(`Cannot list tracked sources at trusted base ${commit}.`);
  const entries = new Map<string, TreeEntry>();
  for (const record of result.stdout.toString('utf8').split('\0')) {
    if (!record) continue;
    const tab = record.indexOf('\t');
    if (tab < 0) fail('git ls-tree returned an invalid record.');
    const [mode, type, oid] = record.slice(0, tab).split(' ');
    if (!mode || !type || !oid) fail('git ls-tree returned an invalid record.');
    const name = normalizedTrackedPath(record.slice(tab + 1));
    entries.set(name, {mode, type, oid});
  }
  const names = selectTrackedSources([...entries.keys()]);
  for (const name of names) {
    const entry = entries.get(name);
    if (!entry || entry.type !== 'blob' || entry.mode === '120000') {
      fail(`Trusted source is not a regular file: ${name}`);
    }
  }
  return names.map((name) => {
    const entry = entries.get(name);
    if (!entry) fail(`Trusted source is missing from the Git tree: ${name}`);
    return {path: name, oid: entry.oid};
  });
}

function readBlobs(entries: readonly SourceBlobEntry[]): Map<string, Buffer> {
  if (entries.length === 0) fail('No tracked sources exist at the trusted base.');
  const output = git(['cat-file', '--batch'], {
    input: Buffer.from(entries.map(({oid}) => oid).join('\n') + '\n', 'utf8')
  });
  if (output.status !== 0) fail(`Cannot read trusted source blobs: ${output.stderr.toString('utf8')}`);
  const blobs = new Map<string, Buffer>();
  let cursor = 0;
  for (const entry of entries) {
    const headerEnd = output.stdout.indexOf(0x0a, cursor);
    if (headerEnd < 0) fail('git cat-file returned an incomplete object header.');
    const header = output.stdout.subarray(cursor, headerEnd).toString('ascii').split(' ');
    if (header.length !== 3 || header[0] !== entry.oid || header[1] !== 'blob') {
      fail(`git cat-file returned an unexpected object for ${entry.path}.`);
    }
    const size = Number(header[2]);
    const start = headerEnd + 1;
    const end = start + size;
    if (!Number.isSafeInteger(size) || size < 0 || end >= output.stdout.length || output.stdout[end] !== 0x0a) {
      fail(`git cat-file returned an invalid blob size for ${entry.path}.`);
    }
    blobs.set(entry.path, output.stdout.subarray(start, end));
    cursor = end + 1;
  }
  if (cursor !== output.stdout.length) fail('git cat-file returned unexpected trailing bytes.');
  return blobs;
}

export function collectFindings(files: readonly string[], sourcePrefix = ''): ComplexityFinding[] {
  const lintFiles = files.map((file) => (sourcePrefix ? `${sourcePrefix}/${file}` : file));
  const diagnostics = runOxlint(lintFiles);
  const lintToSource = new Map(lintFiles.map((lintFile, index) => [lintFile, files[index]]));
  const findings = diagnostics.map((diagnostic) => {
    const lintFile = normalizedTrackedPath(diagnostic.filename.replaceAll('\\', '/'));
    const file = lintToSource.get(lintFile);
    if (!file) fail(`Oxlint reported a file outside the tracked input set: ${lintFile}`);
    const source = verifySourceFile(lintFile);
    return diagnosticIdentity(diagnostic, source, file);
  });
  validateFindings(findings, 'Current Oxlint report');
  return findings;
}

export function scanTrustedCommit(commit: string): ComplexityFinding[] {
  const entries = sourceTreeAtCommit(commit);
  const blobs = readBlobs(entries);
  const cacheDirectory = path.join(ROOT, 'node_modules', '.cache');
  fs.mkdirSync(cacheDirectory, {recursive: true});
  const temporaryRoot = fs.mkdtempSync(path.join(cacheDirectory, 'sonarjs-base-'));
  const sourcePrefix = path.relative(ROOT, temporaryRoot).replaceAll('\\', '/');
  try {
    for (const {path: relativePath} of entries) {
      const destination = path.join(temporaryRoot, ...relativePath.split('/'));
      fs.mkdirSync(path.dirname(destination), {recursive: true});
      fs.writeFileSync(destination, blobs.get(relativePath)!);
    }
    const baseFiles = entries.map(({path: name}) => name);
    for (const file of baseFiles) {
      const source = verifySourceFile(`${sourcePrefix}/${file}`);
      const violations = findComplexitySuppressions(source);
      if (violations.length) fail(`Trusted base contains a cognitive complexity suppression in ${file}.`);
    }
    return collectFindings(baseFiles, sourcePrefix);
  } finally {
    fs.rmSync(temporaryRoot, {recursive: true, force: true});
  }
}

export function compareFindingsToTrusted(
  currentFindings: readonly ComplexityFinding[],
  trustedFindings: readonly ComplexityFinding[]
): void {
  const current = validateFindings(currentFindings, 'Current SonarJS findings');
  const trusted = validateFindings(trustedFindings, 'Trusted SonarJS findings');
  for (const [key, score] of current) {
    const previous = trusted.get(key);
    if (previous === undefined)
      fail(`New SonarJS cognitive complexity violation: ${key.replace('\0', ' ')} (${score}).`);
    if (score > previous)
      fail(`SonarJS cognitive complexity increased: ${key.replace('\0', ' ')} ${previous} -> ${score}.`);
  }
}

function verifyCurrentMatchesBaseline(current: ComplexityBaseline, reviewed: ValidatedBaseline): void {
  const reviewedSnapshot = {
    schemaVersion: reviewed.schemaVersion,
    threshold: reviewed.threshold,
    versions: reviewed.versions,
    generatedExclusions: reviewed.generatedExclusions,
    findings: reviewed.findings
  };
  if (JSON.stringify(current) === JSON.stringify(reviewedSnapshot)) return;

  const expected = reviewed.scores;
  const actual = validateFindings(current.findings, 'Current Oxlint report');
  for (const [key, score] of actual) {
    if (!expected.has(key))
      fail(`Unreviewed SonarJS cognitive complexity violation: ${key.replace('\0', ' ')} (${score}).`);
    if (score !== expected.get(key))
      fail(`Reviewed complexity score differs: ${key.replace('\0', ' ')} ${expected.get(key)} -> ${score}.`);
  }
  for (const key of expected.keys()) {
    if (!actual.has(key)) fail(`Stale complexity baseline entry: ${key.replace('\0', ' ')}.`);
  }
  fail('Reviewed cognitive complexity baseline does not match the current scan.');
}

function verifyAgainstTrustedBase(
  baseSha: string,
  reviewed: ValidatedBaseline,
  findings: readonly ComplexityFinding[]
): void {
  const trusted = trustedBaseline(baseSha);
  const previousVersions = trusted.baseline?.versions ?? versionsAtCommit(trusted.sha);
  ensureNoToolDowngrades(reviewed.versions, previousVersions);
  if (
    trusted.baseline &&
    JSON.stringify(reviewed.generatedExclusions) !== JSON.stringify(trusted.baseline.generatedExclusions)
  ) {
    fail('Cognitive complexity generated exclusions changed from the trusted baseline.');
  }
  if (trusted.baseline && sameToolVersions(reviewed.versions, trusted.baseline.versions)) {
    checkBaseline(reviewed, trusted.baseline);
    return;
  }
  compareFindingsToTrusted(findings, scanTrustedCommit(trusted.sha));
}

function checkSuppressions(files: readonly string[]): void {
  for (const file of files) {
    const source = verifySourceFile(file);
    const violations = findComplexitySuppressions(source);
    if (violations.length) {
      const violation = violations[0]!;
      const line = source.slice(0, violation.offset).split('\n').length;
      fail(`${file}:${line} uses a cognitive complexity suppression: ${violation.directive}`);
    }
  }
}

export function checkBaseline(
  candidateInput: ComplexityBaseline,
  trustedInput: ComplexityBaseline | null = null
): true {
  const candidate = validateBaseline(candidateInput, 'Candidate baseline');
  const packageVersions = versionsFromPackage();
  if (
    candidate.versions.oxlint !== packageVersions.oxlint ||
    candidate.versions.sonarjs !== packageVersions.sonarjs ||
    candidate.versions.typescriptParser !== packageVersions.typescriptParser
  ) {
    fail('Candidate baseline tool versions do not match package.json.');
  }
  const trusted = trustedInput ? validateBaseline(trustedInput, 'Trusted baseline') : null;
  if (trusted && !sameToolVersions(candidate.versions, trusted.versions)) {
    fail('Candidate and trusted baseline tool versions differ.');
  }
  compareToTrusted(candidate, trusted);
  return true;
}

function main(): void {
  const files = workingTreeSources();
  for (const file of files) verifySourceFile(file);
  checkSuppressions(files);

  const findings = collectFindings(files);
  const current = actualBaseline(findings);
  const reviewed = validateBaseline(loadJsonBytes(fs.readFileSync(BASELINE), BASELINE_PATH), 'Checked-in baseline');
  verifyCurrentMatchesBaseline(current, reviewed);

  const baseSha = process.env.BASE_SHA || '';
  if (process.env.GITHUB_ACTIONS === 'true' && !baseSha) {
    fail('BASE_SHA is required in GitHub Actions for trusted-baseline comparison.');
  }
  if (baseSha) verifyAgainstTrustedBase(baseSha, reviewed, findings);
  process.stdout.write(
    `SonarJS cognitive complexity ratchet passed: ${files.length} selected source files, ${findings.length} existing findings, threshold ${THRESHOLD}.\n`
  );
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`${errorMessage(error)}\n`);
    process.exitCode = 1;
  }
}
