import {execFileSync} from 'node:child_process';
import {mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const packageDirectory = 'packages/web-decision';
const jsPath = `${packageDirectory}/src/runtime/tvm.generated.mjs`;
const declarationPath = `${packageDirectory}/src/runtime/tvm.generated.d.mts`;
const marker = '?web-decision-runtime';
const header = '// Generated from src/runtime/tvm.ts; do not edit.\n';

type GeneratedBoundary = {readonly javascript: string; readonly declaration: string};

function formatExports(source: string): string {
  return source.replace(
    /export (type )?\{ ([^{}]+) \} from /gu,
    (_match, type: string | undefined, names: string) => `export ${type ?? ''}{${names}} from `
  );
}

/** Keep this generated bridge limited to static re-exports of the pinned TVM modules. */
export function validateTvmEmission(source: string): void {
  const statements = formatExports(source)
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('//'));
  const expected = [
    "export {instantiate} from '../../../../.toolchains/apache-tvm-src/web/src/runtime.ts?web-decision-runtime';",
    "export {createPolyfillWASI} from '../../../../.toolchains/apache-tvm-src/web/src/compact.ts?web-decision-runtime';"
  ];
  if (statements.length !== expected.length || statements.some((line, index) => line !== expected[index])) {
    throw new Error('TVM boundary must contain only the two pinned static re-exports.');
  }
}

/** Compile the checked TS boundary and remove only its two type-resolution markers. */
export function generateTvmBoundary(root: string): GeneratedBoundary {
  const require = createRequire(join(root, 'package.json'));
  const compiler = join(dirname(require.resolve('typescript/package.json')), 'bin/tsc');
  const temporary = mkdtempSync(join(tmpdir(), 'web-decision-tvm-boundary-'));
  try {
    execFileSync(
      process.execPath,
      [
        compiler,
        '--project',
        join(root, packageDirectory, 'tsconfig.tvm-boundary.json'),
        '--noEmit',
        'false',
        '--outDir',
        temporary
      ],
      {cwd: root, stdio: 'pipe'}
    );
    const emitted = readFileSync(join(temporary, 'tvm.js'), 'utf8');
    const declaration = readFileSync(join(temporary, 'tvm.d.ts'), 'utf8');
    validateTvmEmission(emitted);
    if (emitted.split(marker).length !== 3 || declaration.split(marker).length !== 3)
      throw new Error('Expected exactly two TVM type markers in each compiler output.');
    return {
      javascript: header + formatExports(emitted.replaceAll(marker, '')),
      declaration: header + formatExports(declaration)
    };
  } finally {
    rmSync(temporary, {recursive: true, force: true});
  }
}

export function verifyTvmBoundary(root: string, generated: GeneratedBoundary): string[] {
  const files: readonly (readonly [string, string])[] = [
    [jsPath, generated.javascript],
    [declarationPath, generated.declaration]
  ];
  return files.flatMap(([path, expected]) => {
    try {
      return readFileSync(join(root, path), 'utf8') === expected
        ? []
        : [`${path} differs from its generated TypeScript output.`];
    } catch {
      return [`${path} is missing.`];
    }
  });
}

function main(): number {
  const root = fileURLToPath(new URL('../../', import.meta.url));
  const generated = generateTvmBoundary(root);
  if (process.argv[2] === '--write' && process.argv.length === 3) {
    writeFileSync(join(root, jsPath), generated.javascript);
    writeFileSync(join(root, declarationPath), generated.declaration);
    return 0;
  }
  if (process.argv.length !== 2) throw new Error('Use --write to regenerate or no arguments to check.');
  const problems = verifyTvmBoundary(root, generated);
  for (const problem of problems) console.error(problem);
  return problems.length ? 1 : 0;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) process.exitCode = main();
