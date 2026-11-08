import {execFileSync} from 'node:child_process';
import {copyFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = fileURLToPath(new URL('../../', import.meta.url));
const require = createRequire(import.meta.url);
const compiler = join(dirname(require.resolve('typescript/package.json')), 'bin/tsc');
const fixture = join(root, 'tests/types');

/** Type-check the packed package where a consumer has installed it. */
export function checkConsumerTypes(consumer: string, typescriptCompiler = compiler): void {
  for (const name of ['consumer.ts', 'tsconfig.base.json', 'tsconfig.bundler.json', 'tsconfig.nodenext.json']) {
    copyFileSync(join(fixture, name), join(consumer, name));
  }
  for (const project of ['tsconfig.bundler.json', 'tsconfig.nodenext.json']) {
    for (const exactOptionalPropertyTypes of ['false', 'true']) {
      execFileSync(
        process.execPath,
        [typescriptCompiler, '--project', project, '--exactOptionalPropertyTypes', exactOptionalPropertyTypes],
        {
          cwd: consumer,
          stdio: 'inherit'
        }
      );
    }
  }
}
