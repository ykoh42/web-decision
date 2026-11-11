import {execFileSync} from 'node:child_process';
import {copyFileSync, writeFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {APPROVED_MODEL_IDS} from '../../packages/web-decision/src/catalog/public-catalog.generated.ts';

const root = fileURLToPath(new URL('../../', import.meta.url));
const require = createRequire(import.meta.url);
const compiler = join(dirname(require.resolve('typescript/package.json')), 'bin/tsc');
const fixture = join(root, 'tests/types');

/** Type-check the packed package where a consumer has installed it. */
export function checkConsumerTypes(consumer: string, typescriptCompiler = compiler): void {
  for (const name of [
    'public-api-consumer.ts',
    'tsconfig.base.json',
    'tsconfig.bundler.json',
    'tsconfig.nodenext.json'
  ]) {
    copyFileSync(join(fixture, name), join(consumer, name));
  }
  for (const [name, modelId] of [
    ['consumer.ts', 'kev-0.6b-q4f16'],
    ['kanana-consumer.ts', 'kanana-2-v-decide-3b-2611']
  ] as const) {
    const destination = join(consumer, name);
    if (APPROVED_MODEL_IDS.includes(modelId as (typeof APPROVED_MODEL_IDS)[number]))
      copyFileSync(join(fixture, name), destination);
    else writeFileSync(destination, 'export {};\n');
  }
  const pendingModelIds = ['kev-0.6b-q4f16', 'kanana-2-v-decide-3b-2611'].filter(
    (modelId) => !APPROVED_MODEL_IDS.includes(modelId as (typeof APPROVED_MODEL_IDS)[number])
  );
  const pendingChecks = pendingModelIds.map(
    (modelId, index) =>
      '// @ts-expect-error ' +
      modelId +
      ' is not approved for the public package.\n' +
      'const unavailableModelId' +
      index +
      ': ModelId = ' +
      JSON.stringify(modelId) +
      ';'
  );
  writeFileSync(
    join(consumer, 'pending-model-ids.ts'),
    ["import type {ModelId} from 'web-decision';", ...pendingChecks, 'export {};', ''].join('\n')
  );
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
