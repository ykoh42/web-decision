import assert from 'node:assert/strict';
import {mkdirSync, mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import {test} from 'node:test';
import {fileURLToPath} from 'node:url';
import {generateTvmBoundary, validateTvmEmission, verifyTvmBoundary} from './generate-tvm-boundary.mts';

test('the generated TVM boundary matches checked TypeScript and detects drift', (context) => {
  const repository = fileURLToPath(new URL('../../', import.meta.url));
  const generated = generateTvmBoundary(repository);
  assert.deepEqual(verifyTvmBoundary(repository, generated), []);
  assert.doesNotMatch(generated.javascript, /\?web-decision-runtime/u);
  assert.equal(generated.declaration.split('?web-decision-runtime').length, 3);

  const root = mkdtempSync(join(tmpdir(), 'web-decision-tvm-boundary-test-'));
  context.after(() => rmSync(root, {recursive: true, force: true}));
  const write = (path: string, text: string) => {
    const destination = join(root, 'packages/web-decision/src/runtime', path);
    mkdirSync(dirname(destination), {recursive: true});
    writeFileSync(destination, text);
  };
  write('tvm.generated.mjs', generated.javascript);
  write('tvm.generated.d.mts', generated.declaration);
  assert.deepEqual(verifyTvmBoundary(root, generated), []);
  write('tvm.generated.mjs', `${generated.javascript}\n`);
  assert.ok(verifyTvmBoundary(root, generated).some((message) => message.includes('tvm.generated.mjs')));
  write('tvm.generated.d.mts', '');
  assert.ok(verifyTvmBoundary(root, generated).some((message) => message.includes('tvm.generated.d.mts')));
});

test('the TVM boundary rejects extra executable code', () => {
  const repository = fileURLToPath(new URL('../../', import.meta.url));
  const generated = generateTvmBoundary(repository);
  const emitted = generated.javascript.replaceAll(".ts';", ".ts?web-decision-runtime';");
  assert.doesNotThrow(() => validateTvmEmission(emitted));
  assert.throws(() => validateTvmEmission(`${emitted}console.log('unexpected');\n`), /two pinned static re-exports/u);
  assert.throws(
    () => validateTvmEmission(emitted.replace('runtime.ts?', 'other.ts?')),
    /two pinned static re-exports/u
  );
});
