import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdtemp, mkdir, readFile, realpath, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import test from 'node:test';
import artifactLock from '../../packages/web-decision/src/catalog/artifacts.lock.json' with {type: 'json'};
import {sha256} from './curated-gpu-evidence.mjs';
import {
  assertPromotionReceipt,
  assertPromotionSource,
  committedReceipt,
  promotion,
  reportPath,
  requiredReports,
  verifyPromotionArchive
} from './verify-gemma4-v1-promotion.mjs';

function cpuReceipt() {
  return {
    ...structuredClone(promotion),
    schemaVersion: 1,
    status: 'locally-verified-exact-package',
    kananaEvidenceCommit: 'd'.repeat(40),
    localVerificationSha256: sha256(Buffer.from('synthetic CPU gate fixture')),
    models: Object.fromEntries(
      Object.entries(requiredReports).map(([id, scopes]) => {
        const {files} = artifactLock.byModelId[id as keyof typeof requiredReports].manifest;
        return [
          id,
          {
            reports: Object.fromEntries(scopes.map((scope) => [scope, sha256(Buffer.from(id + '/' + scope))])),
            native: Object.fromEntries(
              Object.entries(files)
                .filter(([name]) => name.endsWith('.wasm'))
                .map(([name, {sha256: wasmSha256}]) => [
                  name,
                  {wasmSha256, reportSha256: sha256(Buffer.from(id + '/' + name))}
                ])
            )
          }
        ];
      })
    )
  };
}

test('every public model and its named reports and native builds are required', () => {
  assert.doesNotThrow(() => assertPromotionReceipt(cpuReceipt()));
  for (const [id, reportNames] of Object.entries(requiredReports)) {
    const missing = cpuReceipt();
    delete missing.models[id];
    assert.throws(() => assertPromotionReceipt(missing), /Every published model requires exact evidence/u);
    for (const scope of reportNames) {
      const receipt = cpuReceipt();
      delete receipt.models[id]!.reports[scope];
      assert.throws(() => assertPromotionReceipt(receipt), /Missing GPU report pins/u);
    }
    for (const artifact of Object.keys(cpuReceipt().models[id]!.native)) {
      const noBuild = cpuReceipt();
      delete noBuild.models[id]!.native[artifact];
      assert.throws(() => assertPromotionReceipt(noBuild), /Missing independent native builds/u);
      const wrong = cpuReceipt();
      wrong.models[id]!.native[artifact]!.wasmSha256 = 'a'.repeat(64);
      assert.throws(() => assertPromotionReceipt(wrong), /differs from the committed artifact/u);
    }
  }
});

test('receipt verification rejects another source, another installation and preflight', () => {
  const evidence = cpuReceipt();
  assert.throws(() => assertPromotionReceipt({...evidence, sourceCommit: 'b'.repeat(40)}), /another tested source/u);
  assert.throws(
    () => assertPromotionReceipt({...evidence, archive: {...evidence.archive, sha256: 'b'.repeat(64)}}),
    /another package/u
  );
  assert.throws(() => assertPromotionReceipt({...evidence, status: 'inputs-verified-no-inference'}));
});

test('the exact package guard refuses other archive bytes', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'web-decision-promotion-archive-'));
  t.after(() => rm(root, {recursive: true}));
  const archive = join(root, 'web-decision-0.8.0.tgz');
  await writeFile(archive, Buffer.from('synthetic different archive'));
  await assert.rejects(verifyPromotionArchive(archive), /differs from the exact GPU-tested archive/u);
});

test('CI reads the reviewed Git blob and rejects absence, replacement and product changes', async (t) => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'web-decision-promotion-receipt-')));
  t.after(() => rm(root, {recursive: true}));
  const git = (...args: string[]) =>
    execFileSync('git', args, {cwd: root, stdio: ['ignore', 'pipe', 'pipe'], encoding: 'utf8'});
  git('init', '--quiet');
  git('config', 'user.name', 'CPU gate test');
  git('config', 'user.email', 'gate@example.invalid');
  git('commit', '--allow-empty', '-qm', 'Tested product source');
  const source = git('rev-parse', 'HEAD').trim();
  assert.equal(assertPromotionSource(root, source), source);

  await assert.rejects(committedReceipt(root), {code: 'ENOENT'});
  const path = join(root, reportPath);
  await mkdir(dirname(path), {recursive: true});
  const bytes = JSON.stringify(cpuReceipt());
  await writeFile(path, bytes);
  await assert.rejects(committedReceipt(root));
  git('add', reportPath);
  git('commit', '-qm', 'Review synthetic receipt');
  assert.equal(assertPromotionSource(root, source), git('rev-parse', 'HEAD').trim());
  assert.equal((await committedReceipt(root)).toString(), bytes);

  await writeFile(path, bytes.replace('"browserMatrix":"', '"browserMatrix":"a'));
  await assert.rejects(committedReceipt(root), /differs from the reviewed Git blob/u);
  await writeFile(path, bytes);

  const product = 'runtime.js';
  await writeFile(join(root, product), 'export {}');
  git('add', product);
  git('commit', '-qm', 'Change distributed entry');
  assert.throws(() => assertPromotionSource(root, source), /outside release verification/u);
  assert.equal(await readFile(path, 'utf8'), bytes);
});

test('the release workflow requires receipt verification after packing', async () => {
  const workflow = await readFile(new URL('../../.github/workflows/release-checks.yml', import.meta.url), 'utf8');
  const marker = '- name: Verify the reviewed exact-archive GPU evidence';
  assert.equal(workflow.split(marker).length, 2);
  const step = workflow.split(marker)[1]!.split('- name:')[0]!;
  assert.match(step, /if: steps\.release\.outputs\.version == '0\.8\.0'/u);
  assert.match(
    step,
    /run: node --import tsx scripts\/release\/verify-gemma4-v1-promotion\.mts "\$GITHUB_WORKSPACE\/web-decision-0\.8\.0\.tgz"/u
  );
  assert.ok(workflow.indexOf(marker) > workflow.indexOf('- name: Verify packed release archive'));
});
