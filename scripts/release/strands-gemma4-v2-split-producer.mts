import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile, realpath, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {join, resolve} from 'node:path';
import {sha256} from './curated-gpu-evidence.mjs';
import {gitEnvironmentFor} from '../git-worktree-environment.mjs';
import {
  derivePersistentSpec,
  deriveWarmSpec,
  persistentConfig,
  warmConfig
} from './strands-gemma4-v2-persistent-variant.mjs';
import {persistentMode} from './strands-gemma4-v2-persistent-trace.mjs';
import {corpusTitle, gemmaSplitCandidate} from './strands-gemma4-v2-split-candidate.mjs';

export {corpusTitle};

export const workspace = fileURLToPath(new URL('../../', import.meta.url));
export const scratch = process.env.WEB_DECISION_SPLIT_PRODUCER_DIR ?? '';
export const sourceCommit = process.env.WEB_DECISION_SPLIT_TESTED_COMMIT ?? '';
export const originalPath = 'tests/browser/gemma-v2-public-real-webgpu.spec.ts';
export const producerPath = 'split-public-archive.spec.ts';

export function deriveSplitSpec(source: string) {
  let split = source
    .replaceAll("'../../", "'./checkout/")
    .replaceAll(
      /from '(\.\/gemma-v2-|\.\/kev06-|\.\/mapika2b-)/gu,
      (_, match: string) => "from './checkout/tests/browser/" + match.slice(2)
    );
  function replaceOnce(original: string, replacement: string) {
    assert.equal(split.split(original).length, 2, 'Expected one committed producer marker: ' + original);
    split = split.replace(original, replacement);
  }
  assert.ok(split.includes("new URL('./checkout/', import.meta.url)"));
  replaceOnce('  gemmaV2PublicTitle,\n', '');
  replaceOnce('test(gemmaV2PublicTitle,', 'test(' + JSON.stringify(corpusTitle) + ',');
  replaceOnce(
    '  const references = validateGemmaV2CpuReferences(cpu);',
    "  const references = validateGemmaV2CpuReferences(cpu);\n  delete references['text-32768'];\n" +
      '  const producerPath = fileURLToPath(import.meta.url);\n' +
      '  const producerSha256 = sha256(await readFile(producerPath));\n' +
      "  expect(producerSha256).toBe(requiredEnv('WEB_DECISION_GEMMA_V2_SPLIT_PRODUCER_SHA256'));"
  );
  replaceOnce(
    '    const transport = exactPage?.evidence();',
    '    expect(sha256(await readFile(producerPath))).toBe(producerSha256);\n' +
      '    const transport = exactPage?.evidence();'
  );
  replaceOnce(
    '      schemaVersion: 1,',
    "      schemaVersion: 1,\n      caseMode: 'five-cases-external-32k-required',\n      producerSha256,"
  );
  return split;
}

function committedSource() {
  return execFileSync(
    'git',
    ['--no-optional-locks', '--no-replace-objects', 'show', sourceCommit + ':' + originalPath],
    {
      cwd: workspace,
      env: gitEnvironmentFor(workspace)
    }
  );
}
function pins(source: Buffer, derived: Buffer) {
  return {
    sourceCommit,
    originalSourcePath: originalPath,
    originalSourceSha256: sha256(source),
    producerPath,
    producerSha256: sha256(derived)
  };
}
export function assertDerivedSpec(source: Buffer, actual: Buffer, provided: unknown) {
  const expected = Buffer.from(deriveSplitSpec(source.toString('utf8')));
  assert.ok(actual.equals(expected), 'Derived producer differs from the exact approved transformation.');
  const derived = pins(source, expected);
  assert.deepEqual(provided, derived);
  return derived;
}

export async function inspectSplitProducer() {
  assert.ok(
    scratch && /^[a-f0-9]{40}$/u.test(sourceCommit),
    'Split producer directory and tested commit are required.'
  );
  assert.equal(sourceCommit, gemmaSplitCandidate.sourceCommit);
  assert.equal(
    process.env.WEB_DECISION_SPLIT_PERSISTENT_REUSE_FAILED_PROFILE,
    undefined,
    'The previous archive profile receipt does not establish e729 profile provenance.'
  );
  const source = committedSource();
  assert.deepEqual(source, await readFile(join(workspace, originalPath)));
  const mode = persistentMode();
  if (mode) {
    const warm = mode === 'warm';
    if (warm)
      assert.equal(
        await realpath(join(scratch, 'verifier')),
        await realpath(workspace),
        'Warm runtime checkout differs.'
      );
    const persistentPath = warm ? 'persistent-warm-archive.spec.ts' : 'persistent-public-archive.spec.ts';
    const [file, metadata, config, runtime, derivation, pinnedRuntime, pinnedDerivation] = await Promise.all([
      readFile(join(scratch, persistentPath)),
      readFile(join(scratch, 'producer-pin.json'), 'utf8'),
      readFile(join(scratch, 'playwright.config.mts'), 'utf8'),
      readFile(join(scratch, 'local-checkpoint.mts')),
      readFile(join(scratch, 'persistent-variant.mts')),
      readFile(new URL('./strands-gemma4-v2-local-checkpoint.mts', import.meta.url)),
      readFile(new URL('./strands-gemma4-v2-persistent-variant.mts', import.meta.url))
    ]);
    const base = deriveSplitSpec(source.toString('utf8'));
    const expected = warm ? deriveWarmSpec(base) : derivePersistentSpec(base);
    assert.ok(file.equals(Buffer.from(expected)), 'The persistent producer differs from the allowed derivation.');
    assert.equal(config, warm ? warmConfig : persistentConfig, 'The persistent Playwright configuration differs.');
    assert.ok(runtime.equals(pinnedRuntime), 'The persistent checkpoint runtime differs.');
    assert.ok(derivation.equals(pinnedDerivation), 'The persistent derivation differs.');
    const warmRuntime = warm
      ? {warmRuntimeSha256: sha256(await readFile(new URL('./strands-gemma4-v2-warm-seed.mts', import.meta.url)))}
      : {};
    const pin = {
      sourceCommit,
      originalSourceSha256: sha256(source),
      standardProducerSha256: sha256(Buffer.from(base)),
      producerPath: persistentPath,
      producerSha256: sha256(file),
      configSha256: sha256(Buffer.from(config)),
      checkpointRuntimeSha256: sha256(runtime),
      derivationSha256: sha256(derivation),
      ...warmRuntime
    };
    const provided = JSON.parse(metadata) as Record<string, unknown>;
    for (const [key, value] of Object.entries(pin))
      assert.equal(provided[key], value, 'Persistent producer pin differs: ' + key);
    return {
      ...pin,
      originalSourcePath: originalPath,
      playwrightConfigSha256: pin.configSha256,
      persistentCheckpointMode: mode
    };
  }
  const [file, metadata, config] = await Promise.all([
    readFile(join(scratch, producerPath)),
    readFile(join(scratch, 'split-producer-pin.json')),
    readFile(join(scratch, 'playwright.config.mts'))
  ]);
  return {
    ...assertDerivedSpec(source, file, JSON.parse(metadata.toString('utf8'))),
    playwrightConfigSha256: sha256(config)
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const source = committedSource();
  const derived = Buffer.from(deriveSplitSpec(source.toString('utf8')));
  await writeFile(join(scratch, producerPath), derived);
  await writeFile(join(scratch, 'split-producer-pin.json'), JSON.stringify(pins(source, derived), null, 2) + '\n');
  console.log(JSON.stringify(await inspectSplitProducer(), null, 2));
}
