import assert from 'node:assert/strict';
import {mkdtemp, readFile, readdir, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import test from 'node:test';
import {KANANA_FILES} from '../src/kanana/contract.mjs';
import {prepareKananaTokenizerTestAssets} from '../../../scripts/kanana/prepare-tokenizer-test-assets.mts';

test('prepares only pinned Kanana source assets and exports their directory', async () => {
  const tempDirectory = await mkdtemp(join(tmpdir(), 'kanana-test-support-'));
  try {
    const envFile = join(tempDirectory, 'github-env');
    await writeFile(envFile, 'EXISTING_VALUE=kept\n');
    const reads: string[] = [];
    const openSource = async (options: {cacheEnabled: boolean}) => {
      assert.deepEqual(options, {cacheEnabled: false});
      return {
        async read(name: keyof typeof KANANA_FILES) {
          reads.push(name);
          return new TextEncoder().encode(`verified:${name}`).buffer;
        }
      };
    };

    const snapshotDirectory = await prepareKananaTokenizerTestAssets({tempDirectory, envFile, openSource});
    assert.deepEqual(reads, Object.keys(KANANA_FILES));
    assert.deepEqual((await readdir(snapshotDirectory)).sort(), Object.keys(KANANA_FILES).sort());
    for (const name of Object.keys(KANANA_FILES))
      assert.equal(await readFile(join(snapshotDirectory, name), 'utf8'), `verified:${name}`);
    assert.equal(
      await readFile(envFile, 'utf8'),
      `EXISTING_VALUE=kept\nKANANA_TOKENIZER_SNAPSHOT=${snapshotDirectory}\n`
    );
  } finally {
    await rm(tempDirectory, {recursive: true, force: true});
  }
});

test('removes partial test assets when a pinned source read fails', async () => {
  const tempDirectory = await mkdtemp(join(tmpdir(), 'kanana-test-support-'));
  try {
    const envFile = join(tempDirectory, 'github-env');
    await writeFile(envFile, '');
    const openSource = async () => ({
      async read(name: keyof typeof KANANA_FILES) {
        if (name === Object.keys(KANANA_FILES)[1]) throw new Error('asset fetch failed');
        return new TextEncoder().encode(`verified:${name}`).buffer;
      }
    });

    await assert.rejects(prepareKananaTokenizerTestAssets({tempDirectory, envFile, openSource}), /asset fetch failed/);
    assert.deepEqual(await readdir(tempDirectory), ['github-env']);
    assert.equal(await readFile(envFile, 'utf8'), '');
  } finally {
    await rm(tempDirectory, {recursive: true, force: true});
  }
});
