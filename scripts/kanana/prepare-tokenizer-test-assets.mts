import {appendFile, mkdtemp, mkdir, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {KANANA_FILES} from '../../packages/web-decision/src/kanana/contract.mjs';
import {openKananaSource} from '../../packages/web-decision/src/kanana/source.mjs';

type KananaSource = Readonly<{
  read(name: keyof typeof KANANA_FILES): Promise<ArrayBuffer>;
}>;
type PrepareOptions = Readonly<{
  tempDirectory?: string;
  envFile?: string;
  openSource?: (options: {cacheEnabled: boolean}) => Promise<KananaSource>;
}>;

export async function prepareKananaTokenizerTestAssets({
  tempDirectory = process.env.RUNNER_TEMP ?? tmpdir(),
  envFile = process.env.GITHUB_ENV,
  openSource = openKananaSource
}: PrepareOptions = {}): Promise<string> {
  if (typeof envFile !== 'string' || !envFile)
    throw new Error('GITHUB_ENV is required to export the Kanana test asset path.');
  if (typeof tempDirectory !== 'string' || !tempDirectory)
    throw new TypeError('A temporary directory is required for Kanana test assets.');

  await mkdir(tempDirectory, {recursive: true});
  const snapshotDirectory = await mkdtemp(join(tempDirectory, 'kanana-tokenizer-test-'));
  try {
    const source = await openSource({cacheEnabled: false});
    for (const name of Object.keys(KANANA_FILES) as Array<keyof typeof KANANA_FILES>) {
      const bytes = await source.read(name);
      await writeFile(join(snapshotDirectory, name), new Uint8Array(bytes), {flag: 'wx'});
    }
    await appendFile(envFile, `KANANA_TOKENIZER_SNAPSHOT=${snapshotDirectory}\n`, 'utf8');
    return snapshotDirectory;
  } catch (cause) {
    await rm(snapshotDirectory, {recursive: true, force: true});
    throw cause;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const snapshotDirectory = await prepareKananaTokenizerTestAssets();
  process.stdout.write(`Prepared pinned Kanana tokenizer test assets in ${snapshotDirectory}\n`);
}
