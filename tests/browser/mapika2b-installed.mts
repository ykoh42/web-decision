import {readFile, realpath} from 'node:fs/promises';
import {join} from 'node:path';
import {expect} from '@playwright/test';
import {verifyArchiveInstall} from '../../scripts/release/verify-release-evidence.ts';

export async function verifyInstalledMapikaCandidate() {
  const archivePath = process.env.WEB_DECISION_MAPIKA2B_ARCHIVE;
  const installedPackage = process.env.WEB_DECISION_MAPIKA2B_INSTALLED_PACKAGE;
  const installedBundle = process.env.WEB_DECISION_MAPIKA2B_INSTALLED_BUNDLE;
  if (!archivePath && !installedPackage && !installedBundle) return null;
  if (!archivePath || !installedPackage || !installedBundle)
    throw new Error('The private Mapika candidate needs its archive, installed package and installed bundle paths.');

  const proof = await verifyArchiveInstall(archivePath, installedPackage);
  const metadata = JSON.parse(await readFile(join(proof.installedRoot, 'package.json'), 'utf8')) as unknown;
  expect(metadata).toMatchObject({name: 'web-decision', private: true, version: '0.0.0-mapika2b-candidate'});
  expect(await realpath(installedBundle)).toBe(proof.bundlePath);
  expect(proof.archiveFiles).toEqual(
    expect.arrayContaining(['dist/mapika2b/text.wasm', 'dist/mapika2b/text-bindings.json'])
  );
  return proof;
}
