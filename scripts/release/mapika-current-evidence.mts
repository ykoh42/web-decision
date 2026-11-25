import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {cleanSourceCommit, sha256} from './curated-gpu-evidence.mts';
import {assertSplitVerifierSource} from './strands-gemma4-v2-split-source.mts';

export const mapikaCurrentPin = {
  testedCommit: 'b05bb22b4676b6e919a4c30665c910338b2c58f3',
  archiveSha256: 'bfa9922b25a32e37d09783b67ad26b4c9284432095c5feb307d243857c0c768a',
  historicalCommit: '7e0cfbcb810ac85a29657ab395350dd81e606f39',
  historicalViteSha256: 'cc45deecb79bfd0079020d6f56ccb5b033f10b46dbc9bfc14fb7bf11b8f4ddaf',
  testedViteSha256: 'e00d2cd329a281f9bd32744bf1c62d8fee2fe45b137cc253bd12657eff618180'
} as const;
export function assertMapikaCurrentFacts(input: {
  testedCommit: string;
  historicalViteSha256: string;
  testedViteSha256: string;
}) {
  assert.equal(input.testedCommit, mapikaCurrentPin.testedCommit);
  assert.equal(input.historicalViteSha256, mapikaCurrentPin.historicalViteSha256);
  assert.equal(input.testedViteSha256, mapikaCurrentPin.testedViteSha256);
}

/** Pin an explicitly selected tested tree independently from the verifier tree. */
export async function verifyMapikaCurrentSource(testedRoot: string, verifierRoot: string) {
  const testedCommit = cleanSourceCommit(testedRoot);
  const verifierCommit = cleanSourceCommit(verifierRoot);
  assert.equal(testedCommit, mapikaCurrentPin.testedCommit);
  assert.notEqual(testedCommit, verifierCommit, 'Current verifier must be a separate source commit');
  const {verificationFiles} = assertSplitVerifierSource(verifierRoot, testedCommit, verifierCommit);
  const path = 'tests/browser/vite.config.ts';
  const historical = execFileSync('git', ['-C', testedRoot, 'show', `${mapikaCurrentPin.historicalCommit}:${path}`]);
  const tested = await readFile(join(testedRoot, path));
  const facts = {testedCommit, historicalViteSha256: sha256(historical), testedViteSha256: sha256(tested)};
  assertMapikaCurrentFacts(facts);
  return {...facts, verifierCommit, verificationFiles};
}
