import assert from 'node:assert/strict';

export const gemmaSplitCandidate = {
  sourceCommit: 'b05bb22b4676b6e919a4c30665c910338b2c58f3',
  archive: {
    sha256: 'bfa9922b25a32e37d09783b67ad26b4c9284432095c5feb307d243857c0c768a',
    packageTreeSha256: 'c63c699fc4548251d895afef070abaa6f6bb43ff7c832dde4d8f117165a40269',
    bundleSha256: '186279ad1ec30544155cc46638f2627f3ecc6fdc345080bfa72bf92edc44b2c4',
    packageName: 'web-decision',
    packageVersion: '0.7.0'
  }
} as const;
export const viteConsumerScope = 'installed-archive-vite-consumer; live response bytes not captured';
export const corpusTitle =
  'the exact bfa9922b archive records official subset and five cases, excluding the independent 32K run';

export function assertSplitCandidate(sourceCommit: string, archive: unknown) {
  assert.equal(
    sourceCommit,
    gemmaSplitCandidate.sourceCommit,
    'The recorded tested source differs from the bfa candidate.'
  );
  assert.deepEqual(
    archive,
    gemmaSplitCandidate.archive,
    'The exact installed archive differs from the bfa candidate.'
  );
}
