import assert from 'node:assert/strict';
import {mkdtemp, readFile, realpath, rm, symlink, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import test from 'node:test';
const identity = {
  sourceCommit: 'a'.repeat(40),
  archive: {
    sha256: 'archive',
    packageTreeSha256: 'tree',
    bundleSha256: 'bundle',
    packageName: 'web-decision',
    packageVersion: '0.0.0-test'
  }
};
import {sha256} from './curated-gpu-evidence.mts';
import {
  GEMMA_V2_OFFICIAL_SUBSET_PATH,
  gemma4V2Agreement,
  readGemmaCandidate,
  readPinnedGemma4V2Subset,
  selectGemmaGoldenRows
} from './strands-gemma4-v2-conformance.mjs';
import {assertSplitCandidate, gemmaSplitCandidate} from './strands-gemma4-v2-split-candidate.mjs';
import {GEMMA_V2_CPU_REFERENCE_PATH} from './strands-gemma4-v2-source-reference.mts';
import {checkpointUrl, checkedInventory} from './strands-gemma4-v2-local-checkpoint.mjs';
import {
  assertClosedPrivateProfile,
  assertWarmTrace,
  blockWarmRemote,
  expectedWarmRehash,
  probePinnedPublicRange,
  warmOrigin
} from './strands-gemma4-v2-warm-seed.mjs';
import {assertDerivedSpec, deriveSplitSpec, originalPath, workspace} from './strands-gemma4-v2-split-producer.mjs';
import {
  assertCanonicalRawVectors,
  assertCohortMetrics,
  assertCohortOperational,
  assertMatchingGpu,
  assertRawReceipt,
  attachment,
  passingChromiumRun,
  verifyCompleted32k
} from './strands-gemma4-v2-split-evidence.mjs';

test('rejects a changed split producer even when the companion digest was changed with it', async () => {
  const original = await readFile(join(workspace, originalPath));
  const derived = deriveSplitSpec(original.toString('utf8'));
  const pin = {
    sourceCommit: '',
    originalSourcePath: originalPath,
    originalSourceSha256: sha256(original),
    producerPath: 'split-public-archive.spec.ts',
    producerSha256: sha256(Buffer.from(derived))
  };
  const tampered = Buffer.from(derived.replace("delete references['text-32768'];", "delete references['text-1024'];"));
  assert.notEqual(derived, tampered.toString());
  assert.throws(
    () => assertDerivedSpec(original, tampered, {...pin, producerSha256: sha256(tampered)}),
    /Derived producer differs/
  );
});

test('raw receipt binds lifecycle, actual Playwright report, producer and installed archive', () => {
  const expected = {
    sourceCommit: identity.sourceCommit,
    archive: identity.archive,
    producer: {sha256: 'producer'},
    startedSha256: 'start',
    playwrightSha256: 'report'
  };
  const genuine = {...expected, status: 'completed-exact-raw-cohort', releaseApproved: false};
  assert.doesNotThrow(() => assertRawReceipt(genuine, expected));
  for (const difference of [
    {status: 'started-raw-cohort'},
    {playwrightSha256: 'different'},
    {startedSha256: 'different'},
    {producer: {sha256: 'different'}},
    {archive: {sha256: 'other'}},
    {sourceCommit: 'wrong'}
  ])
    assert.throws(() => assertRawReceipt({...genuine, ...difference}, expected));
});

test('32K verification rejects missing completion and altered reports, exact requests, or archive identity', async () => {
  const tmp = await mkdtemp(join(tmpdir(), 'audit-32k-'));
  try {
    const prefix = join(tmp, 'diagnostic');
    const cpu = await readFile(join(workspace, GEMMA_V2_CPU_REFERENCE_PATH));
    const input = JSON.parse(cpu.toString()).cases['text-32768'];
    const base = {
      mode: 'reference',
      sourceCommit: identity.sourceCommit,
      archive: identity.archive,
      requestSha256: input.requestSha256,
      referenceSha256: input.resultSha256
    };
    const report = '{}';
    await writeFile(prefix + '.playwright.json', report);
    await writeFile(prefix + '.start.json', JSON.stringify({...base, status: 'running-not-release-evidence'}));
    await assert.rejects(verifyCompleted32k(prefix, cpu, identity), {code: 'ENOENT'});
    const completion = {
      ...base,
      status: 'completed-exact-installed-run',
      releaseApproved: false,
      playwrightReportSha256: sha256(Buffer.from(report))
    };
    for (const synthetic of [
      {...completion, requestSha256: 'wrong'},
      {...completion, archive: {...identity.archive, sha256: 'wrong'}},
      {...completion, playwrightReportSha256: 'wrong'}
    ]) {
      await writeFile(prefix + '.completed.json', JSON.stringify(synthetic));
      await assert.rejects(verifyCompleted32k(prefix, cpu, identity));
    }
  } finally {
    await rm(tmp, {recursive: true, force: true});
  }
});

test('missing raw attachments, rounded vectors, and failed upstream metrics are rejected', () => {
  assert.throws(() => attachment({attachments: []}, 'original', 'application/json'), /Both original attachments/);
  const metrics = {questions: 1066, meetsUpstreamMetricThresholds: false} as Parameters<typeof assertCohortMetrics>[1];
  assert.throws(() => assertCohortMetrics({probabilityEncoding: 'chromium-cdp-unrounded-fp32', ...metrics}, metrics));
  assert.throws(() => assertCohortMetrics({probabilityEncoding: 'system-one-four-decimal', ...metrics}, metrics));
});

test('offline requests, cleanup/recovery failures and fallback or different devices are rejected', () => {
  const browser = {name: 'chromium', version: 'test'};
  const adapter = {isFallbackAdapter: false, device: 'test'};
  const hardware = {status: 'Hardware accelerated', renderer: 'test'};
  const good = {
    cases: {'image-single': {result: {type: 'choice'}}},
    cleanup: 'passed',
    recovery: 'passed',
    offline: {cacheStatus: 'complete', blockedRequests: [], result: {type: 'choice'}},
    browser,
    adapter,
    hardware
  };
  assert.doesNotThrow(() => assertCohortOperational(good));
  for (const difference of [
    {offline: {...good.offline, blockedRequests: ['https://example.test']}},
    {cleanup: 'failed'},
    {recovery: 'failed'},
    {adapter: {isFallbackAdapter: true}}
  ])
    assert.throws(() => assertCohortOperational({...good, ...difference}));
  assert.throws(() => assertMatchingGpu(good, {browser, adapter: {...adapter, device: 'other'}, hardware}));
  assert.throws(() => assertMatchingGpu(good, {browser: {...browser, version: 'other'}, adapter, hardware}));
});

test('passing Chromium report checks zero retries, correct spec and an actual passing result', () => {
  const root = '/private/tmp/synthetic-browser-proof';
  const file = 'only.spec.ts';
  const title = 'expected spec';
  const run = {status: 'passed', retry: 0, errors: [], duration: 1};
  const report = (patch: Record<string, unknown>) =>
    Buffer.from(
      JSON.stringify({
        config: {rootDir: root},
        stats: {expected: 1, unexpected: 0, flaky: 0, skipped: 0, startTime: '2026-11-24T12:00:00Z'},
        suites: [
          {
            specs: [
              {
                file,
                title,
                ok: true,
                tests: [
                  {projectName: 'chromium', status: 'expected', expectedStatus: 'passed', results: [{...run, ...patch}]}
                ]
              }
            ]
          }
        ]
      })
    );
  assert.deepEqual(passingChromiumRun(report({}), root, file, title), run);
  for (const patch of [{retry: 1}, {status: 'interrupted'}, {errors: ['incomplete']}]) {
    assert.throws(() => passingChromiumRun(report(patch), root, file, title));
  }
  assert.throws(() => passingChromiumRun(report({}), root, 'different.spec.ts', title));
});

test('32K accepts one complete bound report and rejects skipped or changed browser results', async () => {
  const tmp = await mkdtemp(join(tmpdir(), 'audit-32k-valid-'));
  try {
    const prefix = join(tmp, 'diagnostic');
    const cpu = await readFile(join(workspace, GEMMA_V2_CPU_REFERENCE_PATH));
    const input = JSON.parse(cpu.toString()).cases['text-32768'];
    const reference = JSON.parse(input.resultJson);
    const actual = {...reference, model: 'strands-decider-e2b-gemma4-v2-2611'};
    const base = {
      mode: 'reference',
      sourceCommit: identity.sourceCommit,
      archive: identity.archive,
      requestSha256: input.requestSha256,
      referenceSha256: input.resultSha256
    };
    const event = {
      archiveSha256: identity.archive.sha256,
      bundleSha256: identity.archive.bundleSha256,
      packageVersion: identity.archive.packageVersion,
      requestSha256: input.requestSha256,
      referenceSha256: input.resultSha256,
      reference,
      actual
    };
    const raw = {
      ...event,
      browser: {name: 'chromium', version: 'test'},
      hardware: {status: 'Hardware accelerated', renderer: 'test'},
      adapter: {isFallbackAdapter: false}
    };
    const summary = {...event, maxDelta: 0, tolerance: null};
    const root = join(process.env.WEB_DECISION_SPLIT_TESTED_ROOT ?? '', 'scripts/strands/gemma4_v2/candidate');
    const report = (status: string) =>
      Buffer.from(
        JSON.stringify({
          errors: [],
          config: {rootDir: join(process.cwd(), root)},
          stats: {expected: 1, unexpected: 0, flaky: 0, skipped: 0, startTime: '2026-11-24T00:00:00Z'},
          suites: [
            {
              specs: [
                {
                  file: 'gemma4-v2-installed-reference.spec.ts',
                  title: 'installed archive collects one pinned upstream request',
                  ok: true,
                  tests: [
                    {
                      projectName: 'chromium',
                      expectedStatus: 'passed',
                      status: 'expected',
                      results: [
                        {
                          status,
                          retry: 0,
                          duration: 1,
                          errors: [],
                          stdout: [
                            {text: 'INSTALLED_REFERENCE_RAW ' + JSON.stringify(raw) + '\n'},
                            {text: 'INSTALLED_REFERENCE ' + JSON.stringify(summary) + '\n'}
                          ]
                        }
                      ]
                    }
                  ]
                }
              ]
            }
          ]
        })
      );
    const write = async (bytes: Buffer) => {
      await writeFile(prefix + '.playwright.json', bytes);
      await writeFile(
        prefix + '.completed.json',
        JSON.stringify({
          ...base,
          status: 'completed-exact-installed-run',
          releaseApproved: false,
          playwrightReportSha256: sha256(bytes)
        })
      );
    };
    await writeFile(prefix + '.start.json', JSON.stringify({...base, status: 'running-not-release-evidence'}));
    await write(report('passed'));
    assert.equal((await verifyCompleted32k(prefix, cpu, identity)).maxDelta, 0);
    await write(report('skipped'));
    await assert.rejects(verifyCompleted32k(prefix, cpu, identity));
  } finally {
    await rm(tmp, {recursive: true, force: true});
  }
});

test('Vite replay remains separately pinned and fails closed on wrong provenance or scope', async () => {
  const {assertViteReplayRecord} = await import('./strands-gemma4-v2-vite-replay.mjs');
  const {gemmaV2ArchiveTokenizerSha256} = await import('../../tests/browser/gemma-v2-exact-archive-page.mjs');
  const expected = {
    sourceCommit: 'source',
    archiveSha256: 'archive',
    packageTreeSha256: 'tree',
    startSha256: 'start',
    requestSha256: 'request',
    referenceSha256: 'reference'
  };
  const report = {
    scope: 'CPU Vite replay; no browser response bytes from the live run',
    gpuExecuted: false,
    sourceCommit: 'source',
    archiveSha256: 'archive',
    packageTreeSha256: 'tree',
    recordedStartSha256: 'start',
    fixturePins: {requestSha256: 'request', referenceSha256: 'reference'},
    viteVersion: '8.3.6',
    parserVersion: '8.19.0',
    tokenizerSha256: gemmaV2ArchiveTokenizerSha256,
    issues: [],
    archiveModules: 27,
    matchedModules: 28,
    devShell: {entryImportsExactInstalledBundle: true, pageModuleScripts: ['/@vite/client', '/harness.ts']},
    modules: Array.from({length: 28}, () => ({})),
    assets: Array.from({length: 17}, () => ({}))
  };
  assert.doesNotThrow(() => assertViteReplayRecord(report, expected));
  for (const mutation of [
    {scope: 'live browser response'},
    {gpuExecuted: true},
    {sourceCommit: 'other'},
    {archiveSha256: 'other'},
    {packageTreeSha256: 'other'},
    {recordedStartSha256: 'other'},
    {fixturePins: {requestSha256: 'wrong', referenceSha256: 'reference'}},
    {issues: ['mismatch']},
    {archiveModules: 26},
    {matchedModules: 27},
    {modules: report.modules.slice(1)},
    {assets: report.assets.slice(1)}
  ])
    assert.throws(() => assertViteReplayRecord({...report, ...mutation}, expected));
});

test('the reviewed Vite replay and script have independent required digests', async () => {
  const {assertViteReplayPins, VITE_REPLAY_ARCHIVE_SHA256, VITE_REPLAY_SHA256, VITE_REPLAY_SCRIPT_SHA256} =
    await import('./strands-gemma4-v2-vite-replay.mjs');
  const pinned = {reportSha256: VITE_REPLAY_SHA256, scriptSha256: VITE_REPLAY_SCRIPT_SHA256};
  assert.doesNotThrow(() => assertViteReplayPins(pinned, VITE_REPLAY_ARCHIVE_SHA256));
  assert.throws(
    () => assertViteReplayPins({...pinned, reportSha256: 'different'}, VITE_REPLAY_ARCHIVE_SHA256),
    /Unreviewed Vite replay report/
  );
  assert.throws(
    () => assertViteReplayPins({...pinned, scriptSha256: 'different'}, VITE_REPLAY_ARCHIVE_SHA256),
    /Unreviewed Vite replay script/
  );
  assert.throws(() => assertViteReplayPins(pinned, '0'.repeat(64)), /No reviewed Vite replay is pinned for archive/);
  assert.throws(
    () => assertViteReplayPins(pinned, gemmaSplitCandidate.archive.sha256),
    /No reviewed Vite replay is pinned for archive/
  );
});

test('split gate rejects a missing file or modified reviewed Vite replay report', async () => {
  const {verifyViteReplay, VITE_REPLAY_ARCHIVE_SHA256} = await import('./strands-gemma4-v2-vite-replay.mjs');
  const tmp = await mkdtemp(join(tmpdir(), 'vite-replay-integrity-'));
  try {
    const reportPath = join(tmp, 'audit.json');
    const scriptPath = join(tmp, 'audit.mts');
    const input = {
      reportPath,
      scriptPath,
      workspace,
      sourceCommit: 'a'.repeat(40),
      archive: {archiveSha256: VITE_REPLAY_ARCHIVE_SHA256} as Parameters<typeof verifyViteReplay>[0]['archive'],
      startSha256: 'a',
      requestSha256: 'b',
      referenceSha256: 'c'
    };
    await assert.rejects(verifyViteReplay(input), {code: 'ENOENT'});
    await writeFile(reportPath, '{}');
    await writeFile(scriptPath, 'modified');
    await assert.rejects(verifyViteReplay(input), /Unreviewed Vite replay report/);
  } finally {
    await rm(tmp, {recursive: true, force: true});
  }
});

test('the persistent online path rejects a local checkpoint trace and a mismatched range digest', async () => {
  const {assertPersistentTrace, failedPersistentStartSha256} = await import('./strands-gemma4-v2-persistent-trace.mjs');
  const {checkpointUrl, requiredOfflineWeightBytes} = await import('./strands-gemma4-v2-local-checkpoint.mjs');
  const sha256 = 'a'.repeat(64);
  const inventory = {
    file: 'model.safetensors',
    bytes: 6,
    sha256,
    chunks: [
      [0, 3, sha256],
      [3, 3, 'b'.repeat(64)]
    ] as const
  };
  const estimate = {quota: requiredOfflineWeightBytes, usage: 0};
  const storage = {
    mode: 'persistent-chromium-profile',
    graphObservation: 'model-load-response',
    requiredOfflineWeightBytes,
    before: estimate,
    afterOffline: estimate
  };
  const firstObservedRange = {
    url: checkpointUrl,
    host: 'us.aws.cdn.hf.co',
    requestRange: 'bytes=0-2',
    status: 206,
    contentRange: 'bytes 0-2/6',
    bytes: 3,
    sha256
  };
  const checkpointTransport = {
    mode: 'pinned-public-upstream-http-range',
    wholeCheckpointOnlineDownloadVerified: false,
    sourceUrl: checkpointUrl,
    observation: 'model-load-range',
    firstObservedRange
  };
  assert.doesNotThrow(() => assertPersistentTrace({storage, checkpointTransport}, 'online', inventory));
  assert.throws(() =>
    assertPersistentTrace({storage, checkpointTransport: {...checkpointTransport, localRoute: {}}}, 'online', inventory)
  );
  assert.throws(() =>
    assertPersistentTrace(
      {
        storage,
        checkpointTransport: {...checkpointTransport, firstObservedRange: {...firstObservedRange, sha256: 'wrong'}}
      },
      'online',
      inventory
    )
  );

  const warmStorage = {
    ...storage,
    graphObservation: 'independent-archive-response',
    reusedProfileFromStartSha256: failedPersistentStartSha256
  };
  const probe = {...checkpointTransport, observation: 'independent-online-range-probe'};
  const warm = {storage: warmStorage, checkpointTransport: probe};
  assert.doesNotThrow(() => assertPersistentTrace(warm, 'online', inventory, failedPersistentStartSha256));
  assert.throws(() => assertPersistentTrace(warm, 'online', inventory));
  assert.throws(() => assertPersistentTrace(warm, 'local', inventory, failedPersistentStartSha256));
  assert.throws(() =>
    assertPersistentTrace({storage: warmStorage, checkpointTransport}, 'online', inventory, failedPersistentStartSha256)
  );
  assert.throws(() =>
    assertPersistentTrace({storage, checkpointTransport: probe}, 'online', inventory, failedPersistentStartSha256)
  );
  assert.throws(() =>
    assertPersistentTrace(
      {...warm, storage: {...warmStorage, graphObservation: 'model-load-response'}},
      'online',
      inventory,
      failedPersistentStartSha256
    )
  );
  const otherPinnedRange = {
    ...firstObservedRange,
    requestRange: 'bytes=3-5',
    contentRange: 'bytes 3-5/6',
    sha256: 'b'.repeat(64)
  };
  assert.doesNotThrow(() =>
    assertPersistentTrace(
      {storage, checkpointTransport: {...checkpointTransport, firstObservedRange: otherPinnedRange}},
      'online',
      inventory
    )
  );
  for (const transport of [
    {...probe, firstObservedRange: otherPinnedRange},
    {...probe, wholeCheckpointOnlineDownloadVerified: true}
  ]) {
    assert.throws(() =>
      assertPersistentTrace({...warm, checkpointTransport: transport}, 'online', inventory, failedPersistentStartSha256)
    );
  }
});

test('official raw corpus uses physical golden line numbers across the omitted interval', async () => {
  const subset = readPinnedGemma4V2Subset(await readFile(join(workspace, GEMMA_V2_OFFICIAL_SUBSET_PATH)));
  assert.deepEqual(subset, [
    ...Array.from({length: 300}, (_, row) => row),
    ...Array.from({length: 60}, (_, row) => row + 430)
  ]);
  const q = {codes: ['C', 'D'], probabilities: [0.5, 0.5]};
  const key = (line: number) => JSON.stringify([line, 'q']);
  const source = new Map([
    [key(299), q],
    [key(300), q],
    [key(430), q]
  ]);
  const target = selectGemmaGoldenRows(source, [299, 430]);
  const row = (line: number) => JSON.stringify({line, questions: {q}});
  const exact = readGemmaCandidate(Buffer.from(row(430) + '\n' + row(299) + '\n'));
  assert.equal(gemma4V2Agreement(target, exact).top1Agreement, 1);
  assert.throws(() => gemma4V2Agreement(target, readGemmaCandidate(Buffer.from(row(299)))), /same questions/);
  assert.throws(
    () => gemma4V2Agreement(target, readGemmaCandidate(Buffer.from(row(299) + '\n' + row(300)))),
    /Missing candidate question/
  );
  assert.throws(() => readGemmaCandidate(Buffer.from(row(299) + '\n' + row(299))), /Duplicate candidate row/);
});

test('raw vector encoding rejects duplicate object members that JSON.parse collapses', () => {
  const value = {codes: ['C', 'D'], probabilities: [0.5, 0.5]};
  const valid = JSON.stringify({line: 430, questions: {q: value}}) + '\n';
  assert.doesNotThrow(() => assertCanonicalRawVectors(Buffer.from(valid)));
  const q = '"q":' + JSON.stringify(value);
  const duplicate = '{"line":430,"questions":{' + q + ',' + q + '}}\n';
  assert.throws(() => assertCanonicalRawVectors(Buffer.from(duplicate)), /encoded by the exact producer/);
  assert.throws(() => assertCanonicalRawVectors(Buffer.from(valid.trimEnd())), /newline-terminated/);
});

test('raw and gold ties choose the first code in pinned order and delta averages question maxima', () => {
  const codes = ['D', 'C', 'A'];
  const row = (probabilities: number[]) => ({codes, probabilities});
  const reference = new Map([
    ['candidate-tie', row([0.25, 0.5, 0.25])],
    ['reference-tie', row([0.375, 0.375, 0.25])]
  ]);
  const candidate = new Map([
    ['candidate-tie', row([0.375, 0.375, 0.25])],
    ['reference-tie', row([0.5, 0.25, 0.25])]
  ]);
  const score = gemma4V2Agreement(reference, candidate);
  assert.equal(score.top1Agreement, 0.5);
  assert.equal(score.meanAbsDp, 0.125);
});

test('the 301f candidate rejects legacy source, legacy archive and changed installation identity', () => {
  const {sourceCommit, archive} = gemmaSplitCandidate;
  assert.doesNotThrow(() => assertSplitCandidate(sourceCommit, archive));
  assert.throws(
    () => assertSplitCandidate('65fe1823326961c09d4074c8f436b56b901ebe94', archive),
    /tested source differs/
  );
  for (const change of [
    {sha256: '7c3bb65bdd8a4f3195a613435f6d2a23c8699ff0c435db4614e6d2a88af78edd'},
    {packageTreeSha256: '0'.repeat(64)},
    {bundleSha256: '0'.repeat(64)}
  ])
    assert.throws(() => assertSplitCandidate(sourceCommit, {...archive, ...change}), /exact installed archive differs/);
});

test('an e729 32K start receipt alone cannot verify a run', async () => {
  const root = await mkdtemp(join(tmpdir(), 'gemma-e729-incomplete-'));
  try {
    const prefix = join(root, '32k');
    await writeFile(
      prefix + '.start.json',
      JSON.stringify({
        status: 'running-not-release-evidence',
        mode: 'reference',
        ...gemmaSplitCandidate
      })
    );
    const cpu = await readFile(join(workspace, GEMMA_V2_CPU_REFERENCE_PATH));
    await assert.rejects(verifyCompleted32k(prefix, cpu, gemmaSplitCandidate), {code: 'ENOENT'});
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});

test('warm reuse only admits a closed, private ordinary profile directory', async () => {
  const profile = await realpath(await mkdtemp(join(tmpdir(), 'gemma-warm-profile-')));
  try {
    await assert.doesNotReject(assertClosedPrivateProfile(profile));
    const lock = join(profile, 'SingletonLock');
    await symlink('held-by-chromium', lock);
    await assert.rejects(assertClosedPrivateProfile(profile), /profile is locked/);
    await rm(lock);
    await symlink('elsewhere', join(profile, 'unexpected-link'));
    await assert.rejects(assertClosedPrivateProfile(profile), /special entry/);
  } finally {
    await rm(profile, {recursive: true, force: true});
  }
});

test('warm cohort requires the original receipt, origin, distinct clone path, and complete actual rehash', async () => {
  const inventory = await checkedInventory(join(workspace, 'packages/web-decision'));
  const rehash = expectedWarmRehash(inventory);
  assert.equal(rehash.verifiedRanges, 9179);
  assert.equal(rehash.verifiedBytes, 9_638_447_838);
  const source = {
    sourceCommit: 'a',
    archive: gemmaSplitCandidate.archive,
    verifierCommit: 'b',
    producerSha256: 'c',
    origin: warmOrigin,
    previousProfileSha256: 'd',
    previousProfileSeedStartSha256: 'e',
    startedSha256: 'f',
    completedSha256: 'g',
    playwrightSha256: 'h'
  };
  const path = '/tmp/new-profile';
  const warmSeed = {
    ...source,
    copy: {mode: 'closed-profile-copy', copiedProfilePathSha256: sha256(Buffer.from(path))},
    rehash,
    browserRemoteAttempts: []
  };
  assert.deepEqual(assertWarmTrace({warmSeed}, source, path, inventory), warmSeed);
  for (const change of [
    {origin: 'http://127.0.0.1:42000'},
    {completedSha256: undefined},
    {playwrightSha256: 'different'},
    {copy: {...warmSeed.copy, copiedProfilePathSha256: sha256(Buffer.from('/tmp/old-profile'))}},
    {browserRemoteAttempts: ['https://huggingface.co']},
    {rehash: {...rehash, verifiedRanges: 9178}},
    {rehash: {...rehash, rangeDigestsSha256: sha256(Buffer.from('one entry missing'))}}
  ])
    assert.throws(() => assertWarmTrace({warmSeed: {...warmSeed, ...change}}, source, path, inventory));
});

test('warm browser routing refuses a new origin and aborts unmatched network requests', async () => {
  let handle!: (route: unknown) => Promise<void>;
  const context = {
    async route(pattern: string, handler: typeof handle) {
      assert.equal(pattern, '**/*');
      handle = handler;
    }
  } as unknown as Parameters<typeof blockWarmRemote>[0];
  await assert.rejects(blockWarmRemote(context, 'http://127.0.0.1:42000'));
  const attempts = await blockWarmRemote(context, warmOrigin);
  let aborted = false;
  await handle({
    request: () => ({url: () => checkpointUrl}),
    abort: async () => {
      aborted = true;
    }
  });
  assert.equal(aborted, true);
  assert.deepEqual(attempts, ['https://huggingface.co']);
});

test('independent public probe accepts only a pinned complete 206 body', async () => {
  const body = Buffer.from('warm');
  const inventory = {file: 'model.safetensors', sha256: 'unused', bytes: 100, chunks: [[0, 4, sha256(body)]]} as const;
  const fake = (status: number, content: Buffer) => async (input: Parameters<typeof fetch>[0], init?: RequestInit) => {
    assert.equal(input, checkpointUrl);
    assert.equal((init?.headers as Record<string, string>).Range, 'bytes=0-3');
    assert.equal(init?.credentials, 'omit');
    assert.equal(init?.cache, 'no-store');
    const response = new Response(new Uint8Array(content), {
      status,
      headers: {'content-range': 'bytes 0-3/100', 'content-length': '4'}
    });
    Object.defineProperty(response, 'url', {value: 'https://huggingface.co/redirect-omitted'});
    return response;
  };
  assert.equal((await probePinnedPublicRange(inventory, fake(206, body))).sha256, sha256(body));
  await assert.rejects(probePinnedPublicRange(inventory, fake(200, body)));
  await assert.rejects(probePinnedPublicRange(inventory, fake(206, Buffer.from('cold'))));
});
