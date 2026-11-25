import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {lstat, readFile, readdir, realpath} from 'node:fs/promises';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import type {BrowserContext, Page} from '@playwright/test';
import {cleanSourceCommit, record, sha256} from './curated-gpu-evidence.mjs';
import {
  checkpointUrl,
  inventorySha256,
  requiredOfflineWeightBytes,
  type checkedInventory
} from './strands-gemma4-v2-local-checkpoint.mjs';

export const warmOrigin = 'http://127.0.0.1:42713';
export const warmProfileName = 'chromium-rehashed-seed';
const previousRoot = '/private/tmp/gemma4-persistent-graph-fix.Vxel4PMd';
const previousRun = join(previousRoot, 'online-retry');
const previousReport = join(previousRun, 'corpus.playwright.json');
const previousProfile = join(previousRun, 'profiles/chromium-resumed');
const previousVerifier = join(previousRoot, 'verifier');
const previousCommit = '65fe1823326961c09d4074c8f436b56b901ebe94';
const previousVerifierCommit = '9faa74ec3212d435707d1bfec0925ae7f4ca3e9c';
const previousStartSha256 = '420820b4b967a7ac340217af697578e4760ca5feb3361b3629e5049f8afdd19f';
const previousConfigSha256 = '3530e2ce6b29c778f7ecce44f95d55deaa0e83f4c3220e4040556bfa53e9e991';
const previousArchive = {
  sha256: '7c3bb65bdd8a4f3195a613435f6d2a23c8699ff0c435db4614e6d2a88af78edd',
  packageTreeSha256: '5c36d7ad47e5268a5531e91d791fd97658111905f981ac962a2443937ac840c9',
  bundleSha256: '773e7f6f630044d69680297a84e503806830affd69f63919d7b4a3a1db94e41c',
  packageName: 'web-decision',
  packageVersion: '0.7.0'
};

export function assertPreviousCompletion(startBytes: Buffer, doneBytes: Buffer, reportBytes: Buffer, config: Buffer) {
  const parse = (bytes: Buffer) => record(JSON.parse(bytes.toString('utf8')), 'Invalid previous corpus receipt.');
  const start = parse(startBytes);
  const done = parse(doneBytes);
  assert.equal(sha256(startBytes), previousStartSha256, 'The previous run start receipt differs.');
  assert.equal(sha256(config), previousConfigSha256, 'The previous run browser configuration differs.');
  assert.equal(start.status, 'started-raw-cohort');
  assert.equal(start.sourceCommit, previousCommit);
  assert.deepEqual(start.archive, previousArchive);
  const producer = record(start.producer, 'Missing previous producer identity.');
  assert.equal(producer.sourceCommit, previousCommit);
  assert.equal(producer.configSha256, previousConfigSha256);
  assert.equal(producer.persistentCheckpointMode, 'online');
  assert.equal(done.status, 'completed-exact-raw-cohort');
  assert.equal(done.releaseApproved, false);
  assert.equal(done.sourceCommit, previousCommit);
  assert.deepEqual(done.archive, previousArchive);
  assert.deepEqual(done.producer, producer);
  assert.equal(done.startedSha256, previousStartSha256);
  assert.equal(done.playwrightSha256, sha256(reportBytes));
  return {
    sourceCommit: previousCommit,
    archive: previousArchive,
    verifierCommit: previousVerifierCommit,
    producerSha256: producer.producerSha256,
    origin: warmOrigin,
    previousProfileSha256: sha256(Buffer.from(previousProfile)),
    previousProfileSeedStartSha256: producer.reusedProfileFromStartSha256,
    startedSha256: previousStartSha256,
    completedSha256: sha256(doneBytes),
    playwrightSha256: sha256(reportBytes)
  };
}

export async function completedWarmSeed() {
  // Require completion before examining a profile or running the historical verifier.
  const done = await readFile(previousReport + '.completed.json').catch((cause: NodeJS.ErrnoException) => {
    if (cause.code === 'ENOENT')
      throw new Error('A completed 7c3 corpus receipt is required for profile reuse.', {cause});
    throw cause;
  });
  const [start, report, config] = await Promise.all([
    readFile(join(previousRun, 'started.json')),
    readFile(previousReport),
    readFile(join(previousRoot, 'playwright.config.mts'))
  ]);
  const provenance = assertPreviousCompletion(start, done, report, config);
  assert.equal(cleanSourceCommit(previousVerifier), previousVerifierCommit);
  const previousPackage = '/private/tmp/web-decision-asset-audit/repair-070-candidate';
  const verify = pathToFileURL(join(previousVerifier, 'scripts/release/strands-gemma4-v2-split-evidence.mts'));
  const check = `import {verifyRawCohort} from ${JSON.stringify(verify.href)};
    const result = await verifyRawCohort(process.argv[1]);
    console.log(result.playwrightSha256);`;
  const verified = execFileSync(
    process.execPath,
    ['--import', 'tsx', '--input-type=module', '-e', check, previousReport],
    {
      cwd: previousRoot,
      encoding: 'utf8',
      env: {
        PATH: process.env.PATH,
        WEB_DECISION_SPLIT_TESTED_ROOT: join(previousRoot, 'checkout'),
        WEB_DECISION_SPLIT_TESTED_COMMIT: previousCommit,
        WEB_DECISION_SPLIT_PRODUCER_DIR: previousRoot,
        WEB_DECISION_SPLIT_PERSISTENT_CHECKPOINT_MODE: 'online',
        WEB_DECISION_SPLIT_PERSISTENT_REUSE_FAILED_PROFILE: '1',
        WEB_DECISION_RELEASE_ARCHIVE: join(previousPackage, 'web-decision-0.7.0.tgz'),
        WEB_DECISION_INSTALLED_PACKAGE: join(previousPackage, 'consumer/node_modules/web-decision'),
        WEB_DECISION_GEMMA_V2_OFFICIAL_GOLDEN: process.env.WEB_DECISION_GEMMA_V2_OFFICIAL_GOLDEN
      }
    }
  );
  assert.equal(verified.trim(), provenance.playwrightSha256, 'The verified previous report changed.');
  return provenance;
}

export async function assertClosedPrivateProfile(path: string) {
  assert.equal(await realpath(path), path, 'Profile path resolves elsewhere.');
  const details = await lstat(path);
  assert.ok(details.isDirectory() && details.uid === process.getuid?.() && (details.mode & 0o077) === 0);
  const entries = await readdir(path, {withFileTypes: true, recursive: true});
  assert.ok(!entries.some((entry) => entry.name.startsWith('Singleton')), 'Chromium profile is locked.');
  assert.ok(
    entries.every((entry) => entry.isFile() || entry.isDirectory()),
    'Profile contains a special entry.'
  );
  return details;
}

export async function cloneCompletedWarmSeed(profileRoot: string) {
  const provenance = await completedWarmSeed();
  const previous = await assertClosedPrivateProfile(previousProfile);
  assert.equal(await realpath(profileRoot), profileRoot);
  assert.deepEqual(await readdir(profileRoot), []);
  const clone = join(profileRoot, warmProfileName);
  execFileSync('/bin/cp', ['-cRP', previousProfile, clone]);
  const cloned = await assertClosedPrivateProfile(clone);
  const after = await assertClosedPrivateProfile(previousProfile);
  assert.deepEqual([after.dev, after.ino, after.mtimeMs], [previous.dev, previous.ino, previous.mtimeMs]);
  assert.ok(cloned.dev !== previous.dev || cloned.ino !== previous.ino, 'The profile must be a distinct copy.');
  return {...provenance, copiedProfilePathSha256: sha256(Buffer.from(clone))};
}

type Inventory = Awaited<ReturnType<typeof checkedInventory>>;
const pinnedRangeCount = 9_179;

export function requiredWarmRanges(inventory: Inventory) {
  const {tensors, headerBytes} = inventory as Inventory & {
    headerBytes: number;
    tensors: Record<string, {offset: number; length: number}>;
  };
  const spans = [{offset: 0, length: headerBytes}];
  for (const [name, span] of Object.entries(tensors)) {
    if (/^model\.(?:language_model|vision_tower|embed_vision)\./u.test(name)) spans.push(span);
  }
  const ranges = inventory.chunks.filter(([start, size]) =>
    spans.some(({offset, length}) => start < offset + length && offset < start + size)
  );
  const digests = ranges.map((range) => range[2]).sort();
  assert.equal(new Set(digests).size, pinnedRangeCount);
  assert.equal(
    ranges.reduce((bytes, range) => bytes + range[1], 0),
    requiredOfflineWeightBytes
  );
  return digests;
}

export async function probePinnedPublicRange(inventory: Inventory, fetchRange: typeof fetch = fetch) {
  const [start, bytes, digest] = inventory.chunks[0]!;
  const end = start + bytes - 1;
  const requestRange = `bytes=${start}-${end}`;
  const contentRange = `bytes ${start}-${end}/${inventory.bytes}`;
  const response = await fetchRange(checkpointUrl, {
    headers: {Range: requestRange},
    credentials: 'omit',
    cache: 'no-store',
    signal: AbortSignal.timeout(60_000)
  });
  try {
    const {hostname: host, protocol} = new URL(response.url);
    assert.equal(protocol, 'https:');
    assert.ok(host === 'huggingface.co' || host.endsWith('.huggingface.co') || host.endsWith('.hf.co'));
    assert.equal(response.status, 206);
    assert.equal(response.headers.get('content-range'), contentRange);
    assert.equal(response.headers.get('content-length'), String(bytes));
    const body = await response.arrayBuffer();
    assert.equal(body.byteLength, bytes);
    assert.equal(sha256(new Uint8Array(body)), digest);
    return {url: checkpointUrl, host, requestRange, status: 206, contentRange, bytes, sha256: digest};
  } finally {
    if (!response.bodyUsed) await response.body?.cancel();
  }
}

export async function blockWarmRemote(context: BrowserContext, origin: string) {
  assert.equal(origin, warmOrigin);
  const attempts: string[] = [];
  // The later exact archive route intercepts the allowlisted origin before this catch-all.
  await context.route('**/*', async (route) => {
    attempts.push(new URL(route.request().url()).origin);
    await route.abort();
  });
  return attempts;
}

export function expectedWarmRehash(inventory: Inventory) {
  const digests = requiredWarmRanges(inventory);
  return {
    mode: 'offline-prepareOffline-before-first-inference',
    inventorySha256,
    verifiedRanges: digests.length,
    verifiedBytes: requiredOfflineWeightBytes,
    rangeDigestsSha256: sha256(Buffer.from(JSON.stringify(digests)))
  };
}

export async function warmRehash(page: Page, modelId: string, inventory: Inventory) {
  const pinned = requiredWarmRanges(inventory);
  const observed = await page.evaluate(
    async ({id, required}) => {
      const api = Reflect.get(window, 'webDecision') as unknown as {
        loadDecisionModel(options: {modelId: string; cache: true; offline: true}): Promise<{
          prepareOffline(): Promise<void>;
          isOfflineReady(): Promise<boolean>;
          dispose(): Promise<void>;
        }>;
      };
      const model = await api.loadDecisionModel({modelId: id, cache: true, offline: true});
      const wanted = new Set(required);
      const hits = new Set<string>();
      const match = Cache.prototype.match;
      Cache.prototype.match = async function (request, options) {
        const response = await match.call(this, request, options);
        const url = new URL(request instanceof Request ? request.url : request, location.href);
        const digest = url.searchParams.get('digest');
        if (response && url.pathname === '/.web-decision-cache/sha256/' && digest && wanted.has(digest))
          hits.add(digest);
        return response;
      };
      try {
        await model.prepareOffline();
        if (!(await model.isOfflineReady())) throw new Error('The rehashed offline profile is incomplete.');
        Reflect.set(window, 'gemmaV2ReleaseSession', model);
        return [...hits].sort();
      } catch (error) {
        await model.dispose();
        throw error;
      } finally {
        Cache.prototype.match = match;
      }
    },
    {id: modelId, required: pinned}
  );
  assert.deepEqual(observed, pinned, 'The offline preparation did not read every required cached range.');
  return expectedWarmRehash(inventory);
}

export type WarmClone = Awaited<ReturnType<typeof cloneCompletedWarmSeed>>;

export function assertWarmTrace(
  data: Record<string, unknown>,
  expectedSeed: Awaited<ReturnType<typeof completedWarmSeed>>,
  expectedClonePath: string,
  inventory: Inventory
) {
  const trace = record(data.warmSeed, 'Missing rehashed profile evidence.');
  const {rehash, browserRemoteAttempts, copy, ...rest} = trace;
  assert.deepEqual(rest, expectedSeed);
  assert.deepEqual(copy, {
    mode: 'closed-profile-copy',
    copiedProfilePathSha256: sha256(Buffer.from(expectedClonePath))
  });
  assert.deepEqual(browserRemoteAttempts, [], 'The warm browser attempted network access.');
  assert.deepEqual(rehash, expectedWarmRehash(inventory));
  return trace;
}
