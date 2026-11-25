import assert from 'node:assert/strict';
import {record} from './curated-gpu-evidence.mjs';
import {
  checkpointUrl,
  inventorySha256,
  requiredOfflineWeightBytes,
  type checkedInventory
} from './strands-gemma4-v2-local-checkpoint.mjs';

export type PersistentMode = 'online' | 'local';
type Inventory = Awaited<ReturnType<typeof checkedInventory>>;
export const failedPersistentStartSha256 = 'c64d1bd2389ad25b50cba933bd85546a8a7b1dacc8a4c1a540f205f8af49820f';

export function persistentProfileSeed(mode: PersistentMode | undefined) {
  const requested = process.env.WEB_DECISION_SPLIT_PERSISTENT_REUSE_FAILED_PROFILE;
  assert.ok(
    requested === undefined || (requested === '1' && mode === 'online'),
    'Failed-profile reuse is supported only for the pinned online rerun.'
  );
  return requested ? failedPersistentStartSha256 : undefined;
}

export function persistentMode(): PersistentMode | 'warm' | undefined {
  const mode = process.env.WEB_DECISION_SPLIT_PERSISTENT_CHECKPOINT_MODE;
  assert.ok(
    mode === undefined || mode === 'online' || mode === 'local' || mode === 'warm',
    'Expected persistent checkpoint mode online, local, or warm.'
  );
  return mode;
}

export function assertPersistentTrace(
  data: Record<string, unknown>,
  mode: PersistentMode,
  inventory: Inventory,
  reusedProfileFromStartSha256?: string,
  rehashedSeed = false
) {
  const independent = Boolean(reusedProfileFromStartSha256 || rehashedSeed);
  const storage = record(data.storage, 'Missing persistent profile evidence.');
  assert.equal(storage.mode, 'persistent-chromium-profile');
  assert.equal(storage.requiredOfflineWeightBytes, requiredOfflineWeightBytes);
  assert.equal(storage.reusedProfileFromStartSha256, reusedProfileFromStartSha256);
  assert.equal(storage.graphObservation, independent ? 'independent-archive-response' : 'model-load-response');
  if (reusedProfileFromStartSha256) {
    assert.equal(mode, 'online');
    assert.equal(reusedProfileFromStartSha256, failedPersistentStartSha256);
  }
  for (const name of ['before', 'afterOffline']) {
    const estimate = record(storage[name], 'Missing browser storage estimate.');
    assert.ok(
      typeof estimate.quota === 'number' &&
        Number.isSafeInteger(estimate.quota) &&
        estimate.quota >= requiredOfflineWeightBytes
    );
    assert.ok(typeof estimate.usage === 'number' && Number.isSafeInteger(estimate.usage) && estimate.usage >= 0);
  }
  const transport = record(data.checkpointTransport, 'Missing persistent checkpoint transport evidence.');
  assert.equal(transport.sourceUrl, checkpointUrl);
  assert.equal(transport.observation, independent ? 'independent-online-range-probe' : 'model-load-range');
  assert.equal(transport.wholeCheckpointOnlineDownloadVerified, false);
  assert.equal(
    transport.mode,
    mode === 'online' ? 'pinned-public-upstream-http-range' : 'verified-local-checkpoint-http-range'
  );
  const pins = new Map(
    inventory.chunks.map(([offset, length, sha256]) => [
      'bytes=' + offset + '-' + (offset + length - 1),
      {
        status: 206,
        contentRange: 'bytes ' + offset + '-' + (offset + length - 1) + '/' + inventory.bytes,
        bytes: length,
        sha256
      }
    ])
  );
  function assertRange(range: Record<string, unknown>) {
    const expected = pins.get(String(range.requestRange));
    assert.ok(expected, 'Checkpoint response did not use an installed range pin.');
    for (const [key, value] of Object.entries(expected))
      assert.equal(range[key], value, 'Observed checkpoint ' + key + ' differs.');
  }
  const observed = record(transport.firstObservedRange, 'Missing observed checkpoint response.');
  assert.equal(observed.url, checkpointUrl);
  const host = String(observed.host);
  assert.ok(host === 'huggingface.co' || host.endsWith('.huggingface.co') || host.endsWith('.hf.co'));
  assertRange(observed);
  if (independent) {
    const [offset, length] = inventory.chunks[0]!;
    assert.equal(observed.requestRange, 'bytes=' + offset + '-' + (offset + length - 1));
  }
  if (mode === 'online') {
    assert.equal(transport.localRoute, undefined, 'A local fixture does not satisfy the online mode.');
  } else {
    assert.equal(host, 'huggingface.co');
    const local = record(transport.localRoute, 'Missing local range audit.');
    assert.deepEqual(
      [
        local.mode,
        local.wholeCheckpointOnlineDownloadVerified,
        local.fileSha256,
        local.fileBytes,
        local.inventorySha256
      ],
      ['verified-local-checkpoint-http-range', false, inventory.sha256, inventory.bytes, inventorySha256]
    );
    assert.ok(Array.isArray(local.ranges) && local.ranges.length > 0);
    const used = new Set();
    for (const value of local.ranges) {
      const range = record(value, 'Invalid served checkpoint range.');
      assertRange(range);
      assert.ok(typeof range.count === 'number' && Number.isSafeInteger(range.count) && range.count > 0);
      assert.ok(!used.has(range.requestRange), 'Duplicate range audit entry.');
      used.add(range.requestRange);
    }
    assert.ok(used.has(observed.requestRange), 'Observed response is absent from the local range audit.');
  }
  return {storage, checkpointTransport: transport};
}
