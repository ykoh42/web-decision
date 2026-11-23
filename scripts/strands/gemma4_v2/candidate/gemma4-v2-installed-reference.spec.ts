import {createHash} from 'node:crypto';
import {open, readFile, realpath, type FileHandle} from 'node:fs/promises';
import {dirname, resolve} from 'node:path';
import {expect, test, type Route} from '@playwright/test';
import {readAdapter, readHardwareReport} from '../../../../tests/browser/mapika2b-hardware.mts';
import {verifyArchiveInstall} from '../../../release/verify-release-evidence.ts';

const modelId = 'strands-decider-e2b-gemma4-v2-2611';
const upstream =
  'https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v2-2611/resolve/a5f32ea1bffe2e91c25c0c641d444e8115e28212/';
const names = new Set([
  'config.json',
  'decision_config.json',
  'processor_config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'readout.safetensors'
]);
const sha256 = (data: Uint8Array) => createHash('sha256').update(data).digest('hex');

async function serveRange(route: Route, checkpoint: FileHandle) {
  const match = /^bytes=(\d+)-(\d+)$/u.exec(route.request().headers().range ?? '');
  if (!match) throw new Error('Missing checkpoint byte range.');
  const start = Number(match[1]);
  const end = Number(match[2]);
  const data = Buffer.alloc(end - start + 1);
  if ((await checkpoint.read(data, 0, data.length, start)).bytesRead !== data.length)
    throw new Error('Short checkpoint read.');
  await route.fulfill({
    status: 206,
    headers: {
      'content-range': `bytes ${start}-${end}/10246621918`,
      'access-control-allow-origin': '*',
      'access-control-expose-headers': '*'
    },
    body: data
  });
}

function compare(
  reference: unknown,
  actual: unknown,
  path: string,
  differences: Array<{path: string; reference: number; actual: number; delta: number}>
): void {
  if (typeof reference === 'number') {
    expect(typeof actual, path).toBe('number');
    expect(Number.isFinite(actual), path).toBe(true);
    differences.push({path, reference, actual: actual as number, delta: Math.abs(reference - (actual as number))});
  } else if (reference && typeof reference === 'object') {
    expect(actual, path).not.toBeNull();
    expect(typeof actual, path).toBe('object');
    for (const [key, value] of Object.entries(reference))
      compare(value, (actual as Record<string, unknown>)[key], `${path}.${key}`, differences);
  } else expect(actual, path).toEqual(reference);
}

test('installed archive collects one pinned upstream request', async ({page, browser}) => {
  test.skip(process.env.WEB_DECISION_GEMMA_V2_RELEASE_REFERENCE !== '1', 'Enable the installed archive GPU test.');
  const archive = process.env.WEB_DECISION_RELEASE_ARCHIVE;
  const archiveHash = process.env.WEB_DECISION_RELEASE_ARCHIVE_SHA256;
  const bundle = process.env.WEB_DECISION_INSTALLED_BUNDLE;
  const requestPath = process.env.WEB_DECISION_REFERENCE_REQUEST;
  const referencePath = process.env.WEB_DECISION_REFERENCE_RESULT;
  const timeout = Number(process.env.WEB_DECISION_LONG_TIMEOUT_MS ?? '1800000');
  const toleranceText = process.env.WEB_DECISION_REFERENCE_TOLERANCE;
  const tolerance = toleranceText === undefined ? null : Number(toleranceText);
  if (!archive || !bundle || !requestPath || !referencePath || !/^[0-9a-f]{64}$/u.test(archiveHash ?? ''))
    throw new Error('Set the archive, archive SHA-256, installed bundle, request, and reference result paths.');
  if (!Number.isSafeInteger(timeout) || timeout < 1 || timeout > 2_147_483_647) throw new Error('Invalid timeout.');
  if (tolerance !== null && (!Number.isFinite(tolerance) || tolerance < 0))
    throw new Error('Invalid reference tolerance.');
  test.setTimeout(timeout);
  const selectedBundle = await realpath(bundle);
  const installed = resolve(dirname(selectedBundle), '..');
  const evidence = await verifyArchiveInstall(archive, installed);
  expect(evidence.archiveSha256).toBe(archiveHash);
  expect(await realpath(evidence.bundlePath)).toBe(selectedBundle);
  const requestBytes = await readFile(requestPath);
  const referenceBytes = await readFile(referencePath);
  const request = JSON.parse(requestBytes.toString('utf8'));
  const reference = JSON.parse(referenceBytes.toString('utf8'));
  const checkpointRoot = resolve(process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT!, 'strands-e2b-v2');
  const checkpoint = await open(resolve(checkpointRoot, 'model.safetensors'), 'r');
  try {
    await page.route(`${upstream}**`, async (route) => {
      const name = new URL(route.request().url()).pathname.split('/').at(-1)!;
      if (name === 'model.safetensors') return serveRange(route, checkpoint);
      if (!names.has(name)) throw new Error(`Unexpected upstream asset: ${name}`);
      await route.fulfill({
        status: 200,
        headers: {'access-control-allow-origin': '*'},
        body: await readFile(resolve(checkpointRoot, name))
      });
    });
    const hardware = await readHardwareReport(page);
    expect(hardware.status).toBe('Hardware accelerated');
    await page.goto('/');
    await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');
    const adapter = await readAdapter(page);
    expect(adapter?.isFallbackAdapter).toBe(false);
    const browserEvidence = {name: browser.browserType().name(), version: browser.version()};
    const actual = await page.evaluate(
      async ({id, input}) => {
        const api = window.webDecision as unknown as {
          loadDecisionModel(
            options: unknown
          ): Promise<{systemOne(request: unknown): Promise<unknown>; dispose(): Promise<void>}>;
        };
        const model = await api.loadDecisionModel({modelId: id, cache: false});
        try {
          return await model.systemOne(input);
        } finally {
          await model.dispose();
        }
      },
      {id: modelId, input: request}
    );
    console.log(
      `INSTALLED_REFERENCE_RAW ${JSON.stringify({archiveSha256: evidence.archiveSha256, bundleSha256: evidence.bundleSha256, packageVersion: evidence.packageVersion, requestSha256: sha256(requestBytes), referenceSha256: sha256(referenceBytes), browser: browserEvidence, hardware, adapter, reference, actual})}`
    );
    const differences: Array<{path: string; reference: number; actual: number; delta: number}> = [];
    const parsed = actual as {answers: unknown; usage: unknown};
    compare(reference.answers, parsed.answers, 'answers', differences);
    compare(reference.usage, parsed.usage, 'usage', differences);
    const maxDelta = Math.max(0, ...differences.map((item) => item.delta));
    console.log(
      `INSTALLED_REFERENCE ${JSON.stringify({archiveSha256: evidence.archiveSha256, bundleSha256: evidence.bundleSha256, packageVersion: evidence.packageVersion, requestSha256: sha256(requestBytes), referenceSha256: sha256(referenceBytes), reference, actual, differences, maxDelta, tolerance})}`
    );
    expect(parsed.usage).toEqual(reference.usage);
    if (tolerance !== null) expect(maxDelta).toBeLessThanOrEqual(tolerance);
    const after = await verifyArchiveInstall(archive, installed);
    expect(after.archiveSha256).toBe(evidence.archiveSha256);
    expect(after.packageTreeSha256).toBe(evidence.packageTreeSha256);
    expect(sha256(await readFile(requestPath))).toBe(sha256(requestBytes));
    expect(sha256(await readFile(referencePath))).toBe(sha256(referenceBytes));
  } finally {
    await checkpoint.close();
  }
});
