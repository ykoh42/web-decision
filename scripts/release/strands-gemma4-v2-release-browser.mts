import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {
  gemmaV2PublicRunAttachment,
  gemmaV2PublicSpec,
  gemmaV2PublicTitle,
  gemmaV2PublicVectorsAttachment
} from '../../tests/browser/gemma-v2-public-evidence.mjs';
import {record} from './curated-gpu-evidence.mjs';

function only(items: unknown, description: string): Record<string, unknown> {
  assert.ok(Array.isArray(items) && items.length === 1, description);
  return record(items[0], description);
}

function attachment(items: unknown[], name: string, contentType: string): Buffer {
  const result = only(
    items.filter((item) => record(item, 'Invalid Playwright attachment.').name === name),
    name
  );
  assert.equal(result.contentType, contentType);
  assert.equal(result.path, undefined, 'Gemma evidence attachments must be embedded in the Playwright report.');
  assert.ok(typeof result.body === 'string' && result.body.length > 0, 'Missing Playwright attachment bytes: ' + name);
  const bytes = Buffer.from(result.body, 'base64');
  assert.equal(bytes.toString('base64'), result.body, 'Invalid embedded Playwright attachment: ' + name);
  return bytes;
}

/** Read the output of one passing, unretried Chromium run of the checked-in public-package spec. */
export function readGemmaV2PlaywrightRun(report: unknown, workspaceRoot: string) {
  const data = record(report, 'Expected the Playwright JSON run report.');
  assert.deepEqual(data.errors ?? [], [], 'Playwright reported errors.');
  const config = record(data.config, 'Missing Playwright configuration.');
  assert.equal(config.rootDir, resolve(workspaceRoot, 'tests/browser'));
  const stats = record(data.stats, 'Missing Playwright run statistics.');
  assert.deepEqual([stats.expected, stats.unexpected, stats.flaky, stats.skipped], [1, 0, 0, 0]);
  assert.ok(typeof stats.startTime === 'string' && Number.isFinite(Date.parse(stats.startTime)));
  const suite = only(data.suites, 'Use a Playwright run containing only the exact public Gemma spec.');
  assert.ok(!Array.isArray(suite.suites) || suite.suites.length === 0, 'Unexpected nested Playwright suites.');
  const spec = only(suite.specs, 'Missing the exact public Gemma test result.');
  assert.equal(spec.title, gemmaV2PublicTitle);
  assert.ok(typeof spec.file === 'string');
  assert.equal(resolve(config.rootDir, spec.file), resolve(workspaceRoot, gemmaV2PublicSpec));
  assert.equal(spec.ok, true);
  const test = only(spec.tests, 'Expected exactly one Chromium project.');
  assert.equal(test.projectName, 'chromium');
  assert.equal(test.expectedStatus, 'passed');
  assert.equal(test.status, 'expected');
  const run = only(test.results, 'Expected exactly one browser test execution with no retries.');
  assert.equal(run.retry, 0);
  assert.equal(run.status, 'passed');
  assert.deepEqual(run.errors ?? [], [], 'Chromium reported errors.');
  assert.ok(typeof run.duration === 'number' && Number.isFinite(run.duration) && run.duration > 0);
  const attachments = run.attachments;
  assert.ok(Array.isArray(attachments) && attachments.length === 2, 'Both browser-run files are required.');
  const runBytes = attachment(attachments, gemmaV2PublicRunAttachment, 'application/json');
  const vectors = attachment(attachments, gemmaV2PublicVectorsAttachment, 'application/x-ndjson');
  return {runBytes, vectors};
}
