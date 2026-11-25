import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {open, readFile, type FileHandle} from 'node:fs/promises';
import {join} from 'node:path';
import type {BrowserContext, Route} from '@playwright/test';

export const checkpointUrl =
  'https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v2-2611/resolve/a5f32ea1bffe2e91c25c0c641d444e8115e28212/model.safetensors';
export const inventorySha256 = 'c4c1f9ca91d272d88f2559dd186fa0d37e878fa70d378c44a20de978179366aa';
export const requiredOfflineWeightBytes = 9_638_447_838;

type Chunk = readonly [offset: number, length: number, sha256: string];
type Inventory = {file: string; bytes: number; sha256: string; chunks: readonly Chunk[]};
const sha = (data: Uint8Array) => createHash('sha256').update(data).digest('hex');
const formatRange = ([offset, length]: Chunk) => 'bytes=' + offset + '-' + (offset + length - 1);

export async function checkedInventory(installedRoot: string): Promise<Inventory> {
  const bytes = await readFile(join(installedRoot, 'assets/strands-gemma4-v2/source-inventory.json'));
  assert.equal(sha(bytes), inventorySha256, 'Installed checkpoint range manifest differs.');
  const inventory = JSON.parse(bytes.toString('utf8')) as Inventory;
  assert.equal(inventory.file, 'model.safetensors');
  assert.equal(inventory.bytes, 10_246_621_918);
  assert.equal(inventory.sha256, '593a28a67754653c7ffffff80a5186f18966cc6262c2f4f097414c187e5fd48c');
  return inventory;
}

export async function verifyCheckpointFile(handle: FileHandle, expected: Pick<Inventory, 'bytes' | 'sha256'>) {
  const before = await handle.stat({bigint: true});
  assert.ok(before.isFile(), 'The local checkpoint must be a regular file.');
  assert.equal(before.size, BigInt(expected.bytes), 'Pinned checkpoint file size differs.');
  const sha256 = createHash('sha256');
  for await (const part of handle.createReadStream({autoClose: false, start: 0})) sha256.update(part);
  assert.equal(sha256.digest('hex'), expected.sha256, 'Pinned checkpoint file SHA-256 differs.');
  const after = await handle.stat({bigint: true});
  assert.deepEqual(
    [before.size, before.ino, before.mtimeNs, before.ctimeNs],
    [after.size, after.ino, after.mtimeNs, after.ctimeNs],
    'The local checkpoint changed during full SHA-256 verification.'
  );
}

export function checkpointResponses(inventory: Inventory, read: (offset: number, length: number) => Promise<Buffer>) {
  const chunks = new Map(inventory.chunks.map((chunk) => [formatRange(chunk), chunk]));
  const audit = new Map<
    string,
    {requestRange: string; contentRange: string; status: number; bytes: number; sha256: string; count: number}
  >();
  return {
    async serve(route: Route) {
      try {
        const request = route.request();
        assert.equal(request.method(), 'GET');
        const range = request.headers().range ?? '';
        const chunk = chunks.get(range);
        assert.ok(chunk, 'Expected one exact pinned checkpoint range.');
        const [offset, length, digest] = chunk;
        const bytes = await read(offset, length);
        assert.equal(bytes.byteLength, length, 'Short local checkpoint range.');
        assert.equal(sha(bytes), digest, 'Local checkpoint range SHA-256 differs.');
        const contentRange = 'bytes ' + offset + '-' + (offset + length - 1) + '/' + inventory.bytes;
        await route.fulfill({
          status: 206,
          headers: {
            'content-type': 'application/octet-stream',
            'accept-ranges': 'bytes',
            'content-range': contentRange,
            'content-length': String(length),
            'access-control-allow-origin': '*',
            'access-control-expose-headers': 'Content-Range, Content-Length'
          },
          body: bytes
        });
        const entry = audit.get(range);
        if (entry) entry.count++;
        else
          audit.set(range, {requestRange: range, contentRange, status: 206, bytes: length, sha256: digest, count: 1});
      } catch (error) {
        await route.abort('failed').catch(() => {});
        throw error;
      }
    },
    evidence() {
      return [...audit.values()].sort((a, b) => a.requestRange.localeCompare(b.requestRange));
    }
  };
}

export async function routeVerifiedLocalCheckpoint(context: BrowserContext, path: string, inventory: Inventory) {
  const handle = await open(path, 'r');
  try {
    await verifyCheckpointFile(handle, inventory);
    const responder = checkpointResponses(inventory, async (offset, length) => {
      const buffer = Buffer.alloc(length);
      let size = 0;
      while (size < length) {
        const {bytesRead} = await handle.read(buffer, size, length - size, offset + size);
        if (bytesRead === 0) break;
        size += bytesRead;
      }
      return buffer.subarray(0, size);
    });
    const handler = (route: Route) => responder.serve(route);
    await context.route(checkpointUrl, handler);
    return {
      evidence() {
        const ranges = responder.evidence();
        assert.ok(ranges.length, 'No pinned local range was served.');
        return {
          mode: 'verified-local-checkpoint-http-range',
          wholeCheckpointOnlineDownloadVerified: false,
          fileSha256: inventory.sha256,
          fileBytes: inventory.bytes,
          inventorySha256,
          ranges
        };
      },
      async close() {
        try {
          await context.unroute(checkpointUrl, handler);
        } finally {
          await handle.close();
        }
      }
    };
  } catch (error) {
    await handle.close();
    throw error;
  }
}

/** Bind the first observed browser response to the installed inventory without logging redirects. */
export async function observedCheckpointRange(response: import('@playwright/test').Response, inventory: Inventory) {
  assert.equal(response.status(), 206);
  let request = response.request();
  while (request.redirectedFrom()) request = request.redirectedFrom()!;
  assert.equal(request.url(), checkpointUrl);
  const requestRange = response.request().headers().range ?? '';
  const chunk = inventory.chunks.find((item) => formatRange(item) === requestRange);
  assert.ok(chunk, 'Browser checkpoint request did not use a pinned range.');
  const [offset, bytes, digest] = chunk;
  const contentRange = 'bytes ' + offset + '-' + (offset + bytes - 1) + '/' + inventory.bytes;
  assert.equal(response.headers()['content-range'], contentRange);
  const body = await response.body();
  assert.equal(body.byteLength, bytes);
  assert.equal(sha(body), digest, 'Observed browser checkpoint body hash differs.');
  return {
    url: checkpointUrl,
    host: new URL(response.url()).hostname,
    requestRange,
    status: 206,
    contentRange,
    bytes,
    sha256: digest
  };
}
