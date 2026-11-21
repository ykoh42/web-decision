import {open} from 'node:fs/promises';
import {join} from 'node:path';
import {setTimeout} from 'node:timers/promises';
import type {Page, Route} from '@playwright/test';
import {MAPIKA2B_REPOSITORY, MAPIKA2B_REVISION} from '../../packages/web-decision/src/mapika2b/contract.mts';

const source = `https://huggingface.co/${MAPIKA2B_REPOSITORY}/resolve/${MAPIKA2B_REVISION}/`;
const maxRangeBytes = 8 * 1024 * 1024;
const visionWeights = 'vision/visual.safetensors';
const visionPath = new URL(visionWeights, source).pathname;
const weightFiles = ['model.safetensors', visionWeights];
const corsHeaders = {
  'access-control-allow-origin': '*',
  'access-control-allow-headers': 'range',
  'access-control-expose-headers': 'content-length, content-range'
};

async function serveRange(route: Route, filePath: string, bytes: string): Promise<number> {
  const range = /^bytes=(\d+)-(\d+)$/.exec(bytes);
  if (!range) throw new Error('Expected a bounded Mapika snapshot byte range.');
  const start = Number(range[1]);
  const end = Number(range[2]);
  const file = await open(filePath, 'r');
  try {
    const {size} = await file.stat();
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || end >= size)
      throw new RangeError('Range is outside the Mapika snapshot.');
    const length = end - start + 1;
    if (length > maxRangeBytes) throw new RangeError('Mapika snapshot read exceeds the verified range limit.');
    const body = Buffer.alloc(length);
    const {bytesRead} = await file.read(body, 0, length, start);
    if (bytesRead !== length) throw new Error('Could not read the complete Mapika snapshot byte range.');
    await route.fulfill({
      status: 206,
      headers: {
        ...corsHeaders,
        'content-type': 'application/octet-stream',
        'content-range': `bytes ${start}-${end}/${size}`,
        'content-length': String(length)
      },
      body
    });
    return length;
  } finally {
    await file.close();
  }
}

export async function routeMapikaSnapshot(
  page: Page,
  root: string,
  log: (message: string) => void,
  {visionDelayMs = 0, liveVision = false}: {visionDelayMs?: number; liveVision?: boolean} = {}
): Promise<void> {
  await page.route(
    (url) => url.href.startsWith(source) && (!liveVision || url.pathname !== visionPath),
    async (route) => {
      const key = new URL(route.request().url()).pathname.slice(new URL(source).pathname.length);
      if (!key.split('/').every((piece) => /^[\w-]+\.(?:json|safetensors)$|^vision$/.test(piece))) {
        await route.abort();
        return;
      }
      if (route.request().method() === 'OPTIONS') {
        await route.fulfill({status: 204, headers: corsHeaders});
        return;
      }
      const path = join(root, key);
      const range = route.request().headers().range;
      try {
        if (range) {
          if (visionDelayMs && key === visionWeights) await setTimeout(visionDelayMs);
          await serveRange(route, path, range);
        } else if (weightFiles.includes(key)) {
          throw new Error('The candidate requested an entire Mapika checkpoint instead of a verified range.');
        } else {
          await route.fulfill({path, headers: corsHeaders});
        }
      } catch (cause) {
        log(`Snapshot response failed for ${key}: ${String(cause)}`);
        await route.abort();
      }
    }
  );
}
