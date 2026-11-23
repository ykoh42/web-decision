import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {dirname, join} from 'node:path';
import type {BrowserContext} from '@playwright/test';
import {sha256} from '../../scripts/release/curated-gpu-evidence.mjs';

export const gemmaV2ArchiveTokenizerSha256 = '1c4a848f86156201c033d6cd08c4cf81ae1088552acec3367f31ef086edf5af0';
const rootPath = '/__gemma_public_archive/';
const contentTypes: Record<string, string> = {
  '.js': 'text/javascript',
  '.json': 'application/json',
  '.bin': 'application/octet-stream',
  '.wasm': 'application/wasm'
};
const html = `<!doctype html>
<link rel="icon" href="data:,">
<script type="importmap">{"imports":{"@huggingface/tokenizers":"${rootPath}tokenizers.mjs"}}</script>
<script type="module">
import * as api from "./dist/index.js";
window.webDecision = api;
</script>`;

export async function verifyGemmaArchiveResources(
  installedRoot: string,
  archiveFiles: readonly string[],
  observed: Record<string, unknown>,
  required: readonly string[]
) {
  for (const entry of required)
    assert.ok(Object.hasOwn(observed, entry), 'Missing served Gemma archive resource: ' + entry);
  for (const [path, digest] of Object.entries(observed)) {
    assert.ok(
      path.startsWith('dist/') && archiveFiles.includes(path),
      'Gemma served an asset outside the archive: ' + path
    );
    assert.equal(
      digest,
      sha256(await readFile(join(installedRoot, path))),
      'Served Gemma bytes differ from the archive: ' + path
    );
  }
}

export async function routeGemmaExactArchive(
  context: BrowserContext,
  origin: string,
  installedRoot: string,
  archiveFiles: readonly string[]
) {
  const tokenizerPath = join(dirname(installedRoot), '@huggingface/tokenizers/dist/tokenizers.mjs');
  const tokenizer = await readFile(tokenizerPath);
  assert.equal(sha256(tokenizer), gemmaV2ArchiveTokenizerSha256, 'Installed Gemma tokenizer bytes differ.');
  assert.ok(archiveFiles.includes('dist/index.js'), 'Missing exact public bundle entry.');
  const paths = new Map(archiveFiles.filter((path) => path.startsWith('dist/')).map((path) => [rootPath + path, path]));
  const served = new Map<string, string>();
  let tokenizerServed = false;
  await context.route(origin + '/**', async (route) => {
    const url = new URL(route.request().url());
    if (route.request().method() !== 'GET' || url.search || url.hash) return route.abort();
    if (url.pathname === rootPath) return route.fulfill({contentType: 'text/html', body: html});
    if (url.pathname === rootPath + 'tokenizers.mjs') {
      tokenizerServed = true;
      return route.fulfill({contentType: 'text/javascript', body: tokenizer});
    }
    const asset = paths.get(url.pathname);
    const contentType = asset && contentTypes[asset.slice(asset.lastIndexOf('.'))];
    if (!asset || !contentType) return route.abort();
    const bytes = await readFile(join(installedRoot, asset));
    const digest = sha256(bytes);
    if (served.has(asset))
      assert.equal(digest, served.get(asset), 'Served archive resource changed during the run: ' + asset);
    served.set(asset, digest);
    return route.fulfill({contentType, body: bytes});
  });
  return {
    url: (path = '') => new URL(rootPath + path, origin).href,
    evidence: () => ({
      served: Object.fromEntries([...served].sort(([a], [b]) => a.localeCompare(b))),
      tokenizerSha256: gemmaV2ArchiveTokenizerSha256,
      tokenizerServed
    })
  };
}
