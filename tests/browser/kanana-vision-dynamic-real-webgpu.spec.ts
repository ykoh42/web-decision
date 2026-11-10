import {createReadStream} from 'node:fs';
import {stat} from 'node:fs/promises';
import {createServer, type IncomingMessage, type ServerResponse} from 'node:http';
import {extname, resolve, sep} from 'node:path';
import {expect, test} from '@playwright/test';

const enabled = process.env.WEB_DECISION_KANANA_DYNAMIC_VISION === '1';
const assetsDirectory = resolve(process.env.WEB_DECISION_KANANA_DYNAMIC_VISION_ASSETS ?? '.');
const checkpointPath = resolve(process.env.WEB_DECISION_KANANA_CHECKPOINT ?? '.');

function serveAsset(request: IncomingMessage, response: ServerResponse) {
  response.setHeader('Access-Control-Allow-Origin', '*');
  response.setHeader('Access-Control-Allow-Headers', 'Range');
  response.setHeader('Access-Control-Expose-Headers', 'Content-Length, Content-Range');
  if (request.method === 'OPTIONS') {
    response.writeHead(204).end();
    return;
  }

  void (async () => {
    const requestUrl = new URL(request.url ?? '/', 'http://127.0.0.1');
    const relativePath = decodeURIComponent(requestUrl.pathname.slice('/assets/'.length));
    const filePath = relativePath === 'model.safetensors' ? checkpointPath : resolve(assetsDirectory, relativePath);
    if (filePath !== checkpointPath && !filePath.startsWith(`${assetsDirectory}${sep}`)) {
      response.writeHead(404).end();
      return;
    }
    const info = await stat(filePath);
    if (!info.isFile()) {
      response.writeHead(404).end();
      return;
    }
    const range = request.headers.range;
    let start = 0;
    let end = info.size - 1;
    let status = 200;
    if (range) {
      const match = /^bytes=(\d+)-(\d+)$/.exec(range);
      if (!match) {
        response.writeHead(416, {'Content-Range': `bytes */${info.size}`}).end();
        return;
      }
      start = Number(match[1]);
      end = Number(match[2]);
      if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || end >= info.size) {
        response.writeHead(416, {'Content-Range': `bytes */${info.size}`}).end();
        return;
      }
      status = 206;
    }
    const headers: Record<string, string | number> = {
      'Accept-Ranges': 'bytes',
      'Content-Length': end - start + 1,
      'Content-Type': extname(filePath) === '.json' ? 'application/json' : 'application/octet-stream'
    };
    if (status === 206) headers['Content-Range'] = `bytes ${start}-${end}/${info.size}`;
    response.writeHead(status, headers);
    if (request.method === 'HEAD') {
      response.end();
      return;
    }
    createReadStream(filePath, {start, end})
      .on('error', () => response.destroy())
      .pipe(response);
  })().catch((error: unknown) => {
    if (!response.headersSent) response.writeHead(500);
    response.end(error instanceof Error ? error.message : String(error));
  });
}

test('runs dynamic Kanana vision inputs through the real TVM WebGPU graph', async ({page, browser}) => {
  test.skip(
    !enabled || !process.env.WEB_DECISION_KANANA_DYNAMIC_VISION_ASSETS || !process.env.WEB_DECISION_KANANA_CHECKPOINT,
    'Set the local Kanana dynamic vision graph and checkpoint paths to run the real model.'
  );
  test.setTimeout(45 * 60 * 1000);
  const server = createServer(serveAsset);
  await new Promise<void>((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolveListen);
  });
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Kanana vision asset server did not start.');
  const assetsBase = `http://127.0.0.1:${address.port}/assets/`;
  const progress: Record<string, unknown>[] = [];

  try {
    await page.exposeFunction('__kananaVisionProgress', (event: Record<string, unknown>) => {
      progress.push(event);
      console.log(`KANANA_VISION_PROGRESS ${JSON.stringify(event)}`);
    });
    await page.goto('/');
    await page.addScriptTag({path: resolve('.toolchains/apache-tvm-src/web/lib/index.js')});
    const result = await page.evaluate(
      async ({assetsBase}) => {
        const runnerUrl = new URL('/kanana-vision-dynamic-real-webgpu.ts', location.href).href;
        const runner = await import(/* @vite-ignore */ runnerUrl);
        return runner.runKananaDynamicVisionRealWebGpu(assetsBase);
      },
      {assetsBase}
    );
    console.log(`KANANA_DYNAMIC_VISION_RESULT ${JSON.stringify({browser: browser.version(), result, progress})}`);
    expect(result.cases.map((item: {name: string}) => item.name)).toEqual(['minimum-256', 'middle-384', 'large-512']);
    expect(result.cases.map((item: {shape: number[]}) => item.shape)).toEqual([
      [64, 2560],
      [96, 2560],
      [128, 2560]
    ]);
    for (const item of result.cases) {
      expect(item.maxAbsDelta).toBeLessThan(0.02);
      expect(item.rmse).toBeLessThan(0.002);
    }
    expect(result.webgpuErrors).toEqual([]);
  } finally {
    await new Promise<void>((resolveClose, reject) =>
      server.close((error) => (error ? reject(error) : resolveClose()))
    );
  }
});
