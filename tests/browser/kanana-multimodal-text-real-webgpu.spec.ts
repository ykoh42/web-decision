import {createReadStream} from 'node:fs';
import {stat} from 'node:fs/promises';
import {createServer, type IncomingMessage, type ServerResponse} from 'node:http';
import {extname, resolve, sep} from 'node:path';
import {expect, test} from '@playwright/test';

const enabled = process.env.WEB_DECISION_KANANA_MULTIMODAL_TEXT === '1';
const checkpointOneLayer = process.env.WEB_DECISION_KANANA_MULTIMODAL_TEXT_CHECKPOINT === '1';
const checkpointFull = process.env.WEB_DECISION_KANANA_MULTIMODAL_TEXT_FULL === '1';
const assetsDirectory = resolve(process.env.WEB_DECISION_KANANA_MULTIMODAL_TEXT_ASSETS ?? '.');
const checkpointPath = process.env.WEB_DECISION_KANANA_CHECKPOINT
  ? resolve(process.env.WEB_DECISION_KANANA_CHECKPOINT)
  : undefined;

type MultimodalTextResult = Readonly<{
  classification: string;
  shape: readonly number[];
  maxAbsDelta: number;
  rmse: number;
  dynamicCases: readonly Readonly<{sequenceLength: number}>[];
  decisionScoreParity?: Readonly<{
    withinOutputTolerance: boolean;
    maxAbsScoreDelta: number;
    maxAbsProbabilityDelta: number;
    maxAbsAnswerDelta: number;
    maxAbsConfidenceDelta: number;
  }>;
  webgpuErrors: readonly unknown[];
}>;

function expectedModelValue<T>(fullCheckpoint: T, oneLayer: T, reduced: T): T {
  if (checkpointFull) return fullCheckpoint;
  if (checkpointOneLayer) return oneLayer;
  return reduced;
}

function assertModelResult(result: MultimodalTextResult): void {
  expect(result.classification).toContain(
    expectedModelValue('pinned-checkpoint full 32-layer', 'pinned-checkpoint one-text-layer', 'reduced random-init')
  );
  expect(result.shape).toEqual(expectedModelValue([1, 8, 2560], [1, 8, 2560], [1, 8, 32]));
  expect(result.maxAbsDelta).toBeLessThan(expectedModelValue(0.12, 0.05, 0.03));
  expect(result.rmse).toBeLessThan(expectedModelValue(0.02, 0.008, 0.005));
  expect(result.dynamicCases.map(({sequenceLength}) => sequenceLength)).toEqual(
    expectedModelValue([1, 8, 9], [1, 8, 9, 33, 64, 128], [1, 8, 9, 33, 64])
  );
  if (checkpointFull) {
    const parity = result.decisionScoreParity;
    if (!parity) throw new Error('Full-checkpoint result has no decision-score parity report.');
    expect(parity.withinOutputTolerance).toBe(true);
    expect(parity.maxAbsScoreDelta).toBeLessThan(0.3);
    expect(parity.maxAbsProbabilityDelta).toBeLessThan(0.002);
    expect(parity.maxAbsAnswerDelta).toBeLessThan(0.003);
    expect(parity.maxAbsConfidenceDelta).toBeLessThan(0.002);
  }
  expect(result.webgpuErrors).toEqual([]);
}

function serveAsset(request: IncomingMessage, response: ServerResponse) {
  response.setHeader('Access-Control-Allow-Origin', '*');
  response.setHeader('Access-Control-Allow-Headers', 'Range');
  response.setHeader('Access-Control-Expose-Headers', 'Accept-Ranges, Content-Length, Content-Range');
  if (request.method === 'OPTIONS') {
    response.writeHead(204).end();
    return;
  }

  const relativePath = decodeURIComponent(
    new URL(request.url ?? '/', 'http://127.0.0.1').pathname.slice('/assets/'.length)
  );
  const filePath =
    relativePath === 'model.safetensors' && checkpointPath ? checkpointPath : resolve(assetsDirectory, relativePath);
  if (filePath !== checkpointPath && !filePath.startsWith(`${assetsDirectory}${sep}`)) {
    response.writeHead(404).end();
    return;
  }
  void (async () => {
    const info = await stat(filePath);
    if (!info.isFile()) {
      response.writeHead(404).end();
      return;
    }

    let start = 0;
    let end = info.size - 1;
    let status = 200;
    const range = request.headers.range;
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
  })().catch(() => {
    if (!response.headersSent) response.writeHead(404);
    response.end();
  });
}

test('runs the Kanana multimodal text graph through its real TVM WebGPU session', async ({page, browser}) => {
  test.skip(
    !enabled || !process.env.WEB_DECISION_KANANA_MULTIMODAL_TEXT_ASSETS || (checkpointFull && !checkpointPath),
    'Set the local Kanana graph path and, for the pinned full checkpoint, its Safetensors path.'
  );
  test.setTimeout(checkpointFull ? 45 * 60 * 1000 : 10 * 60 * 1000);
  const server = createServer(serveAsset);
  await new Promise<void>((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolveListen);
  });
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Kanana text graph server did not start.');
  const assetsBase = `http://127.0.0.1:${address.port}/assets/`;

  try {
    await page.goto('/');
    await page.addScriptTag({path: resolve('.toolchains/apache-tvm-src/web/lib/index.js')});
    const result: MultimodalTextResult = await page.evaluate(
      async ({assetsBase}) => {
        const runnerUrl = new URL('/kanana-multimodal-text-real-webgpu.ts', location.href).href;
        const runner = await import(/* @vite-ignore */ runnerUrl);
        return runner.runKananaMultimodalTextRealWebGpu(assetsBase);
      },
      {assetsBase}
    );
    console.log(`KANANA_MULTIMODAL_TEXT_WEBGPU ${JSON.stringify({browser: browser.version(), result})}`);
    assertModelResult(result);
  } finally {
    await new Promise<void>((resolveClose, reject) =>
      server.close((error) => (error ? reject(error) : resolveClose()))
    );
  }
});
