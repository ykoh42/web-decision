import {createReadStream} from 'node:fs';
import {stat} from 'node:fs/promises';
import {createServer} from 'node:http';
import type {AddressInfo} from 'node:net';

/** Serve an explicit local file map and exact HTTP ranges without copying checkpoints. */
export async function startLocalRangeServer(files: ReadonlyMap<string, string>) {
  const allowed = new Map(files);
  let count = 0;
  let bytes = 0;
  const server = createServer(async (request, response) => {
    response.setHeader('Access-Control-Allow-Origin', '*');
    response.setHeader('Access-Control-Allow-Headers', 'Range');
    response.setHeader('Access-Control-Expose-Headers', 'Content-Range, Content-Length');
    if (request.method === 'OPTIONS') {
      response.writeHead(204).end();
      return;
    }
    if (request.method !== 'GET') {
      response.writeHead(405).end();
      return;
    }
    const pathname = new URL(request.url ?? '/', 'http://localhost').pathname;
    const path = allowed.get(pathname);
    if (!path) {
      response.writeHead(404).end();
      return;
    }
    try {
      const length = (await stat(path)).size;
      const range = request.headers.range;
      let start = 0;
      let end = length - 1;
      if (range !== undefined) {
        const match = /^bytes=(\d+)-(\d+)$/u.exec(range);
        if (!match) {
          response.writeHead(416).end();
          return;
        }
        start = Number(match[1]);
        end = Number(match[2]);
        if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start || end >= length) {
          response.writeHead(416, {'Content-Range': `bytes */${length}`}).end();
          return;
        }
        response.setHeader('Content-Range', `bytes ${start}-${end}/${length}`);
      }
      const size = end - start + 1;
      response.setHeader('Content-Length', String(size));
      response.setHeader('Content-Type', 'application/octet-stream');
      response.setHeader('Accept-Ranges', 'bytes');
      response.writeHead(range === undefined ? 200 : 206);
      count++;
      bytes += size;
      const stream = createReadStream(path, {start, end});
      stream.on('error', () => response.destroy());
      response.on('close', () => stream.destroy());
      stream.pipe(response);
    } catch {
      if (!response.headersSent) response.writeHead(500).end();
      else response.destroy();
    }
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      server.off('error', reject);
      resolve();
    });
  });
  const port = (server.address() as AddressInfo).port;
  return {
    base: `http://127.0.0.1:${port}/`,
    stats: () => ({requests: count, bytes}),
    async close() {
      await new Promise<void>((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
    }
  };
}
