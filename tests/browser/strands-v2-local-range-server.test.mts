import assert from 'node:assert/strict';
import {mkdtemp, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {test} from 'node:test';
import {startLocalRangeServer} from './strands-v2-local-range-server.mts';

test('local checkpoint server streams exact bounded ranges and denies other paths', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'web-decision-range-test-'));
  const path = join(directory, 'model.safetensors');
  await writeFile(
    path,
    Uint8Array.from({length: 64}, (_, index) => index)
  );
  const server = await startLocalRangeServer(new Map([['/checkpoint/model.safetensors', path]]));
  try {
    const response = await fetch(new URL('checkpoint/model.safetensors', server.base), {headers: {Range: 'bytes=5-9'}});
    assert.equal(response.status, 206);
    assert.equal(response.headers.get('content-range'), 'bytes 5-9/64');
    assert.deepEqual(Array.from(new Uint8Array(await response.arrayBuffer())), [5, 6, 7, 8, 9]);
    assert.equal(
      (await fetch(new URL('checkpoint/model.safetensors', server.base), {headers: {Range: 'bytes=62-64'}})).status,
      416
    );
    assert.equal((await fetch(new URL('other', server.base))).status, 404);
    assert.deepEqual(server.stats(), {requests: 1, bytes: 5});
  } finally {
    await server.close();
    await rm(directory, {recursive: true, force: true});
  }
});
