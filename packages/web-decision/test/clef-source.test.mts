import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {indexClefInventory, openClefSource} from '../src/clef/source.mjs';
import {readClefRange} from '../src/clef/range.mjs';
import {CLEF_INVENTORY_SHA256} from '../src/clef/contract.mjs';
import type {ClefRangePart, ClefTensorEntry} from '../src/clef/types.js';

interface MutableInventoryFixture {
  parameters: Array<{position: number; path: string; chunks: ClefRangePart[]}>;
  inputBanks: Array<{rows: number}>;
}

const hash = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex');
const inventoryPath = new URL('./fixtures/clef-inventory.json', import.meta.url);
const inventorySkip = existsSync(inventoryPath) ? false : 'The local CLEF inventory fixture is not included.';

test(
  'pinned inventory has 797 ordered fixed entries and rejects metadata corruption',
  {skip: inventorySkip},
  async () => {
    const bytes = await readFile(inventoryPath);
    assert.equal(hash(bytes), CLEF_INVENTORY_SHA256);
    const value = JSON.parse(bytes as unknown as string) as MutableInventoryFixture,
      parsed = indexClefInventory(value);
    assert.equal(parsed.entries.length, 797);
    assert.equal(parsed.inputBanks.length, 4);
    assert.equal(parsed.outputBanks.length, 4);
    for (const mutate of [
      (v: MutableInventoryFixture) => {
        v.parameters[0]!.position++;
      },
      (v: MutableInventoryFixture) => {
        v.parameters[0]!.chunks[0]!.sha256 = 'x';
      },
      (v: MutableInventoryFixture) => {
        v.parameters[1]!.path = v.parameters[0]!.path;
      },
      (v: MutableInventoryFixture) => {
        v.inputBanks[0]!.rows--;
      },
      (v: MutableInventoryFixture) => {
        v.parameters[0]!.chunks[0]!.offset++;
      }
    ]) {
      const altered = structuredClone(value);
      mutate(altered);
      assert.throws(() => indexClefInventory(altered), {code: 'INVALID_ASSET'});
    }
  }
);

test('range request enforces response geometry, identity encoding and authenticated bytes', async (t) => {
  const saved = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = saved;
  });
  const data = Uint8Array.of(1, 2, 3, 4),
    part: ClefRangePart = {offset: 4, length: 4, sha256: hash(data)},
    entry: ClefTensorEntry = {path: 'source/t0001.q4', bytes: 12} as ClefTensorEntry,
    base = new URL('https://example.test/assets/');
  let sent: {url: string; options: RequestInit & {headers: {Range: string}}} | undefined;
  globalThis.fetch = async (url, options) => {
    sent = {
      url: String(url),
      options
    } as {url: string; options: RequestInit & {headers: {Range: string}}};
    return new Response(data, {status: 206, headers: {'Content-Range': 'bytes 4-7/12', 'Content-Length': '4'}});
  };
  assert.deepEqual(new Uint8Array(await readClefRange(base, entry, part)), data);
  assert.equal(sent!.options.headers.Range, 'bytes=4-7');
  assert.equal(sent!.options.redirect, 'error');
  for (const response of [
    () => new Response(data, {status: 200}),
    () => new Response(data, {status: 206, headers: {'Content-Range': 'bytes 0-3/12'}}),
    () => new Response(data, {status: 206, headers: {'Content-Range': 'bytes 4-7/12', 'Content-Encoding': 'gzip'}}),
    () => new Response(Uint8Array.of(1, 2, 3), {status: 206, headers: {'Content-Range': 'bytes 4-7/12'}}),
    () => new Response(Uint8Array.of(1, 2, 3, 5), {status: 206, headers: {'Content-Range': 'bytes 4-7/12'}})
  ]) {
    globalThis.fetch = async () => response();
    await assert.rejects(readClefRange(base, entry, part), {code: 'ASSET_LOAD_FAILED'});
  }
});

test('initial file failure aborts sibling fetches and cancels late response bodies', async (t) => {
  const saved = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = saved;
  });
  let rejectFirst!: (reason?: unknown) => void,
    cancelled = 0;
  const first = new Promise<Response>((resolve, reject) => {
    rejectFirst = reject as typeof reject & typeof resolve;
  });
  const late: Array<(response: Response) => void> = [],
    signals: Array<AbortSignal | null | undefined> = [];
  globalThis.fetch = (url, options) => {
    signals.push(options!.signal);
    if (signals.length === 1) return first;
    return new Promise<Response>((resolve) => late.push(resolve));
  };
  const loading = openClefSource(new URL('https://example.test/assets/'));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(signals.length, 4);
  rejectFirst(new Error('source failed'));
  await assert.rejects(loading, {code: 'ASSET_LOAD_FAILED'});
  assert.ok(signals.every((signal) => signal!.aborted));
  for (const resolve of late)
    resolve({
      body: {
        cancel() {
          cancelled++;
        }
      }
    } as Response);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(cancelled, 3);
});
