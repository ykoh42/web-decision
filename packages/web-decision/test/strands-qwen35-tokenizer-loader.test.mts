import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import test, {type TestContext} from 'node:test';
import {join} from 'node:path';
import {createQwen35TokenizerLoader, openPinnedQwen35Tokenizer} from '../src/strands/qwen35-tokenizer.mts';
import type {Qwen35PinnedTokenizer} from '../src/strands/qwen35-tokenizer.mts';
import {packQwen35Request} from '../src/strands/qwen35-protocol.mts';

const TEST_BASE = new URL('https://tokenizer.example.test/');
const CHECKPOINT_REPOSITORY = 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610';
const CHECKPOINT_REVISION = '216d1ccb46a9e9b036eb9187b248db98480f77d4';

function hash(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function memoryCache(entries = new Map<string, Response>()) {
  return {
    entries,
    async match(key: string) {
      return entries.get(key)?.clone() ?? null;
    },
    async put(key: string, response: Response) {
      entries.set(key, response.clone());
    }
  };
}

function fixtureBytes(): Readonly<Record<'tokenizer.json' | 'tokenizer_config.json', Uint8Array>> {
  return {
    'tokenizer.json': new TextEncoder().encode('{"version":"fixture","model":{}}'),
    'tokenizer_config.json': new TextEncoder().encode('{"model_max_length":262144}')
  };
}

function loaderFor(
  source: Readonly<Record<'tokenizer.json' | 'tokenizer_config.json', Uint8Array>>,
  sourceHash?: string
) {
  const files = Object.entries(source).map(([name, bytes]) => ({
    name: name as 'tokenizer.json' | 'tokenizer_config.json',
    bytes: bytes.byteLength,
    sha256: name === 'tokenizer.json' && sourceHash ? sourceHash : hash(bytes),
    ranges: [{offset: 0, bytes: bytes.byteLength, sha256: hash(bytes)}]
  }));
  const open = createQwen35TokenizerLoader({
    base: TEST_BASE,
    files,
    markerId: 'qwen35-tokenizer-test',
    createTokenizer(tokenizerData, tokenizerConfig) {
      return {
        data: tokenizerData,
        config: tokenizerConfig,
        encode: () => ({ids: [], tokens: []})
      };
    }
  });
  return {open, files};
}

function serveRanges(
  t: TestContext,
  source: Readonly<Record<'tokenizer.json' | 'tokenizer_config.json', Uint8Array>>,
  requests: string[],
  corrupt = false
) {
  return t.mock.method(globalThis, 'fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input));
    const name = Object.keys(source).find((candidate) => url.pathname.endsWith(`/${candidate}`)) as
      | 'tokenizer.json'
      | 'tokenizer_config.json'
      | undefined;
    assert.ok(name, `unexpected tokenizer URL ${url.href}`);
    const range = new Headers(init?.headers).get('Range');
    assert.ok(range);
    requests.push(`${name}:${range}`);
    const match = /^bytes=(\d+)-(\d+)$/.exec(range);
    assert.ok(match);
    const start = Number(match[1]);
    const end = Number(match[2]);
    const bytes = source[name].slice(start, end + 1);
    if (corrupt && requests.length === 1) bytes[0] = bytes[0]! ^ 1;
    return new Response(bytes, {
      status: 206,
      headers: {
        'content-range': `bytes ${start}-${end}/${source[name].byteLength}`,
        'content-length': String(end - start + 1)
      }
    });
  });
}

test('loads tokenizer and config through verified ranges, commits cache, reuses offline and disposes', async (t) => {
  const source = fixtureBytes();
  const {open} = loaderFor(source);
  const cache = memoryCache();
  const requests: string[] = [];
  serveRanges(t, source, requests);

  const loaded = await open({cache});
  assert.equal(requests.length, 2);
  assert.equal(requests[0], `tokenizer.json:bytes=0-${source['tokenizer.json'].byteLength - 1}`);
  assert.equal(requests[1], `tokenizer_config.json:bytes=0-${source['tokenizer_config.json'].byteLength - 1}`);
  assert.equal(await loaded.commit(), true);
  assert.deepEqual(loaded.tokenizer.encode('test'), {ids: [], tokens: []});
  loaded.dispose();
  loaded.dispose();
  assert.throws(() => loaded.tokenizer, /disposed/);
  await assert.rejects(loaded.commit(), /disposed/);

  const offline = await open({cache, offline: true});
  assert.equal(requests.length, 2);
  offline.dispose();
});

test('rejects corrupted bytes and refuses incomplete offline caches', async (t) => {
  const source = fixtureBytes();
  const {open} = loaderFor(source);
  const requests: string[] = [];
  serveRanges(t, source, requests, true);
  await assert.rejects(open({cache: null}), /integrity check failed/);

  const second = loaderFor(source);
  await assert.rejects(second.open({cache: memoryCache(), offline: true}), /not available offline/);
});

test('keeps online tokenizer loading usable when CacheStorage writes exceed quota', async (t) => {
  const source = fixtureBytes();
  const {open} = loaderFor(source);
  const entries = new Map<string, Response>();
  const requests: string[] = [];
  const cache = {
    async match(key: string) {
      return entries.get(key)?.clone() ?? null;
    },
    async put() {
      throw Object.assign(new Error('browser storage quota reached'), {name: 'QuotaExceededError'});
    }
  };
  serveRanges(t, source, requests);

  const loaded = await open({cache});
  assert.deepEqual(loaded.tokenizer.encode('test'), {ids: [], tokens: []});
  assert.equal(requests.length, 2);
  assert.equal(await loaded.commit(), false);
  loaded.dispose();

  const onlineRequestCount = requests.length;
  await assert.rejects(open({cache, offline: true}), {
    code: 'ASSET_LOAD_FAILED',
    message: 'The complete verified model is not available offline.'
  });
  assert.equal(requests.length, onlineRequestCount, 'offline cache failure must not fall through to fetch');
});

test('checks the complete file hash after all individually verified ranges', async (t) => {
  const source = fixtureBytes();
  const wrongFullHash = '0'.repeat(64);
  const {open} = loaderFor(source, wrongFullHash);
  serveRanges(t, source, []);
  await assert.rejects(open(), /tokenizer\.json hash mismatch/);
});

test('rejects a response whose Content-Range does not match the requested window', async (t) => {
  const source = fixtureBytes();
  const {open} = loaderFor(source);
  const body = source['tokenizer.json'].slice();
  t.mock.method(
    globalThis,
    'fetch',
    async () =>
      new Response(body.buffer as ArrayBuffer, {
        status: 206,
        headers: {
          'content-range': `bytes 1-${source['tokenizer.json'].byteLength}/${source['tokenizer.json'].byteLength}`,
          'content-length': String(source['tokenizer.json'].byteLength)
        }
      })
  );
  await assert.rejects(open(), /Invalid or inaccessible Content-Range/);
});

test('requests only the immutable Strands checkpoint tokenizer URLs', async (t) => {
  const requests: string[] = [];
  t.mock.method(globalThis, 'fetch', async (input: RequestInfo | URL) => {
    const url = new URL(String(input));
    assert.equal(url.origin, 'https://huggingface.co');
    assert.ok(url.pathname.startsWith(`/${CHECKPOINT_REPOSITORY}/resolve/${CHECKPOINT_REVISION}/`));
    requests.push(url.pathname);
    throw new Error('stop after verifying the immutable URL');
  });
  await assert.rejects(openPinnedQwen35Tokenizer({cache: null}), /Could not fetch range/);
  assert.deepEqual(requests, [`/${CHECKPOINT_REPOSITORY}/resolve/${CHECKPOINT_REVISION}/tokenizer.json`]);
});

const requestBoundaryGolden = JSON.parse(
  await readFile(new URL('./fixtures/qwen35-source-request-boundaries-golden.json', import.meta.url), 'utf8')
) as {
  readonly source: {readonly tokenizerSha256: string; readonly tokenizerConfigSha256: string};
  readonly tokenizerCase: {
    readonly request: unknown;
    readonly stateText: string;
    readonly questionText: string;
    readonly stateTokenIds: readonly number[];
    readonly questionTokenIds: readonly number[];
    readonly inputIds: readonly number[];
    readonly optionIndices: readonly number[];
    readonly answerIndex: number;
  };
};
const tokenizerDirectory = process.env.WEB_DECISION_QWEN35_TOKENIZER_DIR;
let hasLocalPinnedTokenizer = Boolean(tokenizerDirectory);
if (tokenizerDirectory) {
  try {
    await readFile(join(tokenizerDirectory, 'tokenizer.json'));
    await readFile(join(tokenizerDirectory, 'tokenizer_config.json'));
  } catch {
    hasLocalPinnedTokenizer = false;
  }
}

test(
  'pinned checkpoint tokenizer reproduces source token ID and pointer goldens',
  {
    skip: hasLocalPinnedTokenizer
      ? false
      : 'Set WEB_DECISION_QWEN35_TOKENIZER_DIR to local pinned checkpoint tokenizer files.'
  },
  async (t) => {
    const source = {
      'tokenizer.json': new Uint8Array(await readFile(join(tokenizerDirectory!, 'tokenizer.json'))),
      'tokenizer_config.json': new Uint8Array(await readFile(join(tokenizerDirectory!, 'tokenizer_config.json')))
    };
    assert.equal(hash(source['tokenizer.json']), requestBoundaryGolden.source.tokenizerSha256);
    assert.equal(hash(source['tokenizer_config.json']), requestBoundaryGolden.source.tokenizerConfigSha256);
    const paths = {
      tokenizer: `/${CHECKPOINT_REPOSITORY}/resolve/${CHECKPOINT_REVISION}/tokenizer.json`,
      config: `/${CHECKPOINT_REPOSITORY}/resolve/${CHECKPOINT_REVISION}/tokenizer_config.json`
    };
    t.mock.method(globalThis, 'fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input));
      assert.equal(url.origin, 'https://huggingface.co');
      const name = url.pathname === paths.tokenizer ? 'tokenizer.json' : 'tokenizer_config.json';
      const range = new Headers(init?.headers).get('Range')!;
      const match = /^bytes=(\d+)-(\d+)$/.exec(range)!;
      const start = Number(match[1]);
      const end = Number(match[2]);
      return new Response(source[name].slice(start, end + 1), {
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${end}/${source[name].byteLength}`,
          'content-length': String(end - start + 1)
        }
      });
    });

    const loaded: Qwen35PinnedTokenizer = await openPinnedQwen35Tokenizer({cache: null});
    try {
      const expected = requestBoundaryGolden.tokenizerCase;
      const tokenizer = loaded.tokenizer;
      assert.deepEqual(
        encodedIds(tokenizer.encode(expected.stateText, {add_special_tokens: true})),
        expected.stateTokenIds
      );
      assert.deepEqual(
        encodedIds(tokenizer.encode(expected.questionText, {add_special_tokens: false})),
        expected.questionTokenIds
      );
      const packed = packQwen35Request(tokenizer, expected.request);
      assert.deepEqual(packed.rows[0]!.inputIds, expected.inputIds);
      assert.deepEqual(packed.rows[0]!.optionIndices, expected.optionIndices);
      assert.equal(packed.rows[0]!.answerIndex, expected.answerIndex);
    } finally {
      loaded.dispose();
    }
  }
);

function encodedIds(value: unknown): readonly number[] {
  assert.ok(value && typeof value === 'object' && Array.isArray((value as {ids?: unknown}).ids));
  return (value as {readonly ids: readonly number[]}).ids;
}
