import {createHash} from 'node:crypto';
import {open, readFile, readdir, unlink, mkdtemp, rm, type FileHandle} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {resolve} from 'node:path';
import {expect, test, type Page} from '@playwright/test';
import {readHardwareReport} from './mapika2b-hardware.mts';
import {startLocalRangeServer} from './strands-v2-local-range-server.mts';

const goldenHash = '658445826fa307b949d88722f5a312eb39b6354cbdfe62102d4d4982dfc92ded';
const sha = (data: Uint8Array | string) => createHash('sha256').update(data).digest('hex');
type Answer = {codes: string[]; probabilities: number[]};
type Golden = {
  variant: string;
  request: {images?: string[]};
  images: {sha256: string}[];
  questions: Record<string, {codes: string[]}>;
};
type RecordRow = {line: number; questions: Record<string, Answer>; previousSha256: string; recordSha256: string};

async function sourceFingerprint(root: string) {
  const files = [
    'package.json',
    'pnpm-lock.yaml',
    'packages/web-decision/package.json',
    'tests/browser/vite.config.ts',
    'tests/browser/strands-v2-golden-capture.html',
    'tests/browser/strands-v2-golden-capture.ts',
    'tests/browser/strands-v2-golden-capture.spec.ts'
  ];
  async function visit(path: string) {
    for (const item of await readdir(resolve(root, path), {withFileTypes: true})) {
      const name = `${path}/${item.name}`;
      if (item.isDirectory()) await visit(name);
      else if (item.isFile() || item.isSymbolicLink()) files.push(name);
    }
  }
  await visit('packages/web-decision/src');
  const hash = createHash('sha256');
  for (const name of files.sort()) hash.update(`${name}\0${sha(await readFile(resolve(root, name)))}\n`);
  return hash.digest('hex');
}

function readGolden(lines: string[], index: number): Golden {
  const text = lines[index];
  if (!text) throw new Error(`Golden line ${index} is missing.`);
  const value = JSON.parse(text) as Golden;
  expect(value.variant).toBe('strands-chat');
  expect(value.request.images?.length ?? 0).toBe(value.images.length);
  for (let i = 0; i < value.images.length; i++)
    expect(sha(Buffer.from(value.request.images![i]!, 'base64'))).toBe(value.images[i]!.sha256);
  return value;
}

function rowDigest(line: number, questions: Record<string, Answer>, previousSha256: string) {
  return sha(JSON.stringify({line, questions, previousSha256}));
}

function validateRow(row: RecordRow, expectedLine: number, golden: Golden, previous: string) {
  expect(row.line).toBe(expectedLine);
  expect(row.previousSha256).toBe(previous);
  expect(Object.keys(row.questions)).toEqual(Object.keys(golden.questions));
  for (const [name, item] of Object.entries(row.questions)) {
    expect(item.codes).toEqual(golden.questions[name]!.codes);
    expect(item.probabilities).toHaveLength(item.codes.length);
    expect(item.probabilities.every((value) => typeof value === 'number' && Number.isFinite(value))).toBe(true);
  }
  expect(row.recordSha256).toBe(rowDigest(row.line, row.questions, previous));
}

async function recoverRows(file: FileHandle, lines: string[], start: number, end: number) {
  const size = (await file.stat()).size;
  const bytes = Buffer.alloc(size);
  if ((await file.read(bytes, 0, size, 0)).bytesRead !== size) throw new Error('Short output read.');
  const lastNewline = bytes.lastIndexOf(10);
  const complete = lastNewline + 1;
  if (complete !== bytes.length) {
    await file.truncate(complete);
    await file.sync();
  }
  const rows = bytes.subarray(0, complete).toString('utf8').split('\n').filter(Boolean);
  if (rows.length > end - start) throw new Error('Existing output exceeds the requested range.');
  let previous = '0'.repeat(64);
  for (const [offset, text] of rows.entries()) {
    const row = JSON.parse(text) as RecordRow;
    validateRow(row, start + offset, readGolden(lines, start + offset), previous);
    previous = row.recordSha256;
  }
  return {next: start + rows.length, position: complete, previous};
}

async function writeDurable(file: FileHandle, text: string, position: number) {
  const data = Buffer.from(text);
  let offset = 0;
  while (offset < data.length) {
    const {bytesWritten} = await file.write(data, offset, data.length - offset, position + offset);
    if (!bytesWritten) throw new Error('Short output write.');
    offset += bytesWritten;
  }
  await file.sync();
  return position + offset;
}

async function openOutput(outputPath: string, manifest: string, resume: boolean) {
  if (resume) {
    expect(await readFile(`${outputPath}.manifest.json`, 'utf8')).toBe(manifest);
    return open(outputPath, 'r+');
  }
  const file = await open(outputPath, 'wx+');
  let manifestFile: FileHandle | undefined;
  try {
    manifestFile = await open(`${outputPath}.manifest.json`, 'wx');
    await manifestFile.writeFile(manifest);
    await manifestFile.sync();
    await manifestFile.close();
    return file;
  } catch (error) {
    await manifestFile?.close();
    if (manifestFile) await unlink(`${outputPath}.manifest.json`);
    await file.close();
    await unlink(outputPath);
    throw error;
  }
}

async function closeResources(
  page: Page,
  server: Awaited<ReturnType<typeof startLocalRangeServer>> | undefined,
  file: FileHandle | undefined,
  lock: FileHandle,
  lockPath: string
) {
  const cleanup = [
    async () => {
      if (server && !page.isClosed()) await page.evaluate(() => window.closeGemmaGoldenCapture());
    },
    async () => {
      if (server && !page.isClosed()) await page.close();
    },
    async () => {
      await server?.close();
    },
    async () => {
      await file?.close();
    },
    async () => {
      await lock.close();
    },
    async () => {
      await unlink(lockPath);
    }
  ];
  let failure: unknown;
  let failed = false;
  for (const action of cleanup) {
    try {
      await action();
    } catch (error) {
      if (!failed) {
        failed = true;
        failure = error;
      }
    }
  }
  if (failed) throw failure;
}

function captureSettings() {
  const root = resolve(process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT!);
  const graph = process.env.WEB_DECISION_GEMMA_GOLDEN_WASM;
  const graphHash = process.env.WEB_DECISION_GEMMA_GOLDEN_WASM_SHA256;
  const output = process.env.WEB_DECISION_GEMMA_GOLDEN_OUTPUT;
  if (!graph || !graphHash || !/^[a-f0-9]{64}$/u.test(graphHash) || !output)
    throw new Error('Set graph, SHA and output path.');
  const start = Number(process.env.WEB_DECISION_GEMMA_GOLDEN_LINE ?? '0');
  const end = Number(process.env.WEB_DECISION_GEMMA_GOLDEN_END ?? String(start + 1));
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end <= start)
    throw new Error('Invalid golden range: END is exclusive.');
  const timeout = Number(
    process.env.WEB_DECISION_GEMMA_GOLDEN_TIMEOUT_MS ?? String(Math.min(2_147_483_647, 600_000 * (end - start)))
  );
  if (!Number.isSafeInteger(timeout) || timeout < 1 || timeout > 2_147_483_647) throw new Error('Invalid timeout.');
  if (process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1')
    throw new Error('Disable HMR via WEB_DECISION_STRANDS_V2_PROBE=1.');
  return {root, graph, graphHash, output, start, end, timeout};
}

test('source-graph capture records official golden requests before rounding', async ({page}) => {
  test.skip(
    process.env.WEB_DECISION_GEMMA_GOLDEN_CAPTURE !== '1',
    'Explicitly enable the source-graph GPU diagnostic.'
  );
  const {root, graph, graphHash, output, start, end, timeout} = captureSettings();
  test.setTimeout(timeout);
  const goldenBytes = await readFile(resolve(root, 'e2b-v2-release-diagnosis/golden_vectors.jsonl'));
  expect(sha(goldenBytes)).toBe(goldenHash);
  const lines = goldenBytes.toString('utf8').split(/\r?\n/u);
  readGolden(lines, start);
  readGolden(lines, end - 1);
  const paths = new Map<string, string>();
  for (const name of [
    'config.json',
    'decision_config.json',
    'processor_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'readout.safetensors',
    'model.safetensors'
  ])
    paths.set(`/checkpoint/${name}`, resolve(root, 'strands-e2b-v2', name));
  const graphFiles: Record<string, string> = {
    'linked.wasm': resolve(root, graph),
    'local.bin': resolve(root, 'e2b-v2-positions/local.bin'),
    'global.bin': resolve(root, 'e2b-v2-positions/global.bin'),
    'scoring.bin': resolve(root, 'checkout/packages/web-decision/assets/strands-gemma4-v2/scoring.bin')
  };
  const expected: Record<string, string> = {
    'linked.wasm': graphHash,
    'local.bin': 'cc63341a0ac42a60b986ed638fd0d45b838b72fabeffec059c463eac4ed9ea15',
    'global.bin': '71a80d617056a8be95eeeb385ebcc01c7fb2c73d5430a8ce862ed6c94179d076',
    'scoring.bin': '8976812e52bc591dee1a1b891d427b16eb0bf641ea61299a518276d35e6da3ad'
  };
  const pins: Record<string, {bytes: number; sha256: string}> = {};
  for (const [name, path] of Object.entries(graphFiles)) {
    const bytes = await readFile(path);
    expect(sha(bytes), name).toBe(expected[name]);
    pins[name] = {bytes: bytes.length, sha256: sha(bytes)};
    paths.set(`/graph/${name}`, path);
  }
  const checkout = resolve(root, 'checkout');
  const sourceSha256 = await sourceFingerprint(checkout);
  const manifest =
    JSON.stringify({format: 1, status: 'source-graph-diagnostic', goldenHash, pins, sourceSha256, start, end}) + '\n';
  const outputPath = resolve(output);
  const lockPath = `${outputPath}.lock`;
  const lock = await open(lockPath, 'wx');
  let file: FileHandle | undefined;
  let server: Awaited<ReturnType<typeof startLocalRangeServer>> | undefined;
  let failed = false;
  let failure: unknown;
  try {
    await lock.writeFile(`${process.pid}\n`);
    await lock.sync();
    file = await openOutput(outputPath, manifest, process.env.WEB_DECISION_GEMMA_GOLDEN_RESUME === '1');
    const progress = await recoverRows(file, lines, start, end);
    if (progress.next < end) {
      expect((await readHardwareReport(page)).status).toBe('Hardware accelerated');
      server = await startLocalRangeServer(paths);
      await page.goto('/strands-v2-golden-capture.html');
      await page.waitForFunction(() => typeof window.openGemmaGoldenCapture === 'function');
      await page.evaluate((params) => window.openGemmaGoldenCapture(params), {base: server.base, pins});
      for (let line = progress.next; line < end; line++) {
        expect(await sourceFingerprint(checkout)).toBe(sourceSha256);
        const golden = readGolden(lines, line);
        const captured = await page.evaluate((params) => window.runGemmaGoldenCapture(params), {line, golden} as never);
        expect(await sourceFingerprint(checkout)).toBe(sourceSha256);
        const previousSha256 = progress.previous;
        const row = {
          line: captured.line,
          questions: captured.questions,
          previousSha256,
          recordSha256: rowDigest(captured.line, captured.questions, previousSha256)
        };
        validateRow(row, line, golden, previousSha256);
        progress.position = await writeDurable(file, JSON.stringify(row) + '\n', progress.position);
        progress.previous = row.recordSha256;
        console.log(
          `SOURCE_GRAPH_GOLDEN_CAPTURE ${JSON.stringify({line, graphSha256: graphHash, goldenSha256: goldenHash, recordSha256: row.recordSha256, releaseEvidence: false})}`
        );
      }
    }
  } catch (error) {
    failed = true;
    failure = error;
  } finally {
    try {
      await closeResources(page, server, file, lock, lockPath);
    } catch (error) {
      if (!failed) {
        failed = true;
        failure = error;
      }
    }
  }
  if (failed) throw failure;
});

test('golden capture checkpoints resume only complete chained rows', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'gemma-golden-checkpoint-'));
  const path = join(directory, 'output.jsonl');
  const file = await open(path, 'wx+');
  try {
    expect(await sourceFingerprint(resolve(process.cwd()))).toMatch(/^[a-f0-9]{64}$/u);
    const exclusive = join(directory, 'exclusive.jsonl');
    const initial = await openOutput(exclusive, 'pinned-manifest', false);
    await initial.close();
    await expect(openOutput(exclusive, 'pinned-manifest', false)).rejects.toThrow();
    await expect(openOutput(exclusive, 'changed-manifest', true)).rejects.toThrow();
    const reopened = await openOutput(exclusive, 'pinned-manifest', true);
    await reopened.close();
    const golden = {variant: 'strands-chat', request: {}, images: [], questions: {q: {codes: ['A', 'B']}}};
    const lines = [JSON.stringify(golden), JSON.stringify(golden)];
    const questions = {q: {codes: ['A', 'B'], probabilities: [0.25, 0.75]}};
    const previousSha256 = '0'.repeat(64);
    const first = {line: 0, questions, previousSha256, recordSha256: rowDigest(0, questions, previousSha256)};
    await file.writeFile(JSON.stringify(first) + '\n{"line":1');
    const resumed = await recoverRows(file, lines, 0, 2);
    expect(resumed.next).toBe(1);
    expect(resumed.previous).toBe(first.recordSha256);
    expect((await readFile(path, 'utf8')).endsWith('\n')).toBe(true);
    const second = {
      line: 1,
      questions,
      previousSha256: resumed.previous,
      recordSha256: rowDigest(1, questions, resumed.previous)
    };
    await writeDurable(file, JSON.stringify(second) + '\n', resumed.position);
    await file.close();
    const check = await open(path, 'r+');
    try {
      const done = await recoverRows(check, lines, 0, 2);
      expect(done.next).toBe(2);
      await expect(recoverRows(check, lines, 1, 2)).rejects.toThrow();
    } finally {
      await check.close();
    }
    expect(() => validateRow({...second, recordSha256: 'wrong'}, 1, readGolden(lines, 1), resumed.previous)).toThrow();
  } finally {
    await file.close();
    await rm(directory, {recursive: true, force: true});
  }
});
