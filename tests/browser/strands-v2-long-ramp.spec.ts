import {createHash} from 'node:crypto';
import {open, readFile, writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';
import {readHardwareReport} from './mapika2b-hardware.mts';
import {buildLongRampInput} from './strands-v2-long-ramp-fixture.mts';

type Answer = {choice: string; probabilities: Record<string, number>};
type Phase = {
  durationMs: number;
  layers: number;
  error: string | null;
  gpuErrors: string[];
  logits: number[] | null;
  answer: Answer | null;
};

function assertCompletePhases(phases: (Phase | null)[]): asserts phases is Phase[] {
  for (const phase of phases) {
    expect(phase).not.toBeNull();
    expect(phase!.error).toBeNull();
    expect(phase!.gpuErrors).toEqual([]);
    expect(phase!.layers).toBe(35);
    expect(phase!.logits).toHaveLength(255);
    expect(phase!.logits!.every(Number.isFinite)).toBe(true);
  }
}

function assertMatchingPhases(baseline: Phase, split: Phase) {
  const differences = baseline.logits!.map((value, index) => Math.abs(value - split.logits![index]!));
  const maxDifference = Math.max(...differences);
  console.log(`LONG_RAMP_LOGIT_MAX_DIFFERENCE ${maxDifference}`);
  expect(maxDifference).toBeLessThan(0.01);
}

async function assertTextReference(path: string, phases: Phase[]) {
  const reference = JSON.parse(await readFile(path, 'utf8'));
  for (const phase of phases) {
    expect(phase.answer!.choice).toBe(reference.answers.q.choice);
    expect(Math.abs(phase.answer!.probabilities.yes! - reference.answers.q.probabilities.yes)).toBeLessThan(0.02);
  }
}

async function readLongInput(root: string, target: number) {
  const fixture = JSON.parse(await readFile(resolve(root, 'e2b-v2-text-1024-reference/tokens.json'), 'utf8')) as {
    state: number[];
    questions: number[][];
  };
  const input = buildLongRampInput(fixture, target);
  expect(input.state.length + input.question.length).toBe(target);
  if (target === 1024) expect(input.state).toEqual(fixture.state);
  return input;
}

async function saveHiddenCapture(
  path: string,
  capture: {graphSha256: string; tokens: number; hidden?: Record<string, number[]>; logits: number[] | null}
) {
  const rows = Object.values(capture.hidden ?? {});
  expect(rows).toHaveLength(37);
  for (const row of rows) {
    expect(row).toHaveLength(1536);
    expect(row.every(Number.isFinite)).toBe(true);
  }
  await writeFile(path, JSON.stringify(capture), {flag: 'wx'});
}

test('experimental 35-layer split runner preserves a requested long input', async ({page}) => {
  test.skip(process.env.WEB_DECISION_STRANDS_V2_PROBE !== '1', 'Enable the private long-input GPU probe.');
  const timeoutMs = Number(process.env.WEB_DECISION_LONG_TIMEOUT_MS ?? '1800000');
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 2_147_483_647)
    throw new Error('Set WEB_DECISION_LONG_TIMEOUT_MS to a positive timer-safe integer.');
  test.setTimeout(timeoutMs);
  const root = resolve(process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT!);
  const target = Number(process.env.WEB_DECISION_LONG_TOKENS ?? '1024');
  const chunkTokens = Number(process.env.WEB_DECISION_LONG_CHUNK ?? '64');
  const mode = process.env.WEB_DECISION_LONG_MODE ?? 'both';
  const hostReadout = process.env.WEB_DECISION_LONG_HOST_READOUT === '1';
  const hiddenPath = process.env.WEB_DECISION_LONG_HIDDEN_CAPTURE;
  if (mode !== 'both' && mode !== 'baseline' && mode !== 'split') throw new Error('Invalid long-input phase mode.');
  if (!Number.isSafeInteger(target) || !Number.isSafeInteger(chunkTokens) || chunkTokens < 1)
    throw new Error('Set an explicit integer token count and positive chunk size.');
  const {state, question} = await readLongInput(root, target);
  const paths: Record<string, string> = {
    'linked.wasm': process.env.WEB_DECISION_LONG_WASM ?? 'e2b-v2-split-transposed-full/layer.wasm',
    'readout.bin': 'e2b-v2-dynamic-readout/input-2.bin',
    'local.bin': 'e2b-v2-positions/local.bin',
    'global.bin': 'e2b-v2-positions/global.bin'
  };
  const wasm = await readFile(resolve(root, paths['linked.wasm']!));
  const graphSha256 = createHash('sha256').update(wasm).digest('hex');
  const expectedSha256 =
    process.env.WEB_DECISION_LONG_WASM_SHA256 ??
    (process.env.WEB_DECISION_LONG_WASM ? '' : '88d91503237cdaa5f1c7b63cf02f2a0b82a44f44699d3d13838623738a5c7b47');
  if (!/^[0-9a-f]{64}$/u.test(expectedSha256)) throw new Error('An exact graph SHA-256 is required.');
  expect(graphSha256).toBe(expectedSha256);
  const checkpoint = await open(resolve(root, 'strands-e2b-v2/model.safetensors'), 'r');
  try {
    page.on('console', (message) => {
      if (message.text().startsWith('LONG_RAMP_')) console.log(message.text());
    });
    await page.route('**/strands-v2-long-ramp-source/model.safetensors', async (route) => {
      const match = /^bytes=(\d+)-(\d+)$/u.exec(route.request().headers().range ?? '');
      if (!match) throw new Error('Missing checkpoint range.');
      const start = Number(match[1]);
      const end = Number(match[2]);
      const bytes = Buffer.alloc(end - start + 1);
      const result = await checkpoint.read(bytes, 0, bytes.length, start);
      if (result.bytesRead !== bytes.length) throw new Error('Short checkpoint range.');
      await route.fulfill({status: 206, headers: {'content-range': `bytes ${start}-${end}/10246621918`}, body: bytes});
    });
    await page.route('**/strands-v2-long-ramp-assets/**', async (route) => {
      const name = new URL(route.request().url()).pathname.split('/').at(-1)!;
      const path = paths[name];
      if (!path) throw new Error(`Unexpected fixture ${name}`);
      await route.fulfill({status: 200, body: await readFile(resolve(root, path))});
    });
    expect((await readHardwareReport(page)).status).toBe('Hardware accelerated');
    await page.goto('/strands-v2-long-ramp.html');
    await page.waitForFunction(() => typeof window.runGemma4LongRamp === 'function');
    const started = Date.now();
    const request: Parameters<Window['runGemma4LongRamp']>[0] = {
      state,
      question,
      chunkTokens,
      mode,
      hostReadout,
      captureHidden: hiddenPath !== undefined
    };
    const result = (await page.evaluate((input) => window.runGemma4LongRamp(input), request)) as {
      tokens: number;
      baseline: Phase | null;
      split: (Phase & {hidden?: Record<string, number[]>}) | null;
      buffersAfterDispose: unknown;
    };
    const {hidden, ...split} = result.split ?? {};
    console.log(
      `LONG_RAMP_RESULT ${JSON.stringify({graphSha256, wallMs: Date.now() - started, ...result, split: result.split && split})}`
    );
    expect(result.tokens).toBe(target);
    const phases =
      mode === 'both' ? [result.baseline, result.split] : [mode === 'baseline' ? result.baseline : result.split];
    assertCompletePhases(phases);
    if (hiddenPath)
      await saveHiddenCapture(hiddenPath, {
        graphSha256,
        tokens: target,
        ...(hidden ? {hidden} : {}),
        logits: result.split?.logits ?? null
      });
    if (mode === 'both') assertMatchingPhases(result.baseline!, result.split!);
    if (target === 1024) await assertTextReference(resolve(root, 'e2b-v2-text-1024-reference/result.json'), phases);
  } finally {
    await checkpoint.close();
  }
});
