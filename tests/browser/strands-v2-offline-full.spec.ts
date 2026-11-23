import {createHash} from 'node:crypto';
import {mkdtemp, readFile, rm, stat} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {chromium, expect, test, type BrowserContext, type Page} from '@playwright/test';
import {readHardwareReport} from './mapika2b-hardware.mts';
import {startLocalRangeServer} from './strands-v2-local-range-server.mts';

const HASHES: Record<string, string> = {
  'local.bin': 'cc63341a0ac42a60b986ed638fd0d45b838b72fabeffec059c463eac4ed9ea15',
  'global.bin': '71a80d617056a8be95eeeb385ebcc01c7fb2c73d5430a8ce862ed6c94179d076'
};
const WEIGHT_BYTES = 9_638_447_838;
const STATIC_AND_CODEC_BYTES = 48_623_927;

async function openPage(
  context: BrowserContext,
  params: Parameters<Window['gemmaOfflineBootstrap']>[0]
): Promise<Page> {
  const page = await context.newPage();
  page.on('console', (message) => {
    if (message.type() === 'log') console.log(`[browser] ${message.text()}`);
  });
  await page.goto(
    `http://127.0.0.1:${process.env.WEB_DECISION_BROWSER_PORT || 41773}/strands-v2-offline-full-probe.html`
  );
  await page.waitForFunction(() => typeof window.gemmaOfflineBootstrap === 'function');
  const storage = await page.evaluate((value) => window.gemmaOfflineBootstrap(value), params);
  console.log(`BROWSER_STORAGE ${JSON.stringify(storage)}`);
  if (process.env.WEB_DECISION_REQUIRE_DURABLE_STORAGE === '1') expect(storage.persisted).toBe(true);
  await page.waitForLoadState('networkidle');
  return page;
}

async function assertHardware(context: BrowserContext): Promise<void> {
  const page = await context.newPage();
  try {
    expect((await readHardwareReport(page)).status).toBe('Hardware accelerated');
  } finally {
    await page.close();
  }
}

test('persist verified local checkpoint, restart browser, then load and infer offline', async () => {
  test.skip(
    process.env.WEB_DECISION_STRANDS_OFFLINE_FULL !== '1',
    'Explicitly enable the full offline disk and GPU test'
  );
  test.setTimeout(3_600_000);
  const root = resolve(process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT!);
  const wasmPath = process.env.WEB_DECISION_STRANDS_V2_IMAGE_WASM;
  const wasmSha256 = process.env.WEB_DECISION_STRANDS_V2_IMAGE_WASM_SHA256;
  if (!wasmPath || !wasmSha256 || !/^[0-9a-f]{64}$/u.test(wasmSha256))
    throw new Error('Explicit WASM path and SHA-256 are required for the offline test.');
  const checkpoint = resolve(root, 'strands-e2b-v2');
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
    paths.set(`/checkpoint/${name}`, resolve(checkpoint, name));
  expect((await stat(paths.get('/checkpoint/model.safetensors')!)).size).toBe(10_246_621_918);
  const graphFiles: Record<string, string> = {
    'linked.wasm': resolve(root, wasmPath),
    'local.bin': resolve(root, 'e2b-v2-positions/local.bin'),
    'global.bin': resolve(root, 'e2b-v2-positions/global.bin')
  };
  const pins: Record<string, {bytes: number; sha256: string}> = {};
  for (const [name, path] of Object.entries(graphFiles)) {
    const content = await readFile(path);
    const sha256 = createHash('sha256').update(content).digest('hex');
    expect(sha256, name).toBe(name === 'linked.wasm' ? wasmSha256 : HASHES[name]);
    pins[name] = {bytes: content.length, sha256};
    paths.set(`/graph/${name}`, path);
  }
  const references = JSON.parse(
    await readFile(
      resolve(root, 'checkout/packages/web-decision/test/fixtures/strands-gemma4-v2-encoded-image.json'),
      'utf8'
    )
  ) as Array<{
    name: string;
    tokens: number;
    pixelsSha256: string;
    positionsSha256: string;
  }>;
  const codecFixtures = await Promise.all(
    ['jpeg-orientation-6.jpg', 'webp-alpha.webp'].map(async (name) => {
      const row = references.find((item) => item.name === name);
      if (!row) throw new Error(`Missing Pillow codec reference ${name}`);
      const encoded = (
        await readFile(resolve(root, 'checkout/packages/web-decision/test/fixtures/qwen35-image-input', name))
      ).toString('base64');
      return {
        encoded,
        expected: {tokens: row.tokens, pixelsSha256: row.pixelsSha256, positionsSha256: row.positionsSha256}
      };
    })
  );
  const official = JSON.parse(await readFile(resolve(root, 'e2b-v2-vision-reference/result.json'), 'utf8'));
  const server = await startLocalRangeServer(paths);
  const profile = await mkdtemp(join(tmpdir(), 'web-decision-offline-v2-'));
  const cache = 'web-decision-gemma4-v2-offline-integration';
  const requiredBytes =
    WEIGHT_BYTES + STATIC_AND_CODEC_BYTES + Object.values(pins).reduce((sum, item) => sum + item.bytes, 0);
  const options = {
    headless: process.env.WEB_DECISION_REAL_GPU_HEADLESS === '1',
    args: ['--enable-unsafe-webgpu', '--disable-background-networking']
  };
  let context: BrowserContext | undefined;
  try {
    console.log(`Minimum cache content ${requiredBytes} bytes; actual disk overhead and browser quota may be higher.`);
    context = await chromium.launchPersistentContext(profile, options);
    await assertHardware(context);
    const online = await openPage(context, {base: server.base, cache, pins, requiredBytes, checkCapacity: true});
    const prepared = await online.evaluate(() => window.gemmaOfflinePrepare());
    expect(prepared.ready).toBe(true);
    expect(prepared.progressBytes).toBe(WEIGHT_BYTES);
    const networkAfterPreparation = server.stats();
    console.log(`LOCAL_PREPARATION ${JSON.stringify(networkAfterPreparation)}`);
    await context.close();
    context = await chromium.launchPersistentContext(profile, options);
    await assertHardware(context);
    const offlinePage = await openPage(context, {base: server.base, cache, pins, requiredBytes, checkCapacity: false});
    await context.setOffline(true);
    expect(await offlinePage.evaluate(() => navigator.onLine)).toBe(false);
    const requests: string[] = [];
    offlinePage.on('request', (request) => requests.push(request.url()));
    const verifyAll = process.env.WEB_DECISION_VERIFY_FULL_CACHE !== '0';
    const result = (await offlinePage.evaluate(({verify, fixtures}) => window.gemmaOfflineRun(verify, fixtures), {
      verify: verifyAll,
      fixtures: codecFixtures
    })) as {
      ready: boolean;
      verified: boolean;
      codecResults: unknown[];
      attempts: string[];
      answer: {answers: {color: {choice: string; probabilities: Record<string, number>}}};
    };
    expect(result.ready).toBe(true);
    expect(result.verified).toBe(true);
    expect(result.codecResults).toEqual(codecFixtures.map((fixture) => fixture.expected));
    expect(result.answer.answers.color.choice).toBe(official.answers.color.choice);
    const actualProbability = result.answer.answers.color.probabilities.red!;
    const officialProbability = official.answers.color.probabilities.red as number;
    expect(Math.abs(actualProbability - officialProbability)).toBeLessThan(0.02);
    expect(result.attempts).toEqual([]);
    expect(requests).toEqual([]);
    expect(server.stats()).toEqual(networkAfterPreparation);
    console.log(
      `OFFLINE_RESULT ${JSON.stringify({verifyAll, actualProbability, officialProbability, requests: requests.length})}`
    );
  } finally {
    await context?.close();
    await server.close();
    if (process.env.WEB_DECISION_KEEP_OFFLINE_PROFILE === '1')
      console.log(`Preserved temporary test profile: ${profile}`);
    else await rm(profile, {recursive: true, force: true});
  }
});
