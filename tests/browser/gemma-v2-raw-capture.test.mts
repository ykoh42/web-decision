import assert from 'node:assert/strict';
import {copyFile, mkdir, mkdtemp, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import test from 'node:test';
import {chromium, type Browser} from '@playwright/test';
import {routeGemmaExactArchive, verifyGemmaArchiveResources} from './gemma-v2-exact-archive-page.mjs';
import {findGemmaV2ScorerPin, openGemmaV2RawCapture} from './gemma-v2-raw-capture.mjs';

const modulePath = 'dist/index.js';
const rootUrl = 'http://127.0.0.1:43617';
const scriptUrl = rootUrl + '/' + modulePath;
const source = `function gemma4V2Probabilities(logits, count, config) {
  return logits.slice(0, count).map(Math.fround);
}
function finishGemma4V2Answer(question, logits, config) {
  const count = question.options.length;
  const probabilities = gemma4V2Probabilities(logits, count, config);
  if (question.kind === "noul") return {
    value: Math.round(10000 * probabilities[1]) / 10000
  };
  return {value: Math.round(10000 * probabilities[0]) / 10000};
}
globalThis.syntheticScorer = (kind = "noul") => {
  const question = Object.freeze({name: "q1", kind, options: ["false", "true"], codes: ["A", "B"]});
  const input = Object.freeze([0.06879, 0.93121]);
  return {answer: finishGemma4V2Answer(question, input, {}), input};
};
`;

async function pageFor(browser: Browser, bytes: string) {
  const page = await browser.newPage();
  await page.route(rootUrl + '/**', (route) => {
    if (route.request().url() === scriptUrl) return route.fulfill({contentType: 'text/javascript', body: bytes});
    if (route.request().url() === rootUrl + '/')
      return route.fulfill({contentType: 'text/html', body: '<script type="module" src="/dist/index.js"></script>'});
    return route.abort();
  });
  await page.goto(rootUrl);
  return page;
}

test('Chromium reads the exact pinned scorer frame without changing its inputs or result', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'gemma-v2-cdp-synthetic-'));
  t.after(() => rm(root, {recursive: true, force: true}));
  await mkdir(join(root, 'dist'));
  await writeFile(join(root, modulePath), source);
  const pin = await findGemmaV2ScorerPin(root, [modulePath]);
  assert.equal(pin.lineNumber, 6);
  const browser = await chromium.launch({headless: true, args: ['--disable-gpu', '--disable-software-rasterizer']});
  t.after(() => browser.close());
  const page = await pageFor(browser, source);
  const invoke = () => page.evaluate('globalThis.syntheticScorer()');
  const baseline = await invoke();
  const capture = await openGemmaV2RawCapture(page, scriptUrl, pin);
  try {
    const {result, frames} = await capture.capture(['q1'], invoke);
    assert.deepEqual(result, baseline);
    assert.deepEqual(result, {answer: {value: 0.9312}, input: [0.06879, 0.93121]});
    assert.deepEqual(frames, [
      {
        name: 'q1',
        options: ['false', 'true'],
        codes: ['A', 'B'],
        probabilities: [Math.fround(0.06879), Math.fround(0.93121)]
      }
    ]);
    const {breakpointColumn} = capture.evidence();
    assert.ok(breakpointColumn >= pin.columnNumber && breakpointColumn < pin.conditionEndColumn);
    assert.deepEqual(capture.evidence(), {...pin, breakpointColumn, frames: 1});
    const other = () => page.evaluate('globalThis.syntheticScorer("score")');
    const {result: scored, frames: scoreFrames} = await capture.capture(['q1'], other);
    assert.deepEqual(scored, {answer: {value: 0.0688}, input: [0.06879, 0.93121]});
    assert.deepEqual(scoreFrames, frames);
    assert.deepEqual(capture.evidence(), {...pin, breakpointColumn, frames: 2});
    await assert.rejects(capture.capture(['unrelated'], invoke), /question identities differ/u);
  } finally {
    await capture.close();
  }
  assert.deepEqual(await invoke(), baseline);
  await assert.rejects(openGemmaV2RawCapture(page, scriptUrl, {...pin, lineNumber: pin.lineNumber + 1}));
  const changed = await pageFor(browser, source + '\n// changed transport bytes');
  await assert.rejects(openGemmaV2RawCapture(changed, scriptUrl, pin), /scorer bytes differ/u);
  await writeFile(join(root, modulePath), source.replace('if (question.kind', 'if (question.kind '));
  await assert.rejects(findGemmaV2ScorerPin(root, [modulePath]), /breakpoint statement changed/u);
});

test('the browser transport serves installed bytes and never falls through to the development server', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'gemma-v2-cdp-transport-'));
  t.after(() => rm(root, {recursive: true, force: true}));
  const installed = join(root, 'node_modules/web-decision');
  const dependency = join(root, 'node_modules/@huggingface/tokenizers/dist');
  await Promise.all([mkdir(join(installed, 'dist'), {recursive: true}), mkdir(dependency, {recursive: true})]);
  const tokenizer = fileURLToPath(
    new URL('../../packages/web-decision/node_modules/@huggingface/tokenizers/dist/tokenizers.mjs', import.meta.url)
  );
  await Promise.all([
    writeFile(join(installed, modulePath), source),
    copyFile(tokenizer, join(dependency, 'tokenizers.mjs'))
  ]);
  const browser = await chromium.launch({headless: true, args: ['--disable-gpu', '--disable-software-rasterizer']});
  t.after(() => browser.close());
  const context = await browser.newContext();
  const page = await context.newPage();
  const routes = await routeGemmaExactArchive(context, rootUrl, installed, [modulePath]);
  const observed = page.waitForResponse(routes.url(modulePath));
  await page.goto(routes.url());
  assert.deepEqual(await (await observed).body(), Buffer.from(source));
  const pin = await findGemmaV2ScorerPin(installed, [modulePath]);
  const transport = routes.evidence();
  assert.deepEqual(transport.served, {[modulePath]: pin.sha256});
  await verifyGemmaArchiveResources(installed, [modulePath], transport.served, [modulePath]);
  await assert.rejects(
    verifyGemmaArchiveResources(installed, [modulePath], {[modulePath]: 'modified'}, [modulePath]),
    /Served Gemma bytes differ/u
  );
  const capture = await openGemmaV2RawCapture(page, routes.url(modulePath), pin);
  try {
    const {frames} = await capture.capture(['q1'], () => page.evaluate('globalThis.syntheticScorer()'));
    assert.equal(frames.length, 1);
  } finally {
    await capture.close();
  }
  assert.equal(
    await page.evaluate(() =>
      fetch('/outside.js').then(
        () => true,
        () => false
      )
    ),
    false
  );
  assert.equal(
    await page.evaluate(() =>
      fetch('/__gemma_public_archive/dist/index.js?rewrite').then(
        () => true,
        () => false
      )
    ),
    false
  );
  await writeFile(join(dependency, 'tokenizers.mjs'), 'export {};');
  await assert.rejects(routeGemmaExactArchive(context, rootUrl, installed, [modulePath]), /tokenizer bytes differ/u);
});
