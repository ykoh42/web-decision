import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';

const assetsDirectory = process.env.WEB_DECISION_KANANA_ONLINE_ATTENTION_ASSETS;
const enabled = process.env.WEB_DECISION_KANANA_ONLINE_ATTENTION_SMOKE === '1' && Boolean(assetsDirectory);

test('Kanana one-layer TVM graph runs on WebGPU through the configured sequence boundary', async ({page, browser}) => {
  test.skip(!enabled, 'Set WEB_DECISION_KANANA_ONLINE_ATTENTION_SMOKE=1 and its artifact directory.');
  test.setTimeout(10 * 60 * 1000);
  await page.goto('/');
  await page.addScriptTag({path: resolve('.toolchains/apache-tvm-src/web/lib/index.js')});
  const result = await page.evaluate(async () => {
    const runnerUrl = new URL('/kanana-online-attention-smoke.ts', location.href).href;
    const runner = await import(/* @vite-ignore */ runnerUrl);
    return runner.runKananaOnlineAttentionSmoke(
      new URL('/kanana-online-attention/', location.href).href,
      new URL('pytorch-reference.json', new URL('/kanana-online-attention/', location.href)).href
    );
  });
  console.log(`KANANA_ONLINE_ATTENTION_RESULT ${JSON.stringify({browser: browser.version(), ...result})}`);
  expect(result.webgpuErrors).toEqual([]);
  expect(result.sequenceCases.map(({sequenceLength}: {sequenceLength: number}) => sequenceLength)).toEqual([1, 8]);
  for (const sequenceCase of result.sequenceCases) {
    expect(sequenceCase.maxAbsDelta).toBeLessThan(0.03);
    expect(sequenceCase.rmse).toBeLessThan(0.005);
  }
  expect(result.boundaryCases.map(({sequenceLength}: {sequenceLength: number}) => sequenceLength)).toEqual([
    20_479, 20_480
  ]);
  for (const boundaryCase of result.boundaryCases) {
    expect(boundaryCase.outputShape).toEqual([1, boundaryCase.sequenceLength, 2560]);
    expect(boundaryCase.sampleValues.every(Number.isFinite)).toBe(true);
  }
});
