import {expect, test} from '@playwright/test';

const enabled = process.env.WEB_DECISION_KANANA_ONLINE_FULL_SMOKE === '1';
const checkpointBase = process.env.WEB_DECISION_KANANA_LOCAL_RANGE_BASE ?? 'http://127.0.0.1:41774/';

test('Kanana full TVM graph matches the pinned checkpoint on WebGPU', async ({page, browser}) => {
  test.skip(!enabled, 'Set WEB_DECISION_KANANA_ONLINE_FULL_SMOKE=1 to run the full checkpoint smoke test.');
  test.setTimeout(45 * 60 * 1000);
  await page.goto('/');
  await page.addScriptTag({path: '.toolchains/apache-tvm-src/web/lib/index.js'});
  const result = await page.evaluate(
    async ({rangeBase}) => {
      const runnerUrl = new URL('/kanana-online-attention-full-smoke.ts', location.href).href;
      const runner = await import(/* @vite-ignore */ runnerUrl);
      return runner.runKananaOnlineAttentionFullSmoke(
        new URL('/kanana-online-attention/', location.href).href,
        rangeBase
      );
    },
    {rangeBase: checkpointBase}
  );
  console.log(`KANANA_FULL_ONLINE_ATTENTION_RESULT ${JSON.stringify({browser: browser.version(), ...result})}`);
  expect(result.webgpuErrors).toEqual([]);
  expect(result.loadedWeightBytes).toBeGreaterThan(6_000_000_000);
  expect(
    result.sequenceCases.map(({name, sequenceLength}: {name: string; sequenceLength: number}) => [name, sequenceLength])
  ).toEqual([
    ['no-images', 1],
    ['no-images', 8],
    ['two-image-tokens', 8]
  ]);
  for (const sequenceCase of result.sequenceCases) {
    expect(sequenceCase.maxAbsDelta).toBeLessThan(0.03);
    expect(sequenceCase.rmse).toBeLessThan(0.005);
  }
});
