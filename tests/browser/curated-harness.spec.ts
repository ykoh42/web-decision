import {expect, test} from '@playwright/test';

test('the browser harness imports the curated package', async ({page}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function', null, {timeout: 5000});
  expect(errors).toEqual([]);
  expect(await page.evaluate(() => window.webDecision.listModels().map(({modelId}) => modelId))).toEqual([
    'kev-0.6b-q4f16',
    'kanana-2-v-decide-3b-2611'
  ]);
});
