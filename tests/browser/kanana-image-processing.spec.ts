import {expect, test} from '@playwright/test';
import type {KananaImageProcessingReport} from './kanana-image-processing-entry.js';

test('Kanana image preprocessing module runs in the browser without model or GPU assets', async ({page}) => {
  await page.goto('/');
  await page.addScriptTag({type: 'module', url: '/kanana-image-processing-entry.ts'});

  const report = await page.evaluate(
    () => (window as Window & {kananaImageProcessingReport?: KananaImageProcessingReport}).kananaImageProcessingReport
  );

  expect(report).toEqual({
    imageTokenThw: [1, 4, 16],
    pixelCount: 256 * 3 * 2 * 16 * 16,
    pixelSha256: 'c6d09a833f0da491714d71e112cfeda0bf9cc362dce502450b79eb39d936cdc1',
    pixelType: 'Float32Array',
    visionGridThw: [1, 8, 32],
    width: 512,
    height: 128
  });
});
