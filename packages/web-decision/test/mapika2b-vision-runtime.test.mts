import assert from 'node:assert/strict';
import test from 'node:test';
import {cacheMapika2bVisionAssets, openMapika2bVision} from '../src/mapika2b/vision-runtime.mts';

test('Mapika vision stops before loading artifacts when the request is already aborted', async () => {
  const controller = new AbortController();
  const reason = new Error('cancelled');
  controller.abort(reason);
  await assert.rejects(cacheMapika2bVisionAssets({signal: controller.signal}), (error) => error === reason);
  await assert.rejects(
    openMapika2bVision({
      signal: controller.signal,
      source: {
        readTensor: async () => {
          throw new Error('unexpected read');
        }
      }
    }),
    (error) => error === reason
  );
});
