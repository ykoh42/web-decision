import {open, readFile, stat} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';
import {createKananaReleaseEvidence, KANANA_RELEASE_EVIDENCE_PREFIX} from './kanana-candidate-evidence.ts';
import type {KananaDecisionModel} from '../../packages/web-decision/src/curated/types.d.ts';
import {
  kananaCheckpointSha256,
  kananaModelId,
  kananaRevision,
  kananaSnapshotDirectory,
  kananaUpstreamBase,
  verifyPinnedKananaSnapshot
} from './kanana-snapshot.ts';
const localBuildMode = process.env.WEB_DECISION_KANANA_LOCAL_BUILD_GPU === '1';
const enabled =
  process.env.WEB_DECISION_KANANA_CURATED_REAL_GPU === '1' ||
  process.env.WEB_DECISION_KANANA_LOCAL_GPU === '1' ||
  localBuildMode;
const upstreamPath = new URL(kananaUpstreamBase).pathname;

async function readRange(file: Awaited<ReturnType<typeof open>>, start: number, end: number) {
  const bytes = Buffer.allocUnsafe(end - start + 1);
  let offset = 0;
  while (offset < bytes.byteLength) {
    const {bytesRead} = await file.read(bytes, offset, bytes.byteLength - offset, start + offset);
    if (!bytesRead) throw new Error(`Unexpected end of local Kanana checkpoint at ${start + offset}.`);
    offset += bytesRead;
  }
  return bytes;
}

test('the Kanana candidate runs the pinned image decision on WebGPU', async ({
  page,
  browserName,
  browser
}, testInfo) => {
  test.skip(!enabled, 'Set WEB_DECISION_KANANA_CURATED_REAL_GPU=1 to run the Kanana candidate GPU check.');
  test.setTimeout(45 * 60 * 1000);

  await verifyPinnedKananaSnapshot();
  const modelPath = resolve(kananaSnapshotDirectory, 'model.safetensors');
  const modelFile = await open(modelPath, 'r');
  const modelSize = (await stat(modelPath)).size;
  const imageFixturePath = resolve('tests/fixtures/kanana-synthetic-image.png');
  const imageReferencePath = resolve('tests/fixtures/kanana-synthetic-image-python-reference.json');
  const imageFixture = await readFile(imageFixturePath);
  const imageFixtureSha256 = createHash('sha256').update(imageFixture).digest('hex');
  const imageReference = JSON.parse(await readFile(imageReferencePath, 'utf8')) as {
    input: {bytes: number; sha256: string; width: number; height: number};
    request: {state: string; temperature: number};
    result: {answer: string; probabilities: Record<string, number>; confidence: number};
  };
  expect(imageFixture.byteLength).toBe(imageReference.input.bytes);
  expect(imageFixtureSha256).toBe(imageReference.input.sha256);
  const fetched: {name: string; bytes: number; range?: string}[] = [];
  const visionWasmRequests: string[] = [];

  try {
    page.on('request', (request) => {
      if (new URL(request.url()).pathname.includes('kanana-vision-model')) visionWasmRequests.push(request.url());
    });
    await page.route(`${kananaUpstreamBase}**`, async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      const name = url.pathname.slice(upstreamPath.length);
      const cors = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Range',
        'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        'Access-Control-Expose-Headers': 'Accept-Ranges, Content-Length, Content-Range'
      };

      if (request.method() === 'OPTIONS') {
        await route.fulfill({status: 204, headers: cors});
        return;
      }

      if (name === 'model.safetensors') {
        const range = request.headers().range;
        const match = range && /^bytes=(\d+)-(\d+)$/u.exec(range);
        if (!match) {
          await route.fulfill({status: 416, headers: {...cors, 'Content-Range': `bytes */${modelSize}`}});
          return;
        }
        const start = Number(match[1]);
        const end = Number(match[2]);
        if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || end >= modelSize) {
          await route.fulfill({status: 416, headers: {...cors, 'Content-Range': `bytes */${modelSize}`}});
          return;
        }
        const bytes = await readRange(modelFile, start, end);
        fetched.push({name, bytes: bytes.byteLength, range});
        await route.fulfill({
          status: 206,
          headers: {
            ...cors,
            'Accept-Ranges': 'bytes',
            'Content-Type': 'application/octet-stream',
            'Content-Length': String(bytes.byteLength),
            'Content-Range': `bytes ${start}-${end}/${modelSize}`
          },
          body: bytes
        });
        return;
      }

      if (!['config.json', 'tokenizer_config.json', 'tokenizer.json'].includes(name)) {
        await route.fulfill({status: 404, headers: cors});
        return;
      }

      const bytes = await readFile(resolve(kananaSnapshotDirectory, name));
      fetched.push({name, bytes: bytes.byteLength});
      await route.fulfill({status: 200, headers: {...cors, 'Content-Length': String(bytes.byteLength)}, body: bytes});
    });

    const browserErrors: string[] = [];
    page.on('pageerror', (error) => browserErrors.push(error.message));
    await page.goto('/');
    const {result, imageCountBoundary, adapterInfo, input} = await page.evaluate(
      async ({imageBase64, imageReferenceState}) => {
        const adapter = await navigator.gpu?.requestAdapter({powerPreference: 'high-performance'});
        if (!adapter) throw new Error('Chromium did not provide a WebGPU adapter.');
        const adapterWithFallback = adapter as GPUAdapter & {readonly isFallbackAdapter?: boolean};
        const isFallback = adapterWithFallback.isFallbackAdapter ?? adapter.info?.isFallbackAdapter ?? null;
        if (isFallback !== false) throw new Error(`Expected a hardware WebGPU adapter; got ${String(isFallback)}.`);
        const model = (await window.webDecision.loadDecisionModel({
          modelId: 'kanana-2-v-decide-3b-2611',
          cache: false
        })) as unknown as KananaDecisionModel;
        try {
          const imageBytes = Uint8Array.from(atob(imageBase64), (value) => value.charCodeAt(0));
          const image = new Blob([imageBytes], {type: 'image/png'});
          const bitmap = await createImageBitmap(image);
          if (bitmap.width !== 256 || bitmap.height !== 256) throw new Error('Pinned image must be 256x256.');
          bitmap.close();
          const request = (images: Blob[]) => ({
            state: imageReferenceState,
            images,
            stateFormat: 'text' as const,
            temperature: 1.0,
            questions: {
              action: {
                type: 'choice' as const,
                instructions: 'Which object is most prominent?',
                criteria: {
                  rectangle: 'A gold rectangle on the left is prominent.',
                  circle: 'A teal circle on the right is prominent.'
                }
              }
            }
          });
          const output = await model.systemOne(request([image]));
          const answer = output.answers.action;
          if (!answer || answer.type !== 'choice') throw new Error('Kanana returned no choice result.');
          const tenImages = Array.from({length: 10}, () => image.slice(0, image.size, 'image/png'));
          const tenImageOutput = await model.systemOne(request(tenImages));
          const tenImageAnswer = tenImageOutput.answers.action;
          if (!tenImageAnswer || tenImageAnswer.type !== 'choice')
            throw new Error('Kanana returned no choice result for ten images.');
          let overLimitCode: unknown;
          try {
            await model.systemOne(request([...tenImages, image]));
          } catch (error) {
            overLimitCode = (error as {code?: unknown}).code;
          }
          if (overLimitCode !== 'image_count')
            throw new Error(`Kanana did not reject 11 images at the pinned model limit: ${String(overLimitCode)}.`);
          const imageDigest = await crypto.subtle.digest('SHA-256', await image.arrayBuffer());
          const imageSha256 = Array.from(new Uint8Array(imageDigest), (byte) =>
            byte.toString(16).padStart(2, '0')
          ).join('');
          const checkpointSha256 = model.description.checkpoint.weightSha256;
          if (!checkpointSha256) throw new Error('Kanana descriptor omitted the checkpoint digest.');
          return {
            adapterInfo: {
              vendor: adapter.info?.vendor ?? '',
              architecture: adapter.info?.architecture ?? '',
              device: adapter.info?.device ?? '',
              isFallback,
              isFallbackAdapter: isFallback,
              features: Array.from(adapter.features),
              limits: {
                maxBufferSize: adapter.limits.maxBufferSize,
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
                maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage
              }
            },
            imageCountBoundary: {
              accepted: tenImages.length,
              answer: tenImageAnswer.choice,
              probabilities: tenImageAnswer.probabilities,
              usage: tenImageOutput.usage,
              rejected: tenImages.length + 1,
              rejectionCode: overLimitCode
            },
            input: {kind: 'pinned-png-v1', width: 256, height: 256, bytes: imageBytes.byteLength, sha256: imageSha256},
            result: {
              modelId: model.modelId,
              descriptionModelId: model.description.modelId,
              checkpointRevision: model.description.checkpoint.revision,
              checkpointSha256,
              backend: model.description.backend.name,
              assetManifestSha256: model.description.assets.manifestSha256,
              answer: answer.choice,
              confidence: answer.confidence,
              probabilities: answer.probabilities,
              probabilityEntries: answer.probabilityEntries,
              usage: output.usage
            }
          };
        } finally {
          await model.dispose();
        }
      },
      {imageBase64: imageFixture.toString('base64'), imageReferenceState: imageReference.request.state}
    );

    expect(adapterInfo.isFallback).toBe(false);
    expect(result.modelId).toBe(kananaModelId);
    expect(result.descriptionModelId).toBe(kananaModelId);
    expect(result.checkpointRevision).toBe(kananaRevision);
    expect(result.checkpointSha256).toBe(kananaCheckpointSha256);
    expect(result.backend).toBe('tvm-webgpu');
    expect(result.answer).toBe(imageReference.result.answer);
    expect(Object.keys(result.probabilities)).toEqual(['rectangle', 'circle']);
    expect(Object.values(result.probabilities).every(Number.isFinite)).toBe(true);
    expect(Math.abs(Object.values(result.probabilities).reduce((sum, value) => sum + value, 0) - 1)).toBeLessThan(1e-6);
    expect(result.usage.input_tokens).toBeGreaterThan(0);
    const probabilityDeltas = Object.fromEntries(
      Object.keys(imageReference.result.probabilities).map((key) => [
        key,
        Math.abs(result.probabilities[key]! - imageReference.result.probabilities[key]!)
      ])
    );
    const confidenceDelta = Math.abs(result.confidence - imageReference.result.confidence);
    const imageProbabilityValues = Object.values(result.probabilities);
    const imageMaximum = Math.max(...imageProbabilityValues);
    const computedConfidence =
      imageMaximum - (imageProbabilityValues.reduce((sum, value) => sum + value, 0) - imageMaximum);
    for (const delta of Object.values(probabilityDeltas)) expect(delta).toBeLessThan(0.001);
    expect(confidenceDelta).toBeLessThan(0.001);
    expect(Math.abs(result.confidence - computedConfidence)).toBeLessThan(1e-9);
    console.log(
      `KANANA_IMAGE_REFERENCE_DELTA ${JSON.stringify({
        probabilities: probabilityDeltas,
        confidence: confidenceDelta
      })}`
    );
    expect(imageCountBoundary.accepted).toBe(10);
    expect(['rectangle', 'circle']).toContain(imageCountBoundary.answer);
    expect(Object.keys(imageCountBoundary.probabilities).sort()).toEqual(['circle', 'rectangle']);
    expect(Object.values(imageCountBoundary.probabilities).every(Number.isFinite)).toBe(true);
    expect(
      Math.abs(Object.values(imageCountBoundary.probabilities).reduce((sum, value) => sum + value, 0) - 1)
    ).toBeLessThan(1e-6);
    expect(imageCountBoundary.usage.input_tokens).toBeGreaterThan(0);
    expect(imageCountBoundary.rejected).toBe(11);
    expect(visionWasmRequests).toHaveLength(1);
    expect(fetched.some(({name}) => name === 'model.safetensors')).toBe(true);
    expect(browserErrors).toEqual([]);
    console.log(`KANANA_MAX_IMAGE_COUNT_GPU ${JSON.stringify(imageCountBoundary)}`);
    console.log(
      `KANANA_CURATED_REAL_GPU ${JSON.stringify({browserName, version: browser.version(), adapterInfo, result, fetchedRanges: fetched.length, fetchedBytes: fetched.reduce((sum, item) => sum + item.bytes, 0)})}`
    );
    const userAgent = await page.evaluate(() => navigator.userAgent);
    const evidence = await createKananaReleaseEvidence({
      kind: 'image',
      testInfo,
      browserName,
      browserVersion: browser.version(),
      userAgent,
      model: {
        requestedId: kananaModelId,
        modelId: result.modelId,
        descriptionModelId: result.descriptionModelId,
        checkpointRevision: result.checkpointRevision,
        checkpointSha256: result.checkpointSha256,
        assetManifestSha256: result.assetManifestSha256,
        backend: result.backend
      },
      adapter: adapterInfo,
      observation: {
        result,
        imageCountBoundary,
        input,
        referenceFixtureSha256: createHash('sha256')
          .update(await readFile(imageReferencePath))
          .digest('hex'),
        visionWasmRequests: visionWasmRequests.length,
        fetchedRanges: fetched.length,
        fetchedBytes: fetched.reduce((sum, item) => sum + item.bytes, 0)
      }
    });
    if (evidence) console.log(`${KANANA_RELEASE_EVIDENCE_PREFIX}${JSON.stringify(evidence)}`);
  } finally {
    await modelFile.close();
  }
});
