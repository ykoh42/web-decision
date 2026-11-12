import {open, readFile, stat} from 'node:fs/promises';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';
import type {KananaDecisionModel} from '../../packages/web-decision/src/curated/types.d.ts';
import {verifyArchiveInstall} from '../../scripts/release/verify-release-evidence.ts';
import {
  kananaModelId,
  kananaSnapshotDirectory,
  kananaUpstreamBase,
  verifyPinnedKananaSnapshot
} from './kanana-snapshot.ts';

const enabled = process.env.WEB_DECISION_KANANA_MAX_CONTEXT_GPU === '1';
const requestedMaxLength = Number(process.env.WEB_DECISION_KANANA_MAX_CONTEXT_LENGTH ?? '20480');
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

test('Kanana runs a truncated request at the requested token boundary', async ({page, browserName, browser}) => {
  test.skip(!enabled, 'Set WEB_DECISION_KANANA_MAX_CONTEXT_GPU=1 to run the long-context GPU check.');
  expect(Number.isSafeInteger(requestedMaxLength) && requestedMaxLength > 0 && requestedMaxLength <= 20_480).toBe(true);
  test.setTimeout(45 * 60 * 1000);

  const archivePath = process.env.WEB_DECISION_RELEASE_ARCHIVE;
  const packagePath = process.env.WEB_DECISION_RELEASE_PACKAGE_DIR;
  if (!archivePath || !packagePath) throw new Error('Set the exact candidate archive and installed package paths.');
  const archive = await verifyArchiveInstall(archivePath, packagePath);
  expect(process.env.WEB_DECISION_INSTALLED_BUNDLE).toBe(archive.bundlePath);
  console.log(
    `KANANA_MAX_CONTEXT_PACKAGE ${JSON.stringify({
      archiveSha256: archive.archiveSha256,
      packageTreeSha256: archive.packageTreeSha256,
      bundleSha256: archive.bundleSha256
    })}`
  );

  await verifyPinnedKananaSnapshot();
  const modelFile = await open(resolve(kananaSnapshotDirectory, 'model.safetensors'), 'r');
  const modelSize = (await stat(resolve(kananaSnapshotDirectory, 'model.safetensors'))).size;

  try {
    await page.route(`${kananaUpstreamBase}**`, async (route) => {
      const request = route.request();
      const name = new URL(request.url()).pathname.slice(upstreamPath.length);
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
        const startText = match[1];
        const endText = match[2];
        if (startText === undefined || endText === undefined) {
          await route.fulfill({status: 416, headers: {...cors, 'Content-Range': `bytes */${modelSize}`}});
          return;
        }
        const start = Number(startText);
        const end = Number(endText);
        if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || end >= modelSize) {
          await route.fulfill({status: 416, headers: {...cors, 'Content-Range': `bytes */${modelSize}`}});
          return;
        }
        const bytes = await readRange(modelFile, start, end);
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
      await route.fulfill({status: 200, headers: {...cors, 'Content-Length': String(bytes.byteLength)}, body: bytes});
    });

    const browserErrors: string[] = [];
    page.on('pageerror', (error) => browserErrors.push(error.message));
    page.on('console', (message) => console.log(`KANANA_BROWSER_${message.type().toUpperCase()} ${message.text()}`));
    await page.goto('/');
    await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');
    const result = await page.evaluate(async (maxLength: number) => {
      const devices: GPUDevice[] = [];
      const largeBuffers: {size: number; label: string | undefined}[] = [];
      const recentBuffers: {index: number; size: number; label: string | undefined; usage: number}[] = [];
      const recentDispatches: {index: number; pipeline: string; workgroups: [number, number, number]}[] = [];
      const bufferSizes = new WeakMap<GPUBuffer, number>();
      let liveBufferBytes = 0;
      let peakLiveBufferBytes = 0;
      let bufferCount = 0;
      let dispatchCount = 0;
      let pipelineLabel = '';
      const adapter = await navigator.gpu?.requestAdapter({powerPreference: 'high-performance'});
      if (!adapter) throw new Error('Chromium did not provide a WebGPU adapter.');
      const adapterWithFallback = adapter as GPUAdapter & {readonly isFallbackAdapter?: boolean};
      const isFallback = adapterWithFallback.isFallbackAdapter ?? adapter.info?.isFallbackAdapter ?? null;
      if (isFallback !== false) throw new Error(`Expected a hardware WebGPU adapter; got ${String(isFallback)}.`);
      const requestDevice = GPUAdapter.prototype.requestDevice;
      GPUAdapter.prototype.requestDevice = async function (...arguments_) {
        const device = await requestDevice.apply(this, arguments_);
        devices.push(device);
        device.addEventListener('uncapturederror', (event) => {
          const error = (event as Event & {readonly error?: {readonly message?: string}}).error;
          console.error(`KANANA_GPU_UNCAPTURED ${error?.message ?? 'unknown WebGPU error'}`);
        });
        void device.lost.then((info) => {
          const report = `KANANA_GPU_LOST ${info.reason} ${info.message}`;
          if (info.reason === 'destroyed') console.log(report);
          else console.error(report);
        });
        return device;
      };
      const createBuffer = GPUDevice.prototype.createBuffer;
      GPUDevice.prototype.createBuffer = function (descriptor) {
        try {
          const buffer = createBuffer.call(this, descriptor);
          bufferCount++;
          bufferSizes.set(buffer, descriptor.size);
          liveBufferBytes += descriptor.size;
          peakLiveBufferBytes = Math.max(peakLiveBufferBytes, liveBufferBytes);
          recentBuffers.push({
            index: bufferCount,
            size: descriptor.size,
            label: descriptor.label,
            usage: descriptor.usage
          });
          if (recentBuffers.length > 20) recentBuffers.shift();
          if (descriptor.size >= 500_000_000) largeBuffers.push({size: descriptor.size, label: descriptor.label});
          return buffer;
        } catch (error) {
          console.error(
            `KANANA_GPU_BUFFER_CREATE_FAILURE ${JSON.stringify({
              size: descriptor.size,
              label: descriptor.label,
              usage: descriptor.usage,
              mappedAtCreation: descriptor.mappedAtCreation,
              bufferCount,
              liveBufferBytes,
              error: error instanceof Error ? {name: error.name, message: error.message} : error
            })}`
          );
          throw error;
        }
      };
      const setPipeline = GPUComputePassEncoder.prototype.setPipeline;
      GPUComputePassEncoder.prototype.setPipeline = function (pipeline) {
        pipelineLabel = pipeline.label;
        return setPipeline.call(this, pipeline);
      };
      const dispatchWorkgroups = GPUComputePassEncoder.prototype.dispatchWorkgroups;
      GPUComputePassEncoder.prototype.dispatchWorkgroups = function (x, y, z: number = 1) {
        dispatchCount++;
        const rows = y ?? 1;
        const depth = z ?? 1;
        recentDispatches.push({index: dispatchCount, pipeline: pipelineLabel, workgroups: [x, rows, depth]});
        if (recentDispatches.length > 20) recentDispatches.shift();
        return dispatchWorkgroups.call(this, x, rows, depth);
      };
      const destroyBuffer = GPUBuffer.prototype.destroy;
      GPUBuffer.prototype.destroy = function () {
        const size = bufferSizes.get(this);
        if (size !== undefined) {
          bufferSizes.delete(this);
          liveBufferBytes -= size;
        }
        destroyBuffer.call(this);
      };
      const limits = adapter.limits;
      console.log(
        `KANANA_MAX_CONTEXT_ADAPTER ${JSON.stringify({
          vendor: adapter.info?.vendor,
          architecture: adapter.info?.architecture,
          device: adapter.info?.device,
          maxBufferSize: limits.maxBufferSize,
          maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
          maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
          maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
          maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
          isFallback
        })}`
      );

      const model = (await window.webDecision.loadDecisionModel({
        modelId: 'kanana-2-v-decide-3b-2611',
        cache: false
      })) as unknown as KananaDecisionModel;
      try {
        const modelMaxLength = model.description.features.textChoice.limits.find(
          ({resource, scope}) => resource === 'tokens' && scope === 'question'
        )?.max;
        const activeDevice = devices.at(-1);
        if (!activeDevice) throw new Error('Kanana did not create a WebGPU device.');
        activeDevice.pushErrorScope('internal');
        activeDevice.pushErrorScope('out-of-memory');
        activeDevice.pushErrorScope('validation');
        let output;
        try {
          output = await model.systemOne(
            {
              state: 'x '.repeat(maxLength + 10_000),
              stateFormat: 'text',
              questions: {
                max_context: {
                  type: 'choice',
                  instructions: 'Choose the label for this maximum-context boundary test.',
                  criteria: {first: 'First option', second: 'Second option'}
                }
              }
            },
            {maxLength}
          );
        } catch (error) {
          const scopedErrors = await Promise.all([
            activeDevice.popErrorScope(),
            activeDevice.popErrorScope(),
            activeDevice.popErrorScope()
          ]);
          const deviceLoss = await Promise.race([
            activeDevice.lost,
            new Promise<undefined>((resolveLoss) => setTimeout(() => resolveLoss(undefined), 0))
          ]);
          const detail =
            error instanceof Error ? {name: error.name, message: error.message, stack: error.stack} : error;
          console.error(
            `KANANA_MAX_CONTEXT_FAILURE ${JSON.stringify({
              detail,
              scopedErrors,
              deviceLoss,
              largeBuffers,
              bufferCount,
              liveBufferBytes,
              peakLiveBufferBytes,
              recentBuffers,
              dispatchCount,
              recentDispatches,
              deviceLimits: {
                maxBufferSize: activeDevice.limits.maxBufferSize,
                maxStorageBufferBindingSize: activeDevice.limits.maxStorageBufferBindingSize
              }
            })}`
          );
          throw error;
        }
        const scopedErrors = await Promise.all([
          activeDevice.popErrorScope(),
          activeDevice.popErrorScope(),
          activeDevice.popErrorScope()
        ]);
        if (scopedErrors.some((error) => error !== null))
          throw new Error(`Kanana WebGPU error scope reported ${JSON.stringify(scopedErrors)}.`);
        const answer = output.answers.max_context;
        if (!answer || answer.type !== 'choice') throw new Error('Kanana returned no choice at maximum context.');
        return {
          modelId: model.modelId,
          backend: model.description.backend.name,
          modelMaxLength,
          answer: answer.choice,
          probabilities: answer.probabilities,
          usage: output.usage,
          inputTruncation: output.inputTruncation,
          isFallback,
          bufferUsage: {bufferCount, liveBufferBytes, peakLiveBufferBytes, largeBuffers},
          deviceLimits: {
            maxBufferSize: activeDevice.limits.maxBufferSize,
            maxStorageBufferBindingSize: activeDevice.limits.maxStorageBufferBindingSize
          }
        };
      } finally {
        await model.dispose();
      }
    }, requestedMaxLength);

    expect(result.modelId).toBe(kananaModelId);
    expect(result.backend).toBe('tvm-webgpu');
    expect(result.modelMaxLength).toBe(20_480);
    expect(result.isFallback).toBe(false);
    expect(result.inputTruncation).toEqual({
      truncated: true,
      maxLength: requestedMaxLength,
      queryQuestionIds: ['max_context'],
      candidateDocuments: []
    });
    expect(['first', 'second']).toContain(result.answer);
    expect(Object.keys(result.probabilities)).toEqual(['first', 'second']);
    expect(Object.values(result.probabilities).every(Number.isFinite)).toBe(true);
    expect(Math.abs(Object.values(result.probabilities).reduce((sum, value) => sum + value, 0) - 1)).toBeLessThan(1e-6);
    expect(browserErrors).toEqual([]);
    console.log(`KANANA_MAX_CONTEXT_REAL_GPU ${JSON.stringify({browserName, version: browser.version(), result})}`);
  } finally {
    await modelFile.close();
  }
});
