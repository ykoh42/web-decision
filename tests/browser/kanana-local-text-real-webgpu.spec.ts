import {readFileSync} from 'node:fs';
import {open, readFile, stat} from 'node:fs/promises';
import {resolve} from 'node:path';
import {expect, test} from '@playwright/test';
import type {ModelProgress} from 'web-decision';
import {kananaSnapshotDirectory, kananaUpstreamBase, verifyPinnedKananaSnapshot} from './kanana-snapshot.ts';

const enabled = process.env.WEB_DECISION_KANANA_FIREFOX_INFERENCE === '1';
const expectResourceLimit = process.env.WEB_DECISION_KANANA_EXPECT_RESOURCE_LIMIT === '1';
const probabilityTolerance = 0.001;
const fixtureBytes = readFileSync(new URL('../fixtures/kanana-decision-python-reference.json', import.meta.url));
type FixtureChoiceQuestion = Readonly<{
  type: 'choice';
  instructions: string;
  criteria: Readonly<Record<string, string>>;
}>;
type ChoiceAnswer = {choice: string; confidence: number; probabilities: Record<string, number>};
type SerializedError = Readonly<{
  name: string;
  message: string;
  stack: string | null;
  code: string | null;
  errorScope: string | null;
  size: number | null;
  cause: SerializedError | string | null;
  errors: readonly (SerializedError | string)[];
}>;
type ErrorDescription = SerializedError | string;
const reference = JSON.parse(fixtureBytes.toString('utf8')) as {
  modelId: string;
  revision: string;
  checkpointSha256: string;
  request: {
    state: string;
    state_format: 'text';
    temperature: number;
    questions: Readonly<Record<string, FixtureChoiceQuestion>>;
  };
  result: {
    answers: Readonly<
      Record<string, Readonly<{choice: string; confidence: number; probabilities: Record<string, number>}>>
    >;
  };
};

function findErrorCode(error: ErrorDescription | null, code: string): SerializedError | undefined {
  if (typeof error === 'string' || error === null) return undefined;
  if (error.code === code) return error;
  const cause = findErrorCode(error.cause, code);
  if (cause) return cause;
  for (const nested of error.errors) {
    const match = findErrorCode(nested, code);
    if (match) return match;
  }
  return undefined;
}

declare global {
  interface Window {
    __kananaProgress?(event: ModelProgress): Promise<void>;
    __kananaDeviceLoss?(event: {reason: string; message: string}): Promise<void>;
    __kananaWebGPUError?(event: {name: string; message: string}): Promise<void>;
    __kananaConsoleError?(events: readonly {name: string; message: string}[]): Promise<void>;
  }
}

test(
  expectResourceLimit
    ? 'captures the Kanana text resource limit on real Firefox WebGPU'
    : 'the current package bundle matches Kanana text reference on real Firefox WebGPU',
  async ({page, browser, browserName}) => {
    test.skip(
      !enabled && !expectResourceLimit,
      'Set WEB_DECISION_KANANA_FIREFOX_INFERENCE=1 or WEB_DECISION_KANANA_EXPECT_RESOURCE_LIMIT=1 to run Kanana on Firefox.'
    );
    expect(browserName).toBe('firefox');
    test.setTimeout(45 * 60 * 1000);
    let lastPercent = -10;
    const deviceLosses: {reason: string; message: string}[] = [];
    const webgpuErrors: {name: string; message: string}[] = [];
    const consoleErrors: {name: string; message: string}[][] = [];
    await page.exposeFunction('__kananaProgress', (event: ModelProgress) => {
      const fraction = event.fraction;
      if (typeof fraction !== 'number' || !Number.isFinite(fraction)) return;
      const percent = Math.floor(fraction * 100);
      if (percent < lastPercent + 10 && percent !== 100) return;
      lastPercent = percent;
      console.log(`KANANA_FIREFOX_PROGRESS ${JSON.stringify(event)}`);
    });
    await page.exposeFunction('__kananaDeviceLoss', (event: {reason: string; message: string}) => {
      deviceLosses.push(event);
      console.log(`KANANA_FIREFOX_DEVICE_LOST ${JSON.stringify(event)}`);
    });
    await page.exposeFunction('__kananaWebGPUError', (event: {name: string; message: string}) => {
      webgpuErrors.push(event);
      console.log(`KANANA_FIREFOX_WEBGPU_ERROR ${JSON.stringify(event)}`);
    });
    await page.exposeFunction('__kananaConsoleError', (events: readonly {name: string; message: string}[]) => {
      consoleErrors.push([...events]);
      console.log(`KANANA_FIREFOX_CONSOLE_ERROR ${JSON.stringify(events)}`);
    });
    await verifyPinnedKananaSnapshot();
    const checkpointPath = resolve(kananaSnapshotDirectory, 'model.safetensors');
    const checkpoint = await open(checkpointPath, 'r');
    const checkpointBytes = (await stat(checkpointPath)).size;
    const fetched: {name: string; bytes: number; range?: string}[] = [];
    try {
      await page.route(`${kananaUpstreamBase}**`, async (route) => {
        const request = route.request();
        const url = new URL(request.url());
        const name = url.pathname.slice(new URL(kananaUpstreamBase).pathname.length);
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
            await route.fulfill({status: 416, headers: {...cors, 'Content-Range': `bytes */${checkpointBytes}`}});
            return;
          }
          const start = Number(match[1]);
          const end = Number(match[2]);
          if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || end >= checkpointBytes) {
            await route.fulfill({status: 416, headers: {...cors, 'Content-Range': `bytes */${checkpointBytes}`}});
            return;
          }
          const bytes = Buffer.allocUnsafe(end - start + 1);
          let offset = 0;
          while (offset < bytes.length) {
            const {bytesRead} = await checkpoint.read(bytes, offset, bytes.length - offset, start + offset);
            if (!bytesRead) throw new Error(`Unexpected end of pinned Kanana checkpoint at ${start + offset}.`);
            offset += bytesRead;
          }
          fetched.push({name, bytes: bytes.length, range});
          await route.fulfill({
            status: 206,
            headers: {
              ...cors,
              'Accept-Ranges': 'bytes',
              'Content-Type': 'application/octet-stream',
              'Content-Length': String(bytes.length),
              'Content-Range': `bytes ${start}-${end}/${checkpointBytes}`
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
        fetched.push({name, bytes: bytes.length});
        await route.fulfill({status: 200, headers: {...cors, 'Content-Length': String(bytes.length)}, body: bytes});
      });

      await page.goto('/');
      await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');

      const report = await page.evaluate(async (fixture) => {
        const originalConsoleError = console.error.bind(console);
        console.error = (...values: unknown[]) => {
          const events = values.map((value) => {
            if (value === null || typeof value !== 'object') return {name: typeof value, message: String(value)};
            const error = value as {constructor?: {name?: unknown}; message?: unknown};
            return {
              name: typeof error.constructor?.name === 'string' ? error.constructor.name : 'Object',
              message: typeof error.message === 'string' ? error.message : String(value)
            };
          });
          void window.__kananaConsoleError?.(events);
          originalConsoleError(...values);
        };
        const gpu = navigator.gpu;
        if (gpu) {
          const requestAdapter = gpu.requestAdapter.bind(gpu);
          Object.defineProperty(gpu, 'requestAdapter', {
            configurable: true,
            value: async (options?: GPURequestAdapterOptions) => {
              const adapter = await requestAdapter(options);
              if (!adapter) return null;
              const requestDevice = adapter.requestDevice.bind(adapter);
              Object.defineProperty(adapter, 'requestDevice', {
                configurable: true,
                value: async (descriptor?: GPUDeviceDescriptor) => {
                  const device = await requestDevice(descriptor);
                  device.addEventListener('uncapturederror', (event) => {
                    const error = (event as GPUUncapturedErrorEvent).error;
                    void window.__kananaWebGPUError?.({
                      name: error?.constructor.name ?? 'GPUError',
                      message: error?.message ?? String(error)
                    });
                  });
                  void device.lost.then((info) =>
                    window.__kananaDeviceLoss?.({reason: info.reason, message: info.message})
                  );
                  return device;
                }
              });
              return adapter;
            }
          });
        }
        const adapter = await navigator.gpu?.requestAdapter({powerPreference: 'high-performance'});
        if (!adapter) throw new Error('Firefox did not provide a WebGPU adapter.');
        const fallback = adapter.info?.isFallbackAdapter ?? null;
        if (fallback !== false) throw new Error(`Expected a hardware WebGPU adapter; got ${String(fallback)}.`);
        const describeError = (error: unknown, depth = 0): ErrorDescription => {
          if (depth > 5) return String(error);
          if (error === null || typeof error !== 'object') return String(error);
          const value = error as {
            name?: unknown;
            message?: unknown;
            stack?: unknown;
            code?: unknown;
            errorScope?: unknown;
            size?: unknown;
            cause?: unknown;
            errors?: unknown;
          };
          return {
            name: typeof value.name === 'string' ? value.name : 'Error',
            message: typeof value.message === 'string' ? value.message : String(error),
            stack: typeof value.stack === 'string' ? value.stack : null,
            code: typeof value.code === 'string' ? value.code : null,
            errorScope: typeof value.errorScope === 'string' ? value.errorScope : null,
            size: typeof value.size === 'number' ? value.size : null,
            cause: value.cause === undefined ? null : describeError(value.cause, depth + 1),
            errors: Array.isArray(value.errors) ? value.errors.map((item) => describeError(item, depth + 1)) : []
          };
        };
        type LoadedModel = Awaited<ReturnType<typeof window.webDecision.loadDecisionModel>>;
        let model: LoadedModel | null = null;
        let loadError: ErrorDescription | null = null;
        try {
          model = await window.webDecision.loadDecisionModel({
            modelId: 'kanana-2-v-decide-3b-2611',
            cache: false,
            onProgress: (event) => window.__kananaProgress?.(event)
          });
        } catch (error) {
          loadError = describeError(error);
        }
        let answer: ChoiceAnswer | null = null;
        let inferenceError: ErrorDescription | null = null;
        if (model) {
          try {
            const output = await model.systemOne({
              state: fixture.request.state,
              stateFormat: fixture.request.state_format,
              questions: fixture.request.questions,
              temperature: fixture.request.temperature
            });
            answer = output.answers.delivery_status as ChoiceAnswer;
          } catch (error) {
            inferenceError = describeError(error);
          }
        }
        let disposalError: ErrorDescription | null = null;
        if (model) {
          try {
            await model.dispose();
          } catch (error) {
            disposalError = describeError(error);
          }
        }
        return {
          adapter: {
            vendor: adapter.info?.vendor ?? '',
            architecture: adapter.info?.architecture ?? '',
            maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage
          },
          modelId: model?.modelId ?? null,
          revision: model?.description.checkpoint.revision ?? null,
          checkpointSha256: model?.description.checkpoint.weightSha256 ?? null,
          answer,
          loadError,
          inferenceError,
          disposalError
        };
      }, reference);

      const fetchedBytes = fetched.reduce((sum, item) => sum + item.bytes, 0);
      const reportLabel = expectResourceLimit ? 'KANANA_FIREFOX_RESOURCE_LIMIT' : 'KANANA_FIREFOX_REAL_INFERENCE';
      console.log(
        `${reportLabel} ${JSON.stringify({browser: browser.version(), report, deviceLosses, webgpuErrors, consoleErrors, fetchedBytes, fetchedRanges: fetched.length})}`
      );
      if (expectResourceLimit) {
        const publicError = report.loadError ?? report.inferenceError;
        expect(report.answer).toBeNull();
        if (!publicError || typeof publicError === 'string')
          throw new Error('Kanana did not report a structured resource-limit error.');
        expect(publicError.code).toBe('DEVICE_RESOURCE_LIMIT');
        expect(findErrorCode(publicError, 'GPU_OUT_OF_MEMORY')).toBeDefined();
        return;
      }
      expect(report.loadError).toBeNull();
      expect(report.inferenceError).toBeNull();
      expect(report.disposalError).toBeNull();
      expect(fetchedBytes).toBeLessThan(checkpointBytes);
      const answer = report.answer;
      if (!answer) throw new Error('Kanana Firefox inference did not produce an answer.');
      const expected = reference.result.answers.delivery_status;
      if (!expected) throw new Error('Kanana reference fixture is missing delivery_status.');
      expect(report.modelId).toBe(reference.modelId.replace('kakaocorp/', ''));
      expect(report.revision).toBe(reference.revision);
      expect(report.checkpointSha256).toBe(reference.checkpointSha256);
      expect(answer.choice).toBe(expected.choice);
      expect(Object.keys(answer.probabilities)).toEqual(Object.keys(expected.probabilities));
      for (const [choice, probability] of Object.entries(expected.probabilities))
        expect(Math.abs(answer.probabilities[choice]! - probability), choice).toBeLessThan(probabilityTolerance);
      expect(Math.abs(answer.confidence - expected.confidence)).toBeLessThan(probabilityTolerance);
    } finally {
      await checkpoint.close();
    }
  }
);
