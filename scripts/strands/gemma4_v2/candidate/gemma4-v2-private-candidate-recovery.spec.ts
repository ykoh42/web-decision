import {open, readFile, type FileHandle} from 'node:fs/promises';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {expect, test, type Route} from '@playwright/test';
import {readHardwareReport} from '../../../../tests/browser/mapika2b-hardware.mts';
import {
  GEMMA_V2_CPU_REFERENCE_PATH,
  GEMMA_V2_CPU_REFERENCE_SHA256,
  validateGemmaV2CpuReferences
} from '../../../release/strands-gemma4-v2-source-reference.mts';
import {gemmaV2EvidenceFingerprint, verifyGemmaV2Candidate} from './gemma4-v2-candidate-evidence.mts';

const modelId = 'strands-decider-e2b-gemma4-v2-2611';
const checkpointUrl =
  'https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v2-2611/resolve/a5f32ea1bffe2e91c25c0c641d444e8115e28212/**';
const checkpointBytes = 10_246_621_918;
const metadataFiles = new Set([
  'config.json',
  'decision_config.json',
  'processor_config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'readout.safetensors'
]);
const cors = {'access-control-allow-origin': '*', 'access-control-expose-headers': '*'};

type ImageResult = {
  answers: {color: {type: string; choice: string; confidence: number; probabilities: Record<string, number>}};
  usage: {input_tokens: number; output_tokens: number};
};

async function serveCheckpoint(route: Route, checkpoint: FileHandle, root: string) {
  const name = new URL(route.request().url()).pathname.split('/').at(-1)!;
  if (metadataFiles.has(name)) return route.fulfill({status: 200, path: resolve(root, name), headers: cors});
  if (name !== 'model.safetensors') throw new Error(`Unexpected checkpoint file: ${name}`);
  const match = /^bytes=(\d+)-(\d+)$/u.exec(route.request().headers().range ?? '');
  if (!match) throw new Error('Expected a checkpoint byte range.');
  const start = Number(match[1]);
  const end = Number(match[2]);
  if (!Number.isSafeInteger(start) || start > end || end >= checkpointBytes)
    throw new RangeError('Checkpoint range is out of bounds.');
  const body = Buffer.alloc(end - start + 1);
  if ((await checkpoint.read(body, 0, body.length, start)).bytesRead !== body.length)
    throw new Error('Short checkpoint range read.');
  await route.fulfill({
    status: 206,
    headers: {...cors, 'content-range': `bytes ${start}-${end}/${checkpointBytes}`},
    body
  });
}

test('installed Gemma image inference recovers on the same model after an incomplete upload', async ({page}) => {
  test.skip(process.env.WEB_DECISION_GEMMA_V2_CANDIDATE_RECOVERY !== '1', 'Enable the installed candidate GPU test.');
  test.setTimeout(600_000);
  const report = process.env.WEB_DECISION_GEMMA_V2_CANDIDATE_REPORT;
  const bundle = process.env.WEB_DECISION_INSTALLED_BUNDLE;
  const chainRoot = process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT;
  if (!report || !bundle || !chainRoot) throw new Error('Set the private candidate report, bundle, and fixture root.');
  const archive = await verifyGemmaV2Candidate(report, bundle);
  const fingerprint = await gemmaV2EvidenceFingerprint(report, fileURLToPath(import.meta.url));
  const fixturePath = new URL(`../../../../${GEMMA_V2_CPU_REFERENCE_PATH}`, import.meta.url);
  const fixtureBytes = await readFile(fixturePath);
  validateGemmaV2CpuReferences(fixtureBytes);
  const {requestJson, requestSha256, resultJson, resultSha256} = JSON.parse(fixtureBytes.toString('utf8'))
    .capturedCases['image-single'] as {
    requestJson: string;
    requestSha256: string;
    resultJson: string;
    resultSha256: string;
  };
  const expected = JSON.parse(resultJson) as ImageResult;
  const checkpointRoot = resolve(chainRoot, 'strands-e2b-v2');
  const checkpoint = await open(resolve(checkpointRoot, 'model.safetensors'), 'r');
  const serve = (route: Route) => serveCheckpoint(route, checkpoint, checkpointRoot);
  try {
    expect((await checkpoint.stat()).size).toBe(checkpointBytes);
    await page.route(checkpointUrl, serve);
    const hardware = await readHardwareReport(page);
    expect(hardware.status).toBe('Hardware accelerated');
    await page.goto('/');
    await page.waitForFunction(() => typeof window.webDecision?.loadDecisionModel === 'function');
    const observed = await page.evaluate(
      async ({id, input}) => {
        const api = window.webDecision as unknown as {
          loadDecisionModel(options: unknown): Promise<{
            systemOne(request: unknown): Promise<ImageResult>;
            dispose(): Promise<void>;
          }>;
        };
        function failure(request: Promise<unknown>) {
          return request.then(
            () => null,
            (error: Error & {code?: string}) => ({name: error.name, message: error.message, code: error.code})
          );
        }
        const validRequest = JSON.parse(input);
        // This Base64 string encodes the PNG signature alone, as in a truncated upload.
        const incompleteRequest = {...validRequest, images: ['iVBORw0KGgo=']};
        const model = await api.loadDecisionModel({modelId: id, cache: false});
        let rejected;
        let recovered;
        try {
          rejected = await failure(model.systemOne(incompleteRequest));
          if (!rejected) throw new Error('The incomplete upload was accepted.');
          recovered = await model.systemOne(validRequest);
        } finally {
          await model.dispose();
        }
        return {rejected, recovered, afterDispose: await failure(model.systemOne(validRequest))};
      },
      {id: modelId, input: requestJson}
    );
    expect(observed.rejected).toMatchObject({name: 'TypeError', message: expect.stringMatching(/PNG.*dimensions/u)});
    expect(observed.recovered.usage).toEqual(expected.usage);
    expect(observed.recovered.answers.color).toMatchObject({
      type: expected.answers.color.type,
      choice: expected.answers.color.choice
    });
    for (const option of Object.keys(expected.answers.color.probabilities))
      expect(Number.isFinite(observed.recovered.answers.color.probabilities[option])).toBe(true);
    expect(observed.afterDispose).toMatchObject({code: 'SESSION_CLOSED'});
    await verifyGemmaV2Candidate(report, bundle);
    expect(await gemmaV2EvidenceFingerprint(report, fileURLToPath(import.meta.url))).toBe(fingerprint);
    expect(await readFile(fixturePath)).toEqual(fixtureBytes);
    console.log(
      `PRIVATE_CANDIDATE_IMAGE_RECOVERY ${JSON.stringify({
        archiveSha256: archive.archiveSha256,
        sourceSnapshotSha256: archive.sourceSnapshotSha256,
        fingerprint,
        fixtureSha256: GEMMA_V2_CPU_REFERENCE_SHA256,
        requestSha256,
        referenceSha256: resultSha256,
        hardware,
        expected,
        observed
      })}`
    );
  } finally {
    try {
      await page.unroute(checkpointUrl, serve);
    } finally {
      await checkpoint.close();
    }
  }
});
