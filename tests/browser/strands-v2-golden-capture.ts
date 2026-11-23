import {
  loadGemma4V2Model,
  type Gemma4V2RunnerFactory
} from '../../packages/web-decision/src/strands/gemma4-v2-loader.mts';
import {createGemma4V2RunnerFactory} from '../../packages/web-decision/src/strands/gemma4-v2-runner-factory.mts';
import {createGemma4V2ImageInserter} from '../../packages/web-decision/src/strands/gemma4-v2-image-runner.mts';
import {transposedSplitGraphs} from '../../packages/web-decision/src/strands/gemma4-v2-split-adapter.mts';
import {readVerifiedAsset} from '../../packages/web-decision/src/assets.js';
import {createGemma4V2ScoringKernel} from '../../packages/web-decision/src/strands/gemma4-v2-scoring.mts';

type Pin = {bytes: number; sha256: string};
type Request = Parameters<Awaited<ReturnType<typeof loadGemma4V2Model>>['systemOne']>[0];
type Golden = {
  request: Request;
  state_ids: number[];
  images: {span: [number, number]}[];
  questions: Record<string, {codes: string[]; question_ids: number[]}>;
};
type Params = {base: string; pins: Record<string, Pin>};
type Capture = {line: number; questions: Record<string, {codes: string[]; probabilities: number[]}>; answer: unknown};
const upstream =
  'https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v2-2611/resolve/a5f32ea1bffe2e91c25c0c641d444e8115e28212/';

declare global {
  interface Window {
    openGemmaGoldenCapture(input: Params): Promise<void>;
    runGemmaGoldenCapture(input: {line: number; golden: Golden}): Promise<Capture>;
    closeGemmaGoldenCapture(): Promise<void>;
  }
}

type Session = Awaited<ReturnType<typeof loadGemma4V2Model>>;
let session: Session | undefined;
let restoreFetch: (() => void) | undefined;
let active: {golden: Golden; names: string[]; index: number; questions: Capture['questions']} | undefined;

window.openGemmaGoldenCapture = async ({base: sourceBase, pins}) => {
  if (session || restoreFetch) throw new Error('Golden capture is already open');
  const originalFetch = globalThis.fetch;
  const base = new URL(sourceBase);
  restoreFetch = () => {
    globalThis.fetch = originalFetch;
    restoreFetch = undefined;
  };
  globalThis.fetch = (input, init) => {
    const request = new Request(input, init);
    if (!request.url.startsWith(upstream)) return originalFetch(request);
    return originalFetch(new Request(new URL('checkpoint/' + request.url.slice(upstream.length), base), request));
  };
  try {
    const asset = (name: string) => ({base: new URL('graph/', base), name, metadata: pins[name]!});
    const scoring = asset('scoring.bin');
    const bytes = await readVerifiedAsset(scoring.base, {files: {[scoring.name]: scoring.metadata}}, scoring.name, {});
    const score = createGemma4V2ScoringKernel(bytes);
    const factory = createGemma4V2RunnerFactory({
      assets: {
        wasm: asset('linked.wasm'),
        localFrequency: asset('local.bin'),
        globalFrequency: asset('global.bin'),
        hostReadout: true
      },
      splitExecution: {graphs: transposedSplitGraphs(), chunkTokens: 256},
      createImageInserter: ({runtime, device, vm, source, signal}) =>
        createGemma4V2ImageInserter(runtime, device, vm, source, signal)
    });
    const createRunner: Gemma4V2RunnerFactory = async (options) => {
      const runner = await factory(options);
      return {
        async run(input, signal) {
          const current = active;
          if (!current) throw new Error('No active golden request');
          const name = current.names[current.index++];
          if (!name) throw new Error('Extra question');
          const {golden, questions} = current;
          const expected = golden.questions[name]!;
          const match = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
          if (!match(input.stateIds, golden.state_ids) || !match(input.questionIds, expected.question_ids))
            throw new Error(`Golden tokens differ for ${name}`);
          if (
            !match(
              input.imageSpans.map(({start, end}) => [start, end]),
              golden.images.map(({span}) => span)
            )
          )
            throw new Error('Golden image spans differ');
          const logits = await runner.run(input, signal);
          questions[name] = {codes: expected.codes, probabilities: score(logits, expected.codes.length)};
          return logits;
        },
        releaseCachedMemory: runner.releaseCachedMemory.bind(runner),
        dispose: runner.dispose.bind(runner)
      };
    };
    session = await loadGemma4V2Model({}, {createRunner, scoringAsset: scoring});
  } catch (error) {
    restoreFetch?.();
    throw error;
  }
};

window.runGemmaGoldenCapture = async ({line, golden}) => {
  if (!session || active) throw new Error('Golden capture is not open or is busy');
  const current = {golden, names: Object.keys(golden.questions), index: 0, questions: {}};
  active = current;
  try {
    const answer = await session.systemOne(golden.request);
    if (current.index !== current.names.length) throw new Error('Missing question');
    return {line, questions: current.questions, answer};
  } finally {
    active = undefined;
  }
};

window.closeGemmaGoldenCapture = async () => {
  const model = session;
  session = undefined;
  try {
    await model?.dispose();
  } finally {
    restoreFetch?.();
  }
};
