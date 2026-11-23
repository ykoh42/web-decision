import {createGemma4V2ImagePreparer} from '../../packages/web-decision/src/strands/gemma4-v2-image.mts';
import {createGemma4V2Session} from '../../packages/web-decision/src/strands/gemma4-v2-session.mts';

type Outcome = {ok: boolean; name?: string; message?: string};
declare global {
  interface Window {
    initializeImageRecovery: (jpeg: string, png: string) => void;
    callRecovery: (kind: 'invalid' | 'png' | 'jpeg') => Promise<Outcome>;
    startCancelledImage: () => void;
    abortImage: () => void;
    cancelledImageResult: () => Promise<Outcome>;
    startRetryImage: () => void;
    retryImageResult: () => Promise<Outcome>;
    finishImageRecovery: () => Promise<{calls: number; disposed: boolean; afterDispose: Outcome}>;
  }
}

window.initializeImageRecovery = (jpeg, png) => {
  let calls = 0;
  let disposed = false;
  const session = createGemma4V2Session({
    tokenizer: {
      encode(text) {
        return {ids: Array.from(new TextEncoder().encode(text))};
      }
    },
    runner: {
      async run() {
        calls++;
        return [0, 0];
      },
      releaseCachedMemory() {},
      dispose() {
        disposed = true;
      }
    },
    prepareImage: createGemma4V2ImagePreparer({cache: null}).prepareImage,
    modelName: 'e2b-v2',
    temperature: 1.5456,
    softcap: 30
  });
  const request = (image: string) => ({state: 's', images: [image], questions: {q: {instructions: 'yes?'}}});
  const outcome = (task: Promise<unknown>): Promise<Outcome> =>
    task.then(
      () => ({ok: true}),
      (error: unknown) => ({ok: false, name: (error as Error).name, message: (error as Error).message})
    );
  let controller: AbortController | undefined;
  let cancelled: Promise<Outcome> | undefined;
  let retry: Promise<Outcome> | undefined;
  window.callRecovery = (kind) =>
    outcome(session.systemOne(request(kind === 'invalid' ? '%%%' : kind === 'png' ? png : jpeg)));
  window.startCancelledImage = () => {
    controller = new AbortController();
    cancelled = outcome(session.systemOne(request(jpeg), {signal: controller.signal}));
  };
  window.abortImage = () => controller!.abort();
  window.cancelledImageResult = () => cancelled!;
  window.startRetryImage = () => {
    retry = outcome(session.systemOne(request(jpeg)));
  };
  window.retryImageResult = () => retry!;
  window.finishImageRecovery = async () => {
    await session.dispose();
    return {calls, disposed, afterDispose: await outcome(session.systemOne(request(png)))};
  };
};
