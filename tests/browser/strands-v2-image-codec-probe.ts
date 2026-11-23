import {createGemma4V2ImagePreparer} from '../../packages/web-decision/src/strands/gemma4-v2-image.mts';

type Result = {tokens: number; pixelHash: string; positionHash: string};
declare global {
  interface Window {
    initGemmaImageCache: (name: string, images: string[]) => Promise<Result[]>;
    testGemmaImageOffline: (name: string, images: string[]) => Promise<Result[]>;
    clearGemmaImageCache: (name: string) => Promise<boolean>;
    gemmaImageOfflineReady: (name: string) => Promise<boolean>;
    corruptGemmaImageCache: (name: string) => Promise<void>;
  }
}
async function hash(data: ArrayBuffer): Promise<string> {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', data)), (byte) =>
    byte.toString(16).padStart(2, '0')
  ).join('');
}
async function prepare(name: string, images: string[], offline: boolean) {
  const cache = await caches.open(name);
  const preparer = createGemma4V2ImagePreparer({cache, offline});
  await preparer.prepareOffline();
  const results: Result[] = [];
  for (const image of images) {
    const {tokens, value} = await preparer.prepareImage(image);
    results.push({
      tokens,
      pixelHash: await hash(value.pixels.buffer),
      positionHash: await hash(value.positions.buffer)
    });
  }
  return results;
}
window.initGemmaImageCache = (name, images) => prepare(name, images, false);
window.testGemmaImageOffline = (name, images) => prepare(name, images, true);
window.clearGemmaImageCache = (name) => caches.delete(name);

window.gemmaImageOfflineReady = async (name) =>
  createGemma4V2ImagePreparer({cache: await caches.open(name), offline: true}).isOfflineReady();
window.corruptGemmaImageCache = async (name) => {
  const cache = await caches.open(name);
  const keys = await cache.keys();
  if (keys.length !== 2) throw new Error('Expected both verified image codec entries');
  await cache.put(keys[0]!, new Response(new Uint8Array([0])));
};
