import {openGemma4V2SourceFromInventory} from '../../packages/web-decision/src/strands/gemma4-v2-source.mts';
import {createGemma4V2RunnerFactory} from '../../packages/web-decision/src/strands/gemma4-v2-runner-factory.mts';
import inventory from '../../packages/web-decision/assets/strands-gemma4-v2/source-inventory.json' with {type: 'json'};

type Metadata = {bytes: number; sha256: string};
declare global {
  interface Window {
    runGemma4V2FactoryProbe: (ids: number[], split: number, pins: Record<string, Metadata>) => Promise<number[]>;
  }
}
async function data(name: string) {
  const response = await fetch('/strands-v2-runner-assets/' + name);
  if (!response.ok) throw new Error(`Missing fixture ${name}`);
  return response.arrayBuffer();
}
window.runGemma4V2FactoryProbe = async (ids, split, pins) => {
  const source = await openGemma4V2SourceFromInventory(
    inventory as never,
    'c4c1f9ca91d272d88f2559dd186fa0d37e878fa70d378c44a20de978179366aa',
    new URL('/strands-v2-local-source/', location.href)
  );
  try {
    const base = new URL('/strands-v2-runner-assets/', location.href);
    const asset = (name: string) => ({base, name, metadata: pins[name]!});
    const createRunner = createGemma4V2RunnerFactory({
      assets: {wasm: asset('linked.wasm'), localFrequency: asset('local.bin'), globalFrequency: asset('global.bin')},
      async createImageInserter() {
        return async (word, input) => {
          if (input.images.length || input.imageSpans.length) throw new Error('This probe is text only');
          return word;
        };
      }
    });
    const runner = await createRunner({source, readout: new Uint8Array(await data('readout.bin')), offline: false});
    try {
      return Array.from(
        await runner.run({stateIds: ids.slice(0, split), questionIds: ids.slice(split), images: [], imageSpans: []})
      );
    } finally {
      await runner.dispose();
    }
  } finally {
    source.dispose();
  }
};
