import {loadModel} from '../../packages/web-decision/src/curated/adapters/strands-qwen35.ts';
import {createMapika2bImageCodecs} from '../../packages/web-decision/src/mapika2b/image-codecs.mts';
import {createQwen35Session} from '../../packages/web-decision/src/strands/qwen35-session.mts';

const modelId = 'strands-decider-2b-qwen3.5-v1-2610';
const tokens = [
  ['<|vision_start|>', 248053],
  ['<|image_pad|>', 248056],
  ['<|vision_end|>', 248054],
  ['<|endoftext|>', 248044],
  ['<bos>', 248043]
] as const;

function tokenizer() {
  return {
    decoder: {
      byte_decoder: Object.fromEntries(
        Array.from({length: 256}, (_, byte) => [String.fromCharCode(byte + 0x100), byte])
      )
    },
    get_added_tokens_decoder: () => new Set(tokens.map(([, id]) => id)),
    encode(text: string, options: {add_special_tokens: boolean}) {
      const ids: number[] = [];
      const names: string[] = [];
      if (options.add_special_tokens) {
        ids.push(248043);
        names.push('<bos>');
      }
      for (let i = 0; i < text.length;) {
        const special = tokens.find(([name]) => text.startsWith(name, i));
        if (special) {
          ids.push(special[1]);
          names.push(special[0]);
          i += special[0].length;
          continue;
        }
        let end = i + 1;
        while (end < text.length && !tokens.some(([name]) => text.startsWith(name, end))) end++;
        for (const byte of new TextEncoder().encode(text.slice(i, end).normalize('NFC'))) {
          ids.push(byte + 1);
          names.push(String.fromCharCode(byte + 0x100));
        }
        i = end;
      }
      return {ids, tokens: names};
    }
  };
}

async function hashFloats(values: Float32Array): Promise<string> {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < values.length; i++) view.setFloat32(i * 4, values[i]!, true);
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (byte) =>
    byte.toString(16).padStart(2, '0')
  ).join('');
}

declare global {
  interface Window {
    qwenImageAdapterBoundary: (images: string[], rows: number[]) => Promise<unknown>;
  }
}

window.qwenImageAdapterBoundary = async (images, rows) => {
  const codecs = createMapika2bImageCodecs({cache: null});
  await codecs.prepareJpeg();
  const magickWasm = await codecs.loadMagick();
  const observations: Array<{imageCount: number; hashes: string[]; grid: number[]}> = [];
  let releases = 0;
  let disposals = 0;
  const session = createQwen35Session({
    tokenizer: tokenizer(),
    modelName: modelId,
    graphOptions: {magickWasm},
    runner: {
      async run(chunk, input) {
        const values = input.imageInput?.pixelValues;
        if (!values) throw new Error('Images did not reach the Qwen graph adapter.');
        const hashes: string[] = [];
        let offset = 0;
        for (const row of rows) {
          hashes.push(await hashFloats(values.subarray(offset, offset + row * 1536)));
          offset += row * 1536;
        }
        if (offset !== values.length) throw new Error('Image feature lengths do not match the source order.');
        observations.push({imageCount: input.imageCount, hashes, grid: Array.from(input.imageGridThw)});
        return chunk.questions.map(() => [0.2, 0.8]);
      },
      releaseCachedMemory() {
        releases++;
      },
      dispose() {
        disposals++;
      }
    }
  });
  const model = await loadModel({}, {modelId} as never, {
    openCache: async () => null,
    loadPinnedModel: async () => session as never
  });
  const request = {
    state: 'Inspect the source images.',
    images,
    questions: {q: {type: 'noul' as const, instructions: 'Are these images visible?'}}
  };
  let invalidRejected = false;
  try {
    try {
      await model.systemOne({...request, images: ['%%%']});
    } catch {
      invalidRejected = true;
    }
    const answer = await model.systemOne(request);
    await model.releaseCachedMemory();
    await model.systemOne(request);
    return {
      invalidRejected,
      observations,
      answer,
      releases,
      get disposals() {
        return disposals;
      }
    };
  } finally {
    await model.dispose();
  }
};
