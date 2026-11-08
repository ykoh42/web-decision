import {KANANA_IMAGE_MARKER, renderKananaChatTemplate} from './chat-template.js';

type TokenizerEncodeOptions = {add_special_tokens?: boolean};
type TokenizerDecodeOptions = {skip_special_tokens?: boolean};
export type KananaTokenizer = {
  encode(text: string, options?: TokenizerEncodeOptions): unknown;
  decode?(ids: number[], options?: TokenizerDecodeOptions): string;
};
type EncodedQuery = {prompt: string; inputIds: number[]; tokenCount: number};

const IMAGE_TOKEN = '<|image_pad|>';
const IMAGE_PLACEHOLDER = '<|placeholder|>';

function imageTokenCount(grid: unknown): number {
  if (
    !Array.isArray(grid) ||
    grid.length !== 3 ||
    !grid.every((dimension: unknown) => Number.isSafeInteger(dimension) && (dimension as number) > 0)
  )
    throw new TypeError('Kanana imageTokenThw entries must contain three positive safe integers.');

  const [temporal, height, width] = grid as [number, number, number];
  const count = temporal * height * width;
  if (!Number.isSafeInteger(count)) throw new RangeError('Kanana image token count exceeds safe integer range.');
  return count;
}

function replaceImageMarkers(prompt: string, imageTokenThw: unknown): string {
  if (!Array.isArray(imageTokenThw)) throw new TypeError('Kanana imageTokenThw must be an array.');
  const counts = (imageTokenThw as unknown[]).map(imageTokenCount);

  for (const count of counts) {
    if (!prompt.includes(IMAGE_TOKEN)) throw new RangeError('Kanana image marker count does not match imageTokenThw.');
    // Match the pinned processor's literal placeholder, including its text collision behavior.
    prompt = prompt.replace(IMAGE_TOKEN, IMAGE_PLACEHOLDER.repeat(count));
  }

  if (prompt.includes(IMAGE_TOKEN)) {
    if (prompt.includes(KANANA_IMAGE_MARKER))
      throw new RangeError('Kanana image marker count does not match imageTokenThw.');
    throw new RangeError('Kanana prompt contains an unpaired image token.');
  }
  return counts.length ? prompt.replaceAll(IMAGE_PLACEHOLDER, IMAGE_TOKEN) : prompt;
}

function snapshotTokenIds(encoded: unknown): number[] {
  const source = (encoded as {ids?: unknown} | null | undefined)?.ids;
  const isArray = Array.isArray(source);
  if (!isArray && !(ArrayBuffer.isView(source) && !(source instanceof DataView)))
    throw new TypeError('Kanana tokenizer must return indexed token IDs.');

  const indexedIds = source as ArrayLike<unknown>;
  const ids: number[] = [];
  for (let index = 0; index < indexedIds.length; index++) {
    if (isArray && !Object.hasOwn(source, index))
      throw new TypeError('Kanana tokenizer returned a sparse token ID array.');
    const id = indexedIds[index];
    if (typeof id !== 'number' || !Number.isSafeInteger(id) || id < 0)
      throw new TypeError('Kanana tokenizer returned an invalid token ID.');
    ids.push(id);
  }
  return ids;
}

/** Render and tokenize one query with the pinned chat-template image token expansion. */
export function encodeKananaQuery(tokenizer: unknown, messages: unknown, imageTokenThw: unknown = []): EncodedQuery {
  if (typeof (tokenizer as {encode?: unknown} | null | undefined)?.encode !== 'function')
    throw new TypeError('A Kanana tokenizer with encode() is required.');
  const queryTokenizer = tokenizer as KananaTokenizer;
  const prompt = replaceImageMarkers(renderKananaChatTemplate(messages), imageTokenThw);
  const inputIds = snapshotTokenIds(queryTokenizer.encode(prompt));
  return {prompt, inputIds, tokenCount: inputIds.length};
}
