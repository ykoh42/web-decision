import {KANANA_IMAGE_MARKER, renderKananaChatTemplate} from './chat-template.mjs';

const IMAGE_TOKEN = '<|image_pad|>';
const IMAGE_PLACEHOLDER = '<|placeholder|>';

function imageTokenCount(grid) {
  if (
    !Array.isArray(grid) ||
    grid.length !== 3 ||
    !grid.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0)
  )
    throw new TypeError('Kanana imageTokenThw entries must contain three positive safe integers.');

  const count = grid[0] * grid[1] * grid[2];
  if (!Number.isSafeInteger(count)) throw new RangeError('Kanana image token count exceeds safe integer range.');
  return count;
}

function replaceImageMarkers(prompt, imageTokenThw) {
  if (!Array.isArray(imageTokenThw)) throw new TypeError('Kanana imageTokenThw must be an array.');
  const counts = imageTokenThw.map(imageTokenCount);

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

function snapshotTokenIds(encoded) {
  const source = encoded?.ids;
  const isArray = Array.isArray(source);
  if (!isArray && !(ArrayBuffer.isView(source) && !(source instanceof DataView)))
    throw new TypeError('Kanana tokenizer must return indexed token IDs.');

  const ids = [];
  for (let index = 0; index < source.length; index++) {
    if (isArray && !Object.hasOwn(source, index))
      throw new TypeError('Kanana tokenizer returned a sparse token ID array.');
    const id = source[index];
    if (!Number.isSafeInteger(id) || id < 0) throw new TypeError('Kanana tokenizer returned an invalid token ID.');
    ids.push(id);
  }
  return ids;
}

/** Render and tokenize one query with the pinned chat-template image token expansion. */
export function encodeKananaQuery(tokenizer, messages, imageTokenThw = []) {
  if (typeof tokenizer?.encode !== 'function') throw new TypeError('A Kanana tokenizer with encode() is required.');
  const prompt = replaceImageMarkers(renderKananaChatTemplate(messages), imageTokenThw);
  const inputIds = snapshotTokenIds(tokenizer.encode(prompt));
  return {prompt, inputIds, tokenCount: inputIds.length};
}
