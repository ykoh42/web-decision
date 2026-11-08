import {WebDecisionError} from '../kev.js';

const invalid = () => new WebDecisionError('INVALID_ASSET', 'Unexpected Nari tokenizer.');
const markers = {'<|im_start|>': 248045, '<|im_end|>': 248046, '<think>': 248068, '</think>': 248069};

// Input bytes must already have been verified by the pinned asset source.
export function createNariTokenizer(Tokenizer, tokenizerBytes, configBytes) {
  if (
    typeof Tokenizer !== 'function' ||
    !(tokenizerBytes instanceof Uint8Array) ||
    !(configBytes instanceof Uint8Array)
  )
    throw invalid();
  let data, config;
  try {
    const decoder = new TextDecoder('utf-8', {fatal: true});
    data = JSON.parse(decoder.decode(tokenizerBytes));
    config = JSON.parse(decoder.decode(configBytes));
  } catch {
    throw invalid();
  }
  if (
    config?.tokenizer_class !== 'Qwen3_5Tokenizer' ||
    !Array.isArray(data?.added_tokens) ||
    data.added_tokens.length !== 33
  )
    throw invalid();
  const ids = new Set(),
    contents = new Set();
  for (const item of data.added_tokens) {
    if (
      !Number.isSafeInteger(item?.id) ||
      item.id < 248044 ||
      item.id > 248076 ||
      typeof item.content !== 'string' ||
      !item.content ||
      ids.has(item.id) ||
      contents.has(item.content)
    )
      throw invalid();
    ids.add(item.id);
    contents.add(item.content);
  }
  const tokenizer = new Tokenizer(data, config);
  if (typeof tokenizer?.encode !== 'function') throw invalid();
  for (const [marker, id] of Object.entries(markers)) {
    if (!data.added_tokens.some((item) => item.content === marker && item.id === id)) throw invalid();
    const direct = tokenizer.encode(marker, {add_special_tokens: false})?.ids;
    const embedded = tokenizer.encode(`x${marker}y`, {add_special_tokens: false})?.ids;
    if (
      !Array.isArray(direct) ||
      direct.length !== 1 ||
      direct[0] !== id ||
      !Array.isArray(embedded) ||
      embedded.filter((value) => value === id).length !== 1
    )
      throw invalid();
  }
  return {tokenizer, controlIds: [...ids]};
}
