import {WebDecisionError} from '../kev.js';

export const OPENDECIDER_NANO_TOKENS = Object.freeze({cls: 50281, sep: 50282, pad: 50283, mask: 50284});
const VOCAB_SIZE = 50368;
const invalid = (): WebDecisionError =>
  new WebDecisionError('INVALID_ASSET', 'OpenDecider-nano tokenizer does not match the pinned model.');

interface TokenizerEncoding {
  readonly ids?: unknown;
}

interface TokenizerLike {
  encode(
    text: string,
    options: Readonly<{add_special_tokens: false; split_special_tokens: true}>
  ): TokenizerEncoding | null | undefined;
}

interface TokenizerConstructor {
  new (data: unknown, config: unknown): TokenizerLike;
}

interface TokenizerData {
  readonly added_tokens?: unknown;
  readonly model?: {readonly vocab?: unknown} | null;
}

function isUnknownArray(value: unknown): value is unknown[] {
  return Array.isArray(value);
}

function isVocabId(value: unknown): value is number {
  if (!Number.isSafeInteger(value as number)) return false;
  const id = value as number;
  return id >= 0 && id < VOCAB_SIZE;
}

function textIds(tokenizer: TokenizerLike, text: string): number[] {
  let ids: unknown;
  try {
    ids = tokenizer.encode(text, {add_special_tokens: false, split_special_tokens: true})?.ids;
  } catch (cause) {
    throw new WebDecisionError('INVALID_ASSET', 'OpenDecider-nano tokenizer failed.', {cause});
  }
  if (!isUnknownArray(ids) || ids.some((id) => !isVocabId(id))) throw invalid();
  const validIds = ids as number[];
  if (validIds.some((id) => (Object.values(OPENDECIDER_NANO_TOKENS) as number[]).includes(id))) throw invalid();
  return [...validIds];
}

export function createOpenDeciderNanoTokenizer(
  Tokenizer: unknown,
  data: unknown,
  config: unknown
): Readonly<{encode(text: string): number[]}> {
  const candidate = data as TokenizerData | null | undefined;
  if (
    typeof Tokenizer !== 'function' ||
    !data ||
    !isUnknownArray(candidate?.added_tokens) ||
    !candidate?.model?.vocab ||
    typeof candidate.model.vocab !== 'object' ||
    !config ||
    typeof config !== 'object' ||
    Array.isArray(config)
  )
    throw invalid();
  const validData = candidate as TokenizerData & {
    readonly added_tokens: unknown[];
    readonly model: {readonly vocab: object};
  };
  const ids = [
    ...Object.values(validData.model.vocab),
    ...validData.added_tokens.map((item: unknown) => (item as {readonly id?: unknown} | null | undefined)?.id)
  ];
  if (ids.length < 1 || ids.some((id) => !isVocabId(id)) || Math.max(...ids) !== VOCAB_SIZE - 1) throw invalid();
  const Constructor = Tokenizer as TokenizerConstructor;
  const tokenizer = new Constructor(data, config);
  const checks: ReadonlyArray<readonly [string, readonly number[]]> = [
    ['question: Which team should handle this?', [19751, 27, 6758, 2285, 943, 6016, 436, 32]],
    ['input: I was charged twice for order 1182.', [5423, 27, 309, 369, 6636, 7019, 323, 1340, 12643, 19, 15]]
  ];
  for (const [text, expected] of checks) {
    const actual = textIds(tokenizer, text);
    if (actual.length !== expected.length || actual.some((id, index) => id !== expected[index])) throw invalid();
  }
  return Object.freeze({encode: (text: string) => textIds(tokenizer, text)});
}
