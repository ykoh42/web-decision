import {sha256} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {buildMapikaLabels} from '../mapika.js';
import {MAPIKA4_LABELS_SHA256} from './contract.mjs';

interface MapikaTokenizerEncoding {
  readonly ids?: Iterable<number>;
}

interface MapikaTokenizer {
  encode(text: string, options: Readonly<{add_special_tokens: false}>): MapikaTokenizerEncoding | null | undefined;
}

interface MapikaTokenizerConstructor {
  new (data: unknown, config: unknown): MapikaTokenizer;
}

interface MapikaLabels {
  readonly names: string[];
  readonly ids: number[];
  readonly open: number[];
}

interface MapikaTokenizerData {
  readonly added_tokens?: unknown;
}

interface Mapika4TokenizerResult {
  readonly tokenizer: MapikaTokenizer;
  readonly labels: MapikaLabels;
  readonly limits: Readonly<{
    controlIds: number[];
    maxInputTokens: 128;
    maxStateTokens: 128;
    temperature: 2.098;
  }>;
}

function isControlId(id: unknown): id is number {
  return Number.isSafeInteger(id as number) && (id as number) >= 0 && (id as number) < 248320;
}

/** Construct and bind the tokenizer's control tokens and 255 answer labels. */
export async function createMapika4Tokenizer(
  Tokenizer: unknown,
  data: unknown,
  config: unknown
): Promise<Mapika4TokenizerResult> {
  const candidate = data as MapikaTokenizerData | null | undefined;
  if (
    typeof Tokenizer !== 'function' ||
    !Array.isArray(candidate?.added_tokens) ||
    !config ||
    typeof config !== 'object'
  )
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Mapika 4B tokenizer.');
  const entries = (data as {readonly added_tokens: unknown[]}).added_tokens;
  const unvalidatedControlIds = entries.map(
    (item: unknown) => (item as {readonly id?: unknown} | null | undefined)?.id
  );
  if (
    unvalidatedControlIds.length !== 33 ||
    unvalidatedControlIds.some((id) => !isControlId(id)) ||
    new Set(unvalidatedControlIds).size !== unvalidatedControlIds.length
  )
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Mapika 4B control tokens.');
  const controlIds = unvalidatedControlIds as number[];
  const Constructor = Tokenizer as MapikaTokenizerConstructor;
  const tokenizer = new Constructor(data, config);
  const labels = buildMapikaLabels(tokenizer) as MapikaLabels;
  const binding = new TextEncoder().encode(JSON.stringify([labels.names, labels.ids, labels.open, controlIds]));
  if ((await sha256(binding)) !== MAPIKA4_LABELS_SHA256)
    throw new WebDecisionError('INVALID_ASSET', 'Mapika 4B answer labels do not match the checkpoint.');
  return {tokenizer, labels, limits: {controlIds, maxInputTokens: 128, maxStateTokens: 128, temperature: 2.098}};
}
