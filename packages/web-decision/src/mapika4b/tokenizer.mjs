import {sha256} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {buildMapikaLabels} from '../mapika.js';
import {MAPIKA4_LABELS_SHA256} from './contract.mjs';

/** Construct and bind the tokenizer's control tokens and 255 answer labels. */
export async function createMapika4Tokenizer(Tokenizer, data, config) {
  if (typeof Tokenizer !== 'function' || !Array.isArray(data?.added_tokens) || !config || typeof config !== 'object')
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Mapika 4B tokenizer.');
  const controlIds = data.added_tokens.map((item) => item?.id);
  if (
    controlIds.length !== 33 ||
    controlIds.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= 248320) ||
    new Set(controlIds).size !== controlIds.length
  )
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Mapika 4B control tokens.');
  const tokenizer = new Tokenizer(data, config);
  const labels = buildMapikaLabels(tokenizer);
  const binding = new TextEncoder().encode(JSON.stringify([labels.names, labels.ids, labels.open, controlIds]));
  if ((await sha256(binding)) !== MAPIKA4_LABELS_SHA256)
    throw new WebDecisionError('INVALID_ASSET', 'Mapika 4B answer labels do not match the checkpoint.');
  return {tokenizer, labels, limits: {controlIds, maxInputTokens: 128, maxStateTokens: 128, temperature: 2.098}};
}
