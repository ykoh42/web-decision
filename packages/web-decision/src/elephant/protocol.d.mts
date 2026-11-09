export interface ElephantSlotTokenizer {
  encode(text: string, options: {readonly add_special_tokens: false}): {readonly ids: number[]};
  decode(ids: number[], options: {readonly skip_special_tokens: false}): string;
}

export function verifySlots(tokenizer: ElephantSlotTokenizer): number[];
