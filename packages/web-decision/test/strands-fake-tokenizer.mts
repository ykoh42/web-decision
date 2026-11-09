// Installed only as the fake consumer's external tokenizer dependency.
const decoder = Object.fromEntries(Array.from({length: 128}, (_, id) => [String.fromCharCode(id), String(id)]));
export class Tokenizer {
  declare decoder: {byte_decoder: Record<string, string>};

  constructor() {
    this.decoder = {byte_decoder: decoder};
  }
  get_added_tokens_decoder(): Map<unknown, unknown> {
    return new Map();
  }
  encode(input: string) {
    const tokens = input.normalize('NFC').match(/[^\n]+|\n/g) || [];
    return {tokens, ids: tokens.map((_, index) => index + 1)};
  }
}
