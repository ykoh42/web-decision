import type {Tokenizer} from '@huggingface/tokenizers';

const HASHES = Object.freeze({
  tokenizer: '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42',
  config: '316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8',
  contract: '7eb96aaea6b7dbb495f0840b42f18014ca5a697697305f76f39d46cae890ccb2'
});

// The source JSON records Qwen3.5's newer pattern, while Qwen2Tokenizer in
// Transformers 5.17 overrides it with this training-time pattern. Apply this
// only to this exact pinned tokenizer and validate the vendor's probes.
const QWEN35_PATTERN =
  "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const QWEN2_PATTERN =
  "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

export type H2oTokenizer = Pick<Tokenizer, 'encode'>;
export type H2oTokenizerConstructor = new (data: object, config: object) => H2oTokenizer;
type TokenizerData = {
  added_tokens?: unknown;
  normalizer?: {type?: unknown} | null;
  pre_tokenizer?: {
    type?: unknown;
    pretokenizers?: Array<{
      type?: unknown;
      pattern?: {Regex?: unknown};
      trim_offsets?: unknown;
    }>;
  };
};
type TokenizerConfig = {
  added_tokens_decoder?: unknown;
  pretokenize_regex?: string;
  tokenizer_class?: unknown;
};
type TokenizerProbe = {id: string; text: string; token_ids: number[]};
type TokenizerContract = {probes?: unknown};
export type TokenizerEvidence = {
  data: {added_tokens: unknown};
  config: {tokenizer_class: unknown; added_tokens_decoder: unknown};
  contract: TokenizerContract;
};
type TokenizerBuildResult = {tokenizer: H2oTokenizer; evidence: TokenizerEvidence | undefined};

async function verifiedJson(
  input: Uint8Array | ArrayBuffer,
  expected: string,
  name: string,
  signal?: AbortSignal
): Promise<unknown> {
  signal?.throwIfAborted();
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes as BufferSource));
  signal?.throwIfAborted();
  const actual = [...digest].map((value) => value.toString(16).padStart(2, '0')).join('');
  if (actual !== expected) throw new Error(`Unexpected ${name} SHA-256`);
  return JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(bytes)) as unknown;
}

function sameIds(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, i) => value === right[i]);
}

async function buildH2oTokenizer(
  Tokenizer: H2oTokenizerConstructor,
  tokenizerBytes: Uint8Array | ArrayBuffer,
  configBytes: Uint8Array | ArrayBuffer,
  contractBytes: Uint8Array | ArrayBuffer,
  captureEvidence: boolean,
  signal?: AbortSignal
): Promise<TokenizerBuildResult> {
  signal?.throwIfAborted();
  const [data, config, contract] = (await Promise.all([
    verifiedJson(tokenizerBytes, HASHES.tokenizer, 'tokenizer', signal),
    verifiedJson(configBytes, HASHES.config, 'tokenizer config', signal),
    verifiedJson(contractBytes, HASHES.contract, 'tokenizer contract', signal)
  ])) as [TokenizerData, TokenizerConfig, TokenizerContract];
  signal?.throwIfAborted();
  const pre = data.pre_tokenizer;
  if (
    config.tokenizer_class !== 'Qwen2Tokenizer' ||
    config.pretokenize_regex !== QWEN35_PATTERN ||
    data.normalizer?.type !== 'NFC' ||
    pre?.type !== 'Sequence' ||
    pre.pretokenizers?.length !== 2 ||
    pre.pretokenizers[0]?.type !== 'Split' ||
    pre.pretokenizers[0].pattern?.Regex !== QWEN35_PATTERN ||
    pre.pretokenizers[1]?.type !== 'ByteLevel'
  )
    throw new Error('Unexpected H2O tokenizer structure');
  pre!.pretokenizers![0]!.pattern!.Regex = QWEN2_PATTERN;
  pre!.pretokenizers![1]!.trim_offsets = true;
  config.pretokenize_regex = QWEN2_PATTERN;
  // The tokenizer constructor may mutate its input. Keep only the small parts
  // needed by the adapter before handing either parsed object to it.
  const evidence: TokenizerEvidence | undefined = captureEvidence
    ? (structuredClone({
        data: {added_tokens: data.added_tokens},
        config: {tokenizer_class: config.tokenizer_class, added_tokens_decoder: config.added_tokens_decoder},
        contract
      }) as TokenizerEvidence)
    : undefined;
  signal?.throwIfAborted();
  const tokenizer = new Tokenizer(data, config);
  if (!Array.isArray(contract.probes) || contract.probes.length !== 24)
    throw new Error('Unexpected tokenizer probe list');
  for (const probe of contract.probes as TokenizerProbe[]) {
    signal?.throwIfAborted();
    const ids = tokenizer.encode(probe.text, {add_special_tokens: false}).ids;
    if (!sameIds(ids, probe.token_ids)) throw new Error(`Training tokenizer mismatch: ${probe.id}`);
  }
  signal?.throwIfAborted();
  return {tokenizer, evidence};
}

/** Build only H2O v1.4.2's training tokenizer after checking pinned input bytes. */
export async function createH2oTokenizer(
  Tokenizer: H2oTokenizerConstructor,
  tokenizerBytes: Uint8Array | ArrayBuffer,
  configBytes: Uint8Array | ArrayBuffer,
  contractBytes: Uint8Array | ArrayBuffer,
  signal?: AbortSignal
): Promise<H2oTokenizer> {
  return (await buildH2oTokenizer(Tokenizer, tokenizerBytes, configBytes, contractBytes, false, signal)).tokenizer;
}

/** Provide the checked adapter with its own pre-constructor metadata snapshot. */
export async function createH2oTokenizerWithEvidence(
  Tokenizer: H2oTokenizerConstructor,
  tokenizerBytes: Uint8Array | ArrayBuffer,
  configBytes: Uint8Array | ArrayBuffer,
  contractBytes: Uint8Array | ArrayBuffer,
  signal?: AbortSignal
): Promise<{tokenizer: H2oTokenizer; evidence: TokenizerEvidence}> {
  return buildH2oTokenizer(Tokenizer, tokenizerBytes, configBytes, contractBytes, true, signal) as Promise<{
    tokenizer: H2oTokenizer;
    evidence: TokenizerEvidence;
  }>;
}
