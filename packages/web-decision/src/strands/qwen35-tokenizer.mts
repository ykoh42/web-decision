import {Tokenizer} from '@huggingface/tokenizers';
import {sha256} from '../assets.js';
import type {AssetCache} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {RangeCacheTransaction} from './range-cache.mjs';
import type {RangeCacheReadOptions, RangeCacheSpec} from './range-cache.mjs';
import {TOKENIZER_CONFIG_SHA256, TOKENIZER_SHA256} from './manifest.mjs';

const HF_BASE = new URL('https://huggingface.co/');
const REPOSITORY = 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610';
const REVISION = '216d1ccb46a9e9b036eb9187b248db98480f77d4';
const MAX_RANGE_BYTES = 8 * 1024 * 1024;
const MARKER_ID = `qwen35-tokenizer-v1:${TOKENIZER_SHA256}:${TOKENIZER_CONFIG_SHA256}`;
const invalid = (message: string, cause?: unknown) => new WebDecisionError('INVALID_ASSET', message, {cause});

interface FileRange {
  readonly offset: number;
  readonly bytes: number;
  readonly sha256: string;
}

interface TokenizerFile {
  readonly name: 'tokenizer.json' | 'tokenizer_config.json';
  readonly bytes: number;
  readonly sha256: string;
  readonly ranges: readonly FileRange[];
}

interface TokenizerLoaderDependencies {
  readonly base: URL;
  readonly files: readonly TokenizerFile[];
  readonly markerId: string;
  readonly createTokenizer: (tokenizerData: object, tokenizerConfig: object) => TokenizerLike;
}

interface TokenizerLike {
  encode(text: string, options?: {readonly add_special_tokens?: boolean}): unknown;
}

interface TokenizerOpenOptions {
  readonly cache?: AssetCache | null | undefined;
  readonly offline?: boolean | undefined;
  readonly signal?: AbortSignal | null | undefined;
}

export interface Qwen35PinnedTokenizer {
  readonly tokenizer: TokenizerLike;
  commit(options?: RangeCacheReadOptions): Promise<boolean>;
  dispose(): void;
}

const PINNED_FILES: readonly TokenizerFile[] = Object.freeze([
  Object.freeze({
    name: 'tokenizer.json',
    bytes: 19_989_592,
    sha256: TOKENIZER_SHA256,
    ranges: Object.freeze([
      Object.freeze({
        offset: 0,
        bytes: MAX_RANGE_BYTES,
        sha256: '4ca2408791853d31f4049546249a60dd5b4f034201f3dfd391916d0c999161af'
      }),
      Object.freeze({
        offset: MAX_RANGE_BYTES,
        bytes: MAX_RANGE_BYTES,
        sha256: '2d6f0488857832d7ff1d0cebc284f152bacafab14692fb5cf764c796c87898a2'
      }),
      Object.freeze({
        offset: MAX_RANGE_BYTES * 2,
        bytes: 3_212_376,
        sha256: 'c5f763e067ed597a2ee4c6feb566bc6484935a672f6bcc9a7eb6dbca83d5d88a'
      })
    ])
  }),
  Object.freeze({
    name: 'tokenizer_config.json',
    bytes: 1_128,
    sha256: TOKENIZER_CONFIG_SHA256,
    ranges: Object.freeze([
      Object.freeze({
        offset: 0,
        bytes: 1_128,
        sha256: '8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45'
      })
    ])
  })
]);

function filePath(name: TokenizerFile['name']): string {
  return `${REPOSITORY}/resolve/${REVISION}/${name}`;
}

function rangeSpecs(files: readonly TokenizerFile[]): readonly RangeCacheSpec[] {
  return files.flatMap(({name, bytes, sha256: digest, ranges}) => {
    let nextOffset = 0;
    for (const range of ranges) {
      if (
        range.offset !== nextOffset ||
        !Number.isSafeInteger(range.bytes) ||
        range.bytes < 1 ||
        range.bytes > MAX_RANGE_BYTES ||
        !/^[0-9a-f]{64}$/.test(range.sha256)
      )
        throw invalid(`Invalid pinned Qwen3.5 ${name} range metadata.`);
      nextOffset += range.bytes;
    }
    if (nextOffset !== bytes || !Number.isSafeInteger(bytes) || bytes < 1 || !/^[0-9a-f]{64}$/.test(digest))
      throw invalid(`Invalid pinned Qwen3.5 ${name} file metadata.`);
    return ranges.map(({offset, bytes: length, sha256: digest}) => ({
      path: filePath(name),
      totalBytes: bytes,
      offset,
      length,
      sha256: digest
    }));
  });
}

function parseObject(bytes: Uint8Array, name: string): object {
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(bytes)) as unknown;
  } catch (cause) {
    throw invalid(`Invalid pinned Qwen3.5 ${name}.`, cause);
  }
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw invalid(`Pinned Qwen3.5 ${name} must be an object.`);
  return value;
}

/** Internal loader factory; production exposes only the immutable checkpoint opener below. */
export function createQwen35TokenizerLoader({
  base,
  files,
  markerId,
  createTokenizer
}: TokenizerLoaderDependencies): (options?: TokenizerOpenOptions) => Promise<Qwen35PinnedTokenizer> {
  if (!(base instanceof URL) || !['https:', 'http:'].includes(base.protocol) || base.search || base.hash)
    throw invalid('Invalid Qwen3.5 tokenizer source base URL.');
  if (typeof markerId !== 'string' || !markerId || typeof createTokenizer !== 'function')
    throw invalid('Invalid Qwen3.5 tokenizer loader dependencies.');
  const specs = rangeSpecs(files);

  return async function openQwen35Tokenizer({cache, offline = false, signal}: TokenizerOpenOptions = {}) {
    if (typeof offline !== 'boolean') throw new WebDecisionError('INVALID_INPUT', 'Invalid Qwen3.5 tokenizer options.');
    const transaction = new RangeCacheTransaction({
      base,
      cache,
      offline,
      signal,
      markerId,
      markerPath: 'web-decision-qwen35-tokenizer-offline-complete-v1.json'
    });
    await transaction.open();
    transaction.require(specs);
    transaction.seal();

    const parsed = new Map<string, object>();
    for (const file of files) {
      const bytes = new Uint8Array(file.bytes);
      for (const range of file.ranges) {
        signal?.throwIfAborted();
        const spec: RangeCacheSpec = {
          path: filePath(file.name),
          totalBytes: file.bytes,
          offset: range.offset,
          length: range.bytes,
          sha256: range.sha256
        };
        const {bytes: rangeBytes} = await transaction.read(base, spec, {signal});
        bytes.set(new Uint8Array(rangeBytes), range.offset);
      }
      signal?.throwIfAborted();
      if ((await sha256(bytes)) !== file.sha256) throw invalid(`Pinned Qwen3.5 ${file.name} hash mismatch.`);
      parsed.set(file.name, parseObject(bytes, file.name));
    }

    let tokenizer: TokenizerLike;
    try {
      tokenizer = createTokenizer(parsed.get('tokenizer.json')!, parsed.get('tokenizer_config.json')!);
    } catch (cause) {
      throw invalid('Could not create the pinned Qwen3.5 tokenizer.', cause);
    }
    signal?.throwIfAborted();
    if (!tokenizer || typeof tokenizer.encode !== 'function')
      throw invalid('Invalid pinned Qwen3.5 tokenizer instance.');

    let activeTokenizer: TokenizerLike | null = tokenizer;
    return {
      get tokenizer() {
        if (!activeTokenizer) throw new WebDecisionError('INVALID_INPUT', 'The Qwen3.5 tokenizer has been disposed.');
        return activeTokenizer;
      },
      async commit(options: RangeCacheReadOptions = {}) {
        if (!activeTokenizer) throw new WebDecisionError('INVALID_INPUT', 'The Qwen3.5 tokenizer has been disposed.');
        return transaction.commit(options);
      },
      dispose() {
        activeTokenizer = null;
      }
    };
  };
}

const openPinned = createQwen35TokenizerLoader({
  base: HF_BASE,
  files: PINNED_FILES,
  markerId: MARKER_ID,
  createTokenizer: (tokenizerData, tokenizerConfig) => new Tokenizer(tokenizerData, tokenizerConfig)
});

/** Load the checkpoint tokenizer from its fixed Hugging Face revision without a caller asset URL. */
export function openPinnedQwen35Tokenizer(options?: TokenizerOpenOptions): Promise<Qwen35PinnedTokenizer> {
  return openPinned(options);
}
