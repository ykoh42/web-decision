export interface FlashDecisionConfig {
  codes: unknown[];
  markerId: number;
  escapedTokens: string[];
  addedTokens: string[];
  addedTokenIds: number[];
  maxInputTokens: number;
}

export interface FlashTokenizer {
  encode(text: string, options?: {add_special_tokens?: boolean}): {ids: number[]};
}

export interface FlashPackedGroup {
  index: number;
  options: string[];
}

export interface PackedFlashRequest {
  prompt: string;
  ids: number[];
  groups: FlashPackedGroup[];
}

export type FlashGraphInputDType = 'int32' | 'float16';
export type FlashGraphInputData = Int32Array | Uint16Array;

export interface FlashGraphInput {
  dtype: FlashGraphInputDType;
  shape: number[];
  data: FlashGraphInputData;
}

export interface FlashRange {
  offset: number;
  bytes: number;
  sha256: string;
}

export interface FlashDerivedRotaryEntry {
  position: number;
  bindingKind: 'derived-rotary';
  dtype: 'float32';
  shape: number[];
  values: number[];
}

export interface FlashQ4CandidateEntry {
  position: number;
  bindingKind: 'q4-candidate';
  dtype: 'float16' | 'float32' | 'uint32';
  shape: number[];
  outputPath: string;
  outputBytes: number;
  sourceOffset: number;
  conversion: 'none' | 'fp32-to-fp16';
  ranges: FlashRange[];
}

export type FlashManifestEntry = FlashDerivedRotaryEntry | FlashQ4CandidateEntry;

export interface FlashManifest {
  state: 'pplx-q4-joint-assets-pending';
  entries: FlashManifestEntry[];
}

export interface FlashSource {
  manifest: FlashManifest;
  wasm: Uint8Array;
  tokenizer: unknown;
  tokenizerConfig: unknown;
  readRange(
    entry: FlashQ4CandidateEntry,
    index: number,
    part: FlashRange,
    signal?: AbortSignal | undefined
  ): Promise<ArrayBuffer>;
}

export interface FlashTensor {
  byteOffset: number;
  dtype: string;
  shape: number[];
  copyFrom(data: unknown): unknown;
  getDataPtr(): number;
  toArray(): unknown;
}

export interface FlashTvmRuntime {
  lib?: {
    webGPUContext?: {
      copyRawBytesToBuffer?(bytes: Uint8Array, pointer: number, offset: number, length: number): void;
    };
  };
  empty(shape: number[], dtype: string, device: unknown): FlashTensor;
  cpu(): unknown;
  dispose(): void;
}

export interface FlashRunnerArgs extends Record<string, unknown> {
  runtime: FlashTvmRuntime;
  device: unknown;
  gpu: unknown;
  fn: (...args: unknown[]) => FlashTensor;
  weights: unknown[];
  lifecycle: unknown;
  disposalGuard: unknown;
  uncaptured: unknown[];
  teardownObserver: unknown;
  teardownTimeoutMs: number;
}
