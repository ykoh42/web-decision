export type ClefDType = 'uint32' | 'float16' | 'float32';

export interface ClefRangePart {
  offset: number;
  length: number;
  sha256: string;
}

export interface ClefTensorEntry {
  target: string;
  path: string;
  dtype: ClefDType;
  shape: number[];
  bytes: number;
  sha256: string;
  chunks: ClefRangePart[];
  position?: number;
}

export interface ClefBankStream extends ClefTensorEntry {
  dtype: 'uint32' | 'float16';
}

export interface ClefBank {
  rows: number;
  packed: ClefBankStream;
  scales: ClefBankStream;
}

export interface ClefSourceIndex {
  entries: ClefTensorEntry[];
  inputBanks: ClefBank[];
  outputBanks: ClefBank[];
}

export interface ClefSource extends ClefSourceIndex {
  wasm: Uint8Array;
  tokenizer: unknown;
  tokenizerConfig: unknown;
  readRange(
    entry: ClefTensorEntry,
    part: ClefRangePart,
    signal?: AbortSignal | undefined
  ): Promise<ArrayBuffer | Uint8Array>;
}

export interface ClefTokenIds extends Iterable<number>, ArrayLike<number> {}

export interface ClefTokenizer {
  encode(text: string, options?: {add_special_tokens?: boolean}): {ids?: ClefTokenIds} | null;
}

export interface ClefQuestion {
  question_id: string;
  question_type: number;
  question_span: [number, number];
  option_spans: Array<[number, number]>;
  option_ids: string[];
  options: string[];
}

export interface PackedClefChoice {
  ids: number[];
  questions: ClefQuestion[];
  maxInputTokens: number;
}

export type GraphInputDType = 'int32' | 'float16' | 'float32' | 'uint32';
export type GraphInputData = Int32Array | Uint16Array | Uint32Array | Float32Array;

export interface GraphInput {
  name: string;
  dtype: GraphInputDType;
  shape: number[];
  data: GraphInputData;
}

export interface ClefChoiceGraph {
  packed: PackedClefChoice;
  offsets: number[];
  inputs: GraphInput[];
}

export interface SelectedClefRows {
  unique: number[];
  local: Int32Array;
  rowBanks: Int32Array;
  packed: Uint32Array;
  scales: Uint16Array;
}

export interface ClefRangeReader {
  (entry: ClefTensorEntry, part: ClefRangePart, signal?: AbortSignal | undefined): Promise<ArrayBuffer | Uint8Array>;
}

export interface ClefRowSelectOptions {
  readRange?: ClefRangeReader | undefined;
  signal?: AbortSignal | undefined;
}
