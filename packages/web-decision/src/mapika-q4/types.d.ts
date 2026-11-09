import type {ModelProgress} from '../curated/types.js';

export interface MapikaQ4ManifestCandidate {
  model?: unknown;
  backend?: unknown;
  formatVersion?: unknown;
  experimental?: unknown;
  maxInputTokens?: unknown;
  maxStateTokens?: unknown;
  parameterCount?: unknown;
  vocabSize?: unknown;
  temperature?: unknown;
  quantization?: {scheme?: unknown; scaleMethod?: unknown} | null;
}

export interface MapikaQ4Manifest extends MapikaQ4ManifestCandidate {
  model: 'mapika-0.8b-q4f16';
  backend: 'tvm-webgpu';
  formatVersion: 1;
  experimental: true;
  maxInputTokens: 128;
  maxStateTokens: 128;
  parameterCount: 510;
  vocabSize: 248320;
  temperature: 1.03;
  quantization: {scheme: 'q4-offset8-group32-f16'; scaleMethod: 'maxabs'};
}

export interface MapikaQ4TensorRecord {
  index: number;
  offset: number;
  dtype: string;
  shape: number[];
  nbytes: number;
}

export interface MapikaQ4WeightIndex {
  records: MapikaQ4TensorRecord[];
  shards: {path: string; nbytes: number}[];
  totalBytes: number;
}

export interface MapikaQ4Labels {
  names: string[];
  ids: number[];
  open: number[];
}

export interface MapikaQ4Limits {
  maxInputTokens: number;
  maxStateTokens: number;
  controlIds: number[];
  temperature: number;
}

export interface MapikaQ4Extras {
  labels: MapikaQ4Labels;
  limits: MapikaQ4Limits;
}

export interface MapikaTokenizer {
  encode(text: string, options?: {add_special_tokens?: boolean}): {ids: number[]};
}

export interface MapikaQ4Tensor {
  byteOffset?: number;
  getDataPtr(): unknown;
}

export interface MapikaQ4Runtime {
  empty(shape: number[], dtype: string, device: unknown): MapikaQ4Tensor;
}

export interface MapikaQ4SessionTensor extends MapikaQ4Tensor {
  copyFrom(value: unknown): MapikaQ4SessionTensor;
  toArray(): unknown;
}

export interface MapikaQ4SessionRuntime extends MapikaQ4Runtime {
  empty(shape: number[], dtype: string, device: unknown): MapikaQ4SessionTensor;
  cpu(): unknown;
  beginScope(): unknown;
  endScope(): unknown;
  dispose(): unknown;
  getGlobalFunc(name: string): () => unknown;
}

export interface MapikaQ4Device {
  sync(): Promise<unknown>;
}

export interface MapikaQ4Lifecycle {
  loss: unknown;
  disposed: boolean;
}

export interface MapikaQ4UploadArgs {
  labels: MapikaQ4Labels;
  index: MapikaQ4WeightIndex;
  read(path: string, signal?: AbortSignal | undefined): Promise<ArrayBuffer>;
  runtime: MapikaQ4Runtime;
  device: MapikaQ4Device;
  weights: unknown[];
  signal: AbortSignal | undefined;
  maxPrefetchBytes: number;
  lifecycle: MapikaQ4Lifecycle;
  uncaptured: string[];
}

export interface MapikaQ4SessionArgs extends MapikaQ4Extras {
  runtime: MapikaQ4SessionRuntime;
  device: MapikaQ4Device;
  gpu: GPUDevice;
  fn: (...args: unknown[]) => unknown;
  weights: unknown[];
  tokenizer: MapikaTokenizer;
  uncaptured: string[];
  lifecycle: MapikaQ4Lifecycle;
}

export interface MapikaQ4ModelUploadArgs extends Omit<MapikaQ4UploadArgs, 'labels'> {
  extras: MapikaQ4Extras;
}

export interface MapikaQ4LoadOptions {
  assetsUrl: string;
  cache?: boolean;
  offline?: boolean;
  signal?: AbortSignal;
  onProgress?: (progress: ModelProgress) => void;
  maxPrefetchBytes?: number;
  model?: 'mapika-0.8b-q4f16';
}
