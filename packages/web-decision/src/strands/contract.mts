// The checkpoint's strands_decider_config.json pins max_length to 4,096.
export const STRANDS_CHECKPOINT_MAX_LENGTH = 4096;

// The compiled 4096-token candidate graph exports this full sequence range.
export const STRANDS_FUSED_GRAPH_MAX_TOKENS = STRANDS_CHECKPOINT_MAX_LENGTH;

// Pinned Strands EngineConfig default; each consecutive group is fit together.
export const STRANDS_DEFAULT_MAX_BATCH = 32;

// The pinned Strands and Qwen3.5 base embeddings use the same 248,320-token vocabulary.
export const STRANDS_CHECKPOINT_VOCAB_SIZE = 248320;
