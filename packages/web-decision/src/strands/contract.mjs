// The checkpoint's strands_decider_config.json pins max_length to 4,096.
export const STRANDS_CHECKPOINT_MAX_LENGTH = 4096;

// The compiled 4096-token candidate graph exports this full sequence range.
export const STRANDS_FUSED_GRAPH_MAX_TOKENS = STRANDS_CHECKPOINT_MAX_LENGTH;
