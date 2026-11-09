export interface ElephantSlotTokenizer {
  encode(text: string, options: {readonly add_special_tokens: false}): {readonly ids: number[]};
  decode(ids: number[], options: {readonly skip_special_tokens: false}): string;
}

export interface ElephantRequest {
  readonly state: string;
  readonly questions: Array<{readonly instructions: string; readonly options: string[]}>;
}

export interface ElephantPackedBranch {
  readonly options: string[];
  readonly suffix: string;
  readonly prompt: string;
  readonly ids: number[];
  readonly suffixIds: number[];
}

export interface ElephantPackedRequest {
  readonly prefixIds: number[];
  readonly branches: ElephantPackedBranch[];
  readonly tokenCount: number;
}

export interface ElephantDecision {
  readonly tokenCount: number;
  readonly answers: Array<{
    readonly selected: string | undefined;
    readonly selectedIndex: number;
    readonly probabilities: Array<{
      readonly option: string;
      readonly probability: number | undefined;
      readonly logit: number | undefined;
    }>;
  }>;
}

export function packElephantRequest(
  tokenizer: ElephantSlotTokenizer,
  request: ElephantRequest,
  config: {readonly addedTokens: string[]; readonly maxBranchTokens: number}
): ElephantPackedRequest;
export function unpackElephantScores(rows: Float32Array[], packed: ElephantPackedRequest): ElephantDecision;
export function verifySlots(tokenizer: ElephantSlotTokenizer): number[];
