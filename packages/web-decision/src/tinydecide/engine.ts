// TinyDecide decision model in TypeScript (browser worker or Node). Mirrors the Python
// reference exactly: reflex/format.py (layout + masks), reflex/backbones.py (ELECTRA path),
// reflex/model.py (native path), reflex/heads.py (typed heads, temperatures, prototypes).
// verify.mjs checks it against torch outputs computed from the same int8 weights.

const K_STATE = 0,
  K_QTEXT = 1,
  K_ANS = 2,
  K_OPT = 3,
  K_LV = 4;
const K_BUCKETS = [1, 2, 4, 8];

type QuestionType = 'choice' | 'noul' | 'score' | 'span';
type TokenOffset = [start: number, end: number];
type WordChar = [value: string, start: number, end: number];
type NumericVector = readonly number[] | Float32Array;

interface TinyDecideQuestion {
  readonly type: QuestionType;
  readonly text: string;
  readonly options?: readonly string[];
}

export interface TinyDecideRequest {
  readonly state: string;
  readonly questions: readonly TinyDecideQuestion[];
}

interface TinyDecideFormat {
  readonly ts_max: number;
  readonly p_q: number;
  readonly q_max: number;
  readonly k_max: number;
  readonly span_max: number;
}

interface TinyDecideConfig {
  readonly arch: string;
  readonly d: number;
  readonly dh_head: number;
  readonly emb_dim: number;
  readonly layers: number;
  readonly heads: number;
  readonly ln_eps: number;
  readonly fusion: string;
  readonly L_a?: number;
}

interface WordPieceMetadata {
  readonly vocab: readonly (string | null)[];
  readonly unk: string;
  readonly prefix: string;
  readonly max_chars: number;
}

interface TinyDecideTensorMetadata {
  readonly name: string;
  readonly dtype: string;
  readonly shape: readonly number[];
  readonly offset: number;
  readonly scale_offset: number;
}

export interface TinyDecideWeightsMetadata {
  readonly tensors: readonly TinyDecideTensorMetadata[];
}

export interface TinyDecideMetadata {
  readonly cfg: TinyDecideConfig;
  readonly tokenizer: WordPieceMetadata & {readonly kind: string};
  readonly tensors: readonly TinyDecideTensorMetadata[];
  readonly specials: Readonly<Record<string, number>>;
  readonly format: TinyDecideFormat;
  readonly temp: readonly number[];
  readonly beta: readonly number[];
}

interface WordPieceEncoding {
  readonly ids: number[];
  readonly offsets: TokenOffset[];
}

interface EncodedQuestion {
  readonly type: QuestionType;
  readonly ans: number;
  readonly opt: number[];
}

interface EncodedRequest {
  readonly ids: number[];
  readonly pos: number[];
  readonly blk: number[];
  readonly kind: number[];
  readonly stIdx: number[];
  readonly stOff: TokenOffset[];
  readonly qs: EncodedQuestion[];
  readonly truncated: boolean;
}

interface WeightedTensor {
  readonly data: Float32Array;
  readonly shape: readonly number[];
}

export interface TinyDecidePrototype {
  readonly vec: Float32Array;
  readonly cnt: Int32Array;
  readonly center: Float32Array | null;
  readonly lam?: number;
}

export interface TinyDecideEngineAnswer {
  readonly answers: TinyDecideAnswer[];
  readonly tokens: {readonly state: number; readonly questions: number; readonly total: number};
  readonly truncated: boolean;
  readonly ms: number;
  readonly ids: number[];
}

type TinyDecideAnswer = ChoiceAnswer | NoulAnswer | SpanAnswer;
type ChoiceAnswer =
  | {
      readonly type: 'choice';
      readonly probs: number[];
      readonly pick: number;
      readonly confidence: number;
      readonly qvec: number[];
      readonly proj: boolean;
      readonly z0: number[];
    }
  | {
      readonly type: 'score';
      readonly probs: number[];
      readonly pick: number;
      readonly confidence: number;
      readonly qvec: number[];
      readonly proj: boolean;
      readonly z0: number[];
      readonly score: number;
    };
interface NoulAnswer {
  readonly type: 'noul';
  readonly p: number;
  readonly qvec: number[];
  readonly proj: boolean;
  readonly z0: [number, number];
}
interface SpanAnswer {
  readonly type: 'span';
  readonly p_present: number;
  readonly tok: [number, number];
  readonly p_span: number;
  readonly text: string;
  readonly char: TokenOffset | null;
}

interface PrototypeScorer {
  readonly proto: TinyDecidePrototype | undefined;
  readonly projected: Float32Array | null;
  term(index: number, count: number): number | null;
}

interface AnswerContext {
  readonly model: TinyDecide;
  readonly request: TinyDecideRequest;
  readonly encoded: EncodedRequest;
  readonly hidden: Float32Array;
  readonly normalized: Float32Array;
  readonly dimensions: number;
  readonly headDimensions: number;
}

// ------------------------------------------------------------------ tokenizers
const RE_CTRL = /[\p{Cc}\p{Cf}]/u,
  RE_ZS = /\p{Zs}/u,
  RE_P = /\p{P}/u,
  RE_MN = /\p{Mn}/u;
function isCJK(cp: number): boolean {
  return (
    (cp >= 0x4e00 && cp <= 0x9fff) ||
    (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0x20000 && cp <= 0x2a6df) ||
    (cp >= 0x2a700 && cp <= 0x2b73f) ||
    (cp >= 0x2b740 && cp <= 0x2b81f) ||
    (cp >= 0x2b820 && cp <= 0x2ceaf) ||
    (cp >= 0xf900 && cp <= 0xfaff) ||
    (cp >= 0x2f800 && cp <= 0x2fa1f)
  );
}
function isPunct(ch: string): boolean {
  const c = ch.codePointAt(0)!;
  if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) return true;
  return RE_P.test(ch);
}

function normalizeWordCharacter(ch: string, start: number, end: number): WordChar[] {
  const codePoint = ch.codePointAt(0)!;
  if (codePoint === 0 || codePoint === 0xfffd || (RE_CTRL.test(ch) && ch !== '\t' && ch !== '\n' && ch !== '\r'))
    return [];
  if (ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r' || RE_ZS.test(ch)) return [[' ', start, end]];
  if (isCJK(codePoint))
    return [
      [' ', start, start],
      [ch, start, end],
      [' ', end, end]
    ];
  const normalized: WordChar[] = [];
  for (const decomposed of ch.normalize('NFD')) {
    if (RE_MN.test(decomposed)) continue;
    for (const lower of decomposed.toLowerCase()) normalized.push([lower, start, end]);
  }
  return normalized;
}

function flushWord(tokenizer: WordPiece, current: WordChar[], ids: number[], offsets: TokenOffset[]): void {
  if (current.length) tokenizer.word(current, ids, offsets);
  current.length = 0;
}

function appendWordCharacter(
  tokenizer: WordPiece,
  current: WordChar[],
  char: WordChar,
  ids: number[],
  offsets: TokenOffset[]
): void {
  if (char[0] === ' ' || /\s/u.test(char[0])) flushWord(tokenizer, current, ids, offsets);
  else if (isPunct(char[0])) {
    flushWord(tokenizer, current, ids, offsets);
    tokenizer.word([char], ids, offsets);
  } else if (current.length > tokenizer.maxChars) current[current.length - 1] = char;
  else current.push(char);
}

interface WordPieceMatch {
  readonly id: number;
  readonly end: number;
}

function findLongestWordPiece(
  vocab: ReadonlyMap<string, number>,
  prefix: string,
  word: readonly WordChar[],
  start: number
): WordPieceMatch | undefined {
  for (let end = word.length; end > start; end--) {
    let candidate = '';
    for (let index = start; index < end; index++) candidate += word[index]![0];
    if (start > 0) candidate = prefix + candidate;
    const id = vocab.get(candidate);
    if (id !== undefined) return {id, end};
  }
  return undefined;
}

export class WordPiece {
  readonly vocab: Map<string, number>;
  readonly unk: number;
  readonly prefix: string;
  readonly maxChars: number;

  constructor(metadata: WordPieceMetadata) {
    this.vocab = new Map();
    metadata.vocab.forEach((piece, index) => {
      if (piece !== null) this.vocab.set(piece, index);
    });
    this.unk = this.vocab.get(metadata.unk)!;
    this.prefix = metadata.prefix;
    this.maxChars = metadata.max_chars;
  }

  // -> {ids, offsets} with offsets in JS string (UTF-16) indices of the original text
  encode(text: string): WordPieceEncoding {
    return this.encodeUntil(text, Infinity);
  }

  // Like encode(), but stops reading text once the result has more than `limit` ids. The ids are a
  // prefix of encode(text).ids, so ids.length > limit tells whether the full encoding is longer.
  encodeUntil(text: string, limit: number): WordPieceEncoding {
    const ids: number[] = [];
    const offsets: TokenOffset[] = [];
    const current: WordChar[] = [];
    let offset = 0;
    for (const char of text) {
      if (ids.length > limit) return {ids, offsets};
      const start = offset;
      const end = offset + char.length;
      offset = end;
      for (const normalized of normalizeWordCharacter(char, start, end))
        appendWordCharacter(this, current, normalized, ids, offsets);
    }
    if (ids.length <= limit) flushWord(this, current, ids, offsets);
    return {ids, offsets};
  }

  // greedy longest-match WordPiece over one word; too long or unmatched -> one unknown token
  word(word: readonly WordChar[], ids: number[], offsets: TokenOffset[]): void {
    const whole: TokenOffset = [word[0]![1], word[word.length - 1]![2]];
    if (word.length > this.maxChars) {
      ids.push(this.unk);
      offsets.push(whole);
      return;
    }
    const pieces: Array<{readonly id: number; readonly offset: TokenOffset}> = [];
    let start = 0;
    while (start < word.length) {
      const match = findLongestWordPiece(this.vocab, this.prefix, word, start);
      if (!match) {
        ids.push(this.unk);
        offsets.push(whole);
        return;
      }
      pieces.push({id: match.id, offset: [word[start]![1], word[match.end - 1]![2]]});
      start = match.end;
    }
    for (const piece of pieces) {
      ids.push(piece.id);
      offsets.push(piece.offset);
    }
  }
}

// ------------------------------------------------------------------ weights
function decodeQ4Block(
  nibbles: Uint8Array,
  output: Float32Array,
  nibbleOffset: number,
  valueOffset: number,
  scale: number
): void {
  for (let index = 0; index < 16; index++) {
    const byte = nibbles[nibbleOffset + index]!;
    output[valueOffset + index] = ((byte & 15) - 8) * scale;
    output[valueOffset + index + 16] = ((byte >> 4) - 8) * scale;
  }
}

function decodeQ4(entry: TinyDecideTensorMetadata, buffer: ArrayBuffer, size: number): Float32Array {
  const rows = entry.shape[0]!;
  const columns = entry.shape[1]!;
  const blocksPerRow = columns / 32;
  const nibbles = new Uint8Array(buffer, entry.offset, rows * blocksPerRow * 16);
  const scales = new Uint16Array(buffer, entry.scale_offset, rows * blocksPerRow);
  const floatScale = new Float32Array(1);
  const scaleBits = new Uint32Array(floatScale.buffer);
  const output = new Float32Array(size);
  for (let row = 0; row < rows; row++) {
    for (let block = 0; block < blocksPerRow; block++) {
      const blockIndex = row * blocksPerRow + block;
      scaleBits[0] = scales[blockIndex]! << 16;
      const nibbleOffset = blockIndex * 16;
      const valueOffset = row * columns + block * 32;
      decodeQ4Block(nibbles, output, nibbleOffset, valueOffset, floatScale[0]!);
    }
  }
  return output;
}

function decodeQuantizedRow(
  quantized: Int8Array,
  scales: Float32Array,
  output: Float32Array,
  row: number,
  columns: number
): void {
  const scale = scales[row]!;
  const offset = row * columns;
  for (let column = 0; column < columns; column++) output[offset + column] = quantized[offset + column]! * scale;
}

function decodeQuantized(entry: TinyDecideTensorMetadata, buffer: ArrayBuffer, size: number): Float32Array {
  const rows = entry.shape[0]!;
  const columns = entry.shape[1]!;
  const quantized = new Int8Array(buffer, entry.offset, size);
  const scales = new Float32Array(buffer, entry.scale_offset, rows);
  const output = new Float32Array(size);
  for (let row = 0; row < rows; row++) decodeQuantizedRow(quantized, scales, output, row, columns);
  return output;
}

function decodeTensor(entry: TinyDecideTensorMetadata, buffer: ArrayBuffer): WeightedTensor {
  const size = entry.shape.reduce((product, dimension) => product * dimension, 1);
  const data =
    entry.dtype === 'f32'
      ? new Float32Array(buffer, entry.offset, size).slice()
      : entry.dtype === 'q4'
        ? decodeQ4(entry, buffer, size)
        : decodeQuantized(entry, buffer, size);
  return {data, shape: entry.shape};
}

export class Weights {
  readonly t: Record<string, WeightedTensor>;

  constructor(metadata: TinyDecideWeightsMetadata, buffer: ArrayBuffer) {
    this.t = {};
    for (const entry of metadata.tensors) this.t[entry.name] = decodeTensor(entry, buffer);
  }

  get(name: string): WeightedTensor {
    const tensor = this.t[name];
    if (!tensor) throw new Error('missing tensor ' + name);
    return tensor;
  }

  has(name: string): boolean {
    return name in this.t;
  }
}

// y[t] = W x[t] + b ; W [out, in] row-major, x [T, in]
function linear(
  x: ArrayLike<number>,
  tokens: number,
  inputSize: number,
  weights: ArrayLike<number>,
  bias: ArrayLike<number> | null,
  outputSize: number
): Float32Array {
  const output = new Float32Array(tokens * outputSize);
  for (let token = 0; token < tokens; token++) {
    const inputOffset = token * inputSize;
    const outputOffset = token * outputSize;
    for (let row = 0; row < outputSize; row++) {
      const weightOffset = row * inputSize;
      let sum = bias ? bias[row]! : 0;
      for (let column = 0; column < inputSize; column++)
        sum += weights[weightOffset + column]! * x[inputOffset + column]!;
      output[outputOffset + row] = sum;
    }
  }
  return output;
}
function layernorm(
  x: ArrayLike<number>,
  tokens: number,
  dimensions: number,
  weight: ArrayLike<number>,
  bias: ArrayLike<number> | null,
  epsilon: number
): Float32Array {
  const output = new Float32Array(tokens * dimensions);
  for (let token = 0; token < tokens; token++) {
    const offset = token * dimensions;
    let mean = 0;
    for (let index = 0; index < dimensions; index++) mean += x[offset + index]!;
    mean /= dimensions;
    let variance = 0;
    for (let index = 0; index < dimensions; index++) {
      const centered = x[offset + index]! - mean;
      variance += centered * centered;
    }
    variance /= dimensions;
    const reciprocalStdDev = 1 / Math.sqrt(variance + epsilon);
    for (let index = 0; index < dimensions; index++)
      output[offset + index] =
        (x[offset + index]! - mean) * reciprocalStdDev * weight[index]! + (bias ? bias[index]! : 0);
  }
  return output;
}
// erf, Abramowitz-Stegun 7.1.26 is 1.5e-7; use a higher-precision rational (W. J. Cody style via erfc series)
function erf(x: number): number {
  const s = Math.sign(x),
    a = Math.abs(x);
  if (a < 0.5) {
    // Taylor series, fast convergence near 0
    let sum = a,
      term = a,
      a2 = a * a;
    for (let n = 1; n < 12; n++) {
      term *= -a2 / n;
      sum += term / (2 * n + 1);
    }
    return ((s * 2) / Math.sqrt(Math.PI)) * sum;
  }
  // continued fraction for erfc (Lentz), accurate to ~1e-12 for a >= 0.5
  const t = 1 / (1 + 0.5 * a);
  const y =
    t *
    Math.exp(
      -a * a -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))
    );
  return s * (1 - y);
}
const gelu = (x: number): number => 0.5 * x * (1 + erf(x / Math.SQRT2));

// ------------------------------------------------------------------ request layout
export function encodeRequest(tok: WordPiece, meta: TinyDecideMetadata, req: TinyDecideRequest): EncodedRequest {
  const sp = meta.specials,
    F = meta.format;
  const st = tok.encodeUntil(req.state, F.ts_max - 1);
  const n = Math.min(st.ids.length, F.ts_max - 1);
  const ids = [sp['<|state|>']!, ...st.ids.slice(0, n)];
  const pos = ids.map((_, i) => i),
    blk = ids.map(() => 0),
    kind = ids.map(() => K_STATE);
  const stIdx: number[] = [];
  for (let i = 1; i < ids.length; i++) stIdx.push(i);
  const stOff = st.offsets.slice(0, n);
  const qs: EncodedQuestion[] = [];
  req.questions.forEach((q, qi) => {
    const k = qi + 1,
      nOpt = (q.options || []).length;
    if ((q.type === 'choice' || q.type === 'score') && (nOpt < 2 || nOpt > F.k_max))
      throw new Error(`Question ${k} needs 2 to ${F.k_max} options, not ${nOpt}.`);
    if ((q.type === 'noul' || q.type === 'span') && nOpt)
      throw new Error(`Question ${k} is a ${q.type} question and takes no options.`);
    const bIds = [sp['<|' + q.type + '|>']!],
      bKind = [K_QTEXT];
    const t = tok.encode(q.text).ids;
    bIds.push(...t);
    t.forEach(() => bKind.push(K_QTEXT));
    const optLocal = [];
    if (q.options && q.options.length) {
      bIds.push(sp['<|sep|>']!);
      bKind.push(K_QTEXT);
      const [mk, mkind] = q.type === 'choice' ? ['<|o|>', K_OPT] : ['<|lv|>', K_LV];
      for (const o of q.options) {
        const oi = tok.encode(o).ids;
        bIds.push(...oi);
        oi.forEach(() => bKind.push(K_QTEXT));
        optLocal.push(bIds.length);
        bIds.push(sp[mk]!);
        bKind.push(mkind);
      }
    }
    const ansLocal = bIds.length;
    bIds.push(sp['<|ans|>']!);
    bKind.push(K_ANS);
    if (bIds.length > F.q_max) throw new Error(`Question ${k} is too long (${bIds.length} tokens, max ${F.q_max}).`);
    const base = ids.length;
    bIds.forEach((id, j) => {
      ids.push(id);
      pos.push(F.p_q + j);
      blk.push(k);
      kind.push(bKind[j]!);
    });
    qs.push({type: q.type, ans: base + ansLocal, opt: optLocal.map((j) => base + j)});
  });
  return {ids, pos, blk, kind, stIdx, stOff, qs, truncated: st.ids.length > n};
}

function continues(kind: readonly number[], fusion: string): boolean[] {
  if (fusion === 'all') return kind.map(() => true);
  if (fusion === 'markers') return kind.map((k) => k === K_STATE || k === K_ANS || k === K_OPT || k === K_LV);
  return kind.map((k) => k === K_STATE || k === K_ANS);
}

function attendHead(
  query: ArrayLike<number>,
  key: ArrayLike<number>,
  value: ArrayLike<number>,
  keys: readonly number[],
  token: number,
  head: number,
  dimensions: number,
  headDimensions: number,
  scale: number,
  scores: Float32Array,
  output: Float32Array
): void {
  const queryOffset = token * dimensions + head * headDimensions;
  let maximum = -Infinity;
  for (let index = 0; index < keys.length; index++) {
    const keyOffset = keys[index]! * dimensions + head * headDimensions;
    let score = 0;
    for (let column = 0; column < headDimensions; column++)
      score += query[queryOffset + column]! * key[keyOffset + column]!;
    score *= scale;
    scores[index] = score;
    if (score > maximum) maximum = score;
  }
  let normalizer = 0;
  for (let index = 0; index < keys.length; index++) {
    scores[index] = Math.exp(scores[index]! - maximum);
    normalizer += scores[index]!;
  }
  const outputOffset = token * dimensions + head * headDimensions;
  for (let index = 0; index < keys.length; index++) {
    const probability = scores[index]! / normalizer;
    const valueOffset = keys[index]! * dimensions + head * headDimensions;
    for (let column = 0; column < headDimensions; column++)
      output[outputOffset + column] = output[outputOffset + column]! + probability * value[valueOffset + column]!;
  }
}

function fillLowRankTokenEmbeddings(model: TinyDecide, encoded: EncodedRequest, output: Float32Array): void {
  const cfg = model.cfg;
  const sequence = encoded.ids.length;
  const dimensions = cfg.emb_dim;
  const left = model.w('m.word.a.weight');
  const right = model.w('m.word.b.weight');
  const rank = model.W.get('m.word.a.weight').shape[1]!;
  const positions = model.w('m.posemb.weight');
  const types = model.w('m.type0');
  for (let token = 0; token < sequence; token++) {
    for (let column = 0; column < dimensions; column++) {
      let sum = 0;
      const leftOffset = encoded.ids[token]! * rank;
      const rightOffset = column * rank;
      for (let index = 0; index < rank; index++) sum += left[leftOffset + index]! * right[rightOffset + index]!;
      output[token * dimensions + column] =
        sum + positions[encoded.pos[token]! * dimensions + column]! + types[column]!;
    }
  }
}

function fillFullTokenEmbeddings(model: TinyDecide, encoded: EncodedRequest, output: Float32Array): void {
  const cfg = model.cfg;
  const sequence = encoded.ids.length;
  const dimensions = cfg.emb_dim;
  const word = model.w('m.word.weight');
  const positions = model.w('m.posemb.weight');
  const types = model.w('m.type0');
  for (let token = 0; token < sequence; token++) {
    for (let column = 0; column < dimensions; column++)
      output[token * dimensions + column] =
        word[encoded.ids[token]! * dimensions + column]! +
        positions[encoded.pos[token]! * dimensions + column]! +
        types[column]!;
  }
}

function createElectraEmbeddings(model: TinyDecide, encoded: EncodedRequest): Float32Array {
  const cfg = model.cfg;
  const sequence = encoded.ids.length;
  const embeddingDimensions = cfg.emb_dim;
  const raw = new Float32Array(sequence * embeddingDimensions);
  if (model.W.has('m.word.a.weight')) fillLowRankTokenEmbeddings(model, encoded, raw);
  else fillFullTokenEmbeddings(model, encoded, raw);
  const normalized = layernorm(
    raw,
    sequence,
    embeddingDimensions,
    model.w('m.eln.weight'),
    model.w('m.eln.bias'),
    cfg.ln_eps
  );
  return linear(normalized, sequence, embeddingDimensions, model.w('m.proj.weight'), model.w('m.proj.bias'), cfg.d);
}

function runElectraLayer(
  model: TinyDecide,
  input: Float32Array,
  sequence: number,
  attentionSets: readonly (readonly number[])[],
  layer: number
): Float32Array {
  const cfg = model.cfg;
  const dimensions = cfg.d;
  const prefix = `m.blocks.${layer}.`;
  const query = linear(
    input,
    sequence,
    dimensions,
    model.w(prefix + 'q.weight'),
    model.w(prefix + 'q.bias'),
    dimensions
  );
  const key = linear(input, sequence, dimensions, model.w(prefix + 'k.weight'), model.w(prefix + 'k.bias'), dimensions);
  const value = linear(
    input,
    sequence,
    dimensions,
    model.w(prefix + 'v.weight'),
    model.w(prefix + 'v.bias'),
    dimensions
  );
  const attended = model.attend(
    query,
    key,
    value,
    sequence,
    attentionSets,
    cfg.heads,
    1 / Math.sqrt(dimensions / cfg.heads)
  );
  const projected = linear(
    attended,
    sequence,
    dimensions,
    model.w(prefix + 'o.weight'),
    model.w(prefix + 'o.bias'),
    dimensions
  );
  for (let index = 0; index < sequence * dimensions; index++) projected[index] = projected[index]! + input[index]!;
  const attentionOutput = layernorm(
    projected,
    sequence,
    dimensions,
    model.w(prefix + 'ln1.weight'),
    model.w(prefix + 'ln1.bias'),
    cfg.ln_eps
  );
  const feedForwardDimensions = model.W.get(prefix + 'fc.weight').shape[0]!;
  const feedForward = linear(
    attentionOutput,
    sequence,
    dimensions,
    model.w(prefix + 'fc.weight'),
    model.w(prefix + 'fc.bias'),
    feedForwardDimensions
  );
  for (let index = 0; index < feedForward.length; index++) feedForward[index] = gelu(feedForward[index]!);
  const feedForwardOutput = linear(
    feedForward,
    sequence,
    feedForwardDimensions,
    model.w(prefix + 'fc2.weight'),
    model.w(prefix + 'fc2.bias'),
    dimensions
  );
  for (let index = 0; index < sequence * dimensions; index++)
    feedForwardOutput[index] = feedForwardOutput[index]! + attentionOutput[index]!;
  return layernorm(
    feedForwardOutput,
    sequence,
    dimensions,
    model.w(prefix + 'ln2.weight'),
    model.w(prefix + 'ln2.bias'),
    cfg.ln_eps
  );
}

function dotProduct(left: NumericVector, right: NumericVector): number {
  let sum = 0;
  for (let index = 0; index < left.length; index++) sum += left[index]! * right[index]!;
  return sum;
}

function normalizeVector(values: readonly number[]): number[] {
  const norm = Math.sqrt(dotProduct(values, values)) || 1e-12;
  return values.map((value) => value / norm);
}

function bucketForCount(count: number): number {
  return K_BUCKETS.reduce((bucket, edge) => (count > edge ? bucket + 1 : bucket), 0);
}

function matrixVector(context: AnswerContext, name: string, vector: NumericVector, biasName?: string): Float32Array {
  const model = context.model;
  const bias = biasName ? model.w(biasName) : null;
  return linear(vector, 1, context.dimensions, model.w(name), bias, model.W.get(name).shape[0]!);
}

function createAnswerContext(model: TinyDecide, request: TinyDecideRequest): AnswerContext {
  const encoded = encodeRequest(model.tok, model.meta, request);
  const hidden = model.hidden(encoded);
  const dimensions = model.cfg.d;
  const normalized = layernorm(
    hidden,
    encoded.ids.length,
    dimensions,
    model.w('h.norm.weight'),
    model.w('h.norm.bias'),
    1e-5
  );
  return {
    model,
    request,
    encoded,
    hidden,
    normalized,
    dimensions,
    headDimensions: model.cfg.dh_head
  };
}

function answerRow(context: AnswerContext, index: number): Float32Array {
  return context.normalized.subarray(index * context.dimensions, (index + 1) * context.dimensions);
}

function createPrototypeScorer(
  context: AnswerContext,
  proto: TinyDecidePrototype | null | undefined,
  answerVector: Float32Array
): PrototypeScorer {
  const {model, headDimensions} = context;
  const hasProjection = model.W.has('h.P.weight');
  const projected = hasProjection ? matrixVector(context, 'h.P.weight', answerVector) : null;
  let center: Float32Array | null | undefined = proto && proto.center;
  const centerFor = (count: number): Float32Array => {
    if (center) return center;
    const mean = Array.from({length: headDimensions}, () => 0);
    let total = 0;
    if (proto) {
      for (let index = 0; index < count; index++) {
        const examples = proto.cnt[index]!;
        if (examples > 0) {
          total += examples;
          for (let dimension = 0; dimension < headDimensions; dimension++)
            mean[dimension] = mean[dimension]! + examples * proto.vec[index * headDimensions + dimension]!;
        }
      }
    }
    center = Float32Array.from(mean, (value) => value / total);
    return center;
  };
  const term = (index: number, count: number): number | null => {
    if (!proto || !(proto.cnt[index]! > 0)) return 0;
    if (!projected) return null;
    const centerVector = centerFor(count);
    const answer = new Float32Array(headDimensions);
    const prototype = new Float32Array(headDimensions);
    for (let dimension = 0; dimension < headDimensions; dimension++) {
      answer[dimension] = projected[dimension]! - centerVector[dimension]!;
      prototype[dimension] = proto.vec[index * headDimensions + dimension]! - centerVector[dimension]!;
    }
    const answerNorm = Math.sqrt(dotProduct(answer, answer)) || 1e-12;
    const prototypeNorm = Math.sqrt(dotProduct(prototype, prototype)) || 1e-12;
    return (
      (model.meta.beta[bucketForCount(proto.cnt[index]!)]! * dotProduct(answer, prototype)) /
      (answerNorm * prototypeNorm)
    );
  };
  return {proto: proto ?? undefined, projected, term};
}

function uncenteredPrototypeTerm(
  context: AnswerContext,
  proto: TinyDecidePrototype | null | undefined,
  index: number,
  queryVector: readonly number[]
): number {
  if (!proto || !(proto.cnt[index]! > 0)) return 0;
  const {headDimensions, model} = context;
  const vector = Array.from(proto.vec.subarray(index * headDimensions, (index + 1) * headDimensions));
  return model.meta.beta[bucketForCount(proto.cnt[index]!)]! * dotProduct(queryVector, normalizeVector(vector));
}

function prototypeWeight(proto: TinyDecidePrototype | null | undefined): number {
  return proto && proto.lam != null ? proto.lam : 1;
}

function choiceOptionLogit(
  context: AnswerContext,
  question: EncodedQuestion,
  scorer: PrototypeScorer,
  query: Float32Array,
  normalizedQuery: number[],
  selection: number,
  temperature: number,
  scale: number,
  index: number,
  zeroShot: number[]
): number {
  const output = matrixVector(context, `h.O.${selection}.weight`, answerRow(context, question.opt[index]!));
  let logit = (scale * dotProduct(query, output)) / Math.sqrt(context.headDimensions);
  zeroShot.push(logit / temperature);
  const correction = scorer.term(index, question.opt.length);
  if (correction === null) logit += uncenteredPrototypeTerm(context, scorer.proto, index, normalizedQuery);
  else logit += prototypeWeight(scorer.proto) * correction;
  return logit;
}

function categoricalResult(
  logits: number[],
  temperature: number
): {
  probs: number[];
  pick: number;
  confidence: number;
} {
  const maximum = Math.max(...logits);
  const exponents = logits.map((value) => Math.exp((value - maximum) / temperature));
  const sum = exponents.reduce((total, value) => total + value, 0);
  const probs = exponents.map((value) => value / sum);
  const count = probs.length;
  const entropy = -probs.reduce(
    (total, probability) => total + (probability > 0 ? probability * Math.log(probability) : 0),
    0
  );
  return {probs, pick: probs.indexOf(Math.max(...probs)), confidence: 1 - entropy / Math.log(count)};
}

function answerChoiceOrScore(
  context: AnswerContext,
  question: EncodedQuestion,
  answer: Float32Array,
  scorer: PrototypeScorer
): ChoiceAnswer {
  const model = context.model;
  const selection = question.type === 'score' ? 1 : 0;
  const query = matrixVector(context, `h.A.${selection}.weight`, answer);
  const scale = Math.exp(model.w('h.scale')[selection]!);
  const normalizedQuery = normalizeVector(Array.from(query));
  const zeroShot: number[] = [];
  const temperatureIndex = selection === 1 ? 2 : 0;
  const temperature = model.meta.temp[temperatureIndex]!;
  const logits = question.opt.map((_, index) =>
    choiceOptionLogit(context, question, scorer, query, normalizedQuery, selection, temperature, scale, index, zeroShot)
  );
  const distribution = categoricalResult(logits, temperature);
  const common = {
    probs: distribution.probs,
    pick: distribution.pick,
    confidence: distribution.confidence,
    qvec: scorer.projected ? Array.from(scorer.projected) : normalizedQuery,
    proj: scorer.projected !== null,
    z0: zeroShot
  };
  if (question.type === 'score') {
    return {
      type: 'score',
      ...common,
      score: distribution.probs.reduce(
        (total, probability, index) => total + (probability * index) / (distribution.probs.length - 1),
        0
      )
    };
  }
  return {type: 'choice', ...common};
}

function noulPrototypeCorrection(
  context: AnswerContext,
  scorer: PrototypeScorer,
  query: readonly number[]
): number | undefined {
  const proto = scorer.proto;
  if (!proto) return undefined;
  const weight = prototypeWeight(proto);
  const terms = [0, 1].map((index) => {
    const correction = scorer.term(index, 2);
    if (correction !== null) return weight * correction;
    return uncenteredPrototypeTerm(context, proto, index, query);
  });
  return terms[1]! - terms[0]!;
}

function answerNoul(
  context: AnswerContext,
  question: EncodedQuestion,
  answer: Float32Array,
  scorer: PrototypeScorer
): NoulAnswer {
  const model = context.model;
  let logit = dotProduct(model.w('h.noul.weight'), answer) + model.w('h.noul.bias')[0]!;
  const zeroShot = logit / model.meta.temp[1]!;
  const query = normalizeVector(Array.from(matrixVector(context, 'h.noul_q.weight', answer)));
  const correction = noulPrototypeCorrection(context, scorer, query);
  if (correction !== undefined) logit += correction;
  return {
    type: 'noul',
    p: 1 / (1 + Math.exp(-logit / model.meta.temp[1]!)),
    qvec: scorer.projected ? Array.from(scorer.projected) : query,
    proj: scorer.projected !== null,
    z0: [0, zeroShot]
  };
}

function logSoftmax(values: readonly number[], temperature: number): number[] {
  const maximum = Math.max(...values);
  const logNormalizer = Math.log(values.reduce((sum, value) => sum + Math.exp((value - maximum) / temperature), 0));
  return values.map((value) => (value - maximum) / temperature - logNormalizer);
}

function bestScoreIndex(values: readonly number[], initial: number, count: number, offset: number): number {
  let best = initial;
  for (let index = 0; index < count; index++) if (values[offset + index]! > values[offset + best]!) best = index;
  return best;
}

function answerSpan(context: AnswerContext, question: EncodedQuestion): SpanAnswer {
  const {model, encoded, headDimensions} = context;
  const sequence = encoded.stIdx.length;
  const answer = answerRow(context, question.ans);
  const queryStart = matrixVector(context, 'h.sq.weight', answer);
  const keys = encoded.stIdx.map((index) => matrixVector(context, 'h.sk.weight', answerRow(context, index)));
  const startLogits = [
    dotProduct(model.w('h.snull.weight'), answer) + model.w('h.snull.bias')[0]!,
    ...keys.map((key) => dotProduct(queryStart, key) / Math.sqrt(headDimensions))
  ];
  const logStart = logSoftmax(startLogits, model.meta.temp[3]!);
  const bestStart = bestScoreIndex(logStart, 0, sequence, 1);
  const queryEnd = matrixVector(context, 'h.eq.weight', answer);
  const startEmbedding = matrixVector(context, 'h.es.weight', answerRow(context, encoded.stIdx[bestStart] ?? 0));
  const queryEndWithStart = queryEnd.map((value, index) => value + startEmbedding[index]!);
  const endLogits = encoded.stIdx.map((index, token) =>
    token >= bestStart && token < bestStart + model.meta.format.span_max
      ? dotProduct(queryEndWithStart, matrixVector(context, 'h.ek.weight', answerRow(context, index))) /
        Math.sqrt(headDimensions)
      : -1e4
  );
  const logEnd = sequence ? logSoftmax(endLogits, model.meta.temp[3]!) : [];
  const bestEnd = sequence ? bestScoreIndex(logEnd, bestStart, sequence, 0) : bestStart;
  const startOffset = encoded.stOff[bestStart];
  const endOffset = encoded.stOff[bestEnd];
  let char: TokenOffset | null = null;
  let text = '';
  if (sequence && startOffset && endOffset) {
    char = [startOffset[0], endOffset[1]];
    text = context.request.state.slice(char[0], char[1]).trim();
  }
  return {
    type: 'span',
    p_present: 1 - Math.exp(logStart[0]!),
    tok: [bestStart, bestEnd],
    p_span: sequence ? Math.exp(logStart[1 + bestStart]! + logEnd[bestEnd]!) : 0,
    text,
    char
  };
}

function answerQuestion(
  context: AnswerContext,
  question: EncodedQuestion,
  proto: TinyDecidePrototype | null | undefined
): TinyDecideAnswer {
  const answer = answerRow(context, question.ans);
  const scorer = createPrototypeScorer(context, proto, answer);
  if (question.type === 'choice' || question.type === 'score')
    return answerChoiceOrScore(context, question, answer, scorer);
  if (question.type === 'noul') return answerNoul(context, question, answer, scorer);
  return answerSpan(context, question);
}

// ------------------------------------------------------------------ model
export class TinyDecide {
  readonly meta: TinyDecideMetadata;
  readonly cfg: TinyDecideConfig;
  readonly W: Weights;
  readonly tok: WordPiece;

  constructor(meta: TinyDecideMetadata, buffer: ArrayBuffer) {
    this.meta = meta;
    this.cfg = meta.cfg;
    this.W = new Weights(meta, buffer);
    const tokenizer = meta.tokenizer;
    if (tokenizer.kind !== 'wordpiece') throw new Error('this engine build ships the WordPiece tokenizer only');
    this.tok = new WordPiece(tokenizer);
  }
  w(name: string): Float32Array {
    return this.W.get(name).data;
  }

  // attention over allowed keys; q,k,v [T, d]; returns [T, d]
  attend(
    query: Float32Array,
    key: Float32Array,
    value: Float32Array,
    sequence: number,
    allowed: readonly (readonly number[])[],
    heads: number,
    scale: number
  ): Float32Array {
    const dimensions = this.cfg.d;
    const headDimensions = dimensions / heads;
    const output = new Float32Array(sequence * dimensions);
    const scores = new Float32Array(sequence);
    for (let token = 0; token < sequence; token++) {
      const keys = allowed[token];
      if (!keys) continue;
      for (let head = 0; head < heads; head++)
        attendHead(query, key, value, keys, token, head, dimensions, headDimensions, scale, scores, output);
    }
    return output;
  }

  allowedSets(encoded: EncodedRequest, continuation: readonly boolean[], fusionLayer: boolean): number[][] {
    const sequence = encoded.ids.length;
    const blocks = encoded.blk;
    const byBlock = new Map<number, number[]>();
    for (let index = 0; index < sequence; index++) {
      if (fusionLayer && !continuation[index]) continue;
      if (!byBlock.has(blocks[index]!)) byBlock.set(blocks[index]!, []);
      byBlock.get(blocks[index]!)!.push(index);
    }
    const state = byBlock.get(0) || [];
    const sets: number[][] = Array.from({length: sequence}, () => []);
    for (let index = 0; index < sequence; index++) {
      if (fusionLayer && !continuation[index]) {
        sets[index] = [index];
        continue;
      }
      const block = blocks[index]!;
      const own = byBlock.get(block) || [index];
      sets[index] = !fusionLayer || block === 0 ? own : state.concat(own);
    }
    return sets;
  }

  hidden(encoded: EncodedRequest): Float32Array {
    const cfg = this.cfg;
    const sequence = encoded.ids.length;
    const firstFusionLayer = cfg.L_a || 0;
    const continuation = continues(encoded.kind, cfg.fusion);
    const beforeFusion = this.allowedSets(encoded, continuation, false);
    const afterFusion = this.allowedSets(encoded, continuation, true);
    if (cfg.arch === 'electra') {
      let output = createElectraEmbeddings(this, encoded);
      for (let layer = 0; layer < cfg.layers; layer++) {
        const sets = layer < firstFusionLayer ? beforeFusion : afterFusion;
        output = runElectraLayer(this, output, sequence, sets, layer);
      }
      return output;
    }
    throw new Error('unsupported architecture ' + cfg.arch);
  }

  // protos: per question {vec: Float32Array[K*dh] | null, cnt: Int32Array[K]} or undefined
  answer(
    request: TinyDecideRequest,
    protos?: readonly (TinyDecidePrototype | null | undefined)[] | null
  ): TinyDecideEngineAnswer {
    const startedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const context = createAnswerContext(this, request);
    const answers = context.encoded.qs.map((question, index) => answerQuestion(context, question, protos?.[index]));
    const finishedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const stateTokens = context.encoded.stIdx.length + 1;
    const totalTokens = context.encoded.ids.length;
    return {
      answers,
      tokens: {
        state: stateTokens,
        questions: totalTokens - stateTokens,
        total: totalTokens
      },
      truncated: context.encoded.truncated,
      ms: finishedAt - startedAt,
      ids: context.encoded.ids
    };
  }
}
