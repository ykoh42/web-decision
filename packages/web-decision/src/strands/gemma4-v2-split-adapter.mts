import {runGemma4V2ChunkedDecoder} from './gemma4-v2-chunked-decoder.mjs';

type Context = Parameters<typeof runGemma4V2ChunkedDecoder>[0];
type Input = Parameters<typeof runGemma4V2ChunkedDecoder>[1];
type Output = Awaited<ReturnType<typeof runGemma4V2ChunkedDecoder>>;
type Tensor = Input['hidden'];
type Graphs = Context['graphs'];
type Family = 0 | 4 | 15 | 19;
type Runner = (context: Context, input: Input) => Promise<Output>;
/** Names from the private four-family transposed candidate, not a release artifact. */
export function transposedSplitGraphs(): Readonly<Record<Family, Graphs>> {
  const graph = (family: Family): Graphs => ({
    attention: `split${family}attn_packed_main`,
    unpack: `split${family}unpack_t_main`,
    tail: `split${family}optimized_tail_main`,
    slice: 'split_slice_main',
    concat: 'split_concat_main'
  });
  return {0: graph(0), 4: graph(4), 15: graph(15), 19: graph(19)};
}
const tailNames = [
  'pre_feedforward_layernorm.weight',
  'mlp.gate_proj.weight',
  'mlp.up_proj.weight',
  'mlp.down_proj.weight',
  'post_feedforward_layernorm.weight',
  'per_layer_input_gate.weight',
  'per_layer_projection.weight',
  'post_per_layer_input_norm.weight'
] as const;
const unsharedAttentionNames = [
  'self_attn.q_proj.weight',
  'self_attn.q_norm.weight',
  'self_attn.k_norm.weight',
  'self_attn.k_proj.weight',
  'self_attn.v_proj.weight',
  'self_attn.o_proj.weight',
  'input_layernorm.weight',
  'post_attention_layernorm.weight'
] as const;
const sharedAttentionNames = [
  'self_attn.q_proj.weight',
  'self_attn.q_norm.weight',
  'self_attn.o_proj.weight',
  'input_layernorm.weight',
  'post_attention_layernorm.weight'
] as const;

export function gemma4V2LayerFamily(index: number): Family {
  if (!Number.isInteger(index) || index < 0 || index >= 35) throw new RangeError('Invalid Gemma 4 layer index.');
  return index < 15 ? (index % 5 === 4 ? 4 : 0) : index % 5 === 4 ? 19 : 15;
}

export type SplitAdapterInput = Omit<Input, 'attentionWeights' | 'tailWeights' | 'cache' | 'retainKeyValues'> & {
  index: number;
  weights: readonly {name: string; value: Tensor}[];
  sharedCache?: readonly [Tensor, Tensor];
};

/** Graph names and the execution plan must match the pinned artifact. */
export async function runSplitAdapter(
  context: Omit<Context, 'graphs'> & {graphs: Readonly<Record<Family, Graphs>>},
  input: SplitAdapterInput,
  run: Runner = runGemma4V2ChunkedDecoder
): Promise<Output> {
  context.signal?.throwIfAborted();
  const family = gemma4V2LayerFamily(input.index);
  const names = family < 15 ? unsharedAttentionNames : sharedAttentionNames;
  if (family >= 15 !== (input.sharedCache !== undefined)) throw new Error('Invalid Gemma 4 shared KV state.');
  const prefix = `model.language_model.layers.${input.index}.`;
  const weights = new Map<string, Tensor>();
  for (const item of input.weights) {
    if (!item.name.startsWith(prefix) || weights.has(item.name)) throw new Error('Invalid split decoder weight.');
    weights.set(item.name, item.value);
  }
  if (weights.size !== names.length + tailNames.length) throw new Error('Invalid split decoder weight count.');
  const select = (keys: readonly string[]) =>
    keys.map((name) => {
      const weight = weights.get(prefix + name);
      if (!weight) throw new Error(`Missing split decoder weight: ${name}`);
      return weight;
    });
  const graphs = context.graphs[family];
  if (
    !graphs ||
    [graphs.attention, graphs.unpack, graphs.tail, graphs.slice, graphs.concat].some(
      (name) => typeof name !== 'string' || !name
    )
  )
    throw new Error('Missing split decoder graph.');
  const {index, weights: _weights, sharedCache, ...base} = input;
  const output = await run(
    {...context, graphs},
    {
      ...base,
      attentionWeights: select(names),
      tailWeights: select(tailNames),
      cache: sharedCache ?? [],
      retainKeyValues: index === 13 || index === 14
    }
  );
  if (context.signal?.aborted) {
    for (const tensor of [output.hidden, output.key, output.value]) {
      try {
        tensor?.dispose();
      } catch {
        // Preserve the caller's abort reason.
      }
    }
    context.signal.throwIfAborted();
  }
  return output;
}
