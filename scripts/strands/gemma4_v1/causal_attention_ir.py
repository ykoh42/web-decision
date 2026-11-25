"""Lower one bounded row tile of upstream Gemma 4 v1 eager attention."""

import torch
from bf16_lowering import lower_bf16_program
from causal_layer_ir import MAX_LENGTH

MAX_QUERY_TILE = 64
QUERY_HEADS = 8


class GemmaCausalAttentionTile(torch.nn.Module):
    def forward(self, query, key, value, causal_mask):
        # E2B has one K/V head. Folding query heads into rows avoids a batched
        # matmul that materializes all eight copies of each key and value.
        batch, heads, rows, width = query.shape
        keys = key.shape[2]
        query_rows = query.reshape(heads * rows, width)
        key_columns = key.reshape(keys, width).transpose(0, 1)
        # Keep the original eager scale, mask, FP32 softmax and BF16 cast.
        weights = (torch.matmul(query_rows, key_columns) * 1.0).reshape(
            batch, heads, rows, keys
        )
        weights = weights + causal_mask
        probabilities = torch.nn.functional.softmax(
            weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        values = torch.matmul(
            probabilities.reshape(heads * rows, keys), value.reshape(keys, width)
        )
        return values.reshape(batch, heads, rows, width).transpose(1, 2).contiguous()


def import_attention_tile(example, max_keys=MAX_LENGTH):
    queries = torch.export.Dim("queries", min=1, max=MAX_QUERY_TILE)
    keys = torch.export.Dim("keys", min=1, max=max_keys)
    shapes = ({2: queries}, {2: keys}, {2: keys}, {2: queries, 3: keys})
    program = torch.export.export(
        GemmaCausalAttentionTile().eval(), example, dynamic_shapes=shapes, strict=False
    )
    return lower_bf16_program(program, example, shapes)


def query_tiles(length):
    for start in range(0, length, MAX_QUERY_TILE):
        yield slice(start, min(start + MAX_QUERY_TILE, length))
