import {WebDecisionError} from '../kev.js';
import {uploadTensorShards} from '../tensor-shard-upload.js';
import {MAPIKA_Q4_LABEL_INDEX} from './contract.js';
import type {MapikaQ4UploadArgs} from './types.js';

/** Check tokenizer label IDs against the pinned graph while streaming weights. */
export async function uploadMapikaQ4Weights({labels, ...args}: MapikaQ4UploadArgs): Promise<void> {
  const expected = new Uint8Array(255 * 4),
    view = new DataView(expected.buffer);
  labels.ids.forEach((id, i) => view.setInt32(i * 4, id, true));
  return uploadTensorShards({
    ...args,
    inspectFragment(index: number, bytes: Uint8Array, offset: number) {
      if (index !== MAPIKA_Q4_LABEL_INDEX) return;
      for (let i = 0; i < bytes.length; i++)
        if (bytes[i] !== expected[offset + i])
          throw new WebDecisionError('INVALID_ASSET', 'Tokenizer labels do not match the compiled Mapika Q4 model.');
    }
  });
}
