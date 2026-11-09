import {WebDecisionError} from '../kev.js';
import {uploadTensorShards} from '../tensor-shard-upload.js';

type MapikaUploadOptions = Parameters<typeof uploadTensorShards>[0] & {labels: {ids: number[]}};

/** Check that the compiled 255 label IDs agree with the tokenizer while uploading. */
export async function uploadMapikaWeights({labels, ...args}: MapikaUploadOptions): Promise<void> {
  const expected = new Uint8Array(255 * 4),
    view = new DataView(expected.buffer);
  labels.ids.forEach((id, i) => view.setInt32(i * 4, id, true));
  return uploadTensorShards({
    ...args,
    inspectFragment(index, bytes, offset) {
      if (index !== 320) return;
      for (let i = 0; i < bytes.length; i++)
        if (bytes[i] !== expected[offset + i])
          throw new WebDecisionError('INVALID_ASSET', 'Tokenizer labels do not match the compiled Mapika model.');
    }
  });
}
