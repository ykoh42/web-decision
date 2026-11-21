import {WebDecisionError} from '../kev.js';
import {
  MAPIKA2B_VISION,
  type Mapika2bImageAsset,
  type Mapika2bLimits,
  type Mapika2bPlan,
  type Mapika2bRow,
  type Mapika2bTemperatures,
  type Mapika2bTokenizer
} from './contract.mjs';
import {
  createMapika2bLabels,
  prefixMapika2bImages,
  prepareMapika2bRequest,
  type Mapika2bImageGrid
} from './protocol.mjs';
import {finishMapika2bResult, type Mapika2bResult} from './readout.mjs';
import {widenMapika2bEmbeddingRows, type Mapika2bTextRow} from './text-runtime.mjs';

export interface Mapika2bVisionFeatures {
  readonly grid: Mapika2bImageGrid;
  readonly embeddings: Float32Array;
}

export interface Mapika2bForwardRow {
  readonly inputIds: readonly number[];
  readonly slotIndices: readonly number[];
  readonly images: readonly Mapika2bVisionFeatures[];
}

interface Mapika2bSessionOptions {
  readonly tokenizer: Mapika2bTokenizer;
  readonly limits?: Mapika2bLimits;
  readonly temperatures?: Mapika2bTemperatures;
  readonly encodeImage: (asset: Mapika2bImageAsset) => Promise<Mapika2bVisionFeatures>;
  readonly run: (row: Mapika2bForwardRow) => Promise<Float32Array>;
  readonly dispose: () => Promise<void>;
}

interface Mapika2bEvaluation {
  readonly result: Mapika2bResult;
  readonly logitsByRow: readonly Float32Array[];
  readonly rows: readonly Mapika2bRow[];
}

const hiddenSize = 2048;

class Mapika2bPositionWriter {
  readonly values: Int32Array;
  cursor = 0;
  position = 0;

  constructor(readonly length: number) {
    this.values = new Int32Array(length * 4);
  }

  write(time: number, height: number, width: number): void {
    const {cursor, length, position} = this;
    this.values[cursor] = cursor;
    this.values[length + cursor] = position + time;
    this.values[2 * length + cursor] = position + height;
    this.values[3 * length + cursor] = position + width;
    this.cursor += 1;
  }

  text(): void {
    this.write(0, 0, 0);
    this.position += 1;
  }

  image(frames: number, height: number, width: number): void {
    for (let time = 0; time < frames; time += 1) {
      for (let row = 0; row < height; row += 1) {
        for (let column = 0; column < width; column += 1) this.write(time, row, column);
      }
    }
    this.position += Math.max(height, width);
  }
}

/** Assemble the Qwen text positions and replace only the v12 image-pad embeddings. */
export function assembleMapika2bTextRow(row: Mapika2bForwardRow, packedEmbeddings: Uint16Array): Mapika2bTextRow {
  const embeddings = widenMapika2bEmbeddingRows(packedEmbeddings);
  if (embeddings.length !== row.inputIds.length * hiddenSize)
    throw new TypeError('Mapika embedding lookup returned an incomplete row.');
  const positions = new Mapika2bPositionWriter(row.inputIds.length);
  for (const {
    grid: [frames, rows, columns],
    embeddings: vision
  } of row.images) {
    const height = rows / MAPIKA2B_VISION.mergeSize;
    const width = columns / MAPIKA2B_VISION.mergeSize;
    const imageTokens = frames * height * width;
    const imageStart = positions.cursor;
    const imageEnd = imageStart + imageTokens + 1;
    if (
      vision.length !== imageTokens * hiddenSize ||
      row.inputIds[imageStart] !== MAPIKA2B_VISION.startToken ||
      row.inputIds[imageEnd] !== MAPIKA2B_VISION.endToken ||
      row.inputIds.slice(imageStart + 1, imageEnd).some((id) => id !== MAPIKA2B_VISION.imageToken)
    )
      throw new TypeError('Mapika vision features differ from the prepared token block.');
    positions.text();
    embeddings.set(vision, positions.cursor * hiddenSize);
    positions.image(frames, height, width);
    positions.text();
  }
  while (positions.cursor < positions.length) positions.text();
  return {embeddings, positionIds: positions.values, slotIndices: Int32Array.from(row.slotIndices)};
}

export function createMapika2bSession(options: Mapika2bSessionOptions) {
  const labels = createMapika2bLabels(options.tokenizer);
  let pending = Promise.resolve();
  let closing: Promise<void> | undefined;

  async function evaluatePrepared(prepared: Mapika2bPlan): Promise<Mapika2bEvaluation> {
    let plan = prepared;
    const unique = new Map<Mapika2bImageAsset, Mapika2bVisionFeatures>();
    const images = [];
    for (const asset of plan.images) {
      const features = unique.get(asset) ?? (await options.encodeImage(asset));
      unique.set(asset, features);
      images.push(features);
    }
    if (images.length > 0)
      plan = prefixMapika2bImages(
        plan,
        images.map(({grid}) => grid)
      );
    const logitsByRow = [];
    for (const row of plan.rows) {
      // Packed empty requests still count the prepared context row, which has no logits.
      if (row.slots.length === 0) {
        logitsByRow.push(new Float32Array(0));
        continue;
      }
      logitsByRow.push(
        await options.run({inputIds: row.inputIds, slotIndices: row.slots.map(({tokenIndex}) => tokenIndex), images})
      );
    }
    return {result: finishMapika2bResult(plan, logitsByRow), logitsByRow, rows: plan.rows};
  }

  function evaluate(request: unknown): Promise<Mapika2bEvaluation> {
    if (closing) return Promise.reject(new WebDecisionError('SESSION_CLOSED', 'The Mapika session is closed.'));
    try {
      const plan = prepareMapika2bRequest(request, options.tokenizer, labels, options.limits, options.temperatures);
      const result = pending.then(() => evaluatePrepared(plan));
      pending = result.then(
        () => {},
        () => {}
      );
      return result;
    } catch (cause) {
      return Promise.reject(cause);
    }
  }

  return Object.freeze({
    evaluate,
    systemOne: async (request: unknown) => (await evaluate(request)).result,
    dispose: () => (closing ??= pending.then(options.dispose))
  });
}
