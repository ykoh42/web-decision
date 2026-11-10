const kananaFloatValues = new WeakMap<KananaFloat, number>();

export class KananaFloat {
  constructor(readonly value: number) {
    if (!Number.isFinite(value)) throw new TypeError('KananaFloat requires a finite number.');
    kananaFloatValues.set(this, value);
    Object.freeze(this);
  }
}

export function kananaFloat(value: number): KananaFloat {
  return new KananaFloat(value);
}

export function readKananaFloat(value: unknown): number | undefined {
  if (typeof value !== 'object' || value === null) return undefined;
  return kananaFloatValues.get(value as KananaFloat);
}
