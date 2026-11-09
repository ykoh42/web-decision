/** Public domain fast-sha256 1.3.0; see LICENSE. */
export class Hash {
  constructor();
  reset(): this;
  clean(): void;
  update(data: Uint8Array, dataLength?: number): this;
  finish(out: Uint8Array): this;
  digest(): Uint8Array;
}
