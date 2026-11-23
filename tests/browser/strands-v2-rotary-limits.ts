/** TVM's pooled allocator rounds each WebGPU allocation to a 4096-byte page. */
export function rotaryAllocationBytes(length: number): number {
  const bytes = length * 256 * Float32Array.BYTES_PER_ELEMENT;
  return Math.ceil(bytes / 4096) * 4096;
}
