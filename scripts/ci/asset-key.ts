/** Identify an exact quoted object key in a generated asset chunk. */
export function hasQuotedAssetKey(source: string, name: string): boolean {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/gu, '\\$&');
  return new RegExp(`(["'])${escaped}\\1\\s*:`, 'u').test(source);
}
