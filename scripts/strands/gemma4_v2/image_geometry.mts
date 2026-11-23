/** Gemma 4 image size and placeholder count, after EXIF orientation. */
export function gemma4ImageGeometry(width: number, height: number) {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1)
    throw new RangeError('Image dimensions must be positive integers');
  const side = 48;
  const maxTokens = 280;
  const factor = Math.sqrt((maxTokens * side * side) / (height * width));
  let outHeight = Math.floor((factor * height) / side) * side;
  let outWidth = Math.floor((factor * width) / side) * side;
  if (outHeight === 0) {
    outHeight = side;
    outWidth = Math.min(Math.floor(width / height) * side, maxTokens * side);
  } else if (outWidth === 0) {
    outWidth = side;
    outHeight = Math.min(Math.floor(height / width) * side, maxTokens * side);
  }
  return {width: outWidth, height: outHeight, tokens: (outWidth / side) * (outHeight / side)};
}
