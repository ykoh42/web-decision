function scanString(text, start, containers) {
  let end = start;
  while (++end < text.length) {
    if (text[end] === '\\') end++;
    else if (text[end] === '"') break;
  }
  let next = end + 1;
  while (text[next] === ' ' || text[next] === '\t' || text[next] === '\n' || text[next] === '\r') next++;
  if (text[next] === ':') {
    const key = JSON.parse(text.slice(start, end + 1));
    const keys = containers[containers.length - 1];
    if (keys.has(key)) throw new SyntaxError('Duplicate JSON object key');
    keys.add(key);
  }
  return end;
}

function checkDuplicateKeys(text) {
  // JSON.parse validates the grammar first. Therefore any string token followed
  // by a colon is an object key, and every closing delimiter has a matching
  // opening delimiter. Scan iteratively to avoid recursion on deeply nested JSON.
  const containers = [];
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === '{') containers.push(new Set());
    else if (char === '[') containers.push(null);
    else if (char === '}' || char === ']') containers.pop();
    else if (char === '"') i = scanString(text, i, containers);
  }
}

/** Parse bounded UTF-8 JSON, rejecting duplicate object member names. */
export function parseStrictJson(bytes, {maxBytes = 1024 * 1024} = {}) {
  if (!(bytes instanceof Uint8Array)) throw new TypeError('JSON input must be a Uint8Array');
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 0)
    throw new RangeError('maxBytes must be a non-negative safe integer');
  if (bytes.byteLength > maxBytes) throw new RangeError('JSON input exceeds maxBytes');
  // Preserve a leading BOM so JSON.parse applies its own JSON grammar to it.
  const text = new TextDecoder('utf-8', {fatal: true, ignoreBOM: true}).decode(bytes);
  const value = JSON.parse(text);
  checkDuplicateKeys(text);
  return value;
}
