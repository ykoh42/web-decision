// Query-message slice of chat_template.jinja at
// 1489062def4145ae3a0a9261fbac4f4d88a8c6d1 (SHA-256 30d1c04dab7a30deba10156ef6ce6ca13dbdbc7f443c6a638d501d0b41c00702).
const PYTHON_WHITESPACE =
  '[\\u0009-\\u000d\\u001c-\\u0020\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]';
const stripStart = new RegExp(`^${PYTHON_WHITESPACE}+`, 'u');
const stripEnd = new RegExp(`${PYTHON_WHITESPACE}+$`, 'u');
export const KANANA_IMAGE_MARKER = '<|vision_start|><|image_pad|><|vision_end|>';
const GENERATION_PROMPT = '<|im_start|>assistant\n<think>\n\n</think>\n\n';

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function pythonStrip(value) {
  return value.replace(stripStart, '').replace(stripEnd, '');
}

function renderPart(part, role) {
  if (!isRecord(part)) throw new TypeError('Kanana chat content parts must be objects.');
  if (part.type === 'text' && typeof part.text === 'string') return part.text;
  if (part.type === 'image') {
    if (role === 'system') throw new TypeError('Kanana system messages cannot contain images.');
    if (!Object.hasOwn(part, 'image')) throw new TypeError('Kanana image parts must include an image source.');
    return KANANA_IMAGE_MARKER;
  }
  throw new TypeError('Kanana query messages support text and image content parts.');
}

function renderParts(parts, role) {
  if (!Array.isArray(parts)) throw new TypeError('Kanana query message content must be an array.');
  return pythonStrip(parts.map((part) => renderPart(part, role)).join(''));
}

/** Render one system and one user query with the pinned Kanana default generation prompt. */
export function renderKananaChatTemplate(messages) {
  if (
    !Array.isArray(messages) ||
    messages.length !== 2 ||
    !isRecord(messages[0]) ||
    !isRecord(messages[1]) ||
    messages[0].role !== 'system' ||
    messages[1].role !== 'user'
  )
    throw new TypeError('Kanana query messages must contain one system message followed by one user message.');

  const system = renderParts(messages[0].content, 'system');
  const user = renderParts(messages[1].content, 'user');
  return `<|im_start|>system\n${system}<|im_end|>\n<|im_start|>user\n${user}<|im_end|>\n${GENERATION_PROMPT}`;
}
