const checks = [
  ['private key', /-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----/],
  ['npm auth token', /(?:_authToken|npm_token)\s*[:=]\s*[^\s"']+/i],
  ['macOS home path', /\/Users\/[^/\s"']+\//],
  ['Linux home path', /\/home\/(?!web_user(?:\/|\b))[^/\s"']+\//]
];

/** Find a small set of common accidental disclosures without printing their contents. */
export function findPackedContentIssues(bytes) {
  const text = bytes.toString('utf8');
  return checks.filter(([, pattern]) => pattern.test(text)).map(([name]) => name);
}
