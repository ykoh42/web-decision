type ContentIssue = 'private key' | 'npm auth token' | 'macOS home path' | 'Linux home path';
type ContentCheck = readonly [issue: ContentIssue, pattern: RegExp];

const checks = [
  ['private key', /-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----/],
  ['npm auth token', /(?:_authToken|npm_token)\s*[:=]\s*[^\s"']+/i],
  ['macOS home path', /\/Users\/[^/\s"']+\//],
  ['Linux home path', /\/home\/(?!web_user(?:\/|\b))[^/\s"']+\//]
] as const satisfies readonly ContentCheck[];

/** Find common accidental disclosures without printing their contents. */
export function findPackedContentIssues(bytes: Buffer): ContentIssue[] {
  const text = bytes.toString('utf8');
  return checks.filter(([, pattern]) => pattern.test(text)).map(([issue]) => issue);
}
