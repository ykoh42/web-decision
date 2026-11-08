import {assetUrl, sha256, waitForAbort} from '../assets.js';
import {cancelResponse, readBoundedResponse} from '../bounded-response.js';
import {WebDecisionError} from '../kev.js';

export async function readClefRange(base, entry, part, signal) {
  if (
    !entry ||
    !part ||
    !Number.isSafeInteger(entry.bytes) ||
    !Number.isSafeInteger(part.offset) ||
    !Number.isSafeInteger(part.length) ||
    part.length < 1 ||
    part.length > 4 * 1024 * 1024 ||
    part.offset < 0 ||
    part.offset > entry.bytes - part.length ||
    !/^[a-f0-9]{64}$/.test(part.sha256)
  )
    throw new WebDecisionError('INVALID_ASSET', 'Invalid Clef range metadata.');
  let response;
  try {
    const url = assetUrl(base, entry.path);
    response = await waitForAbort(
      () =>
        fetch(url, {
          signal,
          redirect: 'error',
          headers: {
            Range: `bytes=${part.offset}-${part.offset + part.length - 1}`
          }
        }),
      signal,
      cancelResponse
    );
    const encoding = response.headers.get('content-encoding'),
      length = response.headers.get('content-length');
    if (
      response.status !== 206 ||
      response.redirected ||
      response.headers.get('content-range') !==
        `bytes ${part.offset}-${part.offset + part.length - 1}/${entry.bytes}` ||
      (length !== null && length !== String(part.length)) ||
      (encoding !== null && encoding.toLowerCase() !== 'identity')
    )
      throw new Error('Invalid Clef range response');
    const bytes = await readBoundedResponse(response, part.length, {exact: true, signal});
    if ((await waitForAbort(() => sha256(bytes), signal)) !== part.sha256) throw new Error('Clef range hash mismatch');
    signal?.throwIfAborted();
    return bytes;
  } catch (cause) {
    cancelResponse(response);
    signal?.throwIfAborted();
    if (cause instanceof WebDecisionError) throw cause;
    throw new WebDecisionError('ASSET_LOAD_FAILED', 'Could not read an authenticated Clef range.', {cause});
  }
}
