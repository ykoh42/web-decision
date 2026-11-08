interface PinnedTextModelContract<Manifest, Index, Tokenizer, Extras, UploadArgs, SessionArgs, Session> {
  name: string;
  offlineMissingMessage: string;
  readManifest(response: Response, signal?: AbortSignal): Promise<Manifest>;
  validateIndex(manifest: Manifest, index: unknown): Index;
  requestDevice(gpu: GPU, signal?: AbortSignal): Promise<GPUDevice>;
  prepare(data: unknown, tokenizer: Tokenizer, manifest: Manifest): Extras;
  upload(args: UploadArgs): Promise<unknown>;
  createSession(args: SessionArgs): Session;
}

export function loadPinnedTextModel<Manifest, Index, Tokenizer, Extras, UploadArgs, SessionArgs, Session>(
  options: unknown,
  model: PinnedTextModelContract<Manifest, Index, Tokenizer, Extras, UploadArgs, SessionArgs, Session>
): Promise<Session>;
