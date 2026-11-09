// Diagnostic-only seam. Each stage's sync must cover every GPU command that
// stage submitted, including any pending TVM readback and WASM destination.
const STAGES = ['inputs', 'graph', 'head', 'letter'] as const;
type StageName = (typeof STAGES)[number];
const NO_FAILURE = Symbol('no failure');

interface CheckpointGpu {
  pushErrorScope(filter: 'validation'): void;
  popErrorScope(): Promise<unknown>;
}

interface CheckpointStage {
  submit(): unknown | Promise<unknown>;
  sync(): unknown | Promise<unknown>;
}

interface CheckpointDependencies {
  readonly gpu: CheckpointGpu;
  readonly stages: Record<StageName, CheckpointStage>;
  readonly retain: () => void;
}

class StageFailure extends Error {
  declare readonly stage: StageName;

  constructor(stage: StageName, cause: unknown) {
    super('Nari diagnostic stage failed', {cause});
    this.stage = stage;
  }
}

async function runStage(
  gpu: CheckpointGpu,
  task: CheckpointStage,
  stage: StageName,
  retain: () => void
): Promise<unknown> {
  let primary: unknown = NO_FAILURE,
    scopeOpened = false,
    uncertain = false,
    popped: unknown,
    value: unknown;
  try {
    gpu.pushErrorScope('validation');
    scopeOpened = true;
    value = await task.submit();
    await task.sync();
  } catch (error) {
    primary = error;
    uncertain = true;
  }
  if (uncertain) retain();
  if (scopeOpened) {
    try {
      popped = await gpu.popErrorScope();
    } catch (error) {
      if (!uncertain) retain();
      if (primary === NO_FAILURE) primary = error;
    }
  }
  if (primary !== NO_FAILURE || popped) throw new StageFailure(stage, primary === NO_FAILURE ? popped : primary);
  return value;
}

export async function runCheckpointedStages({
  gpu,
  stages,
  retain
}: CheckpointDependencies): Promise<Record<StageName, unknown>> {
  if (
    !gpu ||
    typeof gpu.pushErrorScope !== 'function' ||
    typeof gpu.popErrorScope !== 'function' ||
    typeof retain !== 'function' ||
    !stages ||
    STAGES.some((name) => typeof stages[name]?.submit !== 'function' || typeof stages[name]?.sync !== 'function')
  )
    throw Error('invalid diagnostic dependencies');
  const outputs = {} as Record<StageName, unknown>;
  for (const stage of STAGES) {
    outputs[stage] = await runStage(gpu, stages[stage], stage, retain);
  }
  return outputs;
}
