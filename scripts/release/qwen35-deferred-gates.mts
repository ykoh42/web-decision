export interface Qwen35GateFailure {
  readonly stage: string;
  readonly label: string;
  readonly message: string;
}

export async function recordQwen35GateFailure(
  failures: Qwen35GateFailure[],
  stage: string,
  label: string,
  check: () => void | Promise<void>
): Promise<void> {
  try {
    await check();
  } catch (error) {
    failures.push({
      stage,
      label,
      message: error instanceof Error ? error.message : String(error)
    });
  }
}

export async function writeQwen35ReportThenThrow(
  failures: readonly Qwen35GateFailure[],
  writeReport: (failures: readonly Qwen35GateFailure[]) => void | Promise<void>
): Promise<void> {
  await writeReport(failures);
  if (failures.length === 0) return;

  const summary = failures.map(({stage, label, message}) => `${stage}/${label}: ${message}`).join('\n');
  throw new Error(`Qwen browser evidence gate failed after all checks completed:\n${summary}`);
}
