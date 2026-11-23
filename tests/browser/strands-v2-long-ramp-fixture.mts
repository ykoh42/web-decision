export function buildLongRampInput(fixture: {state: number[]; questions: number[][]}, target: number) {
  if (!Number.isSafeInteger(target)) throw new Error('The requested token count must be a safe integer.');
  const question = fixture.questions[0];
  if (!Array.isArray(question) || fixture.state.length < 15) throw new Error('Invalid source token fixture.');
  const prefix = fixture.state.slice(0, 9);
  const suffix = fixture.state.slice(-5);
  const repeat = fixture.state[9]!;
  if (fixture.state.slice(9, -5).some((id) => id !== repeat)) throw new Error('Unexpected source fixture layout.');
  const repeats = target - question.length - prefix.length - suffix.length;
  if (repeats < 0) throw new Error('The requested length is too short for this exact fixture format.');
  return {state: [...prefix, ...Array.from({length: repeats}, () => repeat), ...suffix], question: [...question]};
}
