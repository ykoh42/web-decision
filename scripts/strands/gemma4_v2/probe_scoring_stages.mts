import {readFile} from 'node:fs/promises';
const rows = JSON.parse(await readFile('../e2b-v2-scoring-stages.json', 'utf8'));
const f = Math.fround;
for (const row of rows) {
  const q = row.z.map((z: number) => f(z / f(30)));
  const t = q.map((x: number) => f(Math.tanh(x)));
  const c = t.map((x: number) => f(x * f(30)));
  const s = c.map((x: number) => f(x / f(1.5456)));
  const max = Math.max(...s);
  const shifted = s.map((x: number) => f(x - max));
  const exps = shifted.map((x: number) => f(Math.exp(x)));
  const total = exps.reduce((a: number, b: number) => f(a + b), 0);
  const diff = (a: number[], b: number[]) => ({
    count: a.filter((x, i) => x !== b[i]).length,
    max: Math.max(...a.map((x, i) => Math.abs(x - b[i]!)))
  });
  console.log(
    JSON.stringify({
      index: row.index,
      quotient: diff(q, row.quotient),
      tanh: diff(t, row.tangent),
      capped: diff(c, row.capped),
      scaled: diff(s, row.scaled),
      shifted: diff(shifted, row.shifted),
      exp: diff(exps, row.exp),
      sum: {js: total, numpy: row.sum, diff: total - row.sum}
    })
  );
}
