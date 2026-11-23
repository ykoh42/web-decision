import {readFileSync} from 'node:fs';
import {once} from 'node:events';
import {gemma4V2PortableExp} from '../../../packages/web-decision/src/strands/gemma4-v2-portable-exp.mjs';

const path = process.argv[2];
if (!path) throw new Error('Provide the scoring input path');
const raw = readFileSync(path);
const scores = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
const output = Buffer.allocUnsafe(262144);
let offset = 0;
for (let maximum = 0; maximum < scores.length; maximum++) {
  for (let index = 0; index <= maximum; index++) {
    output.writeFloatLE(gemma4V2PortableExp(Math.fround(scores[index]! - scores[maximum]!)), offset);
    offset += 4;
    if (offset === output.length) {
      if (!process.stdout.write(output)) await once(process.stdout, 'drain');
      offset = 0;
    }
  }
}
if (offset) process.stdout.write(output.subarray(0, offset));
