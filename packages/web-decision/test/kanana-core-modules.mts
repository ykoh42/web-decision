import {join} from 'node:path';
import {pathToFileURL} from 'node:url';

const outputDirectory = process.env.KANANA_CORE_DIST;
if (!outputDirectory) throw new Error('Run Kanana core tests through the scoped TypeScript test script.');

async function load(name: string): Promise<unknown> {
  return import(pathToFileURL(join(outputDirectory!, name)).href);
}

export const [
  protocol,
  scoring,
  candidates,
  chatTemplate,
  truncation,
  tokenizer,
  browserImage,
  encodedImage,
  imageProcessing,
  magickImage,
  pnmImage,
  tiff16Image
] = (await Promise.all([
  load('protocol.js'),
  load('scoring.js'),
  load('candidates.js'),
  load('chat-template.js'),
  load('truncation.js'),
  load('tokenizer.js'),
  load('browser-image.js'),
  load('encoded-image.js'),
  load('image-processing.js'),
  load('magick-image.js'),
  load('pnm-image.js'),
  load('tiff16-image.js')
])) as [
  typeof import('../src/kanana/protocol.js'),
  typeof import('../src/kanana/scoring.js'),
  typeof import('../src/kanana/candidates.js'),
  typeof import('../src/kanana/chat-template.js'),
  typeof import('../src/kanana/truncation.js'),
  typeof import('../src/kanana/tokenizer.js'),
  typeof import('../src/kanana/browser-image.js'),
  typeof import('../src/kanana/encoded-image.js'),
  typeof import('../src/kanana/image-processing.js'),
  typeof import('../src/kanana/magick-image.js'),
  typeof import('../src/kanana/pnm-image.js'),
  typeof import('../src/kanana/tiff16-image.js')
];
