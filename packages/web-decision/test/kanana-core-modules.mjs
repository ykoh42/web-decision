import {join} from 'node:path';
import {pathToFileURL} from 'node:url';

const outputDirectory = process.env.KANANA_CORE_DIST;
if (!outputDirectory) throw new Error('Run Kanana core tests through the scoped TypeScript test script.');

async function load(name) {
  return import(pathToFileURL(join(outputDirectory, name)).href);
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
] = await Promise.all([
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
]);
