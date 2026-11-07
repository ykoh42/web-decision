import {defineConfig} from 'vite';
import {fileURLToPath} from 'node:url';
import {describeModel} from 'web-decision';
import {limitsPlugin} from './limits-plugin.mjs';

// Serve prepared model assets locally for this example. A deployed application
// can host these files separately and pass that directory's URL instead.
export default defineConfig({
  plugins: [limitsPlugin(describeModel)],
  publicDir: fileURLToPath(new URL('../../models/web-decision-assets/', import.meta.url)),
  // Deploy the prepared assets separately; do not copy 349 MB into each build.
  build: {copyPublicDir: false, manifest: true}
});
