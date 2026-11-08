import {defineConfig} from 'vite';
import {fileURLToPath} from 'node:url';

const workspaceRoot = fileURLToPath(new URL('../../', import.meta.url));
const browserRoot = fileURLToPath(new URL('./', import.meta.url));
const packageBundle = fileURLToPath(new URL('../../packages/web-decision/dist/index.js', import.meta.url));

export default defineConfig({
  root: browserRoot,
  resolve: {alias: {'web-decision': packageBundle}},
  server: {
    host: '127.0.0.1',
    port: Number(process.env.WEB_DECISION_UPSTREAM_TEST_PORT || 41784),
    strictPort: true,
    fs: {allow: [workspaceRoot]}
  }
});
