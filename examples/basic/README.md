# Browser example

Build the package and install a clean copy in this example from the repository root:

```sh
pnpm build:package
npm ci --prefix examples/basic --ignore-scripts
npm --prefix examples/basic run dev -- --port 45140 --strictPort
```

Open `http://127.0.0.1:45140/`, select an available model and load it. The package downloads and verifies its pinned model files; no asset URL is required. The first download can be large. Unloading disposes the session but does not clear the browser cache. Browser storage and GPU support vary by device.

This example keeps a separate npm lockfile to exercise a clean consumer of the workspace package. Reinstall the example after rebuilding the package.
