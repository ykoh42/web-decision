# Local browser example

After preparing the package and Kev model assets as described in the [library README](../../packages/web-decision/README.md), run from the repository root:

```sh
npm ci --prefix examples/basic --ignore-scripts
npm --prefix examples/basic run dev -- --port 45140 --strictPort
```

Open `http://127.0.0.1:45140/`, select a model, enter the URL of its matching
asset directory, and submit a question with one option per line. The example
serves `models/web-decision-assets/` as its public directory; assets elsewhere
need a separately configured server. The model selector does not download or
prepare assets. Its token limits come from the installed package's model
descriptors when the example server or build starts; restart the server after
reinstalling the package. Tokens include model protocol content as described
in the displayed limit, not just the visible text. **Unload model** waits for the model session to dispose and
lets you load it again after successful disposal. If disposal fails, the example
disables further actions and asks you to refresh the page. A failed H2O, Nari,
Imajev, Perplexity, Mapika 4B, Elephant, Clef or Strands load also requires a refresh because GPU
resources may remain. Other load failures advise refreshing for the same reason.
Unloading does not clear downloaded files from the browser cache or guarantee
immediate return of GPU memory to the operating system.

The example intentionally uses a separate npm lockfile to test an npm consumer,
even though development of the library uses the root pnpm workspace. It installs a copy of the local package rather than linking to its
source directory, so it works without the package's `node_modules`. Build the
package before installing the example; rerun `npm ci` in the example after
rebuilding the package to pick up changes. When changing the package version,
update the example lockfile before running `npm ci`.
The local clean-install and browser smoke-check details are recorded in
[`experiments/release-license24/CLEAN-EXAMPLE.md`](../../experiments/release-license24/CLEAN-EXAMPLE.md).

This example runs only on a compatible WebGPU browser and requires the separately prepared assets. For a production build, `npm --prefix examples/basic run build` deliberately leaves model files out of `dist`; serve the assets yourself. See the library README for device requirements and the limitations of model scores.

When serving a prepared asset set, make its `notices/` files accessible with
the model files and provide their location to recipients. The model loader
fetches the inference manifest and files, not those notices; verify the actual
deployment's notices and distribution conditions separately.

For the separately built **experimental K32** model, run `bash scripts/setup_kev_k32_candidate.sh` from the repository root after the standard preview setup, then open `http://127.0.0.1:45140/?model=k32`. See [the experiment](../../experiments/kev-perf/README.md) for prerequisites and alternative build paths. It has been tested on one Apple GPU in Chrome and Safari; other devices may be slower or fail. The ordinary URL keeps using the existing model. Both variants are private previews.

For the experimental English DeBERTa model, prepare assets in `models/web-decision-assets/open-jev-deberta-v3-large-q4f16` as described in [the loader experiment](../../experiments/deberta-loader/README.md), then open `http://127.0.0.1:45140/?model=deberta`. The model has different input limits and accuracy characteristics from Kev; see the library README. The preview remains private.

For experimental Elephant 9B Q4, open `http://127.0.0.1:45140/?model=elephant`
and enter the URL of a separately prepared matching HTTP Range asset server.
The compiled graph accepts at most 128 tokens per question branch, needs a
hardware `shader-f16` GPU and approximately 4.47 GB of model weights, and Q4
can change model choices. The package does not provide those assets. See the
[library limits and validation](../../packages/web-decision/README.md#experimental-elephant-9b-q4)
and [local asset experiment](../../experiments/elephant-installed-20261101/README.md).
For the distinct 512-token graph, use `?model=elephant-512` and an asset URL
serving its separately pinned WASM with the same weight and tokenizer files.
For an existing staged set, the [loopback server](../../experiments/elephant-installed-20261101/README.md#serve-a-staged-set-on-loopback)
accepts an explicit directory and matching 128 or 512 variant.

For experimental Clef 9B selected-row Q4, open
`http://127.0.0.1:45140/?model=clef` and enter the URL of separately prepared
matching assets. The package and example do not supply those files. The server
must support exact HTTP 206 ranges; for a different origin it must allow the
example origin and expose `Content-Range`. This route is online only, without
cache or progress callbacks. Each decision authenticates the table ranges
needed for its selected rows; transfer size varies with the request. Unrequested
ranges are not checked, and the server can observe a request-dependent range
pattern. The complete packed request is limited to 512 tokens, including context,
instructions and options; this example sends one question with 2–64 distinct,
nonempty text options. Images, video and native abstention are not supported.
A hardware WebGPU adapter with `shader-f16` and sufficient memory is required.
See the [library documentation](../../packages/web-decision/README.md#experimental-clef-9b-selected-row-q4)
for the full limits and [asset staging and loopback server](../../experiments/clef-portable-assets-20261101/#serve-a-staged-set-on-loopback)
for locally prepared files. Neither provides a hosted download. One installed
package request ran on one GPU; it does not establish general model quality or
other device compatibility, and a matching CPU score comparison is pending.

To try the experimental packed-result readback, add `&readback=packed` to the DeBERTa URL. It has only been measured on one GPU; see the library README for limitations.

The selector also includes Kev reflexes, Mapika 0.8B FP16/Q4 and 4B Q4, Kev 0.8B, H2O,
Nari, Imajev, Perplexity and Strands. The files in each prepared directory must match the selected model;
the example does not treat different precisions or checkpoints as interchangeable.
For models outside the public directory, provide the URL of a server for that
specific prepared directory. If it is a different origin, configure CORS and
the model's required range responses; see the library README. Optionally set
the initial choice and URL with `?model=mapika-q4&assetsUrl=...`, URL-encoding
the asset URL. An unknown model is rejected until a supported one is selected.

### Asset preparation entry points

The model selector does not obtain model files. The table points to the source
checkout's preparation or serving instructions for the ten decision-model
candidates. These are private experimental routes; a server for existing files
is not a fresh build or a promise of a hosted asset bundle. Each graph variant
needs its matching files and model ID. Follow the linked guide's resource,
verification and distribution requirements before preparing or sharing files.

| Model               | Entry point and current preparation scope                                                                                                                                                                |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Perplexity Flash 4B | [Local staging and server](../../experiments/pplx-package-assets-20261101/README.md) require already converted files; they do not reproduce conversion.                                                  |
| Imajev 4B           | [Local asset server](#existing-local-imajev-assets) requires an existing pinned layout; a complete portable preparation recipe remains open.                                                             |
| Strands 2B          | [Preparation guide](../../experiments/strands-asset-preparation-guide-20261030/README.md) inspects an existing set; no reproducible fresh build is available.                                            |
| H2O 4B              | [Portable server and staging](../../experiments/h2o-portable-asset-server-20261101/README.md) require existing converted assets and the matching graph variant.                                          |
| Mapika 4B           | [Local graph-specific server](#existing-local-mapika-4b-assets) requires an existing pinned conversion.                                                                                                  |
| Mapika 0.8B         | [FP16 preparation](../../experiments/mapika-package-20261030/README.md) and [Q4 preparation](../../experiments/mapika-0.8b-integration-20261030/Q4-README.md) depend on existing build inputs.           |
| Nari 4B             | [Full-Q4 server](../../experiments/nari-portable-assets-20261101/README.md) and [mixed-16 server](../../experiments/nari-mixed-q4-assets-20261101/README.md) require separately prepared matching files. |
| Elephant 9B         | [Staging guide](../../experiments/elephant-installed-20261101/README.md#draft-preparing-assets-for-another-host) copies an existing set; the 128- and 512-token graphs are distinct.                     |
| Clef 9B             | [Selected-row staging and loopback server](../../experiments/clef-portable-assets-20261101/README.md) require an existing local conversion.                                                              |
| Kev 0.8B            | [FP16 packager](../../experiments/kev08-package-20261030/README.md) verifies existing experimental artifacts; it is not a general converter.                                                             |

The default Kev 0.6B route has an [isolated fresh-toolchain build record](../../experiments/kev06-clean-pnpm-build-20261104/README.md). Newly fetched model and build inputs reproduced the package, WASM and asset hashes; all 904 package tests passed. The matching package and assets already have real-GPU evidence in Chromium, Firefox and WebKit on one M4 Pro.

### Existing local Nari full-Q4 assets

The following paths describe the previously prepared files in this checkout.
They must already exist; the server does not download or convert the model.
Replace `/absolute/path/to/web-decision` with the actual checkout path and
save this JSON in a private file, for example `/absolute/path/to/nari-paths.json`:

```json
{
  "metadata": "/absolute/path/to/web-decision/experiments/nari-portable-assets-20261101/metadata.json",
  "weights": "/absolute/path/to/web-decision/models/nari-integration-20261030/nari-stream-147a3aa5be4338a3/pending",
  "model": "/absolute/path/to/web-decision/experiments/nari-toq-dynamic-meta-20261101/patched-candidate/model.patched.wasm",
  "tokenizer": "/absolute/path/to/web-decision/models/nari-integration-20261030/source/tokenizer.json",
  "tokenizerConfig": "/absolute/path/to/web-decision/models/nari-integration-20261030/source/tokenizer_config.json",
  "config": "/absolute/path/to/web-decision/models/nari-integration-20261030/source/config.json"
}
```

With the example running at `http://127.0.0.1:45140`, start the existing
[full-Q4 server](../../experiments/nari-portable-assets-20261101/README.md):

```sh
node experiments/nari-portable-assets-20261101/serve.mjs /absolute/path/to/nari-paths.json 41783 http://127.0.0.1:45140
```

Open `http://127.0.0.1:45140/?model=nari` and enter
`http://127.0.0.1:41783/nari-assets/`. The server checks its pinned metadata;
the package checks each asset range as it loads. Neither the configuration nor
the server establishes that a model will fit or run on a particular device.

### Existing local Nari mixed assets

For the separately prepared mixed FP16/Q4 variant, start its loopback server
with the example's exact origin as described in the [mixed asset guide](../../experiments/nari-mixed-q4-assets-20261101/README.md).
Open `http://127.0.0.1:45140/?model=nari-mixed` and enter
`http://127.0.0.1:41784/nari-assets/`. This server uses an already prepared
matching asset set and does not download or convert it. The full-Q4 Nari
variant requires its own matching assets and server.

### Existing local Mapika 4B assets

The local router serves already prepared pinned assets; it does not download,
convert or distribute them. Start it for the same example origin, then select
the matching graph and enter its URL:

```sh
node experiments/mapika4b-numerical-repro-20261101/serve_package.mjs http://127.0.0.1:45140 41952 dlight-all
```

If the matching prepared files are elsewhere, preserve their original directory
layout and pass the absolute model root:

```sh
node experiments/mapika4b-numerical-repro-20261101/serve_package.mjs \
  http://127.0.0.1:45140 41952 dlight-all \
  --model-root /absolute/path/to/mapika-4b-integration-20261030
```

See the [Mapika 4B asset guide](../../experiments/mapika4b-numerical-repro-20261101/README.md#existing-local-assets)
for the required inventory, graph, tokenizer and weight paths, and for the
scope of the relocated-path checks. This option uses existing files; it does
not prepare or copy them.

Open `http://127.0.0.1:45140/?model=mapika4b-dlight-all` and enter
`http://127.0.0.1:41952/`. Use `baseline` / `?model=mapika4b` or `dlight` /
`?model=mapika4b-dlight` for the other graphs. Stop the router with Ctrl-C. A one-request real GPU UI smoke and
the limitations of the experiment are recorded [here](../../experiments/mapika4b-numerical-repro-20261101/README.md).

### Existing local H2O assets

H2O assets are **not distributed** by this project, and the repository's local
router requires the previously prepared pinned inventory and files. If those
files already exist, start this separate loopback router from the repository
root while the example server runs on port 45140. Check `node --version` and
install a Node version supported by the workspace if needed (`^20.19.0`,
`^22.13.0`, or `>=24.0.0`). These are the workspace requirements, not a claim
that this router has been tested on every supported Node version:

```sh
node experiments/h2o-public-asset-router-20261031/router.mjs --port 41782 --allow-origin http://127.0.0.1:45140
```

Open `http://127.0.0.1:45140/?model=h2o` and enter
`http://127.0.0.1:41782/h2o-assets/` in the asset URL field. The router is a
local diagnostic for those specific existing files, not a public distribution
or a fresh asset preparation recipe. H2O does not support cache, offline mode,
progress callbacks or nonzero prefetch; the example disables cache and does
not pass a progress callback. It uses H2O's own question/option semantics,
not the default Kev checkpoint's. Stop both servers when finished.

On 2026-10-31 this example loaded those local assets, ran one 94-token request
and unloaded the model in Chromium on an Apple M4 Pro. The selection matched
the Q4 CPU reference. This single-device smoke test does not establish general
model quality, GPU memory reclamation, or compatibility on other devices; see
the [test record](../../experiments/h2o-basic-ui-smoke-20261031/README.md).

The selector also offers `?model=h2o-dual` for the separate dual-projection
graph. Prepare its matching graph and metadata using the
[private preparation](../../experiments/h2o-q4-dual-private-preparation-20261101/README.md)
and serve them with the [portable loopback server](../../experiments/h2o-portable-asset-server-20261101/README.md).
An earlier `c98f3fbf…` installed bundle was checked on five fixed inputs on
one Apple GPU. The current `630125d4…` build, other devices and broader model
quality have not been established by that check.

The selector also offers `?model=h2o-dlight` for the separate experimental
optimized graph. It has the same online-only, no-progress and refresh-on-load
failure behavior as H2O. Supply an asset server containing the original public
metadata, inventory, manifest, tokenizer and weights plus
`model.dlight.pending.wasm` and `h2o-dlight-metadata.json`; the example and the
existing H2O router do not provide those DLight files. Neither H2O asset set is
distributed by this project. The DLight development measurements cover one
device and are internal runner timings, not public `decide()` timings.

For a local checkout with the separately prepared original weights and the
candidate files already present in
`experiments/h2o-dlight-private-preparation-20261031/outputs/local-20261031-v1/`,
the diagnostic router combines them at one loopback URL. Stop any other server
using port 41782, then run from the repository root:

```bash
node experiments/h2o-dlight-package-integration-20261031/router.mjs --port 41782 --allow-origin http://127.0.0.1:45140
```

Open `http://127.0.0.1:45140/?model=h2o-dlight` and enter
`http://127.0.0.1:41782/h2o-assets/`. The router and output directory are local
diagnostic aids; users must prepare the matching assets separately. They are not
an asset distribution or a general-purpose hosting service.

If you already have the matching files in another directory, the
[portable loopback server](../../experiments/h2o-portable-asset-server-20261101/README.md)
can serve that directory without the checkout-specific router paths. It does
not generate or distribute model files. A companion staging command can assemble
existing files with hard links. Its tests cover local HTTP responses, small-file
loading and one weight range; this server has not yet been tested with full GPU inference.

From this checkout, you can check a running asset server before loading H2O:

```bash
node scripts/check-h2o-assets.mjs --url http://127.0.0.1:41782/h2o-assets/ --model experimental-h2o-lightning-4b-q4f16-dlight
```

Use `experimental-h2o-lightning-4b-q4f16` to check the original graph. The
default checks the pinned metadata, graph, tokenizer and one largest weight
range. Add `--all-weights` to read and verify every pinned weight range; this
downloads about 2.37 GB from the server and can take time. The command runs in
Node from this repository, not from the installed npm package. It checks HTTP
range responses and hashes, but Node does not enforce browser CORS and the
check does not run the GPU or verify model quality. Asset redistribution and
license obligations remain separate.

### Existing local Perplexity assets

The Perplexity route requires separately prepared Q4 weights, graph and
tokenizer. With the files already staged as described in the
[asset guide](../../experiments/pplx-package-assets-20261101/README.md), start
the diagnostic server from the repository root while the example runs on
port 45140:

```sh
node experiments/pplx-package-assets-20261101/server.mjs \
  "$PWD/models/pplx-flash-v1.2-source-9ff7cf98/pplx-q4-portable-assets-20261101-v1" \
  41943 http://127.0.0.1:45140
```

Open `http://127.0.0.1:45140/?model=pplx` and enter
`http://127.0.0.1:41943/` as the asset URL. This route is online-only and
does not report loading progress. Its Q4 choices can differ from the source
model, as described in the [library README](../../packages/web-decision/README.md#experimental-perplexity-decider-flash-v12-q4).
The server and assets are local diagnostics, not an asset distribution.
The example was checked with one 85-token input on Apple Metal 3: the selected
option matched the Q4 CPU reference, and load and unload completed. See the
[local test record](../../models/pplx-flash-v1.2-source-9ff7cf98/pplx-installed-consumer-20261101-v2/basic-ui-90a7ddb30f17.json).
This does not establish general quality or compatibility with other devices.
The same example and installed package were also checked on 30 fixed CLINC
banking and out-of-scope requests; the selections matched the Q4 CPU reference
in all 30. The sample is small, training overlap is unknown, and the offered
`none` option is not native abstention. See the
[independent aggregation](../../models/pplx-flash-v1.2-source-9ff7cf98/pplx-installed-consumer-20261101-v2/basic-ui-clinc-0ba90c4f2e38-audit.json).

### Existing local Imajev assets

The experimental Imajev route requires the previously prepared local Q4
weights, graph, tokenizer, readout, calibration and pinned metadata under
`models/imajev-integration-20261030/`. Those assets are not distributed and
the commands below do not create them. With that specific local preparation,
start this loopback-only diagnostic router while the example runs on port 45140:

```sh
node experiments/imajev-range-hashes-20261101/serve-loader-assets.mjs \
  --port 41783 --allow-origin http://127.0.0.1:45140
```

For an existing matching copy elsewhere, preserve the original layout and add
an absolute model root:

```sh
node experiments/imajev-range-hashes-20261101/serve-loader-assets.mjs \
  --port 41783 --allow-origin http://127.0.0.1:45140 \
  --model-root /absolute/path/to/imajev-integration-20261030
```

The root must contain `imajev-dlight-metadata-20261101.json`,
`imajev-dlight-q4-link-20261101/model.pending.wasm`, the two tokenizer files
in `base/`, readout and calibration files in `adapter/`, and the matching
weight files in `q4-candidate-full-20261030-v1/`. The default root is
`models/imajev-integration-20261030/`. This option neither creates nor copies
assets. A relocated-path check also verified the actual metadata and calibration
files; it did not validate a complete relocated set or GPU execution. See the
[server record](../../experiments/imajev-range-hashes-20261101/README.md).

Open `http://127.0.0.1:45140/?model=imajev` and enter
`http://127.0.0.1:41783/assets/`. The router permits only the pinned file
ranges and the specified browser origin; the loader verifies hashes as it reads. It
does not generate, authenticate, or grant redistribution rights to assets,
and it is not a general-purpose public server. It listens only on loopback but
also accepts requests without an `Origin` header, so it is not a security
boundary against other local processes. Use trusted files without concurrent
modification and stop both servers afterwards.
The example treats Imajev abstention as no selection. See the [library
documentation](../../packages/web-decision/README.md#experimental-imajev-4b-q4)
for the model's limits and validation scope. The cross-origin example was
tested with three fixed inputs on one Apple M4 Pro in Chromium; see the
[experiment record](../../experiments/imajev-range-hashes-20261101/README.md).

### Existing local Mapika 0.8B Q4 assets

The package does not include these model assets, and a reproducible build on
another machine has not been established. If you have the matching prepared
asset directory, place a copy under
`models/web-decision-assets/mapika-0.8b-q4f16/` in this checkout. With the
example running on port 45140, select **Mapika 0.8B Q4** and enter
`/mapika-0.8b-q4f16/` as the asset URL. The example's public directory is
`models/web-decision-assets/`; a path elsewhere under `models/` is not served
by this URL. A separate HTTP(S) server can be used with its own directory URL
and appropriate CORS headers.

From the repository root, the copied set can be checked independently of the
original FP16 conversion files:

```sh
python3 scripts/verify-mapika08-q4-assets.py --assets models/web-decision-assets/mapika-0.8b-q4f16
```

An absolute asset directory elsewhere can also be supplied to `--assets`.
Optional `--report` output must be a new file under this checkout's `models/`.
This reads and hashes the complete prepared set (about 424 MB of weights).
Matching the local pins checks bytes, not publisher identity, conversion
correctness, redistribution rights or model quality. The
[Q4 preparation notes](../../experiments/mapika-0.8b-integration-20261030/Q4-README.md)
describe the existing source-build limitations; no asset download is supplied.

### Existing local Strands assets

Strands assets are also **not distributed**, and a complete reproducible fresh
build command is not yet available. If a matching set is already available,
see the [asset preparation guide](../../experiments/strands-asset-preparation-guide-20261030/README.md)
for the read-only metadata inspector and remaining limitations. The separate
[local diagnostic server](../../experiments/strands-local-range-server-20261102/README.md)
serves matching files to this example using exact authenticated ranges. Start
it as documented, open `http://127.0.0.1:45140/?model=strands`, and enter
`http://127.0.0.1:41790/assets/`. This example disables cache for Strands;
the package separately supports caching when its requirements are met.
Strands needs a large GPU storage binding and substantial additional memory;
its cancellation, device-loss and retained-resource limitations are described
in the library README. No H2O or Strands assets are included in the example.
