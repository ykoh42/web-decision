import { Tokenizer } from "@huggingface/tokenizers";
import * as defaultOrt from "onnxruntime-web/wasm";
// \0rolldown/runtime.js
var __commonJSMin = (cb, mod) => () => (mod || (cb((mod = { exports: {} }).exports, mod, mod.exports), cb = null), mod.exports);
// source/overlay/packages/web-decision/src/kev.js
/** Kev input packing, kept independent from the GPU runtime. */
var KEV_DELIMITERS = Object.freeze({
	state: 151659,
	question: 151660,
	option_start: 151648,
	option_end: 151649,
	decide: 151661
});
var WebDecisionError = class extends Error {
	constructor(code, message, options) {
		super(message, options);
		this.name = "WebDecisionError";
		this.code = code;
	}
};
function invalid$4(message) {
	return new WebDecisionError("INVALID_INPUT", message);
}
function validateQuestionInstructions(question, index, allowEmpty) {
	if (!question || typeof question !== "object" || typeof question.instructions !== "string" || !allowEmpty && !question.instructions.trim()) throw invalid$4(`Question ${index} needs ${allowEmpty ? "text" : "non-empty instructions"}.`);
}
function validateKevRequest(request) {
	if (!request || typeof request !== "object" || Array.isArray(request)) throw invalid$4("A request object is required.");
	if (typeof request.state !== "string") throw invalid$4("state must be a string.");
	if (!Array.isArray(request.questions) || request.questions.length === 0) throw invalid$4("Provide at least one question.");
	return request;
}
function appendKevTokens(ids, tokens, maxInputTokens) {
	if (maxInputTokens !== void 0 && ids.length + tokens.length > maxInputTokens) throw invalid$4(`Packed input exceeds ${maxInputTokens} tokens; no truncation was applied.`);
	for (const token of tokens) ids.push(token);
}
function validateKevOptions(options, index, allowEmpty, allowDuplicates, maxOptions) {
	if (!Array.isArray(options) || options.length < 1 || options.length > maxOptions) throw invalid$4(`Question ${index} needs 1 to ${maxOptions} options.`);
	const seen = /* @__PURE__ */ new Set();
	for (const option of options) {
		if (typeof option !== "string" || !allowEmpty && !option.trim()) throw invalid$4(`Question ${index} has an empty or non-string option.`);
		if (!allowDuplicates && seen.has(option)) throw invalid$4(`Question ${index} has duplicate options.`);
		seen.add(option);
	}
	return [...options];
}
function packKevQuestion(tokenizer, question, index, ids, append, options, maxOptions, stateTokenCount, maxBranchTokens) {
	const branchStart = ids.length;
	const appendBranch = (tokens) => {
		if (stateTokenCount + (ids.length - branchStart + tokens.length) > maxBranchTokens) throw invalid$4(`Question ${index} state and branch exceed ${maxBranchTokens} tokens; no truncation was applied.`);
		append(tokens);
	};
	validateQuestionInstructions(question, index, options.allowEmptyInstructions);
	const labels = validateKevOptions(question.options, index, options.allowEmptyOptions, options.allowDuplicates, maxOptions);
	appendBranch([KEV_DELIMITERS.question]);
	appendBranch(tokenizeKevText(tokenizer, question.instructions));
	const indices = [];
	const tokenizedOptions = /* @__PURE__ */ new Set();
	for (const label of labels) {
		const optionTokens = tokenizeKevText(tokenizer, label);
		const key = optionTokens.join(",");
		if (!options.allowDuplicates && tokenizedOptions.has(key)) throw invalid$4(`Question ${index} has options that encode to the same tokens.`);
		tokenizedOptions.add(key);
		appendBranch([KEV_DELIMITERS.option_start]);
		appendBranch(optionTokens);
		appendBranch([KEV_DELIMITERS.option_end]);
		indices.push(ids.length - 1);
	}
	appendBranch([KEV_DELIMITERS.decide]);
	return {
		indices,
		options: labels
	};
}
function hasInvalidKevToken(id) {
	return !Number.isSafeInteger(id) || id < 0 || id >= 2 ** 31;
}
function tokenizeKevText(tokenizer, text) {
	for (let i = 0; i < text.length; i++) {
		const code = text.charCodeAt(i);
		if (code >= 55296 && code <= 56319) {
			const next = text.charCodeAt(++i);
			if (!(next >= 56320 && next <= 57343)) throw invalid$4("Text contains an unpaired UTF-16 surrogate.");
		} else if (code >= 56320 && code <= 57343) throw invalid$4("Text contains an unpaired UTF-16 surrogate.");
	}
	return tokenizer.encode(text.replace(/<\|([A-Za-z0-9_]+)\|>/g, "<¦$1¦>"), { add_special_tokens: false }).ids;
}
function packKevRequest(tokenizer, request, limits, options = {}) {
	request = validateKevRequest(request);
	const state = tokenizeKevText(tokenizer, request.state);
	const maxStateTokens = limits.maxStateTokens ?? 8192;
	if (state.length + 1 > maxStateTokens) throw invalid$4(`The state exceeds ${maxStateTokens - 1} tokens.`);
	const ids = [KEV_DELIMITERS.state, ...state];
	const stateTokenCount = ids.length;
	const groups = [];
	const { allowEmptyInstructions = false, allowEmptyOptions = false, allowDuplicateOptionsFor = /* @__PURE__ */ new Set() } = options;
	const maxInputTokens = limits.maxInputTokens;
	const maxBranchTokens = limits.maxBranchTokens ?? 8192;
	const append = (tokens) => appendKevTokens(ids, tokens, maxInputTokens);
	const maxOptions = limits.maxOptions ?? 255;
	for (let qi = 0; qi < request.questions.length; qi++) groups.push(packKevQuestion(tokenizer, request.questions[qi], qi, ids, append, {
		allowEmptyInstructions,
		allowEmptyOptions,
		allowDuplicates: allowDuplicateOptionsFor.has(qi)
	}, maxOptions, stateTokenCount, maxBranchTokens));
	if (ids.some(hasInvalidKevToken)) throw invalid$4("Tokenizer produced an unsupported token ID.");
	return {
		ids,
		groups
	};
}
function unpackKevScores(logits, packed) {
	if (!Array.isArray(logits) || logits.length !== packed.ids.length || logits.some((x) => !Number.isFinite(x))) throw new WebDecisionError("INFERENCE_ERROR", "The model returned invalid scores.");
	return {
		tokenCount: packed.ids.length,
		answers: packed.groups.map((group) => {
			const scores = group.indices.map((i) => logits[i]), max = Math.max(...scores);
			const exps = scores.map((score) => Math.exp(score - max));
			const sum = exps.reduce((a, b) => a + b, 0);
			const probabilities = exps.map((value) => value / sum);
			const selectedIndex = probabilities.indexOf(Math.max(...probabilities));
			return {
				selected: group.options[selectedIndex],
				selectedIndex,
				probabilities: group.options.map((option, i) => ({
					option,
					probability: probabilities[i],
					logit: scores[i]
				}))
			};
		})
	};
}
function decodeFloat16(bits) {
	const sign = bits & 32768 ? -1 : 1, exponent = bits >> 10 & 31, mantissa = bits & 1023;
	return exponent === 0 ? sign * 2 ** -14 * mantissa / 1024 : exponent === 31 ? mantissa ? NaN : sign * Infinity : sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}
// source/overlay/packages/web-decision/src/abort.js
/** Stop waiting on a non-abortable operation without abandoning its rejection. */
function waitForAbort(operation, signal, onLateValue) {
	if (!signal) return Promise.resolve().then(operation);
	signal.throwIfAborted();
	return new Promise((resolve, reject) => {
		let settled = false;
		const finish = (callback, value) => {
			if (settled) {
				if (callback === resolve) try {
					onLateValue?.(value);
				} catch {}
				return;
			}
			settled = true;
			signal.removeEventListener("abort", onAbort);
			callback(value);
		};
		const onAbort = () => finish(reject, signal.reason);
		signal.addEventListener("abort", onAbort, { once: true });
		if (signal.aborted) {
			onAbort();
			return;
		}
		try {
			Promise.resolve(operation()).then((value) => finish(resolve, value), (cause) => finish(reject, cause));
		} catch (cause) {
			finish(reject, cause);
		}
	});
}
// source/overlay/packages/web-decision/src/bounded-response.js
var MAX_MANIFEST_BYTES = 1048576;
var MAX_ASSET_BYTES = 134217728;
var ResponseSizeError = class extends Error {};
/** Request cancellation without waiting for a possibly noncooperative body. */
function cancelBody(body) {
	try {
		Promise.resolve(body?.cancel()).catch(() => {});
	} catch {}
}
/** Dispose responses returned by a fetch or cache wrapper after cancellation. */
function cancelResponse(response) {
	cancelBody(response?.body);
}
function appendChunk(buffer, length, chunk) {
	if (!ArrayBuffer.isView(chunk) || chunk.BYTES_PER_ELEMENT !== 1) throw new TypeError("Response stream did not return bytes.");
	if (chunk.byteLength > buffer.length - length) throw new ResponseSizeError("Response body is too large.");
	buffer.set(chunk, length);
	return length + chunk.byteLength;
}
/** Read decoded response bytes with an effective size limit and abort support. */
async function readBoundedResponse(response, maxBytes, { exact = false, signal } = {}) {
	if (!Number.isSafeInteger(maxBytes) || maxBytes < 0) throw new RangeError("Invalid response size limit.");
	if (signal?.aborted) {
		cancelResponse(response);
		signal.throwIfAborted();
	}
	const body = response?.body;
	if (!body) {
		if (exact && maxBytes !== 0) throw new ResponseSizeError("Response body is too short.");
		return /* @__PURE__ */ new ArrayBuffer(0);
	}
	const reader = body.getReader();
	let length = 0, done = false;
	try {
		const buffer = new Uint8Array(maxBytes);
		for (;;) {
			const item = await waitForAbort(() => reader.read(), signal);
			if (item.done) {
				done = true;
				break;
			}
			length = appendChunk(buffer, length, item.value);
		}
		signal?.throwIfAborted();
		if (exact && length !== maxBytes) throw new ResponseSizeError("Response body is too short.");
		return exact || length === maxBytes ? buffer.buffer : buffer.buffer.slice(0, length);
	} finally {
		if (!done) cancelBody(reader);
		try {
			reader.releaseLock();
		} catch {}
	}
}
// source/overlay/packages/web-decision/src/assets.js
var CACHE_NAME = "web-decision-assets-v1";
var error = (message) => new WebDecisionError("INVALID_ASSET", message);
function assetUrl(base, path) {
	if (typeof path !== "string" || !path || path.startsWith("/") || /[\\?#]/.test(path)) throw error("Invalid model asset path.");
	const directory = new URL(base.href);
	if (!directory.search) directory.search = "";
	if (!directory.hash) directory.hash = "";
	const url = new URL(path, directory);
	if (url.origin !== directory.origin || !url.href.startsWith(directory.href) || url.username || url.password) throw error("Model asset path leaves the configured directory.");
	return url.href;
}
function validateManifest(manifest) {
	if (!manifest || manifest.formatVersion !== 1 || manifest.model !== "kev-0.6b-q4f16" || manifest.backend !== "tvm-webgpu" || Object.hasOwn(manifest, "maxInputTokens") || manifest.maxStateTokens !== 8192 || manifest.maxBranchTokens !== 8192 || manifest.parameterCount !== 394 || manifest.checkpoint?.revision !== "5a827091b3d58fe4fdfc02a5a5b2ce364047d678") throw error("The asset manifest does not match the supported Kev Q4 model.");
	if (!manifest.files || typeof manifest.files !== "object") throw error("The asset manifest has no files.");
	for (const name of [
		"model.wasm",
		"tokenizer.json",
		"tokenizer_config.json",
		"model_config.json",
		"weights/tensor-cache.json"
	]) if (!manifest.files[name]) throw error(`Missing asset: ${name}`);
	for (const [name, entry] of Object.entries(manifest.files)) if (!entry || !/^[0-9a-f]{64}$/.test(entry.sha256) || !Number.isSafeInteger(entry.bytes) || entry.bytes < 0) throw error(`Invalid asset metadata: ${name}`);
	return manifest;
}
async function sha256(bytes) {
	if (!globalThis.crypto?.subtle) throw new WebDecisionError("UNSUPPORTED_ENVIRONMENT", "Web Crypto is required to verify model files. Use HTTPS or localhost.");
	return Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))).map((value) => value.toString(16).padStart(2, "0")).join("");
}
async function openAssetCache(enabled, signal) {
	signal?.throwIfAborted();
	if (!enabled || !globalThis.caches) return null;
	try {
		return await waitForAbort(() => caches.open(CACHE_NAME), signal);
	} catch {
		signal?.throwIfAborted();
		return null;
	}
}
/** Serialize writes to one manifest while allowing a cancelled caller to stop waiting. */
async function commitCachedManifest(cache, url, manifest, signal, locks = globalThis.navigator?.locks) {
	signal?.throwIfAborted();
	const write = () => cache.put(url, new Response(JSON.stringify(manifest), { headers: { "content-type": "application/json" } }));
	if (!locks || typeof locks.request !== "function") {
		await write();
		signal?.throwIfAborted();
		return;
	}
	const options = signal ? {
		mode: "exclusive",
		signal
	} : { mode: "exclusive" };
	return waitForAbort(() => locks.request(`web-decision-manifest-v1:${url}`, options, async () => {
		signal?.throwIfAborted();
		await write();
	}), signal);
}
async function writeAssetCache(cache, key, buffer, signal) {
	if (!cache) return false;
	try {
		await waitForAbort(() => cache.put(key, new Response(buffer)), signal);
		return true;
	} catch {
		signal?.throwIfAborted();
		return false;
	}
}
async function checkedCacheEntry(cache, key, metadata, signal) {
	try {
		const cached = await waitForAbort(() => cache.match(key), signal, cancelResponse);
		if (!cached) return null;
		const buffer = await readBoundedResponse(cached, metadata.bytes, {
			exact: true,
			signal
		});
		return await waitForAbort(() => sha256(buffer), signal) === metadata.sha256 ? buffer : null;
	} catch {
		signal?.throwIfAborted();
		return null;
	}
}
async function findCachedAsset(cache, url, sharedKey, metadata, signal) {
	if (!cache) return null;
	const previousKey = new URL(url);
	previousKey.searchParams.set("__web_decision_sha256", metadata.sha256);
	for (const [key, shared] of [
		[sharedKey, true],
		[previousKey.href, false],
		[url, false]
	]) {
		const buffer = await checkedCacheEntry(cache, key, metadata, signal);
		if (buffer) return {
			buffer,
			cachedByHash: shared || await writeAssetCache(cache, sharedKey, buffer, signal)
		};
	}
	return null;
}
async function fetchVerifiedAsset(url, name, metadata, signal) {
	let response;
	try {
		response = await waitForAbort(() => fetch(url, { signal }), signal, cancelResponse);
	} catch (cause) {
		signal?.throwIfAborted();
		throw new WebDecisionError("ASSET_LOAD_FAILED", `Could not fetch ${name}.`, { cause });
	}
	if (!response.ok) {
		cancelBody(response.body);
		signal?.throwIfAborted();
		throw new WebDecisionError("ASSET_LOAD_FAILED", `Could not fetch ${name}: HTTP ${response.status}.`);
	}
	let buffer;
	try {
		buffer = await readBoundedResponse(response, metadata.bytes, {
			exact: true,
			signal
		});
	} catch (cause) {
		signal?.throwIfAborted();
		if (cause instanceof ResponseSizeError) throw error(`Integrity check failed for ${name}.`);
		throw new WebDecisionError("ASSET_LOAD_FAILED", `Could not read ${name}.`, { cause });
	}
	let digest;
	try {
		digest = await waitForAbort(() => sha256(buffer), signal);
	} catch (cause) {
		signal?.throwIfAborted();
		throw cause;
	}
	signal?.throwIfAborted();
	if (digest !== metadata.sha256) throw error(`Integrity check failed for ${name}.`);
	return buffer;
}
function reportAssetRead(result, { signal, onBytes, onCacheStatus }) {
	try {
		onCacheStatus?.(result.cachedByHash);
		signal?.throwIfAborted();
		onBytes?.(result.buffer.byteLength);
		signal?.throwIfAborted();
	} catch (cause) {
		signal?.throwIfAborted();
		throw cause;
	}
}
/** Verify assets and share identical content between model directories on one asset origin. */
async function readVerifiedAsset(base, manifest, name, { cache, signal, onBytes, onCacheStatus, offline = false, maxAssetBytes = MAX_ASSET_BYTES } = {}) {
	const metadata = manifest.files[name];
	if (!metadata) throw error(`Asset not listed in manifest: ${name}`);
	if (!Number.isSafeInteger(maxAssetBytes) || maxAssetBytes < 0 || maxAssetBytes > 1073741824 || !Number.isSafeInteger(metadata.bytes) || metadata.bytes < 0 || metadata.bytes > maxAssetBytes) throw error(`Invalid model asset size: ${name}.`);
	const url = assetUrl(base, name);
	const cacheKey = new URL("/.web-decision-cache/sha256/", url);
	cacheKey.searchParams.set("digest", metadata.sha256);
	signal?.throwIfAborted();
	let result = await findCachedAsset(cache, url, cacheKey.href, metadata, signal);
	signal?.throwIfAborted();
	if (!result) {
		if (offline) throw new WebDecisionError("ASSET_LOAD_FAILED", `Verified asset is not available offline: ${name}.`);
		const buffer = await fetchVerifiedAsset(url, name, metadata, signal);
		result = {
			buffer,
			cachedByHash: await writeAssetCache(cache, cacheKey.href, buffer, signal)
		};
	}
	signal?.throwIfAborted();
	reportAssetRead(result, {
		signal,
		onBytes,
		onCacheStatus
	});
	return result.buffer;
}
function parseJson(bytes, name) {
	try {
		return JSON.parse(new TextDecoder().decode(bytes));
	} catch (cause) {
		throw new WebDecisionError("INVALID_ASSET", `Invalid JSON asset: ${name}.`, { cause });
	}
}
// [TVM]/ctypes.ts
/**
* Size of common data types.
*/
var SizeOf = /* @__PURE__ */ function(SizeOf) {
	SizeOf[SizeOf["U8"] = 1] = "U8";
	SizeOf[SizeOf["U16"] = 2] = "U16";
	SizeOf[SizeOf["I32"] = 4] = "I32";
	SizeOf[SizeOf["I64"] = 8] = "I64";
	SizeOf[SizeOf["F32"] = 4] = "F32";
	SizeOf[SizeOf["F64"] = 8] = "F64";
	SizeOf[SizeOf["TVMValue"] = 8] = "TVMValue";
	SizeOf[SizeOf["TVMFFIAny"] = 16] = "TVMFFIAny";
	SizeOf[SizeOf["DLDataType"] = 4] = "DLDataType";
	SizeOf[SizeOf["DLDevice"] = 8] = "DLDevice";
	SizeOf[SizeOf["ObjectHeader"] = 24] = "ObjectHeader";
	return SizeOf;
}({});
/**
* Type Index in new TVM FFI.
*
* We are keeping the same style as C API here.
*/
var TypeIndex = /* @__PURE__ */ function(TypeIndex) {
	TypeIndex[TypeIndex["kTVMFFIAny"] = -1] = "kTVMFFIAny";
	/*! \brief None/nullptr value */
	TypeIndex[TypeIndex["kTVMFFINone"] = 0] = "kTVMFFINone";
	/*! \brief POD int value */
	TypeIndex[TypeIndex["kTVMFFIInt"] = 1] = "kTVMFFIInt";
	/*! \brief POD bool value */
	TypeIndex[TypeIndex["kTVMFFIBool"] = 2] = "kTVMFFIBool";
	/*! \brief POD float value */
	TypeIndex[TypeIndex["kTVMFFIFloat"] = 3] = "kTVMFFIFloat";
	/*! \brief Opaque pointer object */
	TypeIndex[TypeIndex["kTVMFFIOpaquePtr"] = 4] = "kTVMFFIOpaquePtr";
	/*! \brief DLDataType */
	TypeIndex[TypeIndex["kTVMFFIDataType"] = 5] = "kTVMFFIDataType";
	/*! \brief DLDevice */
	TypeIndex[TypeIndex["kTVMFFIDevice"] = 6] = "kTVMFFIDevice";
	/*! \brief DLTensor* */
	TypeIndex[TypeIndex["kTVMFFIDLTensorPtr"] = 7] = "kTVMFFIDLTensorPtr";
	/*! \brief const char* */
	TypeIndex[TypeIndex["kTVMFFIRawStr"] = 8] = "kTVMFFIRawStr";
	/*! \brief TVMFFIByteArray* */
	TypeIndex[TypeIndex["kTVMFFIByteArrayPtr"] = 9] = "kTVMFFIByteArrayPtr";
	/*! \brief R-value reference to ObjectRef */
	TypeIndex[TypeIndex["kTVMFFIObjectRValueRef"] = 10] = "kTVMFFIObjectRValueRef";
	/*! \brief Small string on stack */
	TypeIndex[TypeIndex["kTVMFFISmallStr"] = 11] = "kTVMFFISmallStr";
	/*! \brief Small bytes on stack */
	TypeIndex[TypeIndex["kTVMFFISmallBytes"] = 12] = "kTVMFFISmallBytes";
	/*! \brief Start of statically defined objects. */
	TypeIndex[TypeIndex["kTVMFFIStaticObjectBegin"] = 64] = "kTVMFFIStaticObjectBegin";
	/*!
	* \brief Object, all objects starts with TVMFFIObject as its header.
	* \note We will also add other fields
	*/
	TypeIndex[TypeIndex["kTVMFFIObject"] = 64] = "kTVMFFIObject";
	/*!
	* \brief String object, layout = { TVMFFIObject, TVMFFIByteArray, ... }
	*/
	TypeIndex[TypeIndex["kTVMFFIStr"] = 65] = "kTVMFFIStr";
	/*!
	* \brief Bytes object, layout = { TVMFFIObject, TVMFFIByteArray, ... }
	*/
	TypeIndex[TypeIndex["kTVMFFIBytes"] = 66] = "kTVMFFIBytes";
	/*! \brief Error object. */
	TypeIndex[TypeIndex["kTVMFFIError"] = 67] = "kTVMFFIError";
	/*! \brief Function object. */
	TypeIndex[TypeIndex["kTVMFFIFunction"] = 68] = "kTVMFFIFunction";
	/*!
	* \brief Shape object, layout = { TVMFFIObject, { const int64_t*, size_t }, ... }
	*/
	TypeIndex[TypeIndex["kTVMFFIShape"] = 69] = "kTVMFFIShape";
	/*!
	* \brief Tensor object, layout = { TVMFFIObject, DLTensor, ... }
	*/
	TypeIndex[TypeIndex["kTVMFFITensor"] = 70] = "kTVMFFITensor";
	/*! \brief Array object. */
	TypeIndex[TypeIndex["kTVMFFIArray"] = 71] = "kTVMFFIArray";
	/*! \brief Map object. */
	TypeIndex[TypeIndex["kTVMFFIMap"] = 72] = "kTVMFFIMap";
	/*! \brief Runtime dynamic loaded module object. */
	TypeIndex[TypeIndex["kTVMFFIModule"] = 73] = "kTVMFFIModule";
	/*!
	* \brief Opaque python object.
	*
	* This is a special type index to indicate we are storing an opaque PyObject.
	* Such object may interact with callback functions that are registered to support
	* python-related operations.
	*
	* We only translate the objects that we do not recognize into this type index.
	*
	* \sa TVMFFIObjectCreateOpaque
	*/
	TypeIndex[TypeIndex["kTVMFFIOpaquePyObject"] = 74] = "kTVMFFIOpaquePyObject";
	TypeIndex[TypeIndex["kTVMFFIStaticObjectEnd"] = 75] = "kTVMFFIStaticObjectEnd";
	/*! \brief Start of type indices that are allocated at runtime. */
	TypeIndex[TypeIndex["kTVMFFIDynObjectBegin"] = 128] = "kTVMFFIDynObjectBegin";
	return TypeIndex;
}({});
// [TVM]/support.ts
/**
* Check if value is a promise type
*
* @param value The input value
* @returns Whether value is promise
*/
function isPromise(value) {
	return value !== void 0 && (typeof value == "object" || typeof value == "function") && typeof value.then == "function";
}
/**
* Convert string to Uint8array.
* @param str The string.
* @returns The corresponding Uint8Array.
*/
function StringToUint8Array(str) {
	const arr = new TextEncoder().encode(str);
	const resArr = new Uint8Array(arr.length + 1);
	for (let i = 0; i < arr.length; ++i) resArr[i] = arr[i];
	resArr[arr.length] = 0;
	return resArr;
}
/**
* Internal assert helper
* @param condition The condition to fail.
* @param msg The message.
*/
function assert(condition, msg) {
	if (!condition) throw new Error("AssertError:" + (msg || ""));
}
var LinearCongruentialGenerator = class {
	modulus;
	multiplier;
	increment;
	rand_state;
	/**
	* Set modulus, multiplier, and increment. Initialize `rand_state` according to `Date.now()`.
	*/
	constructor() {
		this.modulus = 2147483647;
		this.multiplier = 48271;
		this.increment = 0;
		this.setSeed(Date.now());
	}
	/**
	* Sets `rand_state` after normalized with `modulus` to ensure that it is within range.
	* @param seed Any integer. Used to set `rand_state` after normalized with `modulus`.
	*
	* Postcondition: pass `checkRandState()`, i.e. rand_state > 0 and is an integer.
	*/
	setSeed(seed) {
		if (!Number.isInteger(seed)) throw new Error("Seed should be an integer.");
		this.rand_state = seed % this.modulus;
		if (this.rand_state == 0) this.rand_state = 1;
		this.checkRandState();
	}
	/**
	* Get the current generator state for deterministic restoration.
	*/
	getState() {
		return this.rand_state;
	}
	/**
	* Restore a state returned by `getState()`.
	*/
	setState(state) {
		if (!Number.isInteger(state)) throw new Error("RNG state should be an integer.");
		if (state <= 0 || state >= this.modulus) throw new Error(`RNG state should be an integer in (0, ${this.modulus}).`);
		this.rand_state = state;
	}
	/**
	* Generate the next integer in the range (0, this.modulus) non-inclusive, updating `rand_state`.
	*
	* Postcondition: pass `checkRandState()`, i.e. rand_state > 0 and is an integer.
	*/
	nextInt() {
		const intermediate = this.multiplier * this.rand_state + this.increment;
		this.rand_state = intermediate % this.modulus;
		this.checkRandState();
		return this.rand_state;
	}
	/**
	* Generates random float between (0, 1) non-inclusive, updating `rand_state`.
	*
	* Postcondition: pass `checkRandState()`, i.e. rand_state > 0 and is an integer.
	*/
	randomFloat() {
		return this.nextInt() / this.modulus;
	}
	checkRandState() {
		if (this.rand_state <= 0) throw new Error("Random state is unexpectedly not strictly positive.");
		if (!Number.isInteger(this.rand_state)) throw new Error("Random state is unexpectedly not an integer.");
	}
};
// [TVM]/memory.ts
/**
* Classes to manipulate Wasm memories.
*/
/**
* Wasm Memory wrapper to perform JS side raw memory access.
*/
var Memory = class {
	memory;
	wasm32 = true;
	buffer;
	viewU8;
	viewU16;
	viewI32;
	viewU32;
	viewF32;
	viewF64;
	constructor(memory) {
		this.memory = memory;
		this.buffer = this.memory.buffer;
		this.viewU8 = new Uint8Array(this.buffer);
		this.viewU16 = new Uint16Array(this.buffer);
		this.viewI32 = new Int32Array(this.buffer);
		this.viewU32 = new Uint32Array(this.buffer);
		this.viewF32 = new Float32Array(this.buffer);
		this.viewF64 = new Float64Array(this.buffer);
	}
	loadU8(ptr) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		return this.viewU8[ptr >> 0];
	}
	loadU16(ptr) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		return this.viewU16[ptr >> 1];
	}
	loadU32(ptr) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		return this.viewU32[ptr >> 2];
	}
	loadI32(ptr) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		return this.viewI32[ptr >> 2];
	}
	loadI64(ptr) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		const base = ptr >> 2;
		const value = this.viewI32[base + 1] * 4294967296 + this.viewU32[base];
		if (!Number.isSafeInteger(value)) throw new Error("Cannot represent int64 value as a JavaScript number");
		return value;
	}
	loadU64(ptr) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		const base = ptr >> 2;
		const value = this.viewU32[base] + this.viewU32[base + 1] * 4294967296;
		if (!Number.isSafeInteger(value)) throw new Error("Cannot represent uint64 value as a JavaScript number");
		return value;
	}
	loadF32(ptr) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		return this.viewF32[ptr >> 2];
	}
	loadF64(ptr) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		return this.viewF64[ptr >> 3];
	}
	loadPointer(ptr) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		if (this.wasm32) return this.loadU32(ptr);
		else return this.loadI64(ptr);
	}
	loadUSize(ptr) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		if (this.wasm32) return this.loadU32(ptr);
		else return this.loadI64(ptr);
	}
	sizeofPtr() {
		return this.wasm32 ? SizeOf.I32 : SizeOf.I64;
	}
	/**
	* Load raw bytes from ptr.
	* @param ptr The head address
	* @param numBytes The number
	*/
	loadRawBytes(ptr, numBytes) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		const result = new Uint8Array(numBytes);
		result.set(this.viewU8.subarray(ptr, ptr + numBytes));
		return result;
	}
	/**
	* Return a borrowed view of raw bytes in Wasm memory.
	*
	* The returned view aliases the current WebAssembly.Memory buffer and must
	* not be retained across a call that can grow the memory.
	*
	* @param ptr The head address.
	* @param numBytes The number of bytes.
	*/
	viewRawBytes(ptr, numBytes) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		if (!Number.isSafeInteger(ptr) || ptr < 0) throw new Error(`Invalid Wasm memory pointer: ${ptr}`);
		if (!Number.isSafeInteger(numBytes) || numBytes < 0) throw new Error(`Invalid Wasm memory byte length: ${numBytes}`);
		const end = ptr + numBytes;
		if (!Number.isSafeInteger(end) || end > this.viewU8.byteLength) throw new Error(`Wasm memory range [${ptr}, ${end}) exceeds memory size ${this.viewU8.byteLength}`);
		return this.viewU8.subarray(ptr, end);
	}
	/**
	* Load null-terminated C-string from ptr.
	* @param ptr The head address
	*/
	loadCString(ptr) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		const ret = [];
		let ch = 1;
		while (ch != 0) {
			ch = this.viewU8[ptr];
			if (ch != 0) ret.push(String.fromCharCode(ch));
			++ptr;
		}
		return ret.join("");
	}
	/**
	* Store raw bytes to the ptr.
	* @param ptr The head address.
	* @param bytes The bytes content.
	*/
	storeRawBytes(ptr, bytes) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		this.viewU8.set(bytes, ptr);
	}
	/**
	* Load the object type index from the object handle.
	* @param objectHandle The handle of the object.
	* @returns The object type index.
	*/
	loadObjectTypeIndex(objectHandle) {
		return this.loadI32(objectHandle + SizeOf.I64);
	}
	/**
	* Load the type key from the type info pointer.
	* @param typeInfoPtr The pointer to the type info.
	* @returns The type key.
	*/
	loadTypeInfoTypeKey(typeInfoPtr) {
		const typeKeyPtr = typeInfoPtr + 2 * SizeOf.I32;
		return this.loadByteArrayAsString(typeKeyPtr);
	}
	/**
	* Load small string from value pointer.
	* @param ffiAnyPtr The pointer to the value.
	* @returns The small string.
	*/
	loadSmallStr(ffiAnyPtr) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		const sizePtr = ffiAnyPtr + SizeOf.I32;
		const length = this.loadU32(sizePtr);
		const dataPtr = ffiAnyPtr + SizeOf.I32 + SizeOf.I32;
		const ret = [];
		for (let i = 0; i < length; i++) ret.push(String.fromCharCode(this.viewU8[dataPtr + i]));
		return ret.join("");
	}
	/**
	* Load small bytes from value pointer.
	* @param ffiAnyPtr
	*/
	loadSmallBytes(ffiAnyPtr) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		const sizePtr = ffiAnyPtr + SizeOf.I32;
		const length = this.loadU32(sizePtr);
		const dataPtr = ffiAnyPtr + SizeOf.I32 + SizeOf.I32;
		const result = new Uint8Array(length);
		result.set(this.viewU8.subarray(dataPtr, dataPtr + length));
		return result;
	}
	/**
	* Load bytearray as string from ptr.
	* @param byteArrayPtr The head address of the bytearray.
	*/
	loadByteArrayAsString(byteArrayPtr) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		const ptr = this.loadPointer(byteArrayPtr);
		const length = this.loadUSize(byteArrayPtr + this.sizeofPtr());
		const ret = [];
		for (let i = 0; i < length; i++) ret.push(String.fromCharCode(this.viewU8[ptr + i]));
		return ret.join("");
	}
	/**
	* Load bytearray as bytes from ptr.
	* @param byteArrayPtr The head address of the bytearray.
	*/
	loadByteArrayAsBytes(byteArrayPtr) {
		if (this.buffer != this.memory.buffer) this.updateViews();
		const ptr = this.loadPointer(byteArrayPtr);
		const length = this.loadUSize(byteArrayPtr + this.sizeofPtr());
		const result = new Uint8Array(length);
		result.set(this.viewU8.subarray(ptr, ptr + length));
		return result;
	}
	/**
	* Update memory view after the memory growth.
	*/
	updateViews() {
		this.buffer = this.memory.buffer;
		this.viewU8 = new Uint8Array(this.buffer);
		this.viewU16 = new Uint16Array(this.buffer);
		this.viewI32 = new Int32Array(this.buffer);
		this.viewU32 = new Uint32Array(this.buffer);
		this.viewF32 = new Float32Array(this.buffer);
		this.viewF64 = new Float64Array(this.buffer);
	}
};
/**
* Auxiliary call stack for the FFI calls.
*
* Lifecyle of a call stack.
* - Calls into allocXX to allocate space, mixed with storeXXX to store data.
* - Calls into ptrFromOffset, no further allocation(as ptrFromOffset can change),
*   can still call into storeXX
* - Calls into commitToWasmMemory once.
* - reset.
*/
var CachedCallStack = class {
	/** List of temporay arguments that can be disposed during reset. */
	tempArgs = [];
	memory;
	cAllocSpace;
	cFreeSpace;
	buffer;
	viewU8;
	viewI32;
	viewU32;
	viewF64;
	stackTop = 0;
	basePtr = 0;
	addressToSetTargetValue = [];
	constructor(memory, allocSpace, freeSpace) {
		const initCallStackSize = 128;
		this.memory = memory;
		this.cAllocSpace = allocSpace;
		this.cFreeSpace = freeSpace;
		this.buffer = /* @__PURE__ */ new ArrayBuffer(initCallStackSize);
		this.basePtr = this.cAllocSpace(initCallStackSize);
		this.viewU8 = new Uint8Array(this.buffer);
		this.viewI32 = new Int32Array(this.buffer);
		this.viewU32 = new Uint32Array(this.buffer);
		this.viewF64 = new Float64Array(this.buffer);
		this.updateViews();
	}
	dispose() {
		if (this.basePtr != 0) {
			this.cFreeSpace(this.basePtr);
			this.basePtr = 0;
		}
	}
	/**
	* Rest the call stack so that it can be reused again.
	*/
	reset() {
		this.stackTop = 0;
		assert(this.addressToSetTargetValue.length === 0);
		while (this.tempArgs.length != 0) this.tempArgs.pop().dispose();
	}
	/**
	* Commit all the cached data to WasmMemory.
	* This function can only be called once.
	* No further store function should be called.
	*
	* @param nbytes Number of bytes to be stored.
	*/
	commitToWasmMemory(nbytes = this.stackTop) {
		while (this.addressToSetTargetValue.length != 0) {
			const [targetOffset, valueOffset] = this.addressToSetTargetValue.pop();
			this.storePtr(targetOffset, this.ptrFromOffset(valueOffset));
		}
		this.memory.storeRawBytes(this.basePtr, this.viewU8.subarray(0, nbytes));
	}
	/**
	* Allocate space by number of bytes
	* @param nbytes Number of bytes.
	* Note: This function always allocate space that aligns to 64bit.
	*/
	allocRawBytes(nbytes) {
		nbytes = nbytes + 7 >> 3 << 3;
		if (this.stackTop + nbytes > this.buffer.byteLength) {
			const newSize = Math.max(this.buffer.byteLength * 2, this.stackTop + nbytes);
			const oldU8 = this.viewU8;
			this.buffer = new ArrayBuffer(newSize);
			this.updateViews();
			this.viewU8.set(oldU8);
			if (this.basePtr != 0) this.cFreeSpace(this.basePtr);
			this.basePtr = this.cAllocSpace(newSize);
		}
		const retOffset = this.stackTop;
		this.stackTop += nbytes;
		return retOffset;
	}
	/**
	* Allocate space for pointers.
	* @param count Number of pointers.
	* @returns The allocated pointer array.
	*/
	allocPtrArray(count) {
		return this.allocRawBytes(this.memory.sizeofPtr() * count);
	}
	/**
	* Get the real pointer from offset values.
	* Note that the returned value becomes obsolete if alloc is called on the stack.
	* @param offset The allocated offset.
	*/
	ptrFromOffset(offset) {
		return this.basePtr + offset;
	}
	storePtr(offset, value) {
		if (this.memory.wasm32) this.storeU32(offset, value);
		else this.storeI64(offset, value);
	}
	storeUSize(offset, value) {
		if (this.memory.wasm32) this.storeU32(offset, value);
		else this.storeI64(offset, value);
	}
	storeI32(offset, value) {
		this.viewI32[offset >> 2] = value;
	}
	storeU32(offset, value) {
		this.viewU32[offset >> 2] = value;
	}
	storeI64(offset, value) {
		const low = value & 4294967295;
		const base = offset >> 2;
		this.viewI32[base] = low;
		this.viewI32[base + 1] = value < 0 ? -1 : 0;
	}
	storeF64(offset, value) {
		this.viewF64[offset >> 3] = value;
	}
	storeRawBytes(offset, bytes) {
		this.viewU8.set(bytes, offset);
	}
	/**
	* Allocate a byte array for a string and return the offset of the byte array.
	* @param data The string to allocate.
	* @returns The offset of the byte array.
	*/
	allocByteArrayForString(data) {
		const dataUint8 = StringToUint8Array(data);
		const headerOffset = this.allocRawBytes(this.memory.sizeofPtr() * 2);
		const dataOffset = this.allocRawBytes(dataUint8.length);
		this.storeUSize(headerOffset + this.memory.sizeofPtr(), data.length);
		this.storeRawBytes(dataOffset, dataUint8);
		this.addressToSetTargetValue.push([headerOffset, dataOffset]);
		return headerOffset;
	}
	/**
	* Allocate then set C-String pointer to the offset.
	* This function will call into allocBytes to allocate necessary data.
	* The address won't be set immediately(because the possible change of basePtr)
	* and will be filled when we commit the data.
	*
	* @param offset The offset to set ot data pointer.
	* @param data The string content.
	*/
	allocThenSetArgString(offset, data) {
		const dataUint8 = StringToUint8Array(data);
		const strOffset = this.allocRawBytes(dataUint8.length);
		this.storeRawBytes(strOffset, dataUint8);
		this.addressToSetTargetValue.push([offset, strOffset]);
	}
	/**
	* Allocate then set the argument location with a TVMByteArray.
	* Allocate new temporary space for bytes.
	*
	* @param offset The offset to set ot data pointer.
	* @param data The string content.
	*/
	allocThenSetArgBytes(offset, data) {
		const headerOffset = this.allocRawBytes(this.memory.sizeofPtr() * 2);
		const dataOffset = this.allocRawBytes(data.length);
		this.storeRawBytes(dataOffset, data);
		this.storeUSize(headerOffset + this.memory.sizeofPtr(), data.length);
		this.addressToSetTargetValue.push([offset, headerOffset]);
		this.addressToSetTargetValue.push([headerOffset, dataOffset]);
	}
	/**
	* Update internal cache views.
	*/
	updateViews() {
		this.viewU8 = new Uint8Array(this.buffer);
		this.viewI32 = new Int32Array(this.buffer);
		this.viewU32 = new Uint32Array(this.buffer);
		this.viewF64 = new Float64Array(this.buffer);
	}
};
// [TVM]/environment.ts
/**
* Detect library provider from the importObject.
*
* @param importObject The import object.
*/
function detectLibraryProvider(importObject) {
	if (importObject["wasmLibraryProvider"] && importObject["wasmLibraryProvider"]["start"] && importObject["wasmLibraryProvider"]["imports"] !== void 0) {
		const item = importObject;
		return {
			imports: item.wasmLibraryProvider.imports,
			start: (inst) => {
				item.wasmLibraryProvider.start(inst);
			}
		};
	} else if (importObject["imports"] && importObject["start"] !== void 0) return importObject;
	else if (importObject["wasiImport"] && importObject["start"] !== void 0) return {
		imports: { "wasi_snapshot_preview1": importObject["wasiImport"] },
		start: (inst) => {
			importObject["start"](inst);
		}
	};
	else return;
}
/**
* Environment to impelement most of the JS library functions.
*/
var Environment = class {
	logger;
	imports;
	/**
	* Maintains a table of FTVMWasmPackedCFunc that the C part
	* can call via TVMWasmPackedCFunc.
	*
	* We maintain a separate table so that we can have un-limited amount
	* of functions that do not maps to the address space.
	*/
	packedCFuncTable = [void 0];
	/**
	* Free table index that can be recycled.
	*/
	packedCFuncTableFreeId = [];
	libProvider;
	constructor(importObject = {}, logger = console.log) {
		this.logger = logger;
		this.libProvider = detectLibraryProvider(importObject);
		if (this.libProvider !== void 0) this.imports = this.libProvider.imports;
		else this.imports = importObject;
		this.imports.env = this.environment(this.imports.env);
	}
	/** Mark the start of the instance. */
	start(inst) {
		if (this.libProvider !== void 0) this.libProvider.start(inst);
	}
	environment(initEnv) {
		const defaultEnv = {
			"__cxa_thread_atexit": () => {},
			"emscripten_notify_memory_growth": (index) => {}
		};
		const wasmSafeCall = (self, args, num_args, result) => {
			const cfunc = this.packedCFuncTable[self];
			assert(cfunc !== void 0);
			return cfunc(self, args, num_args, result);
		};
		const wasmFunctionDeleter = (self) => {
			this.packedCFuncTable[self] = void 0;
			this.packedCFuncTableFreeId.push(self);
		};
		return Object.assign(defaultEnv, initEnv, {
			"TVMFFIWasmSafeCall": wasmSafeCall,
			"TVMFFIWasmFunctionDeleter": wasmFunctionDeleter,
			"__console_log": (msg) => {
				this.logger(msg);
			}
		});
	}
};
// [TVM]/asyncify.ts
/**
* enums to check the current state of asynctify
*/
var AsyncifyStateKind = /* @__PURE__ */ function(AsyncifyStateKind) {
	AsyncifyStateKind[AsyncifyStateKind["None"] = 0] = "None";
	AsyncifyStateKind[AsyncifyStateKind["Unwinding"] = 1] = "Unwinding";
	AsyncifyStateKind[AsyncifyStateKind["Rewinding"] = 2] = "Rewinding";
	return AsyncifyStateKind;
}(AsyncifyStateKind || {});
/** The start location of asynctify stack data */
var ASYNCIFY_DATA_ADDR = 16;
/** The data start of stack rewind/unwind */
var ASYNCIFY_DATA_START = 24;
/** The data end of stack rewind/unwind */
var ASYNCIFY_DATA_END = 1024;
/** Hold asynctify handler instance that runtime can use */
var AsyncifyHandler = class {
	/** exports from wasm */
	exports;
	/** current state kind */
	state = AsyncifyStateKind.None;
	/** The stored value before unwind */
	storedPromiseBeforeUnwind = null;
	/** The stored value that is resolved */
	storedValueBeforeRewind = null;
	/** The stored exception */
	storedExceptionBeforeRewind = null;
	constructor(exports, memory) {
		this.exports = exports;
		this.initMemory(memory);
	}
	/**
	* Whether the wasm enables asynctify
	* @returns Whether the wasm enables asynctify
	*/
	enabled() {
		return this.exports.asyncify_stop_rewind !== void 0;
	}
	/**
	* Get the current asynctify state
	*
	* @returns The current asynctify state
	*/
	isNormalStackState() {
		return this.state == AsyncifyStateKind.None;
	}
	/**
	* Get the current asynctify state
	*
	* @returns The current asynctify state
	*/
	getState() {
		return this.state;
	}
	/**
	* Wrap a function that can be used as import of the wasm asynctify layer
	*
	* @param func The input import function
	* @returns The wrapped function that can be registered to the system
	*/
	wrapImport(func) {
		return (...args) => {
			if (this.getState() == AsyncifyStateKind.Rewinding) {
				this.stopRewind();
				if (this.storedValueBeforeRewind !== null) {
					assert(this.storedExceptionBeforeRewind === null);
					const result = this.storedValueBeforeRewind;
					this.storedValueBeforeRewind = null;
					return result;
				} else {
					assert(this.storedValueBeforeRewind === null);
					const error = this.storedExceptionBeforeRewind;
					this.storedExceptionBeforeRewind = null;
					throw error;
				}
			}
			assert(this.getState() == AsyncifyStateKind.None);
			const value = func(...args);
			if (isPromise(value)) {
				this.startUnwind();
				assert(this.storedPromiseBeforeUnwind == null);
				this.storedPromiseBeforeUnwind = value;
				return;
			} else return value;
		};
	}
	/**
	* Warp an exported asynctify function so it can return promise
	*
	* @param func The input function
	* @returns The wrapped async function
	*/
	wrapExport(func) {
		return async (...args) => {
			assert(this.getState() == AsyncifyStateKind.None);
			let result = func(...args);
			while (this.getState() == AsyncifyStateKind.Unwinding) {
				this.stopUnwind();
				const storedPromiseBeforeUnwind = this.storedPromiseBeforeUnwind;
				this.storedPromiseBeforeUnwind = null;
				assert(this.storedExceptionBeforeRewind === null);
				assert(this.storedValueBeforeRewind == null);
				try {
					this.storedValueBeforeRewind = await storedPromiseBeforeUnwind;
				} catch (error) {
					this.storedExceptionBeforeRewind = error;
				}
				assert(!isPromise(this.storedValueBeforeRewind));
				assert(this.getState() == AsyncifyStateKind.None);
				this.startRewind();
				result = func(...args);
			}
			return result;
		};
	}
	startRewind() {
		if (this.exports.asyncify_start_rewind === void 0) throw Error("Asynctify is not enabled, please compile with -s ASYNCIFY=1 in emcc");
		this.exports.asyncify_start_rewind(ASYNCIFY_DATA_ADDR);
		this.state = AsyncifyStateKind.Rewinding;
	}
	stopRewind() {
		if (this.exports.asyncify_stop_rewind === void 0) throw Error("Asynctify is not enabled, please compile with -s ASYNCIFY=1 in emcc");
		this.exports.asyncify_stop_rewind();
		this.state = AsyncifyStateKind.None;
	}
	startUnwind() {
		if (this.exports.asyncify_start_unwind === void 0) throw Error("Asynctify is not enabled, please compile with -s ASYNCIFY=1 in emcc");
		this.exports.asyncify_start_unwind(ASYNCIFY_DATA_ADDR);
		this.state = AsyncifyStateKind.Unwinding;
	}
	stopUnwind() {
		if (this.exports.asyncify_stop_unwind === void 0) throw Error("Asynctify is not enabled, please compile with -s ASYNCIFY=1 in emcc");
		this.exports.asyncify_stop_unwind();
		this.state = AsyncifyStateKind.None;
	}
	/**
	* Initialize the wasm memory to setup necessary meta-data
	* for asynctify handling
	* @param memory The memory ti
	*/
	initMemory(memory) {
		new Int32Array(memory.buffer, ASYNCIFY_DATA_ADDR, 2).set([ASYNCIFY_DATA_START, ASYNCIFY_DATA_END]);
	}
};
// [TVM]/webgpu.ts
function roundUpToFourBytes(nbytes) {
	if (!Number.isSafeInteger(nbytes) || nbytes < 0) throw new Error(`Invalid WebGPU buffer size: ${nbytes}`);
	const aligned = Math.ceil(nbytes / 4) * 4;
	if (!Number.isSafeInteger(aligned)) throw new Error(`WebGPU buffer size is too large to align: ${nbytes}`);
	return aligned;
}
function validateWebGPUCopyOffset(offset, name) {
	if (!Number.isSafeInteger(offset) || offset < 0 || offset % 4 != 0) throw new Error(`${name} must be a nonnegative multiple of four: ${offset}`);
}
/**
* Create GPU buffer with `createBuffer()` but with error catching; destroy if error caught.
* @param device The GPUDevice used to create a buffer.
* @param descriptor The GPUBufferDescriptor passed to `createBuffer()`.
* @returns The buffer created by `createBuffer()`.
*
* Note: We treat any error occurred at `createBuffer()` fatal and expect the user to handle
*   `device.destroy()` with `device.lost.then()`.
*/
function tryCreateBuffer(device, descriptor) {
	device.pushErrorScope("out-of-memory");
	device.pushErrorScope("validation");
	device.pushErrorScope("internal");
	const buffer = device.createBuffer(descriptor);
	Promise.all([
		device.popErrorScope(),
		device.popErrorScope(),
		device.popErrorScope()
	]).then((errors) => {
		const captured = errors.filter((error) => error !== null);
		if (captured.length > 0) {
			device.destroy();
			captured.forEach((error) => console.error(error));
		}
	}).catch((err) => {
		console.error("Failed to pop error scopes:", err);
	});
	return buffer;
}
var canvasRenderWGSL = `
@group(0) @binding(0) var my_sampler : sampler;
@group(0) @binding(1) var my_texture : texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

@vertex
fn vertex_main(@builtin(vertex_index) vidx : u32) -> VertexOutput {
  const pos = array(
    vec2( 1.0,  1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0, -1.0),
    vec2(-1.0,  1.0),
  );

  const uv = array(
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0),
    vec2(0.0, 0.0),
  );

  var output : VertexOutput;
  output.position = vec4(pos[vidx], 0.0, 1.0);
  output.uv = uv[vidx];
  return output;
}

@fragment
fn fragment_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  return textureSample(my_texture, my_sampler, uv);
}

@fragment
fn fragment_clear(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  return vec4(1.0, 1.0, 1.0, 1.0);
}
`;
var CanvasRenderManager = class {
	device;
	canvasContext;
	stagingTexture;
	renderSampler;
	renderPipeline;
	clearPipeline;
	canvasTextureFormat;
	constructor(device, canvas) {
		this.device = device;
		const ctx = canvas.getContext("webgpu");
		if (ctx == null) throw Error("Cannot bind WebGPU context");
		this.canvasContext = ctx;
		this.canvasTextureFormat = navigator.gpu.getPreferredCanvasFormat();
		this.canvasContext.configure({
			device: this.device,
			format: this.canvasTextureFormat,
			alphaMode: "opaque"
		});
		this.renderPipeline = device.createRenderPipeline({
			layout: "auto",
			vertex: {
				module: device.createShaderModule({ code: canvasRenderWGSL }),
				entryPoint: "vertex_main"
			},
			fragment: {
				module: device.createShaderModule({ code: canvasRenderWGSL }),
				entryPoint: "fragment_main",
				targets: [{ format: this.canvasTextureFormat }]
			},
			primitive: { topology: "triangle-list" }
		});
		this.clearPipeline = device.createRenderPipeline({
			layout: "auto",
			vertex: {
				module: device.createShaderModule({ code: canvasRenderWGSL }),
				entryPoint: "vertex_main"
			},
			fragment: {
				module: device.createShaderModule({ code: canvasRenderWGSL }),
				entryPoint: "fragment_clear",
				targets: [{ format: this.canvasTextureFormat }]
			},
			primitive: { topology: "triangle-list" }
		});
		this.renderSampler = device.createSampler({
			magFilter: "linear",
			minFilter: "linear"
		});
		this.stagingTexture = device.createTexture({
			size: [
				canvas.height,
				canvas.width,
				1
			],
			format: "rgba8unorm",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
		});
	}
	clear() {
		const commandEncoder = this.device.createCommandEncoder();
		const passEncoder = commandEncoder.beginRenderPass({ colorAttachments: [{
			view: this.canvasContext.getCurrentTexture().createView(),
			clearValue: {
				r: 0,
				g: 0,
				b: 0,
				a: 1
			},
			loadOp: "clear",
			storeOp: "store"
		}] });
		passEncoder.setPipeline(this.clearPipeline);
		const renderBindingGroup = this.device.createBindGroup({
			layout: this.renderPipeline.getBindGroupLayout(0),
			entries: [{
				binding: 0,
				resource: this.renderSampler
			}, {
				binding: 1,
				resource: this.stagingTexture.createView()
			}]
		});
		passEncoder.setBindGroup(0, renderBindingGroup);
		passEncoder.draw(6, 1, 0, 0);
		passEncoder.end();
		this.device.queue.submit([commandEncoder.finish()]);
	}
	draw(buffer, height, width) {
		if (height != this.stagingTexture.height || width != this.stagingTexture.width) {
			this.stagingTexture.destroy();
			this.stagingTexture = this.device.createTexture({
				size: [
					height,
					width,
					1
				],
				format: "rgba8unorm",
				usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
			});
		}
		const commandEncoder = this.device.createCommandEncoder();
		commandEncoder.copyBufferToTexture({
			buffer,
			offset: 0,
			bytesPerRow: this.stagingTexture.width * 4
		}, { texture: this.stagingTexture }, {
			width: this.stagingTexture.width,
			height: this.stagingTexture.height
		});
		const passEncoder = commandEncoder.beginRenderPass({ colorAttachments: [{
			view: this.canvasContext.getCurrentTexture().createView(),
			clearValue: {
				r: 0,
				g: 0,
				b: 0,
				a: 1
			},
			loadOp: "clear",
			storeOp: "store"
		}] });
		passEncoder.setPipeline(this.renderPipeline);
		const renderBindingGroup = this.device.createBindGroup({
			layout: this.renderPipeline.getBindGroupLayout(0),
			entries: [{
				binding: 0,
				resource: this.renderSampler
			}, {
				binding: 1,
				resource: this.stagingTexture.createView()
			}]
		});
		passEncoder.setBindGroup(0, renderBindingGroup);
		passEncoder.draw(6, 1, 0, 0);
		passEncoder.end();
		this.device.queue.submit([commandEncoder.finish()]);
	}
	dispose() {
		this.stagingTexture.destroy();
	}
};
/**
* WebGPU context
* Manages all the webgpu resources here.
*/
var WebGPUContext = class {
	device;
	memory;
	bufferTable = [void 0];
	bufferTableFreeId = [];
	canvasRenderManager = void 0;
	readStagingBufferPool = [];
	maxReadStagingBuffers = 4;
	pendingGPUToCPUCopy = null;
	pendingGPUToCPUCopyIsQueueTail = false;
	pendingEncoder = null;
	uniformBufferPool = [];
	uniformBufferPoolSizes = [];
	pendingDispatchCount = 0;
	peakAllocatedBytes = 0;
	currAllocatedBytes = 0;
	allAllocatedBytes = 0;
	shaderSubmitCounter = 0;
	debugShaderSubmitLimit = -1;
	debugLogFinish = false;
	constructor(memory, device) {
		this.memory = memory;
		this.device = device;
	}
	/**
	* Flush all pending GPU commands by finishing and submitting the
	* accumulated command encoder.
	*
	* Must be called before:
	* - GPU→CPU readback (deviceCopyFromGPU)
	* - CPU→GPU writes (deviceCopyToGPU, copyRawBytesToBuffer)
	* - Buffer deallocation (deviceFreeDataSpace)
	* - Canvas drawing (drawImageFromBuffer)
	* - Queue sync (sync)
	*/
	flushCommands() {
		if (this.pendingEncoder) {
			this.device.queue.submit([this.pendingEncoder.finish()]);
			this.pendingEncoder = null;
			this.pendingDispatchCount = 0;
			this.pendingGPUToCPUCopyIsQueueTail = false;
		}
	}
	/**
	* Dispose context.
	*/
	dispose() {
		this.flushCommands();
		this.canvasRenderManager?.dispose();
		this.bufferTableFreeId = [];
		while (this.bufferTable.length != 0) this.bufferTable.pop()?.destroy();
		for (const buf of this.uniformBufferPool) buf.destroy();
		this.uniformBufferPool.length = 0;
		this.uniformBufferPoolSizes.length = 0;
		while (this.readStagingBufferPool.length != 0) this.readStagingBufferPool.pop()?.buffer.destroy();
		this.device.destroy();
	}
	/**
	* Wait for all pending GPU tasks to complete
	*/
	async sync() {
		this.flushCommands();
		const pendingRead = this.pendingGPUToCPUCopy;
		const pendingReadIsQueueTail = this.pendingGPUToCPUCopyIsQueueTail;
		this.pendingGPUToCPUCopy = null;
		this.pendingGPUToCPUCopyIsQueueTail = false;
		if (pendingRead && pendingReadIsQueueTail) await pendingRead;
		else {
			const queueDone = this.device.queue.onSubmittedWorkDone();
			if (pendingRead) await Promise.all([pendingRead, queueDone]);
			else await queueDone;
		}
	}
	/**
	* Obtain the runtime information in readable format.
	*/
	runtimeStatsText() {
		let info = "peak-memory=" + Math.ceil(this.peakAllocatedBytes / (1 << 20)) + " MB";
		info += ", all-memory=" + Math.ceil(this.allAllocatedBytes / (1 << 20)) + " MB";
		info += ", shader-submissions=" + this.shaderSubmitCounter;
		return info;
	}
	/**
	* Draw image from data in storage buffer.
	* @param ptr The GPU ptr
	* @param height The height of the image.
	* @param width The width of the image.
	*/
	drawImageFromBuffer(ptr, height, width) {
		if (this.canvasRenderManager == void 0) throw Error("Do not have a canvas context, call bindCanvas first");
		this.flushCommands();
		this.canvasRenderManager.draw(this.gpuBufferFromPtr(ptr), height, width);
		this.pendingGPUToCPUCopyIsQueueTail = false;
	}
	/**
	* Copy raw bytes into buffer ptr.
	*
	* @param rawBytes The raw bytes
	* @param toPtr The target gpu buffer ptr
	* @param toOffset The beginning offset
	* @param nbytes Number of bytes
	*/
	copyRawBytesToBuffer(rawBytes, toPtr, toOffset, nbytes) {
		this.flushCommands();
		this.device.queue.writeBuffer(this.gpuBufferFromPtr(toPtr), toOffset, rawBytes, 0, nbytes);
		this.pendingGPUToCPUCopyIsQueueTail = false;
	}
	/**
	* Clear canvas
	*/
	clearCanvas() {
		if (this.canvasRenderManager) {
			this.canvasRenderManager.clear();
			this.pendingGPUToCPUCopyIsQueueTail = false;
		}
	}
	/**
	* Bind a canvas element to the runtime.
	* @param canvas The HTML canvas/
	*/
	bindCanvas(canvas) {
		this.canvasRenderManager = new CanvasRenderManager(this.device, canvas);
	}
	/**
	* Create a PackedFunc that runs the given shader
	* via createComputePipeline
	*
	* @param finfo The function information already parsed as a record.
	* @param code The shader data(in WGSL)
	* @returns The shader
	*/
	createShader(finfo, code) {
		return this.createShadeInternal(finfo, code, false);
	}
	/**
	* Create a PackedFunc that runs the given shader asynchronously
	* via createComputePipelineAsync
	*
	* @param finfo The function information already parsed as a record.
	* @param code The shader data(in WGSL)
	* @returns The shader
	*/
	async createShaderAsync(finfo, code) {
		return await this.createShadeInternal(finfo, code, true);
	}
	/**
	* Get a uniform buffer from the per-dispatch pool.
	*
	* Each dispatch in a batched encoder needs its own uniform buffer because
	* queue.writeBuffer() executes immediately while compute passes are deferred.
	* Reusing a shared buffer would overwrite data before earlier dispatches
	* consume it.
	*
	* The pool grows as needed. Buffers are reused across flushes (indexed by
	* dispatch position within the current batch). If the pool has no slot for
	* this dispatch, we flush first — this submits all pending passes, resets
	* pendingDispatchCount to 0, and allows reuse from the start of the pool.
	*
	* State after flush: the pending encoder and all bind group / buffer
	* references from prior dispatches are submitted and consumed. The new
	* dispatch starts a fresh encoder, so no stale state carries over.
	*
	* @param nbytes Minimum buffer size in bytes.
	* @returns A GPUBuffer with UNIFORM | COPY_DST usage, at least nbytes large.
	*/
	getUniformFromPool(nbytes) {
		const dispatchIdx = this.pendingDispatchCount++;
		if (dispatchIdx < this.uniformBufferPool.length && this.uniformBufferPoolSizes[dispatchIdx] >= nbytes) return this.uniformBufferPool[dispatchIdx];
		if (dispatchIdx < this.uniformBufferPool.length) this.uniformBufferPool[dispatchIdx].destroy();
		const buffer = this.device.createBuffer({
			size: nbytes,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.uniformBufferPool[dispatchIdx] = buffer;
		this.uniformBufferPoolSizes[dispatchIdx] = nbytes;
		return buffer;
	}
	/**
	* Internal impl of createShader for both async and sync mode.
	*
	* @param finfo The function information already parsed as a record.
	* @param code The shader data(in WGSL)
	* @param asyncMode Whether use async mode.
	* @returns The shader function or promise of shader func.
	*/
	createShadeInternal(finfo, code, asyncMode) {
		const dispatchToDim = [];
		let paramWriteAccess = [];
		for (let i = 0; i < finfo.launch_param_tags.length; ++i) {
			const tag = finfo.launch_param_tags[i];
			if (tag.startsWith("blockIdx.")) {
				const target = tag.charCodeAt(tag.length - 1) - "x".charCodeAt(0);
				assert(target >= 0 && target < 3);
				dispatchToDim.push(target);
			} else if (tag.startsWith("threadIdx.")) {
				const target = tag.charCodeAt(tag.length - 1) - "x".charCodeAt(0);
				assert(target >= 0 && target < 3);
				dispatchToDim.push(target + 3);
			} else if (tag.startsWith("paramWriteAccess:")) paramWriteAccess = JSON.parse(tag.substring(17));
			else throw new Error("Cannot handle thread_axis " + tag);
		}
		const layoutEntries = [];
		const bufferArgIndices = [];
		const podArgIndices = [];
		for (let i = 0; i < finfo.arg_types.length; ++i) {
			const dtype = finfo.arg_types[i];
			if (dtype == "handle") {
				layoutEntries.push({
					binding: bufferArgIndices.length,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: paramWriteAccess[bufferArgIndices.length] ? "storage" : "read-only-storage" }
				});
				bufferArgIndices.push(i);
			} else if (dtype.startsWith("int") || dtype.startsWith("uint") || dtype.startsWith("float")) podArgIndices.push(i);
			else throw new Error("Cannot handle argument type " + dtype + " in WebGPU shader");
		}
		assert(paramWriteAccess.length == bufferArgIndices.length);
		layoutEntries.push({
			binding: bufferArgIndices.length,
			visibility: GPUShaderStage.COMPUTE,
			buffer: { type: "uniform" }
		});
		const bindGroupLayout = this.device.createBindGroupLayout({ entries: layoutEntries });
		const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
		const createShaderFunc = (pipeline) => {
			const submitShader = (...args) => {
				if (this.debugShaderSubmitLimit != -1 && this.shaderSubmitCounter >= this.debugShaderSubmitLimit) {
					this.shaderSubmitCounter += 1;
					return;
				}
				if (!this.pendingEncoder) this.pendingEncoder = this.device.createCommandEncoder();
				const compute = this.pendingEncoder.beginComputePass();
				compute.setPipeline(pipeline);
				const bindGroupEntries = [];
				const numBufferOrPodArgs = bufferArgIndices.length + podArgIndices.length;
				assert(args.length == numBufferOrPodArgs + dispatchToDim.length);
				const workDim = [
					1,
					1,
					1,
					1,
					1,
					1
				];
				for (let i = 0; i < dispatchToDim.length; ++i) workDim[dispatchToDim[i]] = args[numBufferOrPodArgs + i];
				if (workDim[2] != 1) throw Error("WebGPU: blockIdx.z is reserved for internal use");
				const packDimX = workDim[0];
				if (workDim[0] >= 65536) {
					let wl_x = workDim[0];
					let wl_z = workDim[2];
					while (wl_x >= 65536) {
						if (wl_x % 2 == 0) wl_x = wl_x / 2;
						else wl_x = (wl_x + 1) / 2;
						wl_z *= 2;
					}
					workDim[0] = wl_x;
					workDim[2] = wl_z;
					assert(wl_x * wl_z >= packDimX);
				}
				for (let i = 0; i < bufferArgIndices.length; ++i) bindGroupEntries.push({
					binding: i,
					resource: { buffer: this.gpuBufferFromPtr(args[bufferArgIndices[i]]) }
				});
				const bufBytes = (podArgIndices.length + 1) * 4;
				const podArgBuffer = this.getUniformFromPool(bufBytes);
				const i32View = new Int32Array(podArgIndices.length + 1);
				const u32View = new Uint32Array(i32View.buffer);
				const f32View = new Float32Array(i32View.buffer);
				for (let i = 0; i < podArgIndices.length; ++i) {
					const value = args[podArgIndices[i]];
					const dtype = finfo.arg_types[podArgIndices[i]];
					if (dtype.startsWith("int")) i32View[i] = value;
					else if (dtype.startsWith("uint")) u32View[i] = value;
					else if (dtype.startsWith("float")) f32View[i] = value;
					else throw Error("Unknown pod dtype " + dtype);
				}
				u32View[podArgIndices.length] = packDimX;
				this.device.queue.writeBuffer(podArgBuffer, 0, i32View.buffer);
				bindGroupEntries.push({
					binding: bufferArgIndices.length,
					resource: {
						buffer: podArgBuffer,
						size: i32View.buffer.byteLength
					}
				});
				compute.setBindGroup(0, this.device.createBindGroup({
					layout: bindGroupLayout,
					entries: bindGroupEntries
				}));
				compute.dispatchWorkgroups(workDim[0], workDim[1], workDim[2]);
				compute.end();
				if (this.debugLogFinish) {
					this.flushCommands();
					const currCounter = this.shaderSubmitCounter;
					this.device.queue.onSubmittedWorkDone().then(() => {
						console.log("[" + currCounter + "][Debug] finish shader" + finfo.name);
					});
				}
				this.shaderSubmitCounter += 1;
			};
			return submitShader;
		};
		const shaderModule = this.device.createShaderModule({
			code,
			compilationHints: [{
				entryPoint: "main",
				layout: pipelineLayout
			}]
		});
		if (asyncMode) return this.device.createComputePipelineAsync({
			layout: pipelineLayout,
			compute: {
				module: shaderModule,
				entryPoint: finfo.name
			}
		}).then((pipeline) => {
			return createShaderFunc(pipeline);
		});
		else return createShaderFunc(this.device.createComputePipeline({
			layout: pipelineLayout,
			compute: {
				module: shaderModule,
				entryPoint: finfo.name
			}
		}));
	}
	/**
	* Get the device API according to its name
	* @param name The name of the API.
	* @returns The corresponding device api.
	*/
	getDeviceAPI(name) {
		if (name == "deviceAllocDataSpace") return (nbytes) => {
			return this.deviceAllocDataSpace(nbytes);
		};
		else if (name == "deviceFreeDataSpace") return (ptr) => {
			return this.deviceFreeDataSpace(ptr);
		};
		else if (name == "deviceCopyToGPU") return (from, to, toOffset, nbytes) => {
			this.deviceCopyToGPU(from, to, toOffset, nbytes);
		};
		else if (name == "deviceCopyFromGPU") return (from, fromOffset, to, nbytes) => {
			this.deviceCopyFromGPU(from, fromOffset, to, nbytes);
		};
		else if (name == "deviceCopyWithinGPU") return (from, fromOffset, to, toOffset, nbytes) => {
			this.deviceCopyWithinGPU(from, fromOffset, to, toOffset, nbytes);
		};
		else throw new Error("Unknown DeviceAPI function " + name);
	}
	deviceAllocDataSpace(nbytes) {
		const allocationBytes = Math.max(4, roundUpToFourBytes(nbytes));
		const buffer = tryCreateBuffer(this.device, {
			size: allocationBytes,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
		});
		this.currAllocatedBytes += buffer.size;
		this.allAllocatedBytes += buffer.size;
		if (this.currAllocatedBytes > this.peakAllocatedBytes) this.peakAllocatedBytes = this.currAllocatedBytes;
		return this.attachToBufferTable(buffer);
	}
	deviceFreeDataSpace(ptr) {
		const idx = ptr;
		const buffer = this.bufferTable[idx];
		this.bufferTable[idx] = void 0;
		assert(buffer !== void 0);
		this.bufferTableFreeId.push(idx);
		this.currAllocatedBytes -= buffer.size;
		this.flushCommands();
		buffer.destroy();
	}
	deviceCopyToGPU(from, to, toOffset, nbytes) {
		validateWebGPUCopyOffset(toOffset, "WebGPU destination offset");
		this.flushCommands();
		let rawBytes = this.memory.viewRawBytes(from, nbytes);
		if (rawBytes.length % 4 !== 0) {
			const toPad = 4 - rawBytes.length % 4;
			const padded = new Uint8Array(rawBytes.length + toPad);
			padded.set(rawBytes);
			rawBytes = padded;
			nbytes = nbytes + toPad;
		}
		this.device.queue.writeBuffer(this.gpuBufferFromPtr(to), toOffset, rawBytes, 0, nbytes);
		this.pendingGPUToCPUCopyIsQueueTail = false;
	}
	/**
	* Get a MAP_READ staging buffer from the pool, or create one if none fits.
	* Uses first-fit-by-size: returns the first pooled buffer >= nbytes.
	*/
	getOrCreateReadStagingBuffer(nbytes) {
		for (let i = 0; i < this.readStagingBufferPool.length; i++) if (this.readStagingBufferPool[i].size >= nbytes) return this.readStagingBufferPool.splice(i, 1)[0].buffer;
		return tryCreateBuffer(this.device, {
			size: nbytes,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		});
	}
	/**
	* Return a MAP_READ staging buffer to the pool for reuse.
	* Evicts the smallest buffer if the pool is full.
	*/
	recycleReadStagingBuffer(buf) {
		buf.unmap();
		if (this.readStagingBufferPool.length >= this.maxReadStagingBuffers) {
			let minIdx = 0;
			for (let i = 1; i < this.readStagingBufferPool.length; i++) if (this.readStagingBufferPool[i].size < this.readStagingBufferPool[minIdx].size) minIdx = i;
			this.readStagingBufferPool.splice(minIdx, 1)[0].buffer.destroy();
		}
		this.readStagingBufferPool.push({
			buffer: buf,
			size: buf.size
		});
	}
	deviceCopyFromGPU(from, fromOffset, to, nbytes) {
		validateWebGPUCopyOffset(fromOffset, "WebGPU source offset");
		this.flushCommands();
		if (nbytes == 0) {
			this.memory.storeRawBytes(to, /* @__PURE__ */ new Uint8Array(0));
			return;
		}
		const copyBytes = roundUpToFourBytes(nbytes);
		const gpuTemp = this.getOrCreateReadStagingBuffer(copyBytes);
		const copyEncoder = this.device.createCommandEncoder();
		copyEncoder.copyBufferToBuffer(this.gpuBufferFromPtr(from), fromOffset, gpuTemp, 0, copyBytes);
		const copyCommands = copyEncoder.finish();
		this.device.queue.submit([copyCommands]);
		const readPromise = gpuTemp.mapAsync(GPUMapMode.READ).then(() => {
			const data = gpuTemp.getMappedRange(0, copyBytes);
			this.memory.storeRawBytes(to, new Uint8Array(data).subarray(0, nbytes));
			this.recycleReadStagingBuffer(gpuTemp);
		});
		this.pendingGPUToCPUCopy = this.pendingGPUToCPUCopy ? this.pendingGPUToCPUCopy.then(() => readPromise) : readPromise;
		this.pendingGPUToCPUCopyIsQueueTail = true;
	}
	deviceCopyWithinGPU(from, fromOffset, to, toOffset, nbytes) {
		if (!this.pendingEncoder) this.pendingEncoder = this.device.createCommandEncoder();
		this.pendingEncoder.copyBufferToBuffer(this.gpuBufferFromPtr(from), fromOffset, this.gpuBufferFromPtr(to), toOffset, nbytes);
	}
	gpuBufferFromPtr(ptr) {
		const buffer = this.bufferTable[ptr];
		assert(buffer !== void 0);
		return buffer;
	}
	attachToBufferTable(buffer) {
		if (this.bufferTableFreeId.length != 0) {
			const idx = this.bufferTableFreeId.pop();
			this.bufferTable[idx] = buffer;
			return idx;
		} else {
			const idx = this.bufferTable.length;
			this.bufferTable.push(buffer);
			return idx;
		}
	}
};
// [TVM]/cache_state.ts
/**
* A generic LRU (Least Recently Used) cache with bounded size.
*
* Entries are evicted in insertion order when the cache exceeds `maxSize`.
* Uses a Map to maintain insertion order for O(1) LRU eviction.
*
* @typeParam K - Cache key type.
* @typeParam V - Cache value type.
*/
var LRUCache = class {
	cache = /* @__PURE__ */ new Map();
	maxSize;
	/** Optional callback invoked when an entry is evicted. */
	onEvict;
	constructor(maxSize, onEvict) {
		this.maxSize = maxSize;
		this.onEvict = onEvict;
	}
	/**
	* Get a value from the cache, constructing it via `constructor` on miss.
	*
	* On hit: moves the entry to most-recently-used position and returns it.
	* On miss: calls `constructor()` to create the value, inserts it, and
	* returns it. If the cache is full, the least-recently-used entry is
	* evicted first.
	*
	* @param key The cache key.
	* @param constructor Factory function called on cache miss to produce the value.
	* @returns The cached or newly constructed value.
	*/
	get(key, constructor) {
		const existing = this.cache.get(key);
		if (existing !== void 0) {
			this.cache.delete(key);
			this.cache.set(key, existing);
			return existing;
		}
		if (this.cache.size >= this.maxSize) {
			const oldest = this.cache.keys().next().value;
			if (oldest !== void 0) {
				if (this.onEvict) this.onEvict(oldest, this.cache.get(oldest));
				this.cache.delete(oldest);
			}
		}
		const value = constructor();
		this.cache.set(key, value);
		return value;
	}
	/**
	* Check whether eviction would be needed for a new entry.
	*
	* Useful when the caller needs to perform side effects before eviction
	* (e.g. flushing pending GPU commands before destroying an evicted buffer).
	*
	* @param key The key to check.
	* @returns true if inserting `key` would trigger eviction of another entry.
	*/
	needEviction(key) {
		if (this.cache.has(key)) return false;
		return this.cache.size >= this.maxSize;
	}
	/**
	* Clear all cached entries.
	*
	* Does not dispose values — the caller is responsible for cleanup
	* (e.g. destroying GPU buffers) before calling invalidate.
	*/
	invalidate() {
		this.cache.clear();
	}
	/** Number of entries currently in the cache. */
	get size() {
		return this.cache.size;
	}
	/** Iterate over all cached values (for disposal). */
	values() {
		return this.cache.values();
	}
};
/**
* CacheState manages domain-specific caches for the WebGPU runtime.
*
* Currently contains:
* - **shapeCache**: Caches TVM ShapeTuple objects keyed by dimension string.
*   - Why: `makeShapeTuple()` is called on every tensor operation, crossing
*     the JS→WASM FFI boundary each time. During LLM decode, the same shapes
*     repeat every token (e.g. [1,32,128]), so caching avoids thousands of
*     redundant FFI round-trips.
*   - Invalidation: Never. Shape tuples are immutable value objects that
*     remain valid for the lifetime of the TVM instance.
*
* Future additions (follow-up PR):
* - **uniformCache**: Caches GPU uniform buffers keyed by content hash.
*   - Why: Many dispatches use identical scalar arguments (matrix dims, etc.).
*     Reusing the buffer avoids `createBuffer` + `writeBuffer` overhead.
*   - Invalidation: Must invalidate on any GPU buffer deallocation, since
*     buffer pointers can be reused by the allocator, making cached entries
*     that reference the old buffer stale.
*/
var CacheState = class {
	/**
	* Cache for TVM ShapeTuple objects.
	*
	* Key: comma-separated dimension string, e.g. "1,32,128"
	* Value: TVM ShapeTuple object (Disposable)
	*
	* Invalidation rule: None required — shape tuples are immutable.
	*/
	shapeCache;
	constructor(shapeCacheSize = 256) {
		this.shapeCache = new LRUCache(shapeCacheSize, (_key, value) => value.dispose());
	}
	/**
	* Compute the cache key for a shape tuple.
	*
	* @param shape Array of dimension values.
	* @returns String key suitable for shapeCache lookup.
	*/
	static computeShapeKey(shape) {
		return shape.toString();
	}
	/**
	* Dispose all cached objects and clear all caches.
	*/
	dispose() {
		for (const obj of this.shapeCache.values()) obj.dispose();
		this.shapeCache.invalidate();
	}
};
// [TVM]/opfs_store.ts
var HASH_ALGORITHM$1 = "SHA-256";
var OPFS_STORE_ROOT_DIRECTORY = "tvmjs-opfs-store";
var OPFSStore = class OPFSStore {
	scope;
	requestedAccessMode;
	accessMode;
	directoryPromise;
	constructor(scope, accessMode = "async") {
		this.scope = scope;
		this.requestedAccessMode = accessMode;
		this.accessMode = OPFSStore.resolveAccessMode(accessMode);
	}
	static isAvailable() {
		const storage = OPFSStore.getStorageManager();
		return storage !== void 0 && typeof storage.getDirectory === "function";
	}
	static resolveAccessMode(accessMode) {
		if (accessMode !== "auto") return accessMode;
		return OPFSStore.isDedicatedWorkerWithSyncAccessHandle() ? "sync" : "async";
	}
	async has(url) {
		try {
			const entry = await this.getStoredEntry(url);
			if (entry === void 0) return false;
			return this.hasExpectedPayloadSize(entry);
		} catch (err) {
			if (this.handleCacheMissStateError(err)) return false;
			throw err;
		}
	}
	async read(url) {
		try {
			const entry = await this.getStoredEntry(url);
			if (entry === void 0) return;
			const blob = await entry.payloadHandle.getFile();
			if (blob.size !== entry.record.nbytes) return;
			return new Response(blob, this.getResponseInit(entry.record));
		} catch (err) {
			if (this.handleCacheMissStateError(err)) return;
			throw err;
		}
	}
	async readArrayBuffer(url) {
		try {
			const entry = await this.getStoredEntry(url);
			if (entry === void 0) return;
			const payload = await this.readPayload(entry.payloadHandle);
			return payload.byteLength === entry.record.nbytes ? payload : void 0;
		} catch (err) {
			if (this.handleCacheMissStateError(err)) return;
			throw err;
		}
	}
	async write(url, response) {
		try {
			const directory = await this.getScopedDirectory();
			const baseName = await this.hashUrl(url);
			await this.removeEntryIfExists(directory, this.getRecordFilename(baseName));
			const payloadHandle = await directory.getFileHandle(this.getPayloadFilename(baseName), { create: true });
			const nbytes = await this.writePayload(payloadHandle, response);
			const recordHandle = await directory.getFileHandle(this.getRecordFilename(baseName), { create: true });
			const record = {
				url,
				nbytes,
				contentType: response.headers.get("content-type") ?? void 0
			};
			await this.writeRecord(recordHandle, record);
		} catch (err) {
			this.resetDirectoryOnInvalidStateError(err);
			throw err;
		}
	}
	async remove(url) {
		try {
			const directory = await this.getScopedDirectory();
			const baseName = await this.hashUrl(url);
			await this.removeEntryIfExists(directory, this.getPayloadFilename(baseName));
			await this.removeEntryIfExists(directory, this.getRecordFilename(baseName));
		} catch (err) {
			this.resetDirectoryOnInvalidStateError(err);
			throw err;
		}
	}
	async getStoredEntry(url) {
		const directory = await this.getScopedDirectory();
		const baseName = await this.hashUrl(url);
		const recordHandle = await this.getFileHandleIfExists(directory, this.getRecordFilename(baseName), false);
		if (recordHandle === void 0) return;
		const record = await this.readRecord(recordHandle);
		if (record === void 0 || record.url !== url) return;
		const payloadHandle = await this.getFileHandleIfExists(directory, this.getPayloadFilename(baseName), false);
		return payloadHandle === void 0 ? void 0 : {
			payloadHandle,
			record
		};
	}
	static getStorageManager() {
		if (typeof navigator === "undefined") return;
		return navigator.storage;
	}
	static isDedicatedWorkerWithSyncAccessHandle() {
		const scope = globalThis;
		return typeof scope.DedicatedWorkerGlobalScope === "function" && globalThis instanceof scope.DedicatedWorkerGlobalScope && typeof scope.FileSystemFileHandle?.prototype?.createSyncAccessHandle === "function";
	}
	async getScopedDirectory() {
		if (this.directoryPromise !== void 0) return this.directoryPromise;
		this.directoryPromise = (async () => {
			const storage = OPFSStore.getStorageManager();
			if (storage === void 0 || typeof storage.getDirectory !== "function") throw new Error("OPFSStore: OPFS API unavailable.");
			let directory = await storage.getDirectory();
			directory = await directory.getDirectoryHandle(OPFS_STORE_ROOT_DIRECTORY, { create: true });
			const scopeParts = this.scope.split("/").filter((part) => part.length > 0);
			for (const part of scopeParts) directory = await directory.getDirectoryHandle(encodeURIComponent(part), { create: true });
			return directory;
		})();
		return this.directoryPromise;
	}
	async readRecord(fileHandle) {
		try {
			const text = await (await fileHandle.getFile()).text();
			const parsed = JSON.parse(text);
			if (parsed === void 0 || parsed === null || typeof parsed !== "object" || typeof parsed.url !== "string" || !Number.isSafeInteger(parsed.nbytes) || parsed.nbytes < 0) return;
			const record = {
				url: parsed.url,
				nbytes: parsed.nbytes
			};
			if (typeof parsed.contentType === "string") record.contentType = parsed.contentType;
			return record;
		} catch (err) {
			if (OPFSStore.getErrorName(err) === "SyntaxError" || this.handleCacheMissStateError(err)) return;
			throw err;
		}
	}
	getResponseInit(record) {
		return record.contentType !== void 0 ? { headers: { "content-type": record.contentType } } : void 0;
	}
	async writeRecord(handle, record) {
		const writable = await handle.createWritable();
		try {
			await writable.write(new TextEncoder().encode(JSON.stringify(record)));
			await writable.close();
		} catch (err) {
			try {
				await writable.abort();
			} catch {}
			throw err;
		}
	}
	async readPayload(handle) {
		const syncHandle = await this.openSyncAccessHandle(handle, "read-only");
		return syncHandle !== void 0 ? this.readPayloadWithSyncHandle(syncHandle) : (await handle.getFile()).arrayBuffer();
	}
	async hasExpectedPayloadSize(entry) {
		if (this.accessMode === "sync") {
			const syncHandle = await this.openSyncAccessHandle(entry.payloadHandle, "read-only");
			if (syncHandle !== void 0) try {
				return syncHandle.getSize() === entry.record.nbytes;
			} finally {
				syncHandle.close();
			}
		}
		return (await entry.payloadHandle.getFile()).size === entry.record.nbytes;
	}
	async writePayload(handle, response) {
		const syncHandle = await this.openSyncAccessHandle(handle, "readwrite");
		if (syncHandle !== void 0) return this.writePayloadWithSyncHandle(syncHandle, response);
		return this.writePayloadWithWritable(handle, response);
	}
	async writePayloadWithWritable(handle, response) {
		const writable = await handle.createWritable();
		try {
			if (response.body !== null) {
				let nbytes = 0;
				const reader = response.body.getReader();
				try {
					while (true) {
						const { done, value } = await reader.read();
						if (done) break;
						await writable.write(value);
						nbytes += value.byteLength;
					}
				} finally {
					reader.releaseLock();
				}
				await writable.close();
				return nbytes;
			}
			const payload = await response.arrayBuffer();
			await writable.write(payload);
			await writable.close();
			return payload.byteLength;
		} catch (err) {
			try {
				await writable.abort();
			} catch {}
			throw err;
		}
	}
	readPayloadWithSyncHandle(syncHandle) {
		try {
			const size = syncHandle.getSize();
			const payload = new ArrayBuffer(size);
			syncHandle.read(new Uint8Array(payload), { at: 0 });
			return payload;
		} finally {
			syncHandle.close();
		}
	}
	async writePayloadWithSyncHandle(syncHandle, response) {
		try {
			syncHandle.truncate(0);
			let offset = 0;
			if (response.body !== null) {
				const reader = response.body.getReader();
				try {
					while (true) {
						const { done, value } = await reader.read();
						if (done) break;
						syncHandle.write(value, { at: offset });
						offset += value.byteLength;
					}
				} finally {
					reader.releaseLock();
				}
			} else {
				const payload = await response.arrayBuffer();
				syncHandle.write(new Uint8Array(payload), { at: 0 });
				offset = payload.byteLength;
			}
			syncHandle.flush();
			return offset;
		} finally {
			syncHandle.close();
		}
	}
	async openSyncAccessHandle(handle, mode) {
		if (this.accessMode === "async") return;
		if (typeof handle.createSyncAccessHandle !== "function") throw this.createSyncUnavailableError();
		try {
			return await handle.createSyncAccessHandle({ mode });
		} catch (err) {
			const isLockContention = OPFSStore.getErrorName(err) === "NoModificationAllowedError";
			if (this.requestedAccessMode === "auto" && isLockContention) return;
			throw err;
		}
	}
	async getFileHandleIfExists(directory, filename, create) {
		try {
			return await directory.getFileHandle(filename, { create });
		} catch (err) {
			if (OPFSStore.isNotFoundError(err)) return;
			throw err;
		}
	}
	async removeEntryIfExists(directory, filename) {
		try {
			await directory.removeEntry(filename);
		} catch (err) {
			if (OPFSStore.isNotFoundError(err)) return;
			throw err;
		}
	}
	async hashUrl(url) {
		const input = new TextEncoder().encode(url);
		if (typeof crypto === "undefined" || crypto.subtle === void 0 || typeof crypto.subtle.digest !== "function") throw new Error("OPFSStore: crypto.subtle.digest is unavailable.");
		const digest = await crypto.subtle.digest(HASH_ALGORITHM$1, input);
		return Array.from(new Uint8Array(digest)).map((byte) => byte.toString(16).padStart(2, "0")).join("");
	}
	static isNotFoundError(err) {
		return OPFSStore.getErrorName(err) === "NotFoundError";
	}
	static isCacheMissStateError(err) {
		const name = OPFSStore.getErrorName(err);
		return name === "NotFoundError" || name === "InvalidStateError";
	}
	handleCacheMissStateError(err) {
		if (!OPFSStore.isCacheMissStateError(err)) return false;
		this.resetDirectoryOnInvalidStateError(err);
		return true;
	}
	resetDirectoryOnInvalidStateError(err) {
		if (OPFSStore.getErrorName(err) === "InvalidStateError") this.directoryPromise = void 0;
	}
	static getErrorName(err) {
		if (err && typeof err === "object" && "name" in err) {
			const name = err.name;
			return typeof name === "string" ? name : void 0;
		}
	}
	getPayloadFilename(baseName) {
		return `${baseName}.bin`;
	}
	getRecordFilename(baseName) {
		return `${baseName}.record.json`;
	}
	createSyncUnavailableError() {
		const err = /* @__PURE__ */ new Error("OPFSStore: createSyncAccessHandle unavailable; sync OPFS access requires a supported dedicated worker context.");
		err.name = "NotSupportedError";
		return err;
	}
};
// [TVM]/artifact_cache.ts
/**
* Return a borrowed view of one tensor record within a shard.
*
* The returned view aliases the shard and is only valid for as long as the
* shard data remains alive. Tensor-cache decoding consumes it synchronously.
*/
function getTensorCacheRecordBytes(shardData, record) {
	const shardBytes = shardData instanceof Uint8Array ? shardData : new Uint8Array(shardData);
	const { byteOffset, nbytes } = record;
	if (!Number.isSafeInteger(byteOffset) || byteOffset < 0) throw new Error(`Invalid tensor-cache byteOffset: ${byteOffset}`);
	if (!Number.isSafeInteger(nbytes) || nbytes < 0) throw new Error(`Invalid tensor-cache nbytes: ${nbytes}`);
	const endOffset = byteOffset + nbytes;
	if (!Number.isSafeInteger(endOffset) || endOffset > shardBytes.byteLength) throw new Error(`Tensor-cache record range [${byteOffset}, ${endOffset}) exceeds shard size ${shardBytes.byteLength}`);
	return shardBytes.subarray(byteOffset, endOffset);
}
var HASH_ALGORITHM = "SHA-256";
var DEFAULT_FETCH_OPTIONS = { method: "GET" };
var COS_HASH_META_CACHE = "tvmjs-cos-hash-meta";
var crossOriginFallbackWarningLogged = false;
var GLOBAL_HASH_CACHE = /* @__PURE__ */ new Map();
var CrossOriginStorage = class CrossOriginStorage {
	hashCache;
	constructor() {
		this.hashCache = GLOBAL_HASH_CACHE;
	}
	static isAvailable() {
		if (typeof navigator === "undefined") return false;
		return navigator.crossOriginStorage !== void 0;
	}
	async match(request) {
		const url = this.normalizeRequest(request);
		const hash = await this.resolveHashDescriptor(url);
		if (!hash) return;
		try {
			const api = this.getApi();
			if (!api) return;
			const handle = await api.requestFileHandle(hash);
			if (!handle) return;
			const blob = await handle.getFile();
			return new Response(blob);
		} catch {
			return;
		}
	}
	async put(request, response) {
		const url = this.normalizeRequest(request);
		const blob = await response.blob();
		const hash = await this.getBlobHash(blob);
		const api = this.getApi();
		if (!api) throw new Error("Cross-origin storage API unavailable.");
		const handle = await api.requestFileHandle(hash, {
			create: true,
			origins: "*"
		});
		if (!handle) throw new Error("Cross-origin storage API returned no handle.");
		const writableStream = await handle.createWritable();
		await writableStream.write(blob);
		await writableStream.close();
		this.hashCache.set(url, hash);
		await this.persistHashEntry(url, hash);
	}
	async delete(_request) {}
	getApi() {
		if (!CrossOriginStorage.isAvailable()) return;
		return navigator.crossOriginStorage;
	}
	normalizeRequest(request) {
		if (typeof request === "string") return request;
		if (request instanceof URL) return request.href;
		if (request instanceof Request) return request.url;
		if (request && typeof request.url === "string") return request.url;
		throw new Error("CrossOriginStorage: Unsupported request type.");
	}
	async persistHashEntry(url, hash) {
		try {
			if (typeof caches === "undefined") return;
			await (await caches.open(COS_HASH_META_CACHE)).put(url, new Response(JSON.stringify(hash)));
		} catch {}
	}
	async loadPersistedHashEntry(url) {
		try {
			if (typeof caches === "undefined") return null;
			const response = await (await caches.open(COS_HASH_META_CACHE)).match(url);
			if (!response) return null;
			return JSON.parse(await response.text());
		} catch {
			return null;
		}
	}
	async resolveHashDescriptor(url) {
		const cached = this.hashCache.get(url);
		if (cached) return cached;
		const persisted = await this.loadPersistedHashEntry(url);
		if (persisted) {
			this.hashCache.set(url, persisted);
			return persisted;
		}
		const hashValue = await this.getFileHash(url);
		if (!hashValue) return null;
		const descriptor = {
			algorithm: HASH_ALGORITHM,
			value: hashValue
		};
		this.hashCache.set(url, descriptor);
		await this.persistHashEntry(url, descriptor);
		return descriptor;
	}
	async getFileHash(url) {
		if (/\/resolve\//.test(url)) {
			const pointerHash = await this.extractHashFromPointer(url);
			if (pointerHash) return pointerHash;
		}
		return null;
	}
	async extractHashFromPointer(url) {
		const rawUrl = url.replace(/\/resolve\//, "/raw/");
		try {
			const text = await fetch(rawUrl).then((res) => res.text());
			if (!text.includes("oid sha256:")) return null;
			const match = text.match(/oid sha256:([A-Fa-f0-9]+)/);
			return match ? match[1] : null;
		} catch {
			return null;
		}
	}
	async getBlobHash(blob) {
		const arrayBuffer = await blob.arrayBuffer();
		const hashBuffer = await crypto.subtle.digest(HASH_ALGORITHM, arrayBuffer);
		return {
			algorithm: HASH_ALGORITHM,
			value: Array.from(new Uint8Array(hashBuffer)).map((byte) => byte.toString(16).padStart(2, "0")).join("")
		};
	}
};
/**
* Cache to store model related data, implemented with the Cache API.
*/
var ArtifactCache = class {
	scope;
	cache;
	constructor(scope) {
		this.scope = scope;
	}
	/**
	* Convert the Response object to the expected storetype instead
	*/
	async responseTostoretype(response, storetype) {
		if (storetype === void 0) return response;
		else if (storetype.toLowerCase() === "json") return await response.json();
		else if (storetype.toLowerCase() === "arraybuffer") return await response.arrayBuffer();
		else {
			console.error("Unknown storage type " + storetype + ", returning raw response");
			return response;
		}
	}
	/**
	* fetch the corresponding url object in response or stored object format
	* @param url url
	* @param storetype the storage type for indexedDB
	* @param signal an optional abort signal to abort fetching
	* @returns response in json, arraybuffer or pure response format
	*/
	async fetchWithCache(url, storetype, signal) {
		await this.addToCache(url, storetype, signal);
		const result = await this.cache.match(new Request(url));
		if (result === void 0) throw Error("Cannot fetch " + url);
		return await this.responseTostoretype(result, storetype);
	}
	async addToCache(url, storetype, signal) {
		const request = new Request(url, signal ? { signal } : void 0);
		if (this.cache === void 0) this.cache = await caches.open(this.scope);
		if (await this.cache.match(request) === void 0) await this.cache.add(request);
	}
	/**
	* Determine if all keys exist in the cache
	* @param keys the url key list of the strings
	* @returns boolean value indicate if all keys are in cache
	*/
	async hasAllKeys(keys) {
		if (this.cache === void 0) this.cache = await caches.open(this.scope);
		return this.cache.keys().then((requests) => requests.map((request) => request.url)).then((cacheKeys) => keys.every((key) => cacheKeys.indexOf(key) !== -1)).catch(() => false);
	}
	/**
	* Delete the corresponding url object in cache
	* @param url the corresponding url object to be deleted
	*/
	async deleteInCache(url) {
		if (this.cache === void 0) this.cache = await caches.open(this.scope);
		await this.cache.delete(url);
	}
};
/**
* Cache by IndexedDB to support caching model data
*/
var ArtifactIndexedDBCache = class {
	dbName;
	dbVersion = 1;
	db;
	constructor(dbName) {
		this.dbName = dbName;
	}
	/**
	* Init the indexed DB database if it is not initialized.
	*/
	async initDB() {
		if (this.db != null) return;
		return new Promise((resolve, reject) => {
			const request = indexedDB.open(this.dbName, this.dbVersion);
			request.onupgradeneeded = (event) => {
				this.db = event.target.result;
				if (!this.db.objectStoreNames.contains("urls")) this.db.createObjectStore("urls", { keyPath: "url" });
			};
			request.onsuccess = (event) => {
				this.db = event.target.result;
				resolve();
			};
			request.onerror = (event) => {
				console.error("Database error: ", event.target.error);
				reject(event.target.error);
			};
		});
	}
	/**
	* Check if current url object is in indexedDB or not
	* @param url the url link
	* @returns boolean indicate if url object in indexedDB
	*/
	async isUrlInDB(url) {
		return new Promise((resolve, reject) => {
			const transaction = this.db?.transaction(["urls"], "readonly");
			if (transaction === void 0) return false;
			const request = transaction.objectStore("urls").get(url);
			request.onsuccess = () => {
				resolve(request.result !== void 0);
			};
			request.onerror = (event) => {
				reject(event.target.error);
			};
		});
	}
	async asyncGetHelper(url) {
		return new Promise((resolve, reject) => {
			let result;
			const transaction = this.db?.transaction(["urls"], "readonly");
			if (transaction === void 0) return false;
			transaction.oncomplete = () => resolve(result);
			transaction.onerror = () => reject(transaction.error);
			const getRequest = transaction.objectStore("urls").get(url);
			getRequest.onsuccess = () => {
				result = getRequest.result;
			};
		});
	}
	async fetchWithCache(url, storetype, signal) {
		await this.addToCache(url, storetype, signal);
		let result = await this.asyncGetHelper(url);
		if (result === null) {
			await this.deleteInCache(url);
			await this.addToCache(url, storetype);
			result = await this.asyncGetHelper(url);
		}
		if (result != null && typeof result === "object" && "data" in result) return result.data;
		throw Error("ArtifactIndexedDBCache failed to fetch: " + url);
	}
	async addToIndexedDB(url, response, storetype) {
		await this.initDB();
		let data;
		if (storetype != void 0) {
			if (storetype.toLowerCase() === "json") data = await response.json();
			else if (storetype.toLocaleLowerCase() === "arraybuffer") data = await response.arrayBuffer();
			else throw Error("Unsupported storetyp for IndexedDB: " + storetype);
		}
		return new Promise((resolve, reject) => {
			const transaction = this.db?.transaction(["urls"], "readwrite");
			if (transaction === void 0) return;
			const request = transaction.objectStore("urls").add({
				data,
				url
			});
			request.onsuccess = () => resolve();
			request.onerror = (event) => reject(event.target.error);
		});
	}
	async addToCache(url, storetype, signal) {
		await this.initDB();
		if (await this.isUrlInDB(url)) return;
		try {
			const response = await fetch(url, signal ? { signal } : void 0);
			if (!response.ok) throw new Error("Network response was not ok");
			const response_copy = response.clone();
			await this.addToIndexedDB(url, response_copy, storetype);
		} catch (error) {
			throw Error("Failed to store " + url + " with error: " + error);
		}
	}
	async hasAllKeys(keys) {
		await this.initDB();
		if (!this.db) throw new Error("Database is not initialized");
		return new Promise((resolve, reject) => {
			const store = this.db.transaction(["urls"], "readonly").objectStore("urls");
			const promises = keys.map((key) => {
				return new Promise((resolve) => {
					const request = store.get(key);
					request.onsuccess = () => {
						if (request.result === void 0) resolve(false);
						else resolve(true);
					};
					request.onerror = () => {
						resolve(false);
					};
				});
			});
			Promise.all(promises).then((results) => {
				resolve(results.every((exists) => exists));
			}).catch((error) => {
				reject(error);
			});
		});
	}
	async deleteInCache(url) {
		await this.initDB();
		const transaction = this.db?.transaction(["urls"], "readwrite");
		if (transaction === void 0) return;
		const request = transaction.objectStore("urls").delete(url);
		await new Promise((resolve, reject) => {
			request.onsuccess = () => resolve();
			request.onerror = () => reject(request.error);
		});
	}
};
/**
* Cache by Origin Private File System (OPFS).
*/
var ArtifactOPFSCache = class {
	store;
	constructor(scope, accessMode = "async") {
		this.store = new OPFSStore(scope, accessMode);
	}
	static isAvailable() {
		return OPFSStore.isAvailable();
	}
	async fetchWithCache(url, storetype, signal) {
		const cached = await this.readFromCache(url, storetype);
		if (cached !== void 0) return cached;
		await this.addToCache(url, storetype, signal);
		const fetched = await this.readFromCache(url, storetype);
		if (fetched === void 0) throw new Error("ArtifactOPFSCache failed to fetch: " + url);
		return fetched;
	}
	async readFromCache(url, storetype) {
		if (storetype?.toLowerCase() === "arraybuffer") return this.store.readArrayBuffer(url);
		const cachedResponse = await this.store.read(url);
		if (cachedResponse === void 0) return;
		return this.responseToStoreType(cachedResponse, storetype);
	}
	async addToCache(url, _storetype, signal) {
		if (await this.store.has(url)) return;
		const request = new Request(url, signal ? {
			...DEFAULT_FETCH_OPTIONS,
			signal
		} : DEFAULT_FETCH_OPTIONS);
		const response = await fetch(request);
		if (!response.ok) throw new Error(`ArtifactOPFSCache: Unable to fetch ${url}, received status ${response.status}`);
		await this.store.write(url, response);
	}
	async hasAllKeys(keys) {
		return (await Promise.all(keys.map(async (key) => await this.store.has(key)))).every((result) => result);
	}
	async deleteInCache(url) {
		await this.store.remove(url);
	}
	async responseToStoreType(response, storetype) {
		if (storetype === void 0) return response;
		const format = storetype.toLowerCase();
		if (format === "json") return response.json();
		if (format === "arraybuffer") return response.arrayBuffer();
		return response;
	}
};
/**
* Cache by cross-origin storage extension.
*/
var ArtifactCrossOriginStorageCache = class {
	storage;
	constructor(_scope, storage = new CrossOriginStorage()) {
		this.storage = storage;
	}
	async fetchWithCache(url, storetype, signal) {
		const cachedResponse = await this.storage.match(url);
		if (cachedResponse !== void 0) return this.responseToStoreType(cachedResponse, storetype);
		await this.addToCache(url, storetype, signal);
		const hydrated = await this.storage.match(url);
		if (hydrated === void 0) throw new Error(`ArtifactCrossOriginStorageCache: failed to hydrate ${url}`);
		return this.responseToStoreType(hydrated, storetype);
	}
	async addToCache(url, _storetype, signal) {
		if (await this.storage.match(url) !== void 0) return;
		const request = new Request(url, signal ? {
			...DEFAULT_FETCH_OPTIONS,
			signal
		} : DEFAULT_FETCH_OPTIONS);
		const response = await fetch(request);
		if (!response.ok) throw new Error(`ArtifactCrossOriginStorageCache: Unable to fetch ${url}, received status ${response.status}`);
		await this.storage.put(url, response.clone());
	}
	async hasAllKeys(keys) {
		return (await Promise.all(keys.map(async (key) => {
			return await this.storage.match(key) !== void 0;
		}))).every((result) => result);
	}
	async deleteInCache(url) {
		await this.storage.delete(url);
	}
	async responseToStoreType(response, storetype) {
		if (storetype === void 0) return response;
		const format = storetype.toLowerCase();
		if (format === "json") return response.json();
		if (format === "arraybuffer") return response.arrayBuffer();
		return response;
	}
};
function normalizeCacheType(cacheType) {
	if (cacheType === void 0) return "cache";
	const normalized = cacheType.toLowerCase();
	if (normalized === "cache") return "cache";
	if (normalized === "indexeddb") return "indexeddb";
	if (normalized === "cross-origin") return "cross-origin";
	if (normalized === "opfs") return "opfs";
	console.error("Unsupported cacheType: " + cacheType + ", using default ArtifactCache.");
	return "cache";
}
function createArtifactCache(scope, options = {}) {
	if (options.artifactCache !== void 0) return options.artifactCache;
	const cacheType = normalizeCacheType(options.cacheType);
	if (cacheType === "indexeddb") return new ArtifactIndexedDBCache(scope);
	if (cacheType === "cross-origin") {
		if (CrossOriginStorage.isAvailable()) return new ArtifactCrossOriginStorageCache(scope);
		if (!crossOriginFallbackWarningLogged) {
			console.warn("Cross-origin storage backend is unavailable; falling back to ArtifactCache.");
			crossOriginFallbackWarningLogged = true;
		}
	}
	if (cacheType === "opfs") return new ArtifactOPFSCache(scope, options.opfsAccessMode);
	return new ArtifactCache(scope);
}
// __vite-browser-external
var require___vite_browser_external = /* @__PURE__ */ __commonJSMin(((exports, module) => {
	module.exports = {};
}));
// [TVM]/tvmjs_runtime_wasi.js
function EmccWASI() {
	var Module = typeof Module != "undefined" ? Module : {};
	var ENVIRONMENT_IS_WEB = !!globalThis.window;
	var ENVIRONMENT_IS_WORKER = !!globalThis.WorkerGlobalScope;
	var ENVIRONMENT_IS_NODE = globalThis.process?.versions?.node && globalThis.process?.type != "renderer";
	var __wasmLib = {};
	function __wasmLibInstantiateWasm(imports, successCallback) {
		__wasmLib.imports = imports;
		__wasmLib.successCallback = successCallback;
	}
	function __wasmLibStart(wasmInstance) {
		__wasmLib.successCallback(wasmInstance);
	}
	__wasmLib.start = __wasmLibStart;
	var Module = {
		instantiateWasm: __wasmLibInstantiateWasm,
		wasmLibraryProvider: __wasmLib
	};
	var arguments_ = [];
	var thisProgram = "./this.program";
	var quit_ = (status, toThrow) => {
		throw toThrow;
	};
	var _scriptName = globalThis.document?.currentScript?.src;
	if (ENVIRONMENT_IS_WORKER) _scriptName = self.location.href;
	var scriptDirectory = "";
	function locateFile(path) {
		if (Module["locateFile"]) return Module["locateFile"](path, scriptDirectory);
		return scriptDirectory + path;
	}
	var readAsync, readBinary;
	if (ENVIRONMENT_IS_NODE) {
		var fs = require___vite_browser_external();
		scriptDirectory = "undefined/";
		readBinary = (filename) => {
			filename = isFileURI(filename) ? new URL(filename) : filename;
			return fs.readFileSync(filename);
		};
		readAsync = async (filename, binary = true) => {
			filename = isFileURI(filename) ? new URL(filename) : filename;
			return fs.readFileSync(filename, binary ? void 0 : "utf8");
		};
		if (process.argv.length > 1) thisProgram = process.argv[1].replace(/\\/g, "/");
		arguments_ = process.argv.slice(2);
		quit_ = (status, toThrow) => {
			process.exitCode = status;
			throw toThrow;
		};
	} else if (ENVIRONMENT_IS_WEB || ENVIRONMENT_IS_WORKER) {
		try {
			scriptDirectory = new URL(".", _scriptName).href;
		} catch {}
		if (ENVIRONMENT_IS_WORKER) readBinary = (url) => {
			var xhr = new XMLHttpRequest();
			xhr.open("GET", url, false);
			xhr.responseType = "arraybuffer";
			xhr.send(null);
			return new Uint8Array(xhr.response);
		};
		readAsync = async (url) => {
			if (isFileURI(url)) return new Promise((resolve, reject) => {
				var xhr = new XMLHttpRequest();
				xhr.open("GET", url, true);
				xhr.responseType = "arraybuffer";
				xhr.onload = () => {
					if (xhr.status == 200 || xhr.status == 0 && xhr.response) {
						resolve(xhr.response);
						return;
					}
					reject(xhr.status);
				};
				xhr.onerror = reject;
				xhr.send(null);
			});
			var response = await fetch(url, { credentials: "same-origin" });
			if (response.ok) return response.arrayBuffer();
			throw new Error(response.status + " : " + response.url);
		};
	}
	var out = console.log.bind(console);
	var err = console.error.bind(console);
	var wasmBinary;
	var ABORT = false;
	var EXITSTATUS;
	var isFileURI = (filename) => filename.startsWith("file://"), HEAP8, HEAPU8, HEAP32, HEAPU32, HEAP64;
	function updateMemoryViews() {
		var b = wasmMemory.buffer;
		HEAP8 = new Int8Array(b);
		new Int16Array(b);
		HEAPU8 = new Uint8Array(b);
		new Uint16Array(b);
		HEAP32 = new Int32Array(b);
		HEAPU32 = new Uint32Array(b);
		new Float32Array(b);
		new Float64Array(b);
		HEAP64 = new BigInt64Array(b);
		new BigUint64Array(b);
	}
	function preRun() {
		if (Module["preRun"]) {
			if (typeof Module["preRun"] == "function") Module["preRun"] = [Module["preRun"]];
			while (Module["preRun"].length) addOnPreRun(Module["preRun"].shift());
		}
		callRuntimeCallbacks(onPreRuns);
	}
	function initRuntime() {
		if (!Module["noFSInit"] && !FS.initialized) FS.init();
		TTY.init();
		FS.ignorePermissions = false;
	}
	function postRun() {
		if (Module["postRun"]) {
			if (typeof Module["postRun"] == "function") Module["postRun"] = [Module["postRun"]];
			while (Module["postRun"].length) addOnPostRun(Module["postRun"].shift());
		}
		callRuntimeCallbacks(onPostRuns);
	}
	function abort(what) {
		Module["onAbort"]?.(what);
		what = "Aborted(" + what + ")";
		err(what);
		ABORT = true;
		what += ". Build with -sASSERTIONS for more info.";
		throw new WebAssembly.RuntimeError(what);
	}
	var wasmBinaryFile;
	function findWasmBinary() {
		return locateFile("tvmjs_runtime.wasm");
	}
	function getBinarySync(file) {
		if (file == wasmBinaryFile && wasmBinary) return new Uint8Array(wasmBinary);
		if (readBinary) return readBinary(file);
		throw "both async and sync fetching of the wasm failed";
	}
	async function getWasmBinary(binaryFile) {
		if (!wasmBinary) try {
			var response = await readAsync(binaryFile);
			return new Uint8Array(response);
		} catch {}
		return getBinarySync(binaryFile);
	}
	async function instantiateArrayBuffer(binaryFile, imports) {
		try {
			var binary = await getWasmBinary(binaryFile);
			return await WebAssembly.instantiate(binary, imports);
		} catch (reason) {
			err(`failed to asynchronously prepare wasm: ${reason}`);
			abort(reason);
		}
	}
	async function instantiateAsync(binary, binaryFile, imports) {
		if (!binary && !isFileURI(binaryFile) && !ENVIRONMENT_IS_NODE) try {
			var response = fetch(binaryFile, { credentials: "same-origin" });
			return await WebAssembly.instantiateStreaming(response, imports);
		} catch (reason) {
			err(`wasm streaming compile failed: ${reason}`);
			err("falling back to ArrayBuffer instantiation");
		}
		return instantiateArrayBuffer(binaryFile, imports);
	}
	function getWasmImports() {
		return {
			env: wasmImports,
			wasi_snapshot_preview1: wasmImports
		};
	}
	async function createWasm() {
		function receiveInstance(instance, module) {
			wasmExports = instance.exports;
			wasmExports = Asyncify.instrumentWasmExports(wasmExports);
			assignWasmExports(wasmExports);
			updateMemoryViews();
			removeRunDependency("wasm-instantiate");
			return wasmExports;
		}
		addRunDependency("wasm-instantiate");
		function receiveInstantiationResult(result) {
			return receiveInstance(result["instance"]);
		}
		var info = getWasmImports();
		if (Module["instantiateWasm"]) return new Promise((resolve, reject) => {
			Module["instantiateWasm"](info, (inst, mod) => {
				resolve(receiveInstance(inst, mod));
			});
		});
		wasmBinaryFile ??= findWasmBinary();
		return receiveInstantiationResult(await instantiateAsync(wasmBinary, wasmBinaryFile, info));
	}
	class ExitStatus {
		name = "ExitStatus";
		constructor(status) {
			this.message = `Program terminated with exit(${status})`;
			this.status = status;
		}
	}
	var callRuntimeCallbacks = (callbacks) => {
		while (callbacks.length > 0) callbacks.shift()(Module);
	};
	var onPostRuns = [];
	var addOnPostRun = (cb) => onPostRuns.push(cb);
	var onPreRuns = [];
	var addOnPreRun = (cb) => onPreRuns.push(cb);
	var runDependencies = 0;
	var dependenciesFulfilled = null;
	var removeRunDependency = (id) => {
		runDependencies--;
		Module["monitorRunDependencies"]?.(runDependencies);
		if (runDependencies == 0) {
			if (dependenciesFulfilled) {
				var callback = dependenciesFulfilled;
				dependenciesFulfilled = null;
				callback();
			}
		}
	};
	var addRunDependency = (id) => {
		runDependencies++;
		Module["monitorRunDependencies"]?.(runDependencies);
	};
	var noExitRuntime = true;
	function _TVMFFIWasmFunctionDeleter(...args) {
		abort("missing function: TVMFFIWasmFunctionDeleter");
	}
	_TVMFFIWasmFunctionDeleter.stub = true;
	function _TVMFFIWasmSafeCall(...args) {
		abort("missing function: TVMFFIWasmSafeCall");
	}
	_TVMFFIWasmSafeCall.stub = true;
	var _emscripten_get_now = () => performance.now();
	var _emscripten_date_now = () => Date.now();
	var nowIsMonotonic = 1;
	var checkWasiClock = (clock_id) => clock_id >= 0 && clock_id <= 3;
	var INT53_MAX = 9007199254740992;
	var INT53_MIN = -9007199254740992;
	var bigintToI53Checked = (num) => num < INT53_MIN || num > INT53_MAX ? NaN : Number(num);
	function _clock_time_get(clk_id, ignored_precision, ptime) {
		ignored_precision = bigintToI53Checked(ignored_precision);
		if (!checkWasiClock(clk_id)) return 28;
		var now;
		if (clk_id === 0) now = _emscripten_date_now();
		else if (nowIsMonotonic) now = _emscripten_get_now();
		else return 52;
		var nsec = Math.round(now * 1e3 * 1e3);
		HEAP64[ptime >> 3] = BigInt(nsec);
		return 0;
	}
	var _emscripten_notify_memory_growth = (memoryIndex) => {
		updateMemoryViews();
	};
	var ENV = {};
	var getExecutableName = () => thisProgram || "./this.program";
	var getEnvStrings = () => {
		if (!getEnvStrings.strings) {
			var env = {
				USER: "web_user",
				LOGNAME: "web_user",
				PATH: "/",
				PWD: "/",
				HOME: "/home/web_user",
				LANG: (globalThis.navigator?.language ?? "C").replace("-", "_") + ".UTF-8",
				_: getExecutableName()
			};
			for (var x in ENV) if (ENV[x] === void 0) delete env[x];
			else env[x] = ENV[x];
			var strings = [];
			for (var x in env) strings.push(`${x}=${env[x]}`);
			getEnvStrings.strings = strings;
		}
		return getEnvStrings.strings;
	};
	var stringToUTF8Array = (str, heap, outIdx, maxBytesToWrite) => {
		if (!(maxBytesToWrite > 0)) return 0;
		var startIdx = outIdx;
		var endIdx = outIdx + maxBytesToWrite - 1;
		for (var i = 0; i < str.length; ++i) {
			var u = str.codePointAt(i);
			if (u <= 127) {
				if (outIdx >= endIdx) break;
				heap[outIdx++] = u;
			} else if (u <= 2047) {
				if (outIdx + 1 >= endIdx) break;
				heap[outIdx++] = 192 | u >> 6;
				heap[outIdx++] = 128 | u & 63;
			} else if (u <= 65535) {
				if (outIdx + 2 >= endIdx) break;
				heap[outIdx++] = 224 | u >> 12;
				heap[outIdx++] = 128 | u >> 6 & 63;
				heap[outIdx++] = 128 | u & 63;
			} else {
				if (outIdx + 3 >= endIdx) break;
				heap[outIdx++] = 240 | u >> 18;
				heap[outIdx++] = 128 | u >> 12 & 63;
				heap[outIdx++] = 128 | u >> 6 & 63;
				heap[outIdx++] = 128 | u & 63;
				i++;
			}
		}
		heap[outIdx] = 0;
		return outIdx - startIdx;
	};
	var stringToUTF8 = (str, outPtr, maxBytesToWrite) => stringToUTF8Array(str, HEAPU8, outPtr, maxBytesToWrite);
	var _environ_get = (__environ, environ_buf) => {
		var bufSize = 0;
		var envp = 0;
		for (var string of getEnvStrings()) {
			var ptr = environ_buf + bufSize;
			HEAPU32[__environ + envp >> 2] = ptr;
			bufSize += stringToUTF8(string, ptr, Infinity) + 1;
			envp += 4;
		}
		return 0;
	};
	var lengthBytesUTF8 = (str) => {
		var len = 0;
		for (var i = 0; i < str.length; ++i) {
			var c = str.charCodeAt(i);
			if (c <= 127) len++;
			else if (c <= 2047) len += 2;
			else if (c >= 55296 && c <= 57343) {
				len += 4;
				++i;
			} else len += 3;
		}
		return len;
	};
	var _environ_sizes_get = (penviron_count, penviron_buf_size) => {
		var strings = getEnvStrings();
		HEAPU32[penviron_count >> 2] = strings.length;
		var bufSize = 0;
		for (var string of strings) bufSize += lengthBytesUTF8(string) + 1;
		HEAPU32[penviron_buf_size >> 2] = bufSize;
		return 0;
	};
	var PATH = {
		isAbs: (path) => path.charAt(0) === "/",
		splitPath: (filename) => {
			return /^(\/?|)([\s\S]*?)((?:\.{1,2}|[^\/]+?|)(\.[^.\/]*|))(?:[\/]*)$/.exec(filename).slice(1);
		},
		normalizeArray: (parts, allowAboveRoot) => {
			var up = 0;
			for (var i = parts.length - 1; i >= 0; i--) {
				var last = parts[i];
				if (last === ".") parts.splice(i, 1);
				else if (last === "..") {
					parts.splice(i, 1);
					up++;
				} else if (up) {
					parts.splice(i, 1);
					up--;
				}
			}
			if (allowAboveRoot) for (; up; up--) parts.unshift("..");
			return parts;
		},
		normalize: (path) => {
			var isAbsolute = PATH.isAbs(path), trailingSlash = path.slice(-1) === "/";
			path = PATH.normalizeArray(path.split("/").filter((p) => !!p), !isAbsolute).join("/");
			if (!path && !isAbsolute) path = ".";
			if (path && trailingSlash) path += "/";
			return (isAbsolute ? "/" : "") + path;
		},
		dirname: (path) => {
			var result = PATH.splitPath(path), root = result[0], dir = result[1];
			if (!root && !dir) return ".";
			if (dir) dir = dir.slice(0, -1);
			return root + dir;
		},
		basename: (path) => path && path.match(/([^\/]+|\/)\/*$/)[1],
		join: (...paths) => PATH.normalize(paths.join("/")),
		join2: (l, r) => PATH.normalize(l + "/" + r)
	};
	var initRandomFill = () => {
		if (ENVIRONMENT_IS_NODE) {
			var nodeCrypto = require___vite_browser_external();
			return (view) => nodeCrypto.randomFillSync(view);
		}
		return (view) => crypto.getRandomValues(view);
	};
	var randomFill = (view) => {
		(randomFill = initRandomFill())(view);
	};
	var PATH_FS = {
		resolve: (...args) => {
			var resolvedPath = "", resolvedAbsolute = false;
			for (var i = args.length - 1; i >= -1 && !resolvedAbsolute; i--) {
				var path = i >= 0 ? args[i] : FS.cwd();
				if (typeof path != "string") throw new TypeError("Arguments to path.resolve must be strings");
				else if (!path) return "";
				resolvedPath = path + "/" + resolvedPath;
				resolvedAbsolute = PATH.isAbs(path);
			}
			resolvedPath = PATH.normalizeArray(resolvedPath.split("/").filter((p) => !!p), !resolvedAbsolute).join("/");
			return (resolvedAbsolute ? "/" : "") + resolvedPath || ".";
		},
		relative: (from, to) => {
			from = PATH_FS.resolve(from).slice(1);
			to = PATH_FS.resolve(to).slice(1);
			function trim(arr) {
				var start = 0;
				for (; start < arr.length; start++) if (arr[start] !== "") break;
				var end = arr.length - 1;
				for (; end >= 0; end--) if (arr[end] !== "") break;
				if (start > end) return [];
				return arr.slice(start, end - start + 1);
			}
			var fromParts = trim(from.split("/"));
			var toParts = trim(to.split("/"));
			var length = Math.min(fromParts.length, toParts.length);
			var samePartsLength = length;
			for (var i = 0; i < length; i++) if (fromParts[i] !== toParts[i]) {
				samePartsLength = i;
				break;
			}
			var outputParts = [];
			for (var i = samePartsLength; i < fromParts.length; i++) outputParts.push("..");
			outputParts = outputParts.concat(toParts.slice(samePartsLength));
			return outputParts.join("/");
		}
	};
	var UTF8Decoder = globalThis.TextDecoder && new TextDecoder();
	var findStringEnd = (heapOrArray, idx, maxBytesToRead, ignoreNul) => {
		var maxIdx = idx + maxBytesToRead;
		if (ignoreNul) return maxIdx;
		while (heapOrArray[idx] && !(idx >= maxIdx)) ++idx;
		return idx;
	};
	var UTF8ArrayToString = (heapOrArray, idx = 0, maxBytesToRead, ignoreNul) => {
		var endPtr = findStringEnd(heapOrArray, idx, maxBytesToRead, ignoreNul);
		if (endPtr - idx > 16 && heapOrArray.buffer && UTF8Decoder) return UTF8Decoder.decode(heapOrArray.subarray(idx, endPtr));
		var str = "";
		while (idx < endPtr) {
			var u0 = heapOrArray[idx++];
			if (!(u0 & 128)) {
				str += String.fromCharCode(u0);
				continue;
			}
			var u1 = heapOrArray[idx++] & 63;
			if ((u0 & 224) == 192) {
				str += String.fromCharCode((u0 & 31) << 6 | u1);
				continue;
			}
			var u2 = heapOrArray[idx++] & 63;
			if ((u0 & 240) == 224) u0 = (u0 & 15) << 12 | u1 << 6 | u2;
			else u0 = (u0 & 7) << 18 | u1 << 12 | u2 << 6 | heapOrArray[idx++] & 63;
			if (u0 < 65536) str += String.fromCharCode(u0);
			else {
				var ch = u0 - 65536;
				str += String.fromCharCode(55296 | ch >> 10, 56320 | ch & 1023);
			}
		}
		return str;
	};
	var FS_stdin_getChar_buffer = [];
	var intArrayFromString = (stringy, dontAddNull, length) => {
		var len = length > 0 ? length : lengthBytesUTF8(stringy) + 1;
		var u8array = new Array(len);
		var numBytesWritten = stringToUTF8Array(stringy, u8array, 0, u8array.length);
		if (dontAddNull) u8array.length = numBytesWritten;
		return u8array;
	};
	var FS_stdin_getChar = () => {
		if (!FS_stdin_getChar_buffer.length) {
			var result = null;
			if (ENVIRONMENT_IS_NODE) {
				var BUFSIZE = 256;
				var buf = Buffer.alloc(BUFSIZE);
				var bytesRead = 0;
				var fd = process.stdin.fd;
				try {
					bytesRead = fs.readSync(fd, buf, 0, BUFSIZE);
				} catch (e) {
					if (e.toString().includes("EOF")) bytesRead = 0;
					else throw e;
				}
				if (bytesRead > 0) result = buf.slice(0, bytesRead).toString("utf-8");
			} else if (globalThis.window?.prompt) {
				result = window.prompt("Input: ");
				if (result !== null) result += "\n";
			}
			if (!result) return null;
			FS_stdin_getChar_buffer = intArrayFromString(result, true);
		}
		return FS_stdin_getChar_buffer.shift();
	};
	var TTY = {
		ttys: [],
		init() {},
		shutdown() {},
		register(dev, ops) {
			TTY.ttys[dev] = {
				input: [],
				output: [],
				ops
			};
			FS.registerDevice(dev, TTY.stream_ops);
		},
		stream_ops: {
			open(stream) {
				var tty = TTY.ttys[stream.node.rdev];
				if (!tty) throw new FS.ErrnoError(43);
				stream.tty = tty;
				stream.seekable = false;
			},
			close(stream) {
				stream.tty.ops.fsync(stream.tty);
			},
			fsync(stream) {
				stream.tty.ops.fsync(stream.tty);
			},
			read(stream, buffer, offset, length, pos) {
				if (!stream.tty || !stream.tty.ops.get_char) throw new FS.ErrnoError(60);
				var bytesRead = 0;
				for (var i = 0; i < length; i++) {
					var result;
					try {
						result = stream.tty.ops.get_char(stream.tty);
					} catch (e) {
						throw new FS.ErrnoError(29);
					}
					if (result === void 0 && bytesRead === 0) throw new FS.ErrnoError(6);
					if (result === null || result === void 0) break;
					bytesRead++;
					buffer[offset + i] = result;
				}
				if (bytesRead) stream.node.atime = Date.now();
				return bytesRead;
			},
			write(stream, buffer, offset, length, pos) {
				if (!stream.tty || !stream.tty.ops.put_char) throw new FS.ErrnoError(60);
				try {
					for (var i = 0; i < length; i++) stream.tty.ops.put_char(stream.tty, buffer[offset + i]);
				} catch (e) {
					throw new FS.ErrnoError(29);
				}
				if (length) stream.node.mtime = stream.node.ctime = Date.now();
				return i;
			}
		},
		default_tty_ops: {
			get_char(tty) {
				return FS_stdin_getChar();
			},
			put_char(tty, val) {
				if (val === null || val === 10) {
					out(UTF8ArrayToString(tty.output));
					tty.output = [];
				} else if (val != 0) tty.output.push(val);
			},
			fsync(tty) {
				if (tty.output?.length > 0) {
					out(UTF8ArrayToString(tty.output));
					tty.output = [];
				}
			},
			ioctl_tcgets(tty) {
				return {
					c_iflag: 25856,
					c_oflag: 5,
					c_cflag: 191,
					c_lflag: 35387,
					c_cc: [
						3,
						28,
						127,
						21,
						4,
						0,
						1,
						0,
						17,
						19,
						26,
						0,
						18,
						15,
						23,
						22,
						0,
						0,
						0,
						0,
						0,
						0,
						0,
						0,
						0,
						0,
						0,
						0,
						0,
						0,
						0,
						0
					]
				};
			},
			ioctl_tcsets(tty, optional_actions, data) {
				return 0;
			},
			ioctl_tiocgwinsz(tty) {
				return [24, 80];
			}
		},
		default_tty1_ops: {
			put_char(tty, val) {
				if (val === null || val === 10) {
					err(UTF8ArrayToString(tty.output));
					tty.output = [];
				} else if (val != 0) tty.output.push(val);
			},
			fsync(tty) {
				if (tty.output?.length > 0) {
					err(UTF8ArrayToString(tty.output));
					tty.output = [];
				}
			}
		}
	};
	var mmapAlloc = (size) => {
		abort();
	};
	var MEMFS = {
		ops_table: null,
		mount(mount) {
			return MEMFS.createNode(null, "/", 16895, 0);
		},
		createNode(parent, name, mode, dev) {
			if (FS.isBlkdev(mode) || FS.isFIFO(mode)) throw new FS.ErrnoError(63);
			MEMFS.ops_table ||= {
				dir: {
					node: {
						getattr: MEMFS.node_ops.getattr,
						setattr: MEMFS.node_ops.setattr,
						lookup: MEMFS.node_ops.lookup,
						mknod: MEMFS.node_ops.mknod,
						rename: MEMFS.node_ops.rename,
						unlink: MEMFS.node_ops.unlink,
						rmdir: MEMFS.node_ops.rmdir,
						readdir: MEMFS.node_ops.readdir,
						symlink: MEMFS.node_ops.symlink
					},
					stream: { llseek: MEMFS.stream_ops.llseek }
				},
				file: {
					node: {
						getattr: MEMFS.node_ops.getattr,
						setattr: MEMFS.node_ops.setattr
					},
					stream: {
						llseek: MEMFS.stream_ops.llseek,
						read: MEMFS.stream_ops.read,
						write: MEMFS.stream_ops.write,
						mmap: MEMFS.stream_ops.mmap,
						msync: MEMFS.stream_ops.msync
					}
				},
				link: {
					node: {
						getattr: MEMFS.node_ops.getattr,
						setattr: MEMFS.node_ops.setattr,
						readlink: MEMFS.node_ops.readlink
					},
					stream: {}
				},
				chrdev: {
					node: {
						getattr: MEMFS.node_ops.getattr,
						setattr: MEMFS.node_ops.setattr
					},
					stream: FS.chrdev_stream_ops
				}
			};
			var node = FS.createNode(parent, name, mode, dev);
			if (FS.isDir(node.mode)) {
				node.node_ops = MEMFS.ops_table.dir.node;
				node.stream_ops = MEMFS.ops_table.dir.stream;
				node.contents = {};
			} else if (FS.isFile(node.mode)) {
				node.node_ops = MEMFS.ops_table.file.node;
				node.stream_ops = MEMFS.ops_table.file.stream;
				node.usedBytes = 0;
				node.contents = null;
			} else if (FS.isLink(node.mode)) {
				node.node_ops = MEMFS.ops_table.link.node;
				node.stream_ops = MEMFS.ops_table.link.stream;
			} else if (FS.isChrdev(node.mode)) {
				node.node_ops = MEMFS.ops_table.chrdev.node;
				node.stream_ops = MEMFS.ops_table.chrdev.stream;
			}
			node.atime = node.mtime = node.ctime = Date.now();
			if (parent) {
				parent.contents[name] = node;
				parent.atime = parent.mtime = parent.ctime = node.atime;
			}
			return node;
		},
		getFileDataAsTypedArray(node) {
			if (!node.contents) return /* @__PURE__ */ new Uint8Array(0);
			if (node.contents.subarray) return node.contents.subarray(0, node.usedBytes);
			return new Uint8Array(node.contents);
		},
		expandFileStorage(node, newCapacity) {
			var prevCapacity = node.contents ? node.contents.length : 0;
			if (prevCapacity >= newCapacity) return;
			newCapacity = Math.max(newCapacity, prevCapacity * (prevCapacity < 1048576 ? 2 : 1.125) >>> 0);
			if (prevCapacity != 0) newCapacity = Math.max(newCapacity, 256);
			var oldContents = node.contents;
			node.contents = new Uint8Array(newCapacity);
			if (node.usedBytes > 0) node.contents.set(oldContents.subarray(0, node.usedBytes), 0);
		},
		resizeFileStorage(node, newSize) {
			if (node.usedBytes == newSize) return;
			if (newSize == 0) {
				node.contents = null;
				node.usedBytes = 0;
			} else {
				var oldContents = node.contents;
				node.contents = new Uint8Array(newSize);
				if (oldContents) node.contents.set(oldContents.subarray(0, Math.min(newSize, node.usedBytes)));
				node.usedBytes = newSize;
			}
		},
		node_ops: {
			getattr(node) {
				var attr = {};
				attr.dev = FS.isChrdev(node.mode) ? node.id : 1;
				attr.ino = node.id;
				attr.mode = node.mode;
				attr.nlink = 1;
				attr.uid = 0;
				attr.gid = 0;
				attr.rdev = node.rdev;
				if (FS.isDir(node.mode)) attr.size = 4096;
				else if (FS.isFile(node.mode)) attr.size = node.usedBytes;
				else if (FS.isLink(node.mode)) attr.size = node.link.length;
				else attr.size = 0;
				attr.atime = new Date(node.atime);
				attr.mtime = new Date(node.mtime);
				attr.ctime = new Date(node.ctime);
				attr.blksize = 4096;
				attr.blocks = Math.ceil(attr.size / attr.blksize);
				return attr;
			},
			setattr(node, attr) {
				for (const key of [
					"mode",
					"atime",
					"mtime",
					"ctime"
				]) if (attr[key] != null) node[key] = attr[key];
				if (attr.size !== void 0) MEMFS.resizeFileStorage(node, attr.size);
			},
			lookup(parent, name) {
				if (!MEMFS.doesNotExistError) {
					MEMFS.doesNotExistError = new FS.ErrnoError(44);
					MEMFS.doesNotExistError.stack = "<generic error, no stack>";
				}
				throw MEMFS.doesNotExistError;
			},
			mknod(parent, name, mode, dev) {
				return MEMFS.createNode(parent, name, mode, dev);
			},
			rename(old_node, new_dir, new_name) {
				var new_node;
				try {
					new_node = FS.lookupNode(new_dir, new_name);
				} catch (e) {}
				if (new_node) {
					if (FS.isDir(old_node.mode)) for (var i in new_node.contents) throw new FS.ErrnoError(55);
					FS.hashRemoveNode(new_node);
				}
				delete old_node.parent.contents[old_node.name];
				new_dir.contents[new_name] = old_node;
				old_node.name = new_name;
				new_dir.ctime = new_dir.mtime = old_node.parent.ctime = old_node.parent.mtime = Date.now();
			},
			unlink(parent, name) {
				delete parent.contents[name];
				parent.ctime = parent.mtime = Date.now();
			},
			rmdir(parent, name) {
				for (var i in FS.lookupNode(parent, name).contents) throw new FS.ErrnoError(55);
				delete parent.contents[name];
				parent.ctime = parent.mtime = Date.now();
			},
			readdir(node) {
				return [
					".",
					"..",
					...Object.keys(node.contents)
				];
			},
			symlink(parent, newname, oldpath) {
				var node = MEMFS.createNode(parent, newname, 41471, 0);
				node.link = oldpath;
				return node;
			},
			readlink(node) {
				if (!FS.isLink(node.mode)) throw new FS.ErrnoError(28);
				return node.link;
			}
		},
		stream_ops: {
			read(stream, buffer, offset, length, position) {
				var contents = stream.node.contents;
				if (position >= stream.node.usedBytes) return 0;
				var size = Math.min(stream.node.usedBytes - position, length);
				if (size > 8 && contents.subarray) buffer.set(contents.subarray(position, position + size), offset);
				else for (var i = 0; i < size; i++) buffer[offset + i] = contents[position + i];
				return size;
			},
			write(stream, buffer, offset, length, position, canOwn) {
				if (buffer.buffer === HEAP8.buffer) canOwn = false;
				if (!length) return 0;
				var node = stream.node;
				node.mtime = node.ctime = Date.now();
				if (buffer.subarray && (!node.contents || node.contents.subarray)) {
					if (canOwn) {
						node.contents = buffer.subarray(offset, offset + length);
						node.usedBytes = length;
						return length;
					} else if (node.usedBytes === 0 && position === 0) {
						node.contents = buffer.slice(offset, offset + length);
						node.usedBytes = length;
						return length;
					} else if (position + length <= node.usedBytes) {
						node.contents.set(buffer.subarray(offset, offset + length), position);
						return length;
					}
				}
				MEMFS.expandFileStorage(node, position + length);
				if (node.contents.subarray && buffer.subarray) node.contents.set(buffer.subarray(offset, offset + length), position);
				else for (var i = 0; i < length; i++) node.contents[position + i] = buffer[offset + i];
				node.usedBytes = Math.max(node.usedBytes, position + length);
				return length;
			},
			llseek(stream, offset, whence) {
				var position = offset;
				if (whence === 1) position += stream.position;
				else if (whence === 2) {
					if (FS.isFile(stream.node.mode)) position += stream.node.usedBytes;
				}
				if (position < 0) throw new FS.ErrnoError(28);
				return position;
			},
			mmap(stream, length, position, prot, flags) {
				if (!FS.isFile(stream.node.mode)) throw new FS.ErrnoError(43);
				var ptr;
				var allocated;
				var contents = stream.node.contents;
				if (!(flags & 2) && contents && contents.buffer === HEAP8.buffer) {
					allocated = false;
					ptr = contents.byteOffset;
				} else {
					allocated = true;
					ptr = mmapAlloc(length);
					if (!ptr) throw new FS.ErrnoError(48);
					if (contents) {
						if (position > 0 || position + length < contents.length) {
							if (contents.subarray) contents = contents.subarray(position, position + length);
							else contents = Array.prototype.slice.call(contents, position, position + length);
						}
						HEAP8.set(contents, ptr);
					}
				}
				return {
					ptr,
					allocated
				};
			},
			msync(stream, buffer, offset, length, mmapFlags) {
				MEMFS.stream_ops.write(stream, buffer, 0, length, offset, false);
				return 0;
			}
		}
	};
	var FS_modeStringToFlags = (str) => {
		var flags = {
			r: 0,
			"r+": 2,
			w: 577,
			"w+": 578,
			a: 1089,
			"a+": 1090
		}[str];
		if (typeof flags == "undefined") throw new Error(`Unknown file open mode: ${str}`);
		return flags;
	};
	var FS_getMode = (canRead, canWrite) => {
		var mode = 0;
		if (canRead) mode |= 365;
		if (canWrite) mode |= 146;
		return mode;
	};
	var asyncLoad = async (url) => {
		var arrayBuffer = await readAsync(url);
		return new Uint8Array(arrayBuffer);
	};
	var FS_createDataFile = (...args) => FS.createDataFile(...args);
	var getUniqueRunDependency = (id) => id;
	var preloadPlugins = [];
	var FS_handledByPreloadPlugin = async (byteArray, fullname) => {
		if (typeof Browser != "undefined") Browser.init();
		for (var plugin of preloadPlugins) if (plugin["canHandle"](fullname)) return plugin["handle"](byteArray, fullname);
		return byteArray;
	};
	var FS_preloadFile = async (parent, name, url, canRead, canWrite, dontCreateFile, canOwn, preFinish) => {
		var fullname = name ? PATH_FS.resolve(PATH.join2(parent, name)) : parent;
		var dep = getUniqueRunDependency(`cp ${fullname}`);
		addRunDependency(dep);
		try {
			var byteArray = url;
			if (typeof url == "string") byteArray = await asyncLoad(url);
			byteArray = await FS_handledByPreloadPlugin(byteArray, fullname);
			preFinish?.();
			if (!dontCreateFile) FS_createDataFile(parent, name, byteArray, canRead, canWrite, canOwn);
		} finally {
			removeRunDependency(dep);
		}
	};
	var FS_createPreloadedFile = (parent, name, url, canRead, canWrite, onload, onerror, dontCreateFile, canOwn, preFinish) => {
		FS_preloadFile(parent, name, url, canRead, canWrite, dontCreateFile, canOwn, preFinish).then(onload).catch(onerror);
	};
	var FS = {
		root: null,
		mounts: [],
		devices: {},
		streams: [],
		nextInode: 1,
		nameTable: null,
		currentPath: "/",
		initialized: false,
		ignorePermissions: true,
		filesystems: null,
		syncFSRequests: 0,
		readFiles: {},
		ErrnoError: class {
			name = "ErrnoError";
			constructor(errno) {
				this.errno = errno;
			}
		},
		FSStream: class {
			shared = {};
			get object() {
				return this.node;
			}
			set object(val) {
				this.node = val;
			}
			get isRead() {
				return (this.flags & 2097155) !== 1;
			}
			get isWrite() {
				return (this.flags & 2097155) !== 0;
			}
			get isAppend() {
				return this.flags & 1024;
			}
			get flags() {
				return this.shared.flags;
			}
			set flags(val) {
				this.shared.flags = val;
			}
			get position() {
				return this.shared.position;
			}
			set position(val) {
				this.shared.position = val;
			}
		},
		FSNode: class {
			node_ops = {};
			stream_ops = {};
			readMode = 365;
			writeMode = 146;
			mounted = null;
			constructor(parent, name, mode, rdev) {
				if (!parent) parent = this;
				this.parent = parent;
				this.mount = parent.mount;
				this.id = FS.nextInode++;
				this.name = name;
				this.mode = mode;
				this.rdev = rdev;
				this.atime = this.mtime = this.ctime = Date.now();
			}
			get read() {
				return (this.mode & this.readMode) === this.readMode;
			}
			set read(val) {
				val ? this.mode |= this.readMode : this.mode &= ~this.readMode;
			}
			get write() {
				return (this.mode & this.writeMode) === this.writeMode;
			}
			set write(val) {
				val ? this.mode |= this.writeMode : this.mode &= ~this.writeMode;
			}
			get isFolder() {
				return FS.isDir(this.mode);
			}
			get isDevice() {
				return FS.isChrdev(this.mode);
			}
		},
		lookupPath(path, opts = {}) {
			if (!path) throw new FS.ErrnoError(44);
			opts.follow_mount ??= true;
			if (!PATH.isAbs(path)) path = FS.cwd() + "/" + path;
			linkloop: for (var nlinks = 0; nlinks < 40; nlinks++) {
				var parts = path.split("/").filter((p) => !!p);
				var current = FS.root;
				var current_path = "/";
				for (var i = 0; i < parts.length; i++) {
					var islast = i === parts.length - 1;
					if (islast && opts.parent) break;
					if (parts[i] === ".") continue;
					if (parts[i] === "..") {
						current_path = PATH.dirname(current_path);
						if (FS.isRoot(current)) {
							path = current_path + "/" + parts.slice(i + 1).join("/");
							nlinks--;
							continue linkloop;
						} else current = current.parent;
						continue;
					}
					current_path = PATH.join2(current_path, parts[i]);
					try {
						current = FS.lookupNode(current, parts[i]);
					} catch (e) {
						if (e?.errno === 44 && islast && opts.noent_okay) return { path: current_path };
						throw e;
					}
					if (FS.isMountpoint(current) && (!islast || opts.follow_mount)) current = current.mounted.root;
					if (FS.isLink(current.mode) && (!islast || opts.follow)) {
						if (!current.node_ops.readlink) throw new FS.ErrnoError(52);
						var link = current.node_ops.readlink(current);
						if (!PATH.isAbs(link)) link = PATH.dirname(current_path) + "/" + link;
						path = link + "/" + parts.slice(i + 1).join("/");
						continue linkloop;
					}
				}
				return {
					path: current_path,
					node: current
				};
			}
			throw new FS.ErrnoError(32);
		},
		getPath(node) {
			var path;
			while (true) {
				if (FS.isRoot(node)) {
					var mount = node.mount.mountpoint;
					if (!path) return mount;
					return mount[mount.length - 1] !== "/" ? `${mount}/${path}` : mount + path;
				}
				path = path ? `${node.name}/${path}` : node.name;
				node = node.parent;
			}
		},
		hashName(parentid, name) {
			var hash = 0;
			for (var i = 0; i < name.length; i++) hash = (hash << 5) - hash + name.charCodeAt(i) | 0;
			return (parentid + hash >>> 0) % FS.nameTable.length;
		},
		hashAddNode(node) {
			var hash = FS.hashName(node.parent.id, node.name);
			node.name_next = FS.nameTable[hash];
			FS.nameTable[hash] = node;
		},
		hashRemoveNode(node) {
			var hash = FS.hashName(node.parent.id, node.name);
			if (FS.nameTable[hash] === node) FS.nameTable[hash] = node.name_next;
			else {
				var current = FS.nameTable[hash];
				while (current) {
					if (current.name_next === node) {
						current.name_next = node.name_next;
						break;
					}
					current = current.name_next;
				}
			}
		},
		lookupNode(parent, name) {
			var errCode = FS.mayLookup(parent);
			if (errCode) throw new FS.ErrnoError(errCode);
			var hash = FS.hashName(parent.id, name);
			for (var node = FS.nameTable[hash]; node; node = node.name_next) {
				var nodeName = node.name;
				if (node.parent.id === parent.id && nodeName === name) return node;
			}
			return FS.lookup(parent, name);
		},
		createNode(parent, name, mode, rdev) {
			var node = new FS.FSNode(parent, name, mode, rdev);
			FS.hashAddNode(node);
			return node;
		},
		destroyNode(node) {
			FS.hashRemoveNode(node);
		},
		isRoot(node) {
			return node === node.parent;
		},
		isMountpoint(node) {
			return !!node.mounted;
		},
		isFile(mode) {
			return (mode & 61440) === 32768;
		},
		isDir(mode) {
			return (mode & 61440) === 16384;
		},
		isLink(mode) {
			return (mode & 61440) === 40960;
		},
		isChrdev(mode) {
			return (mode & 61440) === 8192;
		},
		isBlkdev(mode) {
			return (mode & 61440) === 24576;
		},
		isFIFO(mode) {
			return (mode & 61440) === 4096;
		},
		isSocket(mode) {
			return (mode & 49152) === 49152;
		},
		flagsToPermissionString(flag) {
			var perms = [
				"r",
				"w",
				"rw"
			][flag & 3];
			if (flag & 512) perms += "w";
			return perms;
		},
		nodePermissions(node, perms) {
			if (FS.ignorePermissions) return 0;
			if (perms.includes("r") && !(node.mode & 292)) return 2;
			else if (perms.includes("w") && !(node.mode & 146)) return 2;
			else if (perms.includes("x") && !(node.mode & 73)) return 2;
			return 0;
		},
		mayLookup(dir) {
			if (!FS.isDir(dir.mode)) return 54;
			var errCode = FS.nodePermissions(dir, "x");
			if (errCode) return errCode;
			if (!dir.node_ops.lookup) return 2;
			return 0;
		},
		mayCreate(dir, name) {
			if (!FS.isDir(dir.mode)) return 54;
			try {
				FS.lookupNode(dir, name);
				return 20;
			} catch (e) {}
			return FS.nodePermissions(dir, "wx");
		},
		mayDelete(dir, name, isdir) {
			var node;
			try {
				node = FS.lookupNode(dir, name);
			} catch (e) {
				return e.errno;
			}
			var errCode = FS.nodePermissions(dir, "wx");
			if (errCode) return errCode;
			if (isdir) {
				if (!FS.isDir(node.mode)) return 54;
				if (FS.isRoot(node) || FS.getPath(node) === FS.cwd()) return 10;
			} else if (FS.isDir(node.mode)) return 31;
			return 0;
		},
		mayOpen(node, flags) {
			if (!node) return 44;
			if (FS.isLink(node.mode)) return 32;
			else if (FS.isDir(node.mode)) {
				if (FS.flagsToPermissionString(flags) !== "r" || flags & 576) return 31;
			}
			return FS.nodePermissions(node, FS.flagsToPermissionString(flags));
		},
		checkOpExists(op, err) {
			if (!op) throw new FS.ErrnoError(err);
			return op;
		},
		MAX_OPEN_FDS: 4096,
		nextfd() {
			for (var fd = 0; fd <= FS.MAX_OPEN_FDS; fd++) if (!FS.streams[fd]) return fd;
			throw new FS.ErrnoError(33);
		},
		getStreamChecked(fd) {
			var stream = FS.getStream(fd);
			if (!stream) throw new FS.ErrnoError(8);
			return stream;
		},
		getStream: (fd) => FS.streams[fd],
		createStream(stream, fd = -1) {
			stream = Object.assign(new FS.FSStream(), stream);
			if (fd == -1) fd = FS.nextfd();
			stream.fd = fd;
			FS.streams[fd] = stream;
			return stream;
		},
		closeStream(fd) {
			FS.streams[fd] = null;
		},
		dupStream(origStream, fd = -1) {
			var stream = FS.createStream(origStream, fd);
			stream.stream_ops?.dup?.(stream);
			return stream;
		},
		doSetAttr(stream, node, attr) {
			var setattr = stream?.stream_ops.setattr;
			var arg = setattr ? stream : node;
			setattr ??= node.node_ops.setattr;
			FS.checkOpExists(setattr, 63);
			setattr(arg, attr);
		},
		chrdev_stream_ops: {
			open(stream) {
				stream.stream_ops = FS.getDevice(stream.node.rdev).stream_ops;
				stream.stream_ops.open?.(stream);
			},
			llseek() {
				throw new FS.ErrnoError(70);
			}
		},
		major: (dev) => dev >> 8,
		minor: (dev) => dev & 255,
		makedev: (ma, mi) => ma << 8 | mi,
		registerDevice(dev, ops) {
			FS.devices[dev] = { stream_ops: ops };
		},
		getDevice: (dev) => FS.devices[dev],
		getMounts(mount) {
			var mounts = [];
			var check = [mount];
			while (check.length) {
				var m = check.pop();
				mounts.push(m);
				check.push(...m.mounts);
			}
			return mounts;
		},
		syncfs(populate, callback) {
			if (typeof populate == "function") {
				callback = populate;
				populate = false;
			}
			FS.syncFSRequests++;
			if (FS.syncFSRequests > 1) err(`warning: ${FS.syncFSRequests} FS.syncfs operations in flight at once, probably just doing extra work`);
			var mounts = FS.getMounts(FS.root.mount);
			var completed = 0;
			function doCallback(errCode) {
				FS.syncFSRequests--;
				return callback(errCode);
			}
			function done(errCode) {
				if (errCode) {
					if (!done.errored) {
						done.errored = true;
						return doCallback(errCode);
					}
					return;
				}
				if (++completed >= mounts.length) doCallback(null);
			}
			for (var mount of mounts) if (mount.type.syncfs) mount.type.syncfs(mount, populate, done);
			else done(null);
		},
		mount(type, opts, mountpoint) {
			var root = mountpoint === "/";
			var pseudo = !mountpoint;
			var node;
			if (root && FS.root) throw new FS.ErrnoError(10);
			else if (!root && !pseudo) {
				var lookup = FS.lookupPath(mountpoint, { follow_mount: false });
				mountpoint = lookup.path;
				node = lookup.node;
				if (FS.isMountpoint(node)) throw new FS.ErrnoError(10);
				if (!FS.isDir(node.mode)) throw new FS.ErrnoError(54);
			}
			var mount = {
				type,
				opts,
				mountpoint,
				mounts: []
			};
			var mountRoot = type.mount(mount);
			mountRoot.mount = mount;
			mount.root = mountRoot;
			if (root) FS.root = mountRoot;
			else if (node) {
				node.mounted = mount;
				if (node.mount) node.mount.mounts.push(mount);
			}
			return mountRoot;
		},
		unmount(mountpoint) {
			var lookup = FS.lookupPath(mountpoint, { follow_mount: false });
			if (!FS.isMountpoint(lookup.node)) throw new FS.ErrnoError(28);
			var node = lookup.node;
			var mount = node.mounted;
			var mounts = FS.getMounts(mount);
			for (var [hash, current] of Object.entries(FS.nameTable)) while (current) {
				var next = current.name_next;
				if (mounts.includes(current.mount)) FS.destroyNode(current);
				current = next;
			}
			node.mounted = null;
			var idx = node.mount.mounts.indexOf(mount);
			node.mount.mounts.splice(idx, 1);
		},
		lookup(parent, name) {
			return parent.node_ops.lookup(parent, name);
		},
		mknod(path, mode, dev) {
			var parent = FS.lookupPath(path, { parent: true }).node;
			var name = PATH.basename(path);
			if (!name) throw new FS.ErrnoError(28);
			if (name === "." || name === "..") throw new FS.ErrnoError(20);
			var errCode = FS.mayCreate(parent, name);
			if (errCode) throw new FS.ErrnoError(errCode);
			if (!parent.node_ops.mknod) throw new FS.ErrnoError(63);
			return parent.node_ops.mknod(parent, name, mode, dev);
		},
		statfs(path) {
			return FS.statfsNode(FS.lookupPath(path, { follow: true }).node);
		},
		statfsStream(stream) {
			return FS.statfsNode(stream.node);
		},
		statfsNode(node) {
			var rtn = {
				bsize: 4096,
				frsize: 4096,
				blocks: 1e6,
				bfree: 5e5,
				bavail: 5e5,
				files: FS.nextInode,
				ffree: FS.nextInode - 1,
				fsid: 42,
				flags: 2,
				namelen: 255
			};
			if (node.node_ops.statfs) Object.assign(rtn, node.node_ops.statfs(node.mount.opts.root));
			return rtn;
		},
		create(path, mode = 438) {
			mode &= 4095;
			mode |= 32768;
			return FS.mknod(path, mode, 0);
		},
		mkdir(path, mode = 511) {
			mode &= 1023;
			mode |= 16384;
			return FS.mknod(path, mode, 0);
		},
		mkdirTree(path, mode) {
			var dirs = path.split("/");
			var d = "";
			for (var dir of dirs) {
				if (!dir) continue;
				if (d || PATH.isAbs(path)) d += "/";
				d += dir;
				try {
					FS.mkdir(d, mode);
				} catch (e) {
					if (e.errno != 20) throw e;
				}
			}
		},
		mkdev(path, mode, dev) {
			if (typeof dev == "undefined") {
				dev = mode;
				mode = 438;
			}
			mode |= 8192;
			return FS.mknod(path, mode, dev);
		},
		symlink(oldpath, newpath) {
			if (!PATH_FS.resolve(oldpath)) throw new FS.ErrnoError(44);
			var parent = FS.lookupPath(newpath, { parent: true }).node;
			if (!parent) throw new FS.ErrnoError(44);
			var newname = PATH.basename(newpath);
			var errCode = FS.mayCreate(parent, newname);
			if (errCode) throw new FS.ErrnoError(errCode);
			if (!parent.node_ops.symlink) throw new FS.ErrnoError(63);
			return parent.node_ops.symlink(parent, newname, oldpath);
		},
		rename(old_path, new_path) {
			var old_dirname = PATH.dirname(old_path);
			var new_dirname = PATH.dirname(new_path);
			var old_name = PATH.basename(old_path);
			var new_name = PATH.basename(new_path);
			var lookup = FS.lookupPath(old_path, { parent: true }), old_dir = lookup.node, new_dir;
			lookup = FS.lookupPath(new_path, { parent: true });
			new_dir = lookup.node;
			if (!old_dir || !new_dir) throw new FS.ErrnoError(44);
			if (old_dir.mount !== new_dir.mount) throw new FS.ErrnoError(75);
			var old_node = FS.lookupNode(old_dir, old_name);
			var relative = PATH_FS.relative(old_path, new_dirname);
			if (relative.charAt(0) !== ".") throw new FS.ErrnoError(28);
			relative = PATH_FS.relative(new_path, old_dirname);
			if (relative.charAt(0) !== ".") throw new FS.ErrnoError(55);
			var new_node;
			try {
				new_node = FS.lookupNode(new_dir, new_name);
			} catch (e) {}
			if (old_node === new_node) return;
			var isdir = FS.isDir(old_node.mode);
			var errCode = FS.mayDelete(old_dir, old_name, isdir);
			if (errCode) throw new FS.ErrnoError(errCode);
			errCode = new_node ? FS.mayDelete(new_dir, new_name, isdir) : FS.mayCreate(new_dir, new_name);
			if (errCode) throw new FS.ErrnoError(errCode);
			if (!old_dir.node_ops.rename) throw new FS.ErrnoError(63);
			if (FS.isMountpoint(old_node) || new_node && FS.isMountpoint(new_node)) throw new FS.ErrnoError(10);
			if (new_dir !== old_dir) {
				errCode = FS.nodePermissions(old_dir, "w");
				if (errCode) throw new FS.ErrnoError(errCode);
			}
			FS.hashRemoveNode(old_node);
			try {
				old_dir.node_ops.rename(old_node, new_dir, new_name);
				old_node.parent = new_dir;
			} catch (e) {
				throw e;
			} finally {
				FS.hashAddNode(old_node);
			}
		},
		rmdir(path) {
			var parent = FS.lookupPath(path, { parent: true }).node;
			var name = PATH.basename(path);
			var node = FS.lookupNode(parent, name);
			var errCode = FS.mayDelete(parent, name, true);
			if (errCode) throw new FS.ErrnoError(errCode);
			if (!parent.node_ops.rmdir) throw new FS.ErrnoError(63);
			if (FS.isMountpoint(node)) throw new FS.ErrnoError(10);
			parent.node_ops.rmdir(parent, name);
			FS.destroyNode(node);
		},
		readdir(path) {
			var node = FS.lookupPath(path, { follow: true }).node;
			return FS.checkOpExists(node.node_ops.readdir, 54)(node);
		},
		unlink(path) {
			var parent = FS.lookupPath(path, { parent: true }).node;
			if (!parent) throw new FS.ErrnoError(44);
			var name = PATH.basename(path);
			var node = FS.lookupNode(parent, name);
			var errCode = FS.mayDelete(parent, name, false);
			if (errCode) throw new FS.ErrnoError(errCode);
			if (!parent.node_ops.unlink) throw new FS.ErrnoError(63);
			if (FS.isMountpoint(node)) throw new FS.ErrnoError(10);
			parent.node_ops.unlink(parent, name);
			FS.destroyNode(node);
		},
		readlink(path) {
			var link = FS.lookupPath(path).node;
			if (!link) throw new FS.ErrnoError(44);
			if (!link.node_ops.readlink) throw new FS.ErrnoError(28);
			return link.node_ops.readlink(link);
		},
		stat(path, dontFollow) {
			var node = FS.lookupPath(path, { follow: !dontFollow }).node;
			return FS.checkOpExists(node.node_ops.getattr, 63)(node);
		},
		fstat(fd) {
			var stream = FS.getStreamChecked(fd);
			var node = stream.node;
			var getattr = stream.stream_ops.getattr;
			var arg = getattr ? stream : node;
			getattr ??= node.node_ops.getattr;
			FS.checkOpExists(getattr, 63);
			return getattr(arg);
		},
		lstat(path) {
			return FS.stat(path, true);
		},
		doChmod(stream, node, mode, dontFollow) {
			FS.doSetAttr(stream, node, {
				mode: mode & 4095 | node.mode & -4096,
				ctime: Date.now(),
				dontFollow
			});
		},
		chmod(path, mode, dontFollow) {
			var node;
			if (typeof path == "string") node = FS.lookupPath(path, { follow: !dontFollow }).node;
			else node = path;
			FS.doChmod(null, node, mode, dontFollow);
		},
		lchmod(path, mode) {
			FS.chmod(path, mode, true);
		},
		fchmod(fd, mode) {
			var stream = FS.getStreamChecked(fd);
			FS.doChmod(stream, stream.node, mode, false);
		},
		doChown(stream, node, dontFollow) {
			FS.doSetAttr(stream, node, {
				timestamp: Date.now(),
				dontFollow
			});
		},
		chown(path, uid, gid, dontFollow) {
			var node;
			if (typeof path == "string") node = FS.lookupPath(path, { follow: !dontFollow }).node;
			else node = path;
			FS.doChown(null, node, dontFollow);
		},
		lchown(path, uid, gid) {
			FS.chown(path, uid, gid, true);
		},
		fchown(fd, uid, gid) {
			var stream = FS.getStreamChecked(fd);
			FS.doChown(stream, stream.node, false);
		},
		doTruncate(stream, node, len) {
			if (FS.isDir(node.mode)) throw new FS.ErrnoError(31);
			if (!FS.isFile(node.mode)) throw new FS.ErrnoError(28);
			var errCode = FS.nodePermissions(node, "w");
			if (errCode) throw new FS.ErrnoError(errCode);
			FS.doSetAttr(stream, node, {
				size: len,
				timestamp: Date.now()
			});
		},
		truncate(path, len) {
			if (len < 0) throw new FS.ErrnoError(28);
			var node;
			if (typeof path == "string") node = FS.lookupPath(path, { follow: true }).node;
			else node = path;
			FS.doTruncate(null, node, len);
		},
		ftruncate(fd, len) {
			var stream = FS.getStreamChecked(fd);
			if (len < 0 || (stream.flags & 2097155) === 0) throw new FS.ErrnoError(28);
			FS.doTruncate(stream, stream.node, len);
		},
		utime(path, atime, mtime) {
			var node = FS.lookupPath(path, { follow: true }).node;
			FS.checkOpExists(node.node_ops.setattr, 63)(node, {
				atime,
				mtime
			});
		},
		open(path, flags, mode = 438) {
			if (path === "") throw new FS.ErrnoError(44);
			flags = typeof flags == "string" ? FS_modeStringToFlags(flags) : flags;
			if (flags & 64) mode = mode & 4095 | 32768;
			else mode = 0;
			var node;
			var isDirPath;
			if (typeof path == "object") node = path;
			else {
				isDirPath = path.endsWith("/");
				var lookup = FS.lookupPath(path, {
					follow: !(flags & 131072),
					noent_okay: true
				});
				node = lookup.node;
				path = lookup.path;
			}
			var created = false;
			if (flags & 64) {
				if (node) {
					if (flags & 128) throw new FS.ErrnoError(20);
				} else if (isDirPath) throw new FS.ErrnoError(31);
				else {
					node = FS.mknod(path, mode | 511, 0);
					created = true;
				}
			}
			if (!node) throw new FS.ErrnoError(44);
			if (FS.isChrdev(node.mode)) flags &= -513;
			if (flags & 65536 && !FS.isDir(node.mode)) throw new FS.ErrnoError(54);
			if (!created) {
				var errCode = FS.mayOpen(node, flags);
				if (errCode) throw new FS.ErrnoError(errCode);
			}
			if (flags & 512 && !created) FS.truncate(node, 0);
			flags &= -131713;
			var stream = FS.createStream({
				node,
				path: FS.getPath(node),
				flags,
				seekable: true,
				position: 0,
				stream_ops: node.stream_ops,
				ungotten: [],
				error: false
			});
			if (stream.stream_ops.open) stream.stream_ops.open(stream);
			if (created) FS.chmod(node, mode & 511);
			if (Module["logReadFiles"] && !(flags & 1)) {
				if (!(path in FS.readFiles)) FS.readFiles[path] = 1;
			}
			return stream;
		},
		close(stream) {
			if (FS.isClosed(stream)) throw new FS.ErrnoError(8);
			if (stream.getdents) stream.getdents = null;
			try {
				if (stream.stream_ops.close) stream.stream_ops.close(stream);
			} catch (e) {
				throw e;
			} finally {
				FS.closeStream(stream.fd);
			}
			stream.fd = null;
		},
		isClosed(stream) {
			return stream.fd === null;
		},
		llseek(stream, offset, whence) {
			if (FS.isClosed(stream)) throw new FS.ErrnoError(8);
			if (!stream.seekable || !stream.stream_ops.llseek) throw new FS.ErrnoError(70);
			if (whence != 0 && whence != 1 && whence != 2) throw new FS.ErrnoError(28);
			stream.position = stream.stream_ops.llseek(stream, offset, whence);
			stream.ungotten = [];
			return stream.position;
		},
		read(stream, buffer, offset, length, position) {
			if (length < 0 || position < 0) throw new FS.ErrnoError(28);
			if (FS.isClosed(stream)) throw new FS.ErrnoError(8);
			if ((stream.flags & 2097155) === 1) throw new FS.ErrnoError(8);
			if (FS.isDir(stream.node.mode)) throw new FS.ErrnoError(31);
			if (!stream.stream_ops.read) throw new FS.ErrnoError(28);
			var seeking = typeof position != "undefined";
			if (!seeking) position = stream.position;
			else if (!stream.seekable) throw new FS.ErrnoError(70);
			var bytesRead = stream.stream_ops.read(stream, buffer, offset, length, position);
			if (!seeking) stream.position += bytesRead;
			return bytesRead;
		},
		write(stream, buffer, offset, length, position, canOwn) {
			if (length < 0 || position < 0) throw new FS.ErrnoError(28);
			if (FS.isClosed(stream)) throw new FS.ErrnoError(8);
			if ((stream.flags & 2097155) === 0) throw new FS.ErrnoError(8);
			if (FS.isDir(stream.node.mode)) throw new FS.ErrnoError(31);
			if (!stream.stream_ops.write) throw new FS.ErrnoError(28);
			if (stream.seekable && stream.flags & 1024) FS.llseek(stream, 0, 2);
			var seeking = typeof position != "undefined";
			if (!seeking) position = stream.position;
			else if (!stream.seekable) throw new FS.ErrnoError(70);
			var bytesWritten = stream.stream_ops.write(stream, buffer, offset, length, position, canOwn);
			if (!seeking) stream.position += bytesWritten;
			return bytesWritten;
		},
		mmap(stream, length, position, prot, flags) {
			if ((prot & 2) !== 0 && (flags & 2) === 0 && (stream.flags & 2097155) !== 2) throw new FS.ErrnoError(2);
			if ((stream.flags & 2097155) === 1) throw new FS.ErrnoError(2);
			if (!stream.stream_ops.mmap) throw new FS.ErrnoError(43);
			if (!length) throw new FS.ErrnoError(28);
			return stream.stream_ops.mmap(stream, length, position, prot, flags);
		},
		msync(stream, buffer, offset, length, mmapFlags) {
			if (!stream.stream_ops.msync) return 0;
			return stream.stream_ops.msync(stream, buffer, offset, length, mmapFlags);
		},
		ioctl(stream, cmd, arg) {
			if (!stream.stream_ops.ioctl) throw new FS.ErrnoError(59);
			return stream.stream_ops.ioctl(stream, cmd, arg);
		},
		readFile(path, opts = {}) {
			opts.flags = opts.flags || 0;
			opts.encoding = opts.encoding || "binary";
			if (opts.encoding !== "utf8" && opts.encoding !== "binary") abort(`Invalid encoding type "${opts.encoding}"`);
			var stream = FS.open(path, opts.flags);
			var length = FS.stat(path).size;
			var buf = new Uint8Array(length);
			FS.read(stream, buf, 0, length, 0);
			if (opts.encoding === "utf8") buf = UTF8ArrayToString(buf);
			FS.close(stream);
			return buf;
		},
		writeFile(path, data, opts = {}) {
			opts.flags = opts.flags || 577;
			var stream = FS.open(path, opts.flags, opts.mode);
			if (typeof data == "string") data = new Uint8Array(intArrayFromString(data, true));
			if (ArrayBuffer.isView(data)) FS.write(stream, data, 0, data.byteLength, void 0, opts.canOwn);
			else abort("Unsupported data type");
			FS.close(stream);
		},
		cwd: () => FS.currentPath,
		chdir(path) {
			var lookup = FS.lookupPath(path, { follow: true });
			if (lookup.node === null) throw new FS.ErrnoError(44);
			if (!FS.isDir(lookup.node.mode)) throw new FS.ErrnoError(54);
			var errCode = FS.nodePermissions(lookup.node, "x");
			if (errCode) throw new FS.ErrnoError(errCode);
			FS.currentPath = lookup.path;
		},
		createDefaultDirectories() {
			FS.mkdir("/tmp");
			FS.mkdir("/home");
			FS.mkdir("/home/web_user");
		},
		createDefaultDevices() {
			FS.mkdir("/dev");
			FS.registerDevice(FS.makedev(1, 3), {
				read: () => 0,
				write: (stream, buffer, offset, length, pos) => length,
				llseek: () => 0
			});
			FS.mkdev("/dev/null", FS.makedev(1, 3));
			TTY.register(FS.makedev(5, 0), TTY.default_tty_ops);
			TTY.register(FS.makedev(6, 0), TTY.default_tty1_ops);
			FS.mkdev("/dev/tty", FS.makedev(5, 0));
			FS.mkdev("/dev/tty1", FS.makedev(6, 0));
			var randomBuffer = /* @__PURE__ */ new Uint8Array(1024), randomLeft = 0;
			var randomByte = () => {
				if (randomLeft === 0) {
					randomFill(randomBuffer);
					randomLeft = randomBuffer.byteLength;
				}
				return randomBuffer[--randomLeft];
			};
			FS.createDevice("/dev", "random", randomByte);
			FS.createDevice("/dev", "urandom", randomByte);
			FS.mkdir("/dev/shm");
			FS.mkdir("/dev/shm/tmp");
		},
		createSpecialDirectories() {
			FS.mkdir("/proc");
			var proc_self = FS.mkdir("/proc/self");
			FS.mkdir("/proc/self/fd");
			FS.mount({ mount() {
				var node = FS.createNode(proc_self, "fd", 16895, 73);
				node.stream_ops = { llseek: MEMFS.stream_ops.llseek };
				node.node_ops = {
					lookup(parent, name) {
						var fd = +name;
						var stream = FS.getStreamChecked(fd);
						var ret = {
							parent: null,
							mount: { mountpoint: "fake" },
							node_ops: { readlink: () => stream.path },
							id: fd + 1
						};
						ret.parent = ret;
						return ret;
					},
					readdir() {
						return Array.from(FS.streams.entries()).filter(([k, v]) => v).map(([k, v]) => k.toString());
					}
				};
				return node;
			} }, {}, "/proc/self/fd");
		},
		createStandardStreams(input, output, error) {
			if (input) FS.createDevice("/dev", "stdin", input);
			else FS.symlink("/dev/tty", "/dev/stdin");
			if (output) FS.createDevice("/dev", "stdout", null, output);
			else FS.symlink("/dev/tty", "/dev/stdout");
			if (error) FS.createDevice("/dev", "stderr", null, error);
			else FS.symlink("/dev/tty1", "/dev/stderr");
			FS.open("/dev/stdin", 0);
			FS.open("/dev/stdout", 1);
			FS.open("/dev/stderr", 1);
		},
		staticInit() {
			FS.nameTable = new Array(4096);
			FS.mount(MEMFS, {}, "/");
			FS.createDefaultDirectories();
			FS.createDefaultDevices();
			FS.createSpecialDirectories();
			FS.filesystems = { MEMFS };
		},
		init(input, output, error) {
			FS.initialized = true;
			input ??= Module["stdin"];
			output ??= Module["stdout"];
			error ??= Module["stderr"];
			FS.createStandardStreams(input, output, error);
		},
		quit() {
			FS.initialized = false;
			for (var stream of FS.streams) if (stream) FS.close(stream);
		},
		findObject(path, dontResolveLastLink) {
			var ret = FS.analyzePath(path, dontResolveLastLink);
			if (!ret.exists) return null;
			return ret.object;
		},
		analyzePath(path, dontResolveLastLink) {
			try {
				var lookup = FS.lookupPath(path, { follow: !dontResolveLastLink });
				path = lookup.path;
			} catch (e) {}
			var ret = {
				isRoot: false,
				exists: false,
				error: 0,
				name: null,
				path: null,
				object: null,
				parentExists: false,
				parentPath: null,
				parentObject: null
			};
			try {
				var lookup = FS.lookupPath(path, { parent: true });
				ret.parentExists = true;
				ret.parentPath = lookup.path;
				ret.parentObject = lookup.node;
				ret.name = PATH.basename(path);
				lookup = FS.lookupPath(path, { follow: !dontResolveLastLink });
				ret.exists = true;
				ret.path = lookup.path;
				ret.object = lookup.node;
				ret.name = lookup.node.name;
				ret.isRoot = lookup.path === "/";
			} catch (e) {
				ret.error = e.errno;
			}
			return ret;
		},
		createPath(parent, path, canRead, canWrite) {
			parent = typeof parent == "string" ? parent : FS.getPath(parent);
			var parts = path.split("/").reverse();
			while (parts.length) {
				var part = parts.pop();
				if (!part) continue;
				var current = PATH.join2(parent, part);
				try {
					FS.mkdir(current);
				} catch (e) {
					if (e.errno != 20) throw e;
				}
				parent = current;
			}
			return current;
		},
		createFile(parent, name, properties, canRead, canWrite) {
			var path = PATH.join2(typeof parent == "string" ? parent : FS.getPath(parent), name);
			var mode = FS_getMode(canRead, canWrite);
			return FS.create(path, mode);
		},
		createDataFile(parent, name, data, canRead, canWrite, canOwn) {
			var path = name;
			if (parent) {
				parent = typeof parent == "string" ? parent : FS.getPath(parent);
				path = name ? PATH.join2(parent, name) : parent;
			}
			var mode = FS_getMode(canRead, canWrite);
			var node = FS.create(path, mode);
			if (data) {
				if (typeof data == "string") {
					var arr = new Array(data.length);
					for (var i = 0, len = data.length; i < len; ++i) arr[i] = data.charCodeAt(i);
					data = arr;
				}
				FS.chmod(node, mode | 146);
				var stream = FS.open(node, 577);
				FS.write(stream, data, 0, data.length, 0, canOwn);
				FS.close(stream);
				FS.chmod(node, mode);
			}
		},
		createDevice(parent, name, input, output) {
			var path = PATH.join2(typeof parent == "string" ? parent : FS.getPath(parent), name);
			var mode = FS_getMode(!!input, !!output);
			FS.createDevice.major ??= 64;
			var dev = FS.makedev(FS.createDevice.major++, 0);
			FS.registerDevice(dev, {
				open(stream) {
					stream.seekable = false;
				},
				close(stream) {
					if (output?.buffer?.length) output(10);
				},
				read(stream, buffer, offset, length, pos) {
					var bytesRead = 0;
					for (var i = 0; i < length; i++) {
						var result;
						try {
							result = input();
						} catch (e) {
							throw new FS.ErrnoError(29);
						}
						if (result === void 0 && bytesRead === 0) throw new FS.ErrnoError(6);
						if (result === null || result === void 0) break;
						bytesRead++;
						buffer[offset + i] = result;
					}
					if (bytesRead) stream.node.atime = Date.now();
					return bytesRead;
				},
				write(stream, buffer, offset, length, pos) {
					for (var i = 0; i < length; i++) try {
						output(buffer[offset + i]);
					} catch (e) {
						throw new FS.ErrnoError(29);
					}
					if (length) stream.node.mtime = stream.node.ctime = Date.now();
					return i;
				}
			});
			return FS.mkdev(path, mode, dev);
		},
		forceLoadFile(obj) {
			if (obj.isDevice || obj.isFolder || obj.link || obj.contents) return true;
			if (globalThis.XMLHttpRequest) abort("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.");
			else try {
				obj.contents = readBinary(obj.url);
			} catch (e) {
				throw new FS.ErrnoError(29);
			}
		},
		createLazyFile(parent, name, url, canRead, canWrite) {
			class LazyUint8Array {
				lengthKnown = false;
				chunks = [];
				get(idx) {
					if (idx > this.length - 1 || idx < 0) return;
					var chunkOffset = idx % this.chunkSize;
					var chunkNum = idx / this.chunkSize | 0;
					return this.getter(chunkNum)[chunkOffset];
				}
				setDataGetter(getter) {
					this.getter = getter;
				}
				cacheLength() {
					var xhr = new XMLHttpRequest();
					xhr.open("HEAD", url, false);
					xhr.send(null);
					if (!(xhr.status >= 200 && xhr.status < 300 || xhr.status === 304)) abort("Couldn't load " + url + ". Status: " + xhr.status);
					var datalength = Number(xhr.getResponseHeader("Content-length"));
					var header;
					var hasByteServing = (header = xhr.getResponseHeader("Accept-Ranges")) && header === "bytes";
					var usesGzip = (header = xhr.getResponseHeader("Content-Encoding")) && header === "gzip";
					var chunkSize = 1048576;
					if (!hasByteServing) chunkSize = datalength;
					var doXHR = (from, to) => {
						if (from > to) abort("invalid range (" + from + ", " + to + ") or no bytes requested!");
						if (to > datalength - 1) abort("only " + datalength + " bytes available! programmer error!");
						var xhr = new XMLHttpRequest();
						xhr.open("GET", url, false);
						if (datalength !== chunkSize) xhr.setRequestHeader("Range", "bytes=" + from + "-" + to);
						xhr.responseType = "arraybuffer";
						if (xhr.overrideMimeType) xhr.overrideMimeType("text/plain; charset=x-user-defined");
						xhr.send(null);
						if (!(xhr.status >= 200 && xhr.status < 300 || xhr.status === 304)) abort("Couldn't load " + url + ". Status: " + xhr.status);
						if (xhr.response !== void 0) return new Uint8Array(xhr.response || []);
						return intArrayFromString(xhr.responseText || "", true);
					};
					var lazyArray = this;
					lazyArray.setDataGetter((chunkNum) => {
						var start = chunkNum * chunkSize;
						var end = (chunkNum + 1) * chunkSize - 1;
						end = Math.min(end, datalength - 1);
						if (typeof lazyArray.chunks[chunkNum] == "undefined") lazyArray.chunks[chunkNum] = doXHR(start, end);
						if (typeof lazyArray.chunks[chunkNum] == "undefined") abort("doXHR failed!");
						return lazyArray.chunks[chunkNum];
					});
					if (usesGzip || !datalength) {
						chunkSize = datalength = 1;
						datalength = this.getter(0).length;
						chunkSize = datalength;
						out("LazyFiles on gzip forces download of the whole file when length is accessed");
					}
					this._length = datalength;
					this._chunkSize = chunkSize;
					this.lengthKnown = true;
				}
				get length() {
					if (!this.lengthKnown) this.cacheLength();
					return this._length;
				}
				get chunkSize() {
					if (!this.lengthKnown) this.cacheLength();
					return this._chunkSize;
				}
			}
			if (globalThis.XMLHttpRequest) {
				if (!ENVIRONMENT_IS_WORKER) abort("Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc");
				var properties = {
					isDevice: false,
					contents: new LazyUint8Array()
				};
			} else var properties = {
				isDevice: false,
				url
			};
			var node = FS.createFile(parent, name, properties, canRead, canWrite);
			if (properties.contents) node.contents = properties.contents;
			else if (properties.url) {
				node.contents = null;
				node.url = properties.url;
			}
			Object.defineProperties(node, { usedBytes: { get: function() {
				return this.contents.length;
			} } });
			var stream_ops = {};
			for (const [key, fn] of Object.entries(node.stream_ops)) stream_ops[key] = (...args) => {
				FS.forceLoadFile(node);
				return fn(...args);
			};
			function writeChunks(stream, buffer, offset, length, position) {
				var contents = stream.node.contents;
				if (position >= contents.length) return 0;
				var size = Math.min(contents.length - position, length);
				if (contents.slice) for (var i = 0; i < size; i++) buffer[offset + i] = contents[position + i];
				else for (var i = 0; i < size; i++) buffer[offset + i] = contents.get(position + i);
				return size;
			}
			stream_ops.read = (stream, buffer, offset, length, position) => {
				FS.forceLoadFile(node);
				return writeChunks(stream, buffer, offset, length, position);
			};
			stream_ops.mmap = (stream, length, position, prot, flags) => {
				FS.forceLoadFile(node);
				var ptr = mmapAlloc(length);
				if (!ptr) throw new FS.ErrnoError(48);
				writeChunks(stream, HEAP8, ptr, length, position);
				return {
					ptr,
					allocated: true
				};
			};
			node.stream_ops = stream_ops;
			return node;
		}
	};
	var UTF8ToString = (ptr, maxBytesToRead, ignoreNul) => ptr ? UTF8ArrayToString(HEAPU8, ptr, maxBytesToRead, ignoreNul) : "";
	var SYSCALLS = {
		calculateAt(dirfd, path, allowEmpty) {
			if (PATH.isAbs(path)) return path;
			var dir;
			if (dirfd === -100) dir = FS.cwd();
			else dir = SYSCALLS.getStreamFromFD(dirfd).path;
			if (path.length == 0) {
				if (!allowEmpty) throw new FS.ErrnoError(44);
				return dir;
			}
			return dir + "/" + path;
		},
		writeStat(buf, stat) {
			HEAPU32[buf >> 2] = stat.dev;
			HEAPU32[buf + 4 >> 2] = stat.mode;
			HEAPU32[buf + 8 >> 2] = stat.nlink;
			HEAPU32[buf + 12 >> 2] = stat.uid;
			HEAPU32[buf + 16 >> 2] = stat.gid;
			HEAPU32[buf + 20 >> 2] = stat.rdev;
			HEAP64[buf + 24 >> 3] = BigInt(stat.size);
			HEAP32[buf + 32 >> 2] = 4096;
			HEAP32[buf + 36 >> 2] = stat.blocks;
			var atime = stat.atime.getTime();
			var mtime = stat.mtime.getTime();
			var ctime = stat.ctime.getTime();
			HEAP64[buf + 40 >> 3] = BigInt(Math.floor(atime / 1e3));
			HEAPU32[buf + 48 >> 2] = atime % 1e3 * 1e3 * 1e3;
			HEAP64[buf + 56 >> 3] = BigInt(Math.floor(mtime / 1e3));
			HEAPU32[buf + 64 >> 2] = mtime % 1e3 * 1e3 * 1e3;
			HEAP64[buf + 72 >> 3] = BigInt(Math.floor(ctime / 1e3));
			HEAPU32[buf + 80 >> 2] = ctime % 1e3 * 1e3 * 1e3;
			HEAP64[buf + 88 >> 3] = BigInt(stat.ino);
			return 0;
		},
		writeStatFs(buf, stats) {
			HEAPU32[buf + 4 >> 2] = stats.bsize;
			HEAPU32[buf + 60 >> 2] = stats.bsize;
			HEAP64[buf + 8 >> 3] = BigInt(stats.blocks);
			HEAP64[buf + 16 >> 3] = BigInt(stats.bfree);
			HEAP64[buf + 24 >> 3] = BigInt(stats.bavail);
			HEAP64[buf + 32 >> 3] = BigInt(stats.files);
			HEAP64[buf + 40 >> 3] = BigInt(stats.ffree);
			HEAPU32[buf + 48 >> 2] = stats.fsid;
			HEAPU32[buf + 64 >> 2] = stats.flags;
			HEAPU32[buf + 56 >> 2] = stats.namelen;
		},
		doMsync(addr, stream, len, flags, offset) {
			if (!FS.isFile(stream.node.mode)) throw new FS.ErrnoError(43);
			if (flags & 2) return 0;
			var buffer = HEAPU8.slice(addr, addr + len);
			FS.msync(stream, buffer, offset, len, flags);
		},
		getStreamFromFD(fd) {
			return FS.getStreamChecked(fd);
		},
		varargs: void 0,
		getStr(ptr) {
			return UTF8ToString(ptr);
		}
	};
	function _fd_close(fd) {
		try {
			var stream = SYSCALLS.getStreamFromFD(fd);
			FS.close(stream);
			return 0;
		} catch (e) {
			if (typeof FS == "undefined" || !(e.name === "ErrnoError")) throw e;
			return e.errno;
		}
	}
	var doReadv = (stream, iov, iovcnt, offset) => {
		var ret = 0;
		for (var i = 0; i < iovcnt; i++) {
			var ptr = HEAPU32[iov >> 2];
			var len = HEAPU32[iov + 4 >> 2];
			iov += 8;
			var curr = FS.read(stream, HEAP8, ptr, len, offset);
			if (curr < 0) return -1;
			ret += curr;
			if (curr < len) break;
			if (typeof offset != "undefined") offset += curr;
		}
		return ret;
	};
	function _fd_read(fd, iov, iovcnt, pnum) {
		try {
			var num = doReadv(SYSCALLS.getStreamFromFD(fd), iov, iovcnt);
			HEAPU32[pnum >> 2] = num;
			return 0;
		} catch (e) {
			if (typeof FS == "undefined" || !(e.name === "ErrnoError")) throw e;
			return e.errno;
		}
	}
	function _fd_seek(fd, offset, whence, newOffset) {
		offset = bigintToI53Checked(offset);
		try {
			if (isNaN(offset)) return 61;
			var stream = SYSCALLS.getStreamFromFD(fd);
			FS.llseek(stream, offset, whence);
			HEAP64[newOffset >> 3] = BigInt(stream.position);
			if (stream.getdents && offset === 0 && whence === 0) stream.getdents = null;
			return 0;
		} catch (e) {
			if (typeof FS == "undefined" || !(e.name === "ErrnoError")) throw e;
			return e.errno;
		}
	}
	var doWritev = (stream, iov, iovcnt, offset) => {
		var ret = 0;
		for (var i = 0; i < iovcnt; i++) {
			var ptr = HEAPU32[iov >> 2];
			var len = HEAPU32[iov + 4 >> 2];
			iov += 8;
			var curr = FS.write(stream, HEAP8, ptr, len, offset);
			if (curr < 0) return -1;
			ret += curr;
			if (curr < len) break;
			if (typeof offset != "undefined") offset += curr;
		}
		return ret;
	};
	function _fd_write(fd, iov, iovcnt, pnum) {
		try {
			var num = doWritev(SYSCALLS.getStreamFromFD(fd), iov, iovcnt);
			HEAPU32[pnum >> 2] = num;
			return 0;
		} catch (e) {
			if (typeof FS == "undefined" || !(e.name === "ErrnoError")) throw e;
			return e.errno;
		}
	}
	var runtimeKeepaliveCounter = 0;
	var keepRuntimeAlive = () => noExitRuntime || runtimeKeepaliveCounter > 0;
	var _proc_exit = (code) => {
		EXITSTATUS = code;
		if (!keepRuntimeAlive()) {
			Module["onExit"]?.(code);
			ABORT = true;
		}
		quit_(code, new ExitStatus(code));
	};
	var exitJS = (status, implicit) => {
		EXITSTATUS = status;
		_proc_exit(status);
	};
	var handleException = (e) => {
		if (e instanceof ExitStatus || e == "unwind") return EXITSTATUS;
		quit_(1, e);
	};
	var runAndAbortIfError = (func) => {
		try {
			return func();
		} catch (e) {
			abort(e);
		}
	};
	var _exit = exitJS;
	var maybeExit = () => {
		if (!keepRuntimeAlive()) try {
			_exit(EXITSTATUS);
		} catch (e) {
			handleException(e);
		}
	};
	var callUserCallback = (func) => {
		if (ABORT) return;
		try {
			func();
			maybeExit();
		} catch (e) {
			handleException(e);
		}
	};
	var Asyncify = {
		instrumentWasmImports(imports) {
			var importPattern = /^(invoke_.*|__asyncjs__.*)$/;
			for (let [x, original] of Object.entries(imports)) if (typeof original == "function") original.isAsync || importPattern.test(x);
		},
		instrumentFunction(original) {
			var wrapper = (...args) => {
				Asyncify.exportCallStack.push(original);
				try {
					return original(...args);
				} finally {
					if (!ABORT) {
						Asyncify.exportCallStack.pop();
						Asyncify.maybeStopUnwind();
					}
				}
			};
			Asyncify.funcWrappers.set(original, wrapper);
			return wrapper;
		},
		instrumentWasmExports(exports) {
			var ret = {};
			for (let [x, original] of Object.entries(exports)) if (typeof original == "function") ret[x] = Asyncify.instrumentFunction(original);
			else ret[x] = original;
			return ret;
		},
		State: {
			Normal: 0,
			Unwinding: 1,
			Rewinding: 2,
			Disabled: 3
		},
		state: 0,
		StackSize: 4096,
		currData: null,
		handleSleepReturnValue: 0,
		exportCallStack: [],
		callstackFuncToId: /* @__PURE__ */ new Map(),
		callStackIdToFunc: /* @__PURE__ */ new Map(),
		funcWrappers: /* @__PURE__ */ new Map(),
		callStackId: 0,
		asyncPromiseHandlers: null,
		sleepCallbacks: [],
		getCallStackId(func) {
			if (!Asyncify.callstackFuncToId.has(func)) {
				var id = Asyncify.callStackId++;
				Asyncify.callstackFuncToId.set(func, id);
				Asyncify.callStackIdToFunc.set(id, func);
			}
			return Asyncify.callstackFuncToId.get(func);
		},
		maybeStopUnwind() {
			if (Asyncify.currData && Asyncify.state === Asyncify.State.Unwinding && Asyncify.exportCallStack.length === 0) {
				Asyncify.state = Asyncify.State.Normal;
				runAndAbortIfError(_asyncify_stop_unwind);
				if (typeof Fibers != "undefined") Fibers.trampoline();
			}
		},
		whenDone() {
			return new Promise((resolve, reject) => {
				Asyncify.asyncPromiseHandlers = {
					resolve,
					reject
				};
			});
		},
		allocateData() {
			var ptr = _malloc(12 + Asyncify.StackSize);
			Asyncify.setDataHeader(ptr, ptr + 12, Asyncify.StackSize);
			Asyncify.setDataRewindFunc(ptr);
			return ptr;
		},
		setDataHeader(ptr, stack, stackSize) {
			HEAPU32[ptr >> 2] = stack;
			HEAPU32[ptr + 4 >> 2] = stack + stackSize;
		},
		setDataRewindFunc(ptr) {
			var bottomOfCallStack = Asyncify.exportCallStack[0];
			var rewindId = Asyncify.getCallStackId(bottomOfCallStack);
			HEAP32[ptr + 8 >> 2] = rewindId;
		},
		getDataRewindFunc(ptr) {
			var id = HEAP32[ptr + 8 >> 2];
			return Asyncify.callStackIdToFunc.get(id);
		},
		doRewind(ptr) {
			var original = Asyncify.getDataRewindFunc(ptr);
			return Asyncify.funcWrappers.get(original)();
		},
		handleSleep(startAsync) {
			if (ABORT) return;
			if (Asyncify.state === Asyncify.State.Normal) {
				var reachedCallback = false;
				var reachedAfterCallback = false;
				startAsync((handleSleepReturnValue = 0) => {
					if (ABORT) return;
					Asyncify.handleSleepReturnValue = handleSleepReturnValue;
					reachedCallback = true;
					if (!reachedAfterCallback) return;
					Asyncify.state = Asyncify.State.Rewinding;
					runAndAbortIfError(() => _asyncify_start_rewind(Asyncify.currData));
					if (typeof MainLoop != "undefined" && MainLoop.func) MainLoop.resume();
					var asyncWasmReturnValue, isError = false;
					try {
						asyncWasmReturnValue = Asyncify.doRewind(Asyncify.currData);
					} catch (err) {
						asyncWasmReturnValue = err;
						isError = true;
					}
					var handled = false;
					if (!Asyncify.currData) {
						var asyncPromiseHandlers = Asyncify.asyncPromiseHandlers;
						if (asyncPromiseHandlers) {
							Asyncify.asyncPromiseHandlers = null;
							(isError ? asyncPromiseHandlers.reject : asyncPromiseHandlers.resolve)(asyncWasmReturnValue);
							handled = true;
						}
					}
					if (isError && !handled) throw asyncWasmReturnValue;
				});
				reachedAfterCallback = true;
				if (!reachedCallback) {
					Asyncify.state = Asyncify.State.Unwinding;
					Asyncify.currData = Asyncify.allocateData();
					if (typeof MainLoop != "undefined" && MainLoop.func) MainLoop.pause();
					runAndAbortIfError(() => _asyncify_start_unwind(Asyncify.currData));
				}
			} else if (Asyncify.state === Asyncify.State.Rewinding) {
				Asyncify.state = Asyncify.State.Normal;
				runAndAbortIfError(_asyncify_stop_rewind);
				_free(Asyncify.currData);
				Asyncify.currData = null;
				Asyncify.sleepCallbacks.forEach(callUserCallback);
			} else abort(`invalid state: ${Asyncify.state}`);
			return Asyncify.handleSleepReturnValue;
		},
		handleAsync: (startAsync) => Asyncify.handleSleep((wakeUp) => {
			startAsync().then(wakeUp);
		})
	};
	FS.createPreloadedFile = FS_createPreloadedFile;
	FS.preloadFile = FS_preloadFile;
	FS.staticInit();
	if (Module["noExitRuntime"]) noExitRuntime = Module["noExitRuntime"];
	if (Module["preloadPlugins"]) preloadPlugins = Module["preloadPlugins"];
	if (Module["print"]) out = Module["print"];
	if (Module["printErr"]) err = Module["printErr"];
	if (Module["wasmBinary"]) wasmBinary = Module["wasmBinary"];
	if (Module["arguments"]) arguments_ = Module["arguments"];
	if (Module["thisProgram"]) thisProgram = Module["thisProgram"];
	if (Module["preInit"]) {
		if (typeof Module["preInit"] == "function") Module["preInit"] = [Module["preInit"]];
		while (Module["preInit"].length > 0) Module["preInit"].shift()();
	}
	var _malloc, _free, __initialize, _asyncify_start_unwind, _asyncify_stop_unwind, _asyncify_start_rewind, _asyncify_stop_rewind, wasmMemory;
	function assignWasmExports(wasmExports) {
		Module["_TVMFFIBacktrace"] = wasmExports["TVMFFIBacktrace"];
		_malloc = wasmExports["malloc"];
		Module["_TVMFFIEnvSetStream"] = wasmExports["TVMFFIEnvSetStream"];
		Module["_TVMFFIEnvGetStream"] = wasmExports["TVMFFIEnvGetStream"];
		Module["_TVMBackendAllocWorkspace"] = wasmExports["TVMBackendAllocWorkspace"];
		Module["_TVMBackendFreeWorkspace"] = wasmExports["TVMBackendFreeWorkspace"];
		Module["_TVMFFITypeGetOrAllocIndex"] = wasmExports["TVMFFITypeGetOrAllocIndex"];
		Module["__ZN3tvm7runtime13GetFileFormatERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES9_"] = wasmExports["_ZN3tvm7runtime13GetFileFormatERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES9_"];
		Module["__ZN3tvm7runtime15GetMetaFilePathERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEE"] = wasmExports["_ZN3tvm7runtime15GetMetaFilePathERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEE"];
		Module["__ZN3tvm7runtime18LoadBinaryFromFileERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEEPS7_"] = wasmExports["_ZN3tvm7runtime18LoadBinaryFromFileERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEEPS7_"];
		Module["__ZN3tvm7runtime16SaveBinaryToFileERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES9_"] = wasmExports["_ZN3tvm7runtime16SaveBinaryToFileERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES9_"];
		Module["__ZN3tvm7runtime18SaveMetaDataToFileERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEERKNS_3ffi3MapINSA_6StringENS0_12FunctionInfoEvEE"] = wasmExports["_ZN3tvm7runtime18SaveMetaDataToFileERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEERKNS_3ffi3MapINSA_6StringENS0_12FunctionInfoEvEE"];
		Module["_TVMFFIDataTypeToString"] = wasmExports["TVMFFIDataTypeToString"];
		Module["__ZN3tvm3ffi4json9StringifyERKNS0_3AnyERKNS0_8OptionalIivEE"] = wasmExports["_ZN3tvm3ffi4json9StringifyERKNS0_3AnyERKNS0_8OptionalIivEE"];
		Module["__ZN3tvm7runtime20LoadMetaDataFromFileERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEEPNS_3ffi3MapINSA_6StringENS0_12FunctionInfoEvEE"] = wasmExports["_ZN3tvm7runtime20LoadMetaDataFromFileERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEEPNS_3ffi3MapINSA_6StringENS0_12FunctionInfoEvEE"];
		Module["__ZN3tvm3ffi4json5ParseERKNS0_6StringEPS2_"] = wasmExports["_ZN3tvm3ffi4json5ParseERKNS0_6StringEPS2_"];
		Module["__ZN3tvm7runtime6Tensor5EmptyENS_3ffi5ShapeE10DLDataType8DLDeviceNS2_8OptionalINS2_6StringEvEE"] = wasmExports["_ZN3tvm7runtime6Tensor5EmptyENS_3ffi5ShapeE10DLDataType8DLDeviceNS2_8OptionalINS2_6StringEvEE"];
		Module["__ZN3tvm7runtime6Tensor11CopyToBytesEPK8DLTensorPvmS5_"] = wasmExports["_ZN3tvm7runtime6Tensor11CopyToBytesEPK8DLTensorPvmS5_"];
		Module["_TVMFFIFunctionGetGlobal"] = wasmExports["TVMFFIFunctionGetGlobal"];
		Module["__ZN3tvm7runtime6Tensor13CopyFromBytesEPK8DLTensorPvmS5_"] = wasmExports["_ZN3tvm7runtime6Tensor13CopyFromBytesEPK8DLTensorPvmS5_"];
		Module["__ZNK3tvm7runtime6Tensor10CreateViewENS_3ffi5ShapeE10DLDataTypey"] = wasmExports["_ZNK3tvm7runtime6Tensor10CreateViewENS_3ffi5ShapeE10DLDataTypey"];
		Module["__ZNK3tvm7runtime6Tensor11CopyToBytesEPvm"] = wasmExports["_ZNK3tvm7runtime6Tensor11CopyToBytesEPvm"];
		Module["__ZN3tvm7runtime6Tensor13CopyFromBytesEPKvm"] = wasmExports["_ZN3tvm7runtime6Tensor13CopyFromBytesEPKvm"];
		Module["__ZNK3tvm7runtime6Tensor6CopyToERK8DLDeviceNS_3ffi8OptionalINS5_6StringEvEE"] = wasmExports["_ZNK3tvm7runtime6Tensor6CopyToERK8DLDeviceNS_3ffi8OptionalINS5_6StringEvEE"];
		Module["__ZN3tvm7runtime6Tensor10CopyFromToEPK8DLTensorPS2_Pv"] = wasmExports["_ZN3tvm7runtime6Tensor10CopyFromToEPK8DLTensorPS2_Pv"];
		Module["__ZN3tvm7runtime6Tensor15IsStorageSharedEPK8DLTensorS4_"] = wasmExports["_ZN3tvm7runtime6Tensor15IsStorageSharedEPK8DLTensorS4_"];
		Module["__ZN3tvm7runtime5Timer5StartE8DLDevice"] = wasmExports["_ZN3tvm7runtime5Timer5StartE8DLDevice"];
		Module["__ZN3tvm7runtime6detail14LogMessageImplERKNSt3__212basic_stringIcNS2_11char_traitsIcEENS2_9allocatorIcEEEEiiSA_"] = wasmExports["_ZN3tvm7runtime6detail14LogMessageImplERKNSt3__212basic_stringIcNS2_11char_traitsIcEENS2_9allocatorIcEEEEiiSA_"];
		Module["_TVMFFIGetCustomAllocator"] = wasmExports["TVMFFIGetCustomAllocator"];
		Module["_TVMFFISetCustomAllocator"] = wasmExports["TVMFFISetCustomAllocator"];
		Module["_TVMFFIDataTypeFromString"] = wasmExports["TVMFFIDataTypeFromString"];
		Module["_TVMFFIErrorSetRaisedFromCStr"] = wasmExports["TVMFFIErrorSetRaisedFromCStr"];
		Module["_TVMFFIErrorSetRaisedFromCStrParts"] = wasmExports["TVMFFIErrorSetRaisedFromCStrParts"];
		Module["_TVMFFIErrorSetRaised"] = wasmExports["TVMFFIErrorSetRaised"];
		Module["_TVMFFIErrorMoveFromRaised"] = wasmExports["TVMFFIErrorMoveFromRaised"];
		Module["_TVMFFIErrorCreate"] = wasmExports["TVMFFIErrorCreate"];
		Module["_TVMFFIErrorCreateWithCauseAndExtraContext"] = wasmExports["TVMFFIErrorCreateWithCauseAndExtraContext"];
		Module["_TVMFFIEnvCheckSignals"] = wasmExports["TVMFFIEnvCheckSignals"];
		Module["_TVMFFIEnvRegisterCAPI"] = wasmExports["TVMFFIEnvRegisterCAPI"];
		Module["_TVMFFIEnvSetDLPackManagedTensorAllocator"] = wasmExports["TVMFFIEnvSetDLPackManagedTensorAllocator"];
		Module["_TVMFFIEnvGetDLPackManagedTensorAllocator"] = wasmExports["TVMFFIEnvGetDLPackManagedTensorAllocator"];
		Module["_TVMFFIEnvTensorAlloc"] = wasmExports["TVMFFIEnvTensorAlloc"];
		Module["__ZN3tvm3ffi6Module19VisitContextSymbolsERKNS0_13TypedFunctionIFvNS0_6StringEPvEEE"] = wasmExports["_ZN3tvm3ffi6Module19VisitContextSymbolsERKNS0_13TypedFunctionIFvNS0_6StringEPvEEE"];
		Module["_TVMFFIEnvModRegisterContextSymbol"] = wasmExports["TVMFFIEnvModRegisterContextSymbol"];
		Module["_TVMFFIEnvModRegisterSystemLibSymbol"] = wasmExports["TVMFFIEnvModRegisterSystemLibSymbol"];
		Module["__ZN3tvm3ffi6Module12LoadFromFileERKNS0_6StringE"] = wasmExports["_ZN3tvm3ffi6Module12LoadFromFileERKNS0_6StringE"];
		Module["_TVMFFIEnvModLookupFromImports"] = wasmExports["TVMFFIEnvModLookupFromImports"];
		Module["_TVMFFIFunctionCreate"] = wasmExports["TVMFFIFunctionCreate"];
		Module["_TVMFFIAnyViewToOwnedAny"] = wasmExports["TVMFFIAnyViewToOwnedAny"];
		Module["_TVMFFIFunctionSetGlobal"] = wasmExports["TVMFFIFunctionSetGlobal"];
		Module["_TVMFFIFunctionSetGlobalFromMethodInfo"] = wasmExports["TVMFFIFunctionSetGlobalFromMethodInfo"];
		Module["_TVMFFIFunctionCall"] = wasmExports["TVMFFIFunctionCall"];
		Module["_TVMFFIHandleInitOnce"] = wasmExports["TVMFFIHandleInitOnce"];
		Module["_TVMFFIHandleDeinitOnce"] = wasmExports["TVMFFIHandleDeinitOnce"];
		Module["_TVMFFIGetVersion"] = wasmExports["TVMFFIGetVersion"];
		Module["_TVMFFIObjectDecRef"] = wasmExports["TVMFFIObjectDecRef"];
		Module["_TVMFFIObjectIncRef"] = wasmExports["TVMFFIObjectIncRef"];
		Module["_TVMFFIObjectCreateOpaque"] = wasmExports["TVMFFIObjectCreateOpaque"];
		Module["_TVMFFITypeKeyToIndex"] = wasmExports["TVMFFITypeKeyToIndex"];
		Module["_TVMFFITypeRegisterField"] = wasmExports["TVMFFITypeRegisterField"];
		Module["_TVMFFITypeRegisterMethod"] = wasmExports["TVMFFITypeRegisterMethod"];
		Module["_TVMFFITypeRegisterMetadata"] = wasmExports["TVMFFITypeRegisterMetadata"];
		Module["_TVMFFITypeRegisterAttr"] = wasmExports["TVMFFITypeRegisterAttr"];
		Module["_TVMFFIGetTypeAttrColumn"] = wasmExports["TVMFFIGetTypeAttrColumn"];
		Module["_TVMFFIGetTypeInfo"] = wasmExports["TVMFFIGetTypeInfo"];
		Module["_TVMFFIStringFromByteArray"] = wasmExports["TVMFFIStringFromByteArray"];
		Module["_TVMFFIBytesFromByteArray"] = wasmExports["TVMFFIBytesFromByteArray"];
		Module["_TVMFFIBigIntFromByteArray"] = wasmExports["TVMFFIBigIntFromByteArray"];
		Module["_TVMFFIBigIntGetContentByteArray"] = wasmExports["TVMFFIBigIntGetContentByteArray"];
		Module["_TVMFFITensorCreateUnsafeView"] = wasmExports["TVMFFITensorCreateUnsafeView"];
		Module["_TVMFFITensorFromDLPack"] = wasmExports["TVMFFITensorFromDLPack"];
		Module["_TVMFFITensorFromDLPackVersioned"] = wasmExports["TVMFFITensorFromDLPackVersioned"];
		Module["_TVMFFITensorToDLPack"] = wasmExports["TVMFFITensorToDLPack"];
		Module["_TVMFFITensorToDLPackVersioned"] = wasmExports["TVMFFITensorToDLPackVersioned"];
		Module["___tvm_ffi_testing_dll_schema_id_int"] = wasmExports["__tvm_ffi_testing_dll_schema_id_int"];
		Module["___tvm_ffi_testing_dll_test_add_with_docstring"] = wasmExports["__tvm_ffi_testing_dll_test_add_with_docstring"];
		Module["_TVMFFITestingDummyTarget"] = wasmExports["TVMFFITestingDummyTarget"];
		Module["__ZN3tvm7runtime6memory7StorageC2ENS1_6BufferEPNS1_9AllocatorE"] = wasmExports["_ZN3tvm7runtime6memory7StorageC2ENS1_6BufferEPNS1_9AllocatorE"];
		Module["__ZN3tvm7runtime6memory10StorageObj17AllocTensorScopedExNS_3ffi5ShapeE10DLDataTypeNS3_6StringE"] = wasmExports["_ZN3tvm7runtime6memory10StorageObj17AllocTensorScopedExNS_3ffi5ShapeE10DLDataTypeNS3_6StringE"];
		Module["__ZN3tvm7runtime6memory10StorageObj11AllocTensorExNS_3ffi5ShapeE10DLDataType"] = wasmExports["_ZN3tvm7runtime6memory10StorageObj11AllocTensorExNS_3ffi5ShapeE10DLDataType"];
		Module["__ZN3tvm7runtime6memory13MemoryManager6GlobalEv"] = wasmExports["_ZN3tvm7runtime6memory13MemoryManager6GlobalEv"];
		Module["__ZN3tvm7runtime6memory13MemoryManager20GetOrCreateAllocatorE8DLDeviceNS1_13AllocatorTypeE"] = wasmExports["_ZN3tvm7runtime6memory13MemoryManager20GetOrCreateAllocatorE8DLDeviceNS1_13AllocatorTypeE"];
		Module["__ZN3tvm7runtime6memory13MemoryManager12GetAllocatorE8DLDeviceNS1_13AllocatorTypeE"] = wasmExports["_ZN3tvm7runtime6memory13MemoryManager12GetAllocatorE8DLDeviceNS1_13AllocatorTypeE"];
		Module["_TVMBackendAnyListSetPackedArg"] = wasmExports["TVMBackendAnyListSetPackedArg"];
		Module["_TVMBackendAnyListResetItem"] = wasmExports["TVMBackendAnyListResetItem"];
		Module["_TVMBackendAnyListMoveFromPackedReturn"] = wasmExports["TVMBackendAnyListMoveFromPackedReturn"];
		Module["__ZN3tvm7runtime2vm19TensorCacheMetadata11LoadFromStrERKNSt3__212basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEESB_"] = wasmExports["_ZN3tvm7runtime2vm19TensorCacheMetadata11LoadFromStrERKNSt3__212basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEESB_"];
		Module["__ZN3tvm7runtime2vm19TensorCacheMetadata4LoadERKNSt3__212basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEE"] = wasmExports["_ZN3tvm7runtime2vm19TensorCacheMetadata4LoadERKNSt3__212basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEE"];
		Module["__ZNK3tvm7runtime2vm19TensorCacheMetadata10FileRecord11ParamRecord4LoadE8DLDevicePKNSt3__212basic_stringIcNS6_11char_traitsIcEENS6_9allocatorIcEEEEPNS_3ffi8OptionalINS0_6TensorEvEE"] = wasmExports["_ZNK3tvm7runtime2vm19TensorCacheMetadata10FileRecord11ParamRecord4LoadE8DLDevicePKNSt3__212basic_stringIcNS6_11char_traitsIcEENS6_9allocatorIcEEEEPNS_3ffi8OptionalINS0_6TensorEvEE"];
		Module["__ZNK3tvm7runtime2vm19TensorCacheMetadata10FileRecord4LoadE8DLDeviceRKNSt3__212basic_stringIcNS5_11char_traitsIcEENS5_9allocatorIcEEEEPSB_PNS_3ffi8OptionalINS0_6TensorEvEE"] = wasmExports["_ZNK3tvm7runtime2vm19TensorCacheMetadata10FileRecord4LoadE8DLDeviceRKNSt3__212basic_stringIcNS5_11char_traitsIcEENS5_9allocatorIcEEEEPSB_PNS_3ffi8OptionalINS0_6TensorEvEE"];
		Module["_TVMBackendParallelLaunch"] = wasmExports["TVMBackendParallelLaunch"];
		Module["_TVMBackendParallelBarrier"] = wasmExports["TVMBackendParallelBarrier"];
		Module["__ZN3tvm7runtime6detail12LogFatalImplERKNSt3__212basic_stringIcNS2_11char_traitsIcEENS2_9allocatorIcEEEEiSA_"] = wasmExports["_ZN3tvm7runtime6detail12LogFatalImplERKNSt3__212basic_stringIcNS2_11char_traitsIcEENS2_9allocatorIcEEEEiSA_"];
		_free = wasmExports["free"];
		Module["__ZN3tvm7runtime6memory7StorageC1ENS1_6BufferEPNS1_9AllocatorE"] = wasmExports["_ZN3tvm7runtime6memory7StorageC1ENS1_6BufferEPNS1_9AllocatorE"];
		Module["_TVMWasmAllocSpace"] = wasmExports["TVMWasmAllocSpace"];
		Module["_TVMWasmFreeSpace"] = wasmExports["TVMWasmFreeSpace"];
		Module["_TVMFFIWasmFunctionCreate"] = wasmExports["TVMFFIWasmFunctionCreate"];
		Module["_TVMFFIWasmGetLastError"] = wasmExports["TVMFFIWasmGetLastError"];
		__initialize = Module["__initialize"] = wasmExports["_initialize"];
		_asyncify_start_unwind = wasmExports["asyncify_start_unwind"];
		_asyncify_stop_unwind = wasmExports["asyncify_stop_unwind"];
		_asyncify_start_rewind = wasmExports["asyncify_start_rewind"];
		_asyncify_stop_rewind = wasmExports["asyncify_stop_rewind"];
		wasmMemory = wasmExports["memory"];
		wasmExports["__indirect_function_table"];
	}
	var wasmImports = {
		TVMFFIWasmFunctionDeleter: _TVMFFIWasmFunctionDeleter,
		TVMFFIWasmSafeCall: _TVMFFIWasmSafeCall,
		clock_time_get: _clock_time_get,
		emscripten_notify_memory_growth: _emscripten_notify_memory_growth,
		environ_get: _environ_get,
		environ_sizes_get: _environ_sizes_get,
		fd_close: _fd_close,
		fd_read: _fd_read,
		fd_seek: _fd_seek,
		fd_write: _fd_write
	};
	function callMain(args = []) {
		var entryFunction = __initialize;
		[thisProgram].concat(args);
		try {
			entryFunction();
			var ret = 0;
			exitJS(ret, true);
			return ret;
		} catch (e) {
			return handleException(e);
		}
	}
	function run(args = arguments_) {
		if (runDependencies > 0) {
			dependenciesFulfilled = run;
			return;
		}
		preRun();
		if (runDependencies > 0) {
			dependenciesFulfilled = run;
			return;
		}
		function doRun() {
			Module["calledRun"] = true;
			if (ABORT) return;
			initRuntime();
			Module["onRuntimeInitialized"]?.();
			if (!(Module["noInitialRun"] || false)) callMain(args);
			postRun();
		}
		if (Module["setStatus"]) {
			Module["setStatus"]("Running...");
			setTimeout(() => {
				setTimeout(() => Module["setStatus"](""), 1);
				doRun();
			}, 1);
		} else doRun();
	}
	var wasmExports;
	createWasm();
	run();
	this.Module = Module;
	this.start = Module.wasmLibraryProvider.start;
	this.imports = Module.wasmLibraryProvider.imports;
	this.wasiImport = this.imports["wasi_snapshot_preview1"];
}
// [TVM]/compact.ts
/**
* Get performance measurement.
*/
function getPerformance() {
	if (typeof performance === "undefined") return require___vite_browser_external().performance;
	else return performance;
}
/**
* Create a WASI based on current environment.
*
* @return A wasi that can run on broswer or local.
*/
function createPolyfillWASI() {
	return new EmccWASI();
}
// [TVM]/runtime.ts
/**
* TVM JS Wasm Runtime library.
*/
/**
* @internal
* FFI Library wrapper, maintains most runtime states.
*/
var FFILibrary = class {
	wasm32;
	memory;
	exports;
	webGPUContext;
	wasmInstance;
	recycledCallStacks = [];
	constructor(wasmInstance, imports) {
		this.wasmInstance = wasmInstance;
		this.memory = new Memory(this.detectWasmMemory(this.wasmInstance, imports));
		assert(this.wasmInstance.exports !== void 0, "Expect the library module contains exports");
		this.exports = this.wasmInstance.exports;
		this.wasm32 = this.memory.wasm32;
		this.validateInstance();
	}
	dispose() {
		while (this.recycledCallStacks.length != 0) this.recycledCallStacks.pop().dispose();
		this.webGPUContext?.dispose();
	}
	sizeofPtr() {
		return this.memory.sizeofPtr();
	}
	checkCall(code) {
		if (code != 0) {
			const msgPtr = this.exports.TVMFFIWasmGetLastError();
			throw new Error(this.memory.loadCString(msgPtr));
		}
	}
	getOrAllocCallStack() {
		if (this.recycledCallStacks.length != 0) return this.recycledCallStacks.pop();
		return new CachedCallStack(this.memory, this.exports.TVMWasmAllocSpace, this.exports.TVMWasmFreeSpace);
	}
	recycleCallStack(callstack) {
		callstack.reset();
		this.recycledCallStacks.push(callstack);
	}
	validateInstance() {
		this.checkExports(["TVMWasmAllocSpace", "TVMWasmFreeSpace"]);
	}
	checkExports(funcNames) {
		const missList = [];
		for (const name of funcNames) if (!(this.exports[name] instanceof Function)) missList.push(name);
		if (missList.length != 0) throw new Error("Cannot find " + missList + " in exports");
	}
	detectWasmMemory(instance, imports) {
		if (instance.exports.memory instanceof WebAssembly.Memory) return instance.exports.memory;
		if (imports.env && imports.env.memory instanceof WebAssembly.Memory) return imports.env.memory;
		throw new Error("Cannt detect wasm memory from imports " + imports + " or exports" + instance.exports);
	}
};
/**
* @internal
* Manages extra runtime context for the runtime.
*/
var RuntimeContext = class {
	functionListGlobalNamesFunctor;
	moduleGetFunction;
	moduleImport;
	tensorEmpty;
	tensorCopyFromTo;
	tensorCopyFromJSBytes;
	tensorCopyToJSBytes;
	arrayGetItem;
	arrayGetSize;
	arrayMake;
	arrayConcat;
	getSysLib;
	tensorCacheGet;
	tensorCacheUpdate;
	tensorCacheRemove;
	tensorCacheClear;
	arrayDecodeStorage;
	arrayDecodeBF16ToF32Inplace;
	paramModuleFromCache;
	paramModuleFromCacheByName;
	makeShapeTuple;
	tensorCreateView;
	sampleTopPFromLogits;
	sampleTopPFromProb;
	applyRepetitionPenalty;
	applyPresenceAndFrequencyPenalty;
	applySoftmaxWithTemperature;
	concatEmbeddings;
	bool;
	autoDisposeScope = [];
	constructor(getGlobalFunc) {
		this.functionListGlobalNamesFunctor = getGlobalFunc("ffi.FunctionListGlobalNamesFunctor");
		this.moduleGetFunction = getGlobalFunc("ffi.ModuleGetFunction");
		this.moduleImport = getGlobalFunc("ffi.ModuleImportModule");
		this.tensorEmpty = getGlobalFunc("runtime.TVMTensorAllocWithScope");
		this.tensorCopyFromTo = getGlobalFunc("runtime.TVMTensorCopyFromTo");
		this.tensorCopyFromJSBytes = getGlobalFunc("tvmjs.runtime.TensorCopyFromBytes");
		this.tensorCopyToJSBytes = getGlobalFunc("tvmjs.runtime.TensorCopyToBytes");
		this.arrayGetItem = getGlobalFunc("ffi.ArrayGetItem");
		this.arrayGetSize = getGlobalFunc("ffi.ArraySize");
		this.arrayMake = getGlobalFunc("ffi.Array");
		this.arrayConcat = getGlobalFunc("tvmjs.runtime.ArrayConcat");
		this.getSysLib = getGlobalFunc("ffi.SystemLib");
		this.tensorCacheGet = getGlobalFunc("vm.builtin.tensor_cache.get");
		this.tensorCacheRemove = getGlobalFunc("vm.builtin.tensor_cache.remove");
		this.tensorCacheUpdate = getGlobalFunc("vm.builtin.tensor_cache.update");
		this.tensorCacheClear = getGlobalFunc("vm.builtin.tensor_cache.clear");
		this.arrayDecodeStorage = getGlobalFunc("tvmjs.array.decode_storage");
		this.arrayDecodeBF16ToF32Inplace = void 0;
		this.paramModuleFromCache = getGlobalFunc("vm.builtin.param_module_from_cache");
		this.paramModuleFromCacheByName = getGlobalFunc("vm.builtin.param_module_from_cache_by_name");
		this.makeShapeTuple = getGlobalFunc("ffi.Shape");
		this.tensorCreateView = getGlobalFunc("runtime.TVMTensorCreateView");
		this.sampleTopPFromLogits = getGlobalFunc("vm.builtin.sample_top_p_from_logits");
		this.sampleTopPFromProb = getGlobalFunc("vm.builtin.sample_top_p_from_prob");
		this.applyRepetitionPenalty = getGlobalFunc("vm.builtin.apply_repetition_penalty");
		this.applyPresenceAndFrequencyPenalty = getGlobalFunc("vm.builtin.apply_presence_and_frequency_penalty");
		this.applySoftmaxWithTemperature = getGlobalFunc("vm.builtin.apply_softmax_with_temperature");
		this.concatEmbeddings = getGlobalFunc("tvmjs.runtime.ConcatEmbeddings");
	}
	dispose() {
		this.tensorCacheClear.dispose();
		this.arrayGetItem.dispose();
		this.arrayGetSize.dispose();
		this.arrayMake.dispose();
		this.arrayConcat.dispose();
		this.tensorCacheGet.dispose();
		this.tensorCacheRemove.dispose();
		this.tensorCacheUpdate.dispose();
		this.arrayDecodeStorage.dispose();
		this.arrayDecodeBF16ToF32Inplace?.dispose();
		this.paramModuleFromCache.dispose();
		this.paramModuleFromCacheByName.dispose();
		this.makeShapeTuple.dispose();
		this.tensorCreateView.dispose();
		this.sampleTopPFromLogits.dispose();
		this.applyRepetitionPenalty.dispose();
		this.applyPresenceAndFrequencyPenalty.dispose();
		this.applySoftmaxWithTemperature.dispose();
		this.concatEmbeddings?.dispose();
	}
	beginScope() {
		this.autoDisposeScope.push([]);
	}
	endScope() {
		if (this.autoDisposeScope.length === 0) throw Error("tvm.endScope called when the stack is empty.");
		const currScope = this.autoDisposeScope.pop();
		for (let i = 0; i < currScope.length; ++i) {
			const val = currScope[i];
			if (val !== void 0) val.dispose();
		}
	}
	/**
	* Track object for dispose in current scope.
	*
	* @param obj The object to be tracked.
	* @returns the same object.
	* Note: This function only needs to be called for raw system C API values.
	*       The return value of PackedFunc will be automatically tracked.
	*/
	attachToCurrentScope(obj) {
		if (this.autoDisposeScope.length === 0) throw Error("Must call beginScope to use functions that returns TVM objects");
		this.autoDisposeScope[this.autoDisposeScope.length - 1].push(obj);
		return obj;
	}
	moveToParentScope(obj) {
		this.detachFromCurrentScope(obj);
		if (this.autoDisposeScope.length < 2) throw Error("moveToParentScope: Parent scope do not exist");
		this.autoDisposeScope[this.autoDisposeScope.length - 2].push(obj);
		return obj;
	}
	detachFromCurrentScope(obj) {
		const currScope = this.autoDisposeScope[this.autoDisposeScope.length - 1];
		let occurrence = 0;
		for (let i = 0; i < currScope.length; ++i) if (currScope[i] === obj) {
			occurrence += 1;
			currScope[i] = void 0;
		}
		if (occurrence === 0) throw Error("Cannot find obj in the current auto conversion pool");
		if (occurrence > 1) throw Error("Value attached to scope multiple times");
		return obj;
	}
};
/**
* A typed scalar constant used to represent a typed number
* argument to PackedFunc calls.
*/
var Scalar = class {
	/** The value. */
	value;
	/** The data type of the scalar. */
	dtype;
	constructor(value, dtype) {
		this.value = value;
		this.dtype = dtype;
	}
};
var DeviceEnumToStr = {
	1: "cpu",
	2: "cuda",
	4: "opencl",
	8: "metal",
	15: "webgpu"
};
var DeviceStrToEnum = {
	cpu: 1,
	cuda: 2,
	cl: 4,
	opencl: 4,
	vulkan: 7,
	metal: 8,
	webgpu: 15
};
/**
* Represent a runtime context where a Tensor can reside.
*/
var DLDevice = class {
	/** The device type code of the device. */
	deviceType;
	/** The device index. */
	deviceId;
	lib;
	constructor(deviceType, deviceId, lib) {
		const tp = typeof deviceType;
		if (tp === "string") {
			this.deviceType = DeviceStrToEnum[deviceType];
			if (this.deviceType === void 0) throw new Error("Cannot recogonize deviceType " + deviceType);
		} else if (tp === "number") this.deviceType = deviceType;
		else throw new Error("Cannot take type " + tp + " as deviceType");
		this.deviceId = deviceId;
		this.lib = lib;
	}
	/**
	* Synchronize the device
	*/
	async sync() {
		if (this.deviceType === DeviceStrToEnum.webgpu) {
			assert(this.lib.webGPUContext !== void 0);
			await this.lib.webGPUContext.sync();
		}
	}
	toString() {
		return DeviceEnumToStr[this.deviceType] + ":" + this.deviceId.toString();
	}
};
var DLDataTypeCodeToStr = {
	0: "int",
	1: "uint",
	2: "float",
	3: "handle"
};
/**
* Runtime data type of Tensor.
*/
var DLDataType = class {
	/** The type code */
	code;
	/** Number of bits in the data type. */
	bits;
	/** Number of vector lanes. */
	lanes;
	constructor(code, bits, lanes) {
		this.code = code;
		this.bits = bits;
		this.lanes = lanes;
	}
	toString() {
		const ret = DLDataTypeCodeToStr[this.code] + this.bits.toString();
		if (this.lanes != 1) return ret + "x" + this.lanes.toString();
		else return ret;
	}
	numStorageBytes() {
		return this.bits * this.lanes + 7 >> 3;
	}
};
/**
* Generic object base
*/
var TVMObject = class {
	handle;
	lib;
	ctx;
	constructor(handle, lib, ctx) {
		this.handle = handle;
		this.lib = lib;
		this.ctx = ctx;
	}
	dispose() {
		if (this.handle != 0) {
			this.lib.checkCall(this.lib.exports.TVMFFIObjectDecRef(this.handle));
			this.handle = 0;
		}
	}
	/**
	* Get handle of module, check it is not null.
	*
	* @param requireNotNull require handle is not null.
	* @returns The handle.
	*/
	getHandle(requireNotNull = true) {
		if (requireNotNull && this.handle === 0) throw Error("Object has already been disposed");
		return this.handle;
	}
	/** get the type index of the object */
	typeIndex() {
		if (this.handle === 0) throw Error("The current Object has already been disposed");
		return this.lib.memory.loadObjectTypeIndex(this.handle);
	}
	/** get the type key of the object */
	typeKey() {
		const type_index = this.typeIndex();
		const typeInfoPtr = this.lib.exports.TVMFFIGetTypeInfo(type_index);
		return this.lib.memory.loadTypeInfoTypeKey(typeInfoPtr);
	}
};
/**
* Cell holds the PackedFunc object.
*/
var PackedFuncCell = class extends TVMObject {
	constructor(handle, lib, ctx) {
		super(handle, lib, ctx);
	}
};
/**
* Tensor( n-dimnesional array).
*/
var Tensor = class Tensor extends TVMObject {
	/** Number of dimensions. */
	ndim;
	/** Data type of the array. */
	dtype;
	/** Shape of the array. */
	shape;
	/** Device of the array. */
	device;
	/** Whether it is a temporary view that can become invalid after the call. */
	isView;
	byteOffset;
	dltensor;
	dataPtr;
	dlDataType;
	constructor(handle, lib, ctx, isView) {
		super(isView ? 0 : handle, lib, ctx);
		this.isView = isView;
		if (this.isView) this.dltensor = handle;
		else this.dltensor = this.getDLTensorFromArrayHandle(this.handle);
		const arrayOffsetContext = 0 + this.lib.sizeofPtr();
		const arrayOffsetDevType = arrayOffsetContext;
		const arrayOffsetDevId = arrayOffsetContext + SizeOf.I32;
		const arrayOffsetNdim = arrayOffsetContext + SizeOf.DLDevice;
		const arrayOffsetDtype = arrayOffsetNdim + SizeOf.I32;
		const arrayOffsetDtypeCode = arrayOffsetDtype;
		const arrayOffsetDtypeBits = arrayOffsetDtype + SizeOf.U8;
		const arrayOffsetDtypeLanes = arrayOffsetDtypeBits + SizeOf.U8;
		const arrayOffsetShape = arrayOffsetDtype + SizeOf.DLDataType;
		const byteOffsetUnaligned = arrayOffsetShape + this.lib.sizeofPtr() + this.lib.sizeofPtr();
		const arrayOffsetByteOffset = Math.ceil(byteOffsetUnaligned / SizeOf.I64) * SizeOf.I64;
		this.dataPtr = lib.memory.loadPointer(this.dltensor);
		this.ndim = lib.memory.loadI32(this.dltensor + arrayOffsetNdim);
		const cshapePtr = lib.memory.loadPointer(this.dltensor + arrayOffsetShape);
		this.shape = [];
		for (let i = 0; i < this.ndim; ++i) this.shape.push(lib.memory.loadI64(cshapePtr + i * SizeOf.I64));
		const code = lib.memory.loadU8(this.dltensor + arrayOffsetDtypeCode);
		const bits = lib.memory.loadU8(this.dltensor + arrayOffsetDtypeBits);
		const lanes = lib.memory.loadU16(this.dltensor + arrayOffsetDtypeLanes);
		this.dlDataType = new DLDataType(code, bits, lanes);
		this.dtype = this.dlDataType.toString();
		const deviceType = lib.memory.loadI32(this.dltensor + arrayOffsetDevType);
		const deviceId = lib.memory.loadI32(this.dltensor + arrayOffsetDevId);
		this.device = new DLDevice(deviceType, deviceId, lib);
		this.byteOffset = lib.memory.loadU64(this.dltensor + arrayOffsetByteOffset);
	}
	/**
	* Create a view of the array.
	* @param shape The shape of the view.
	* @param dtype The data type of the new array.
	* @returns The new sliced ndarray.
	*/
	view(shape, dtype) {
		const shapeArray = shape.map((value) => new Scalar(value, "int"));
		if (dtype === void 0) dtype = this.dtype;
		return this.ctx.tensorCreateView(this, this.ctx.makeShapeTuple(...shapeArray), this.dtype, new Scalar(0, "int"));
	}
	/**
	* Get dataPtr of NDarray
	*
	* @returns The handle.
	*/
	getDataPtr() {
		if (this.handle === 0) throw Error("Tensor has already been disposed");
		return this.dataPtr;
	}
	/**
	* Return the effective address of a CPU Tensor's storage.
	* @returns The address in Wasm linear memory.
	* @internal
	*/
	getCPUDataAddress() {
		if (this.device.deviceType !== DeviceStrToEnum.cpu) throw new Error("Can only obtain a linear-memory address for a CPU Tensor");
		const address = this.getDataPtr() + this.byteOffset;
		if (!Number.isSafeInteger(address) || address < 0) throw new Error("Invalid CPU Tensor storage address");
		return address;
	}
	/**
	* Copy data from another Tensor or javascript array.
	* The number of elements must match.
	*
	* @param data The source data array.
	* @returns this
	*/
	copyFrom(data) {
		if (data instanceof Tensor) {
			this.ctx.tensorCopyFromTo(data, this);
			return this;
		} else {
			const size = this.shape.reduce((a, b) => {
				return a * b;
			}, 1);
			if (data.length != size) throw new Error("data size and shape mismatch data.length" + data.length + " vs " + size);
			let buffer;
			if (this.dtype === "float32") buffer = Float32Array.from(data).buffer;
			else if (this.dtype === "float64") buffer = Float64Array.from(data).buffer;
			else if (this.dtype === "int32") buffer = Int32Array.from(data).buffer;
			else if (this.dtype === "int8") buffer = Int8Array.from(data).buffer;
			else if (this.dtype === "uint8") buffer = Uint8Array.from(data).buffer;
			else if (this.dtype === "uint32") buffer = Uint32Array.from(data).buffer;
			else throw new Error("Unsupported data type " + this.dtype);
			return this.copyFromRawBytes(new Uint8Array(buffer));
		}
	}
	/**
	* Copy data from raw bytes.
	* @param data Uint8Array of bytes.
	* @returns this
	*/
	copyFromRawBytes(data) {
		const nbytes = this.numStorageBytes();
		if (nbytes != data.length) throw new Error("Expect the data's length equals nbytes=" + nbytes);
		if (this.device.deviceType === DeviceStrToEnum.webgpu) {
			if (this.byteOffset % 4 != 0 || data.length % 4 != 0) throw new Error("WebGPU raw byte copies require four-byte-aligned offsets and sizes");
			const webGPUContext = this.lib.webGPUContext;
			if (webGPUContext === void 0) throw new Error("WebGPU context is not initialized");
			webGPUContext.copyRawBytesToBuffer(data, this.getDataPtr(), this.byteOffset, data.length);
			return this;
		}
		this.ctx.tensorCopyFromJSBytes(this, data);
		return this;
	}
	numStorageBytes() {
		let totalBits = this.dlDataType.bits * this.dlDataType.lanes;
		if (!Number.isSafeInteger(totalBits) || totalBits < 0) throw new Error(`Invalid tensor dtype bit width: ${totalBits}`);
		for (const dim of this.shape) {
			if (!Number.isSafeInteger(dim) || dim < 0) throw new Error(`Invalid tensor dimension: ${dim}`);
			totalBits *= dim;
			if (!Number.isSafeInteger(totalBits)) throw new Error("Tensor storage size exceeds JavaScript's safe integer range");
		}
		return Math.ceil(totalBits / 8);
	}
	/**
	* Return a copied Uint8Array of the raw bytes in the Tensor.
	* @returns The result array.
	*/
	toRawBytes() {
		if (this.device.deviceType != DeviceStrToEnum.cpu) throw new Error("Can only sync copy CPU array, use cpu_arr.copyfrom(gpu_arr) then sync instead.");
		return this.ctx.tensorCopyToJSBytes(this);
	}
	/**
	* Return a TypedArray copy of the Tensor, the specific type depends on
	* the dtype of the Tensor.
	* @returns The result array.
	*/
	toArray() {
		const stype = this.dtype;
		if (stype === "float32") return new Float32Array(this.toRawBytes().buffer);
		else if (stype === "float64") return new Float64Array(this.toRawBytes().buffer);
		else if (stype === "int32") return new Int32Array(this.toRawBytes().buffer);
		else if (stype === "int8") return new Int8Array(this.toRawBytes().buffer);
		else if (stype === "uint8") return new Uint8Array(this.toRawBytes().buffer);
		else throw new Error("Unsupported data type " + this.dtype);
	}
	getDLTensorFromArrayHandle(handle) {
		return handle + SizeOf.ObjectHeader;
	}
};
/**
* Runtime Module.
*/
var Module = class extends TVMObject {
	constructor(handle, lib, ctx) {
		super(handle, lib, ctx);
	}
	/**
	* Get a function in the module.
	* @param name The name of the function.
	* @param queryImports Whether to also query imports
	* @returns The result function.
	*/
	getFunction(name, queryImports = true) {
		return this.ctx.moduleGetFunction(this, name, queryImports);
	}
	/**
	* Import another module into the current runtime module.
	* @param mod The module to be imported.
	*/
	importModule(mod) {
		this.ctx.moduleImport(this, mod);
	}
};
/** Runtime array object. */
var TVMArray = class extends TVMObject {
	constructor(handle, lib, ctx) {
		super(handle, lib, ctx);
	}
	/**
	* @returns the size of the array.
	*/
	size() {
		return this.ctx.arrayGetSize(this);
	}
	/**
	* Get index-th element of the array
	* @param index the array index.
	* @returns The element.
	*/
	get(index) {
		return this.ctx.arrayGetItem(this, new Scalar(index, "int32"));
	}
};
var VMAllocatorKind = /* @__PURE__ */ function(VMAllocatorKind) {
	VMAllocatorKind[VMAllocatorKind["NAIVE_ALLOCATOR"] = 1] = "NAIVE_ALLOCATOR";
	VMAllocatorKind[VMAllocatorKind["POOLED_ALLOCATOR"] = 2] = "POOLED_ALLOCATOR";
	return VMAllocatorKind;
}({});
/**
*  VirtualMachine Executor.
*
*  This is a thin wrapper of the underlying TVM module.
*  you can also directly call set_input, run, and get_output
*  of underlying module functions
*/
var VirtualMachine = class {
	mod;
	/**
	* Constructor
	* @param mod The underlying module, need to be detached.
	* @param device The main device ro run VM on.
	*/
	constructor(mod, device) {
		this.mod = mod;
		this.mod.getFunction("vm_initialization")(new Scalar(device.deviceType, "int"), new Scalar(device.deviceId, "int"), new Scalar(VMAllocatorKind.POOLED_ALLOCATOR, "int"), new Scalar(DeviceStrToEnum.cpu, "int"), new Scalar(0, "int"), new Scalar(VMAllocatorKind.POOLED_ALLOCATOR, "int"));
	}
	dispose() {
		this.mod.dispose();
	}
	/**
	* Get a function in the VM module.
	* @param name The name of the function.
	* @returns The result function.
	*/
	getFunction(name) {
		return this.mod.getFunction(name);
	}
	/**
	* Get the internal module.
	*/
	getInternalModule() {
		return this.mod;
	}
};
/** Code used as the first argument of the async callback. */
var AsyncCallbackCode = /* @__PURE__ */ function(AsyncCallbackCode) {
	AsyncCallbackCode[AsyncCallbackCode["kReturn"] = 4] = "kReturn";
	AsyncCallbackCode[AsyncCallbackCode["kException"] = 5] = "kException";
	return AsyncCallbackCode;
}(AsyncCallbackCode || {});
/**
* TVM runtime instance.
*
* All objects(Tensor, Module, PackedFunc) returned by TVM runtim function call
* and PackedFunc instance are tracked through a scope mechanism that will get
* auto-released when we call EndScope.
*
* This is necessarily to be able to release the underlying WASM and WebGPU memory that
* are not tracked through JS native garbage collection mechanism.
*
* This does mean that we have to get familar with the following functions:
* - {@link beginScope}
* - {@link endScope}
* - {@link withNewScope}
* - {@link attachToCurrentScope}
* - {@link detachFromCurrentScope}
*/
var Instance = class {
	memory;
	exports;
	cacheMetadata = {};
	lib;
	env;
	objFactory;
	ctx;
	asyncifyHandler;
	initProgressCallback = [];
	rng;
	deviceLostIsError = true;
	cacheState = new CacheState();
	/**
	* Internal function(registered by the runtime)
	*/
	wasmCreateLibraryModule;
	/**
	* Constructor
	*
	* importObject can also be a {@link LibraryProvider} object,
	* a WASI object, or an object containing wasmLibraryProvider field.
	*
	* @param wasmModule The input module or instance.
	* @param importObject The imports to initialize the wasmInstance if it is not provided.
	* @param wasmInstance Additional wasm instance argument for deferred construction.
	* @param env Directly specified environment module.
	*
	* @see Please use the async version {@link instantiate} when targeting browsers.
	*/
	constructor(wasmModule, importObject = {}, wasmInstance, env) {
		if (wasmInstance instanceof WebAssembly.Instance) assert(env instanceof Environment, "env must be provided when passing in instance");
		else {
			assert(env === void 0);
			env = new Environment(importObject);
			wasmInstance = new WebAssembly.Instance(wasmModule, env.imports);
		}
		env.start(wasmInstance);
		this.env = env;
		this.lib = new FFILibrary(wasmInstance, env.imports);
		this.memory = this.lib.memory;
		this.exports = this.lib.exports;
		this.asyncifyHandler = new AsyncifyHandler(this.exports, this.memory.memory);
		this.objFactory = /* @__PURE__ */ new Map();
		this.ctx = new RuntimeContext((name) => {
			return this.getGlobalFuncInternal(name, false);
		});
		this.ctx.arrayDecodeBF16ToF32Inplace = this.getGlobalFuncInternalOptional("tvmjs.array.decode_bf16_to_f32_inplace", false);
		this.registerEnvGlobalPackedFuncs();
		this.registerObjectFactoryFuncs();
		this.rng = new LinearCongruentialGenerator();
	}
	/**
	* Benchmark stable execution of the run function.
	*
	* @param run The run function.
	* @param dev The device to sync during each run.
	* @param number The number of times to compute the average.
	* @param repeat The number of times to repeat the run.
	*/
	async benchmark(run, dev, number = 10, repeat = 1) {
		const perf = getPerformance();
		const results = [];
		this.withNewScope(run);
		await dev.sync();
		for (let k = 0; k < repeat; ++k) {
			const tstart = perf.now();
			for (let i = 0; i < number; ++i) this.withNewScope(run);
			await dev.sync();
			const tend = perf.now();
			results.push((tend - tstart) / number);
		}
		return results;
	}
	/**
	* Check whether we enabled asyncify mode
	* @returns The asynctify mode toggle
	*/
	asyncifyEnabled() {
		return this.asyncifyHandler.enabled();
	}
	dispose() {
		this.deviceLostIsError = false;
		this.cacheState.dispose();
		this.ctx.dispose();
		this.lib.dispose();
	}
	/**
	* Obtain the runtime information in readable format.
	*/
	runtimeStatsText() {
		if (this.lib.webGPUContext !== void 0) return this.lib.webGPUContext.runtimeStatsText();
		else return "";
	}
	/**
	* Begin a new scope for tracking object disposal.
	*/
	beginScope() {
		this.ctx.beginScope();
	}
	/**
	* End a scope and release all created TVM objects
	* under the current scope.
	*
	* Exception: one can call {@link moveToParentScope} to move
	* a value to parent scope.
	*/
	endScope() {
		this.ctx.endScope();
	}
	/**
	* Perform action under a new scope.
	*
	* @param action The action function.
	* @returns The result value.
	*
	* Note: For action to return a valid value,
	*       we will need to call {@link moveToParentScope}
	*       for the objects that are created in the scope.
	*/
	withNewScope(action) {
		this.beginScope();
		const val = action();
		this.endScope();
		return val;
	}
	/**
	* Attach a detached obj to the auto-release pool of the current scope.
	*
	* @param obj The input obj.
	* Note: Normally user do not need to call this function explicitly, as
	*       all library call return values are explicitly attached to
	*       the current scope. You only need to do so when you call
	*       {@link detachFromCurrentScope} to create a detached object.
	*/
	attachToCurrentScope(obj) {
		return this.ctx.attachToCurrentScope(obj);
	}
	/**
	* Move obj's attachment to the parent scope.
	*
	* This function is useful to make sure objects are still
	* alive when exit the current scope.
	*
	* @param obj The object to be moved.
	* @returns The input obj.
	*/
	moveToParentScope(obj) {
		return this.ctx.moveToParentScope(obj);
	}
	/**
	* Detach the object from the current scope
	* so it won't be released via auto-release during endscope.
	*
	* User needs to either explicitly call obj.dispose(), or
	* {@link attachToCurrentScope} to re-attach to the current scope.
	*
	* This function can be used to return values to the parent scope.
	* @param obj The object.
	*/
	detachFromCurrentScope(obj) {
		return this.ctx.detachFromCurrentScope(obj);
	}
	/**
	* Get system-wide library module in the wasm.
	* System lib is a global module that contains self register functions in startup.
	* @returns The system library module.
	*/
	systemLib() {
		return this.ctx.getSysLib();
	}
	/**
	* List all the global function names registered in the runtime.
	* @returns The name list.
	*/
	listGlobalFuncNames() {
		return this.withNewScope(() => {
			const functor = this.ctx.functionListGlobalNamesFunctor();
			const numNames = functor(new Scalar(-1, "int"));
			const names = new Array(numNames);
			for (let i = 0; i < numNames; i++) names[i] = functor(new Scalar(i, "int"));
			return names;
		});
	}
	/**
	* Register function to be global function in tvm runtime.
	* @param name The name of the function.
	* @param func Function to be registered.
	* @param override Whether overwrite function in existing registry.
	*/
	registerFunc(name, func, override = false) {
		this.withNewScope(() => {
			const packedFunc = this.toPackedFuncInternal(func, true);
			const ioverride = override ? 1 : 0;
			const stack = this.lib.getOrAllocCallStack();
			const nameOffset = stack.allocByteArrayForString(name);
			stack.commitToWasmMemory();
			this.lib.checkCall(this.lib.exports.TVMFFIFunctionSetGlobal(stack.ptrFromOffset(nameOffset), packedFunc._tvmPackedCell.getHandle(), ioverride));
			this.lib.recycleCallStack(stack);
		});
	}
	/**
	* Get global PackedFunc from the runtime.
	* @param name The name of the function.
	* @returns The result function.
	*/
	getGlobalFunc(name) {
		return this.getGlobalFuncInternal(name, true);
	}
	getGlobalFuncInternal(name, autoAttachToScope = true) {
		const ret = this.getGlobalFuncInternalOptional(name, autoAttachToScope);
		if (ret === void 0) throw Error("Cannot find global function " + name);
		return ret;
	}
	getGlobalFuncInternalOptional(name, autoAttachToScope = true) {
		const stack = this.lib.getOrAllocCallStack();
		const nameOffset = stack.allocByteArrayForString(name);
		const outOffset = stack.allocPtrArray(1);
		const outPtr = stack.ptrFromOffset(outOffset);
		stack.commitToWasmMemory(outOffset);
		this.lib.checkCall(this.exports.TVMFFIFunctionGetGlobal(stack.ptrFromOffset(nameOffset), outPtr));
		const handle = this.memory.loadPointer(outPtr);
		this.lib.recycleCallStack(stack);
		if (handle === 0) return;
		const ret = this.makePackedFunc(handle);
		if (autoAttachToScope) this.ctx.attachToCurrentScope(ret);
		return ret;
	}
	/**
	* Check if func is PackedFunc.
	*
	* @param func The input.
	* @returns The check result.
	*/
	isPackedFunc(func) {
		return typeof func === "function" && func.hasOwnProperty("_tvmPackedCell");
	}
	/**
	* Convert func to PackedFunc
	*
	* @param func Input function.
	* @returns The converted function.
	*/
	toPackedFunc(func) {
		return this.toPackedFuncInternal(func, true);
	}
	toPackedFuncInternal(func, autoAttachToScope) {
		if (this.isPackedFunc(func)) return func;
		const ret = this.createPackedFuncFromSafeCallType(this.wrapJSFuncAsSafeCallType(func));
		if (autoAttachToScope) return this.ctx.attachToCurrentScope(ret);
		return ret;
	}
	/**
	* Setup a virtual machine module with given device.
	*
	* @param dev DLDevice the device.
	* @returns The created virtual machime.
	*/
	createVirtualMachine(dev) {
		const mod = this.ctx.detachFromCurrentScope(this.systemLib().getFunction("vm_load_executable")());
		return this.ctx.attachToCurrentScope(new VirtualMachine(mod, dev));
	}
	/**
	* Register a call back for fetch progress.
	*
	* @param cb the fetch progress callback.
	*/
	registerInitProgressCallback(cb) {
		this.initProgressCallback.push(cb);
	}
	/**
	* Get parameters in the form of prefix_i
	*
	* @param prefix The parameter prefix.
	* @param numParams  Number of parameters.
	* @returns
	*/
	getParamsFromCache(prefix, numParams) {
		return this.ctx.paramModuleFromCache(prefix, new Scalar(numParams, "int32")).getFunction("get_params")();
	}
	/**
	* Get parameters based on parameter names provided
	*
	* @param paramNames Names of the parameters.
	* @returns Parameters read.
	*/
	getParamsFromCacheByName(paramNames) {
		return this.ctx.paramModuleFromCacheByName(paramNames).getFunction("get_params")();
	}
	/**
	* Get Tensor from cache.
	* @param name  The name of array.
	* @returns  The result.
	*/
	tensorCacheGet(name) {
		return this.ctx.tensorCacheGet(name);
	}
	/**
	* Get Tensor from cache.
	* @param name  The name of array.
	* @returns  The result.
	*/
	tensorCacheRemove(name) {
		return this.ctx.tensorCacheRemove(name);
	}
	/**
	* Update the tensor cache.
	* @param name The name of the array.
	* @param arr The content.
	*/
	tensorCacheUpdate(name, arr, override = false) {
		this.ctx.tensorCacheUpdate(name, arr, this.scalar(override ? 1 : 0, "int32"));
	}
	/**
	* Clear the tensor cache.
	*/
	tensorCacheClear() {
		this.ctx.tensorCacheClear();
	}
	async fetchTensorCache(tensorCacheUrl, device, cacheScopeOrOptions = "tvmjs", cacheType = "cache", signal) {
		let options;
		if (typeof cacheScopeOrOptions === "object" && cacheScopeOrOptions !== null) options = cacheScopeOrOptions;
		else options = {
			cacheScope: cacheScopeOrOptions,
			signal
		};
		const artifactCache = createArtifactCache(options.cacheScope ?? "tvmjs", {
			...options,
			cacheType: options.cacheType ?? cacheType
		});
		const effectiveSignal = options.signal ?? signal;
		const jsonUrl = new URL("tensor-cache.json", tensorCacheUrl).href;
		const list = await artifactCache.fetchWithCache(jsonUrl, "json", effectiveSignal);
		await this.fetchTensorCacheInternal(tensorCacheUrl, list["records"], device, artifactCache, effectiveSignal);
		this.cacheMetadata = {
			...this.cacheMetadata,
			...list["metadata"]
		};
	}
	/**
	* Consume a tensor-cache record synchronously.
	*
	* Keeping the borrowed record view local to this non-async helper avoids
	* capturing it across the caller's WebGPU synchronization point.
	*/
	loadTensorCacheRecordData(shardBytes, rec, cpuArray, gpuArray) {
		const recBytes = getTensorCacheRecordBytes(shardBytes, rec);
		if (cpuArray !== void 0) {
			if (rec.format === "f32-to-bf16" && rec.dtype === "float32" && this.ctx.arrayDecodeBF16ToF32Inplace !== void 0) {
				this.memory.storeRawBytes(cpuArray.getCPUDataAddress(), recBytes);
				this.ctx.arrayDecodeBF16ToF32Inplace(cpuArray, new Scalar(recBytes.byteLength, "int64"));
			} else this.ctx.arrayDecodeStorage(cpuArray, recBytes, rec.format, rec.dtype);
		}
		if (gpuArray !== void 0) {
			if (cpuArray === void 0) gpuArray.copyFromRawBytes(recBytes);
			else gpuArray.copyFrom(cpuArray);
		}
	}
	/** Return the exact byte size of a packed BF16 tensor. */
	getPackedBF16Bytes(shape) {
		let numElements = 1;
		for (const dim of shape) {
			if (!Number.isSafeInteger(dim) || dim < 0) throw new Error(`Invalid tensor dimension: ${dim}`);
			numElements *= dim;
			if (!Number.isSafeInteger(numElements)) throw new Error("Tensor element count exceeds JavaScript's safe integer range");
		}
		const nbytes = numElements * 2;
		if (!Number.isSafeInteger(nbytes)) throw new Error("Packed BF16 size exceeds JavaScript's safe integer range");
		return nbytes;
	}
	/**
	* Fetch list of Tensor into the TensorCache.
	*
	* @param tensorCacheUrl The cache url.
	* @param list The list of array data.
	* @param device The device to store the data to.
	* @param artifactCache The artifact cache
	* @param signal An optional AbortSignal to abort the fetch
	*/
	async fetchTensorCacheInternal(tensorCacheUrl, list, device, artifactCache, signal) {
		const perf = getPerformance();
		const tstart = perf.now();
		let totalBytes = 0;
		for (let i = 0; i < list.length; ++i) totalBytes += list[i].nbytes;
		let fetchedBytes = 0;
		let fetchedShards = 0;
		let timeElapsed = 0;
		const cacheOnly = await artifactCache.hasAllKeys(list.map((key) => new URL(key.dataPath, tensorCacheUrl).href));
		const reportCallback = (iter, loading = false) => {
			for (let j = 0; j < this.initProgressCallback.length; ++j) {
				let text;
				if (loading) {
					text = "Loading model from cache[" + iter + "/" + list.length + "]: ";
					text += Math.ceil(fetchedBytes / 1048576).toString() + "MB loaded. ";
					text += Math.floor(fetchedBytes * 100 / totalBytes).toString() + "% completed, ";
					text += timeElapsed + " secs elapsed.";
				} else {
					text = "Fetching param cache[" + iter + "/" + list.length + "]: ";
					text += Math.ceil(fetchedBytes / 1048576).toString() + "MB fetched. ";
					text += Math.floor(fetchedBytes * 100 / totalBytes).toString() + "% completed, ";
					text += timeElapsed + " secs elapsed.";
					text += " It can take a while when we first visit this page to populate the cache.";
					text += " Later refreshes will become faster.";
				}
				this.initProgressCallback[j]({
					progress: fetchedBytes / totalBytes,
					timeElapsed,
					text
				});
			}
		};
		for (let j = 0; j < this.initProgressCallback.length; ++j) this.initProgressCallback[j]({
			progress: fetchedBytes / totalBytes,
			timeElapsed: 0,
			text: "Start to fetch params"
		});
		const downloadCache = async (start, end) => {
			for (let i = start; i < end; i++) {
				const shard = list[i];
				const dataUrl = new URL(shard.dataPath, tensorCacheUrl).href;
				try {
					await artifactCache.addToCache(dataUrl, "arraybuffer", signal);
				} catch (err) {
					this.env.logger("Error: Cannot fetch " + dataUrl + " err= " + err);
					throw err;
				}
				timeElapsed = Math.ceil((perf.now() - tstart) / 1e3);
				fetchedBytes += shard.nbytes;
				reportCallback(++fetchedShards, false);
			}
		};
		if (!cacheOnly) {
			const loopSize = Math.floor(list.length / 4);
			await Promise.all([
				downloadCache(0, loopSize),
				downloadCache(loopSize, 2 * loopSize),
				downloadCache(2 * loopSize, 3 * loopSize),
				downloadCache(3 * loopSize, list.length)
			]);
		}
		fetchedBytes = 0;
		fetchedShards = 0;
		for (let i = 0; i < list.length; ++i) {
			const shard = list[i];
			const dataUrl = new URL(shard.dataPath, tensorCacheUrl).href;
			let buffer;
			try {
				buffer = await artifactCache.fetchWithCache(dataUrl, "arraybuffer");
			} catch (err) {
				this.env.logger("Error: Cannot fetch " + dataUrl + " err= " + err);
				throw err;
			}
			const shardBytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
			const shardRecords = shard.records;
			for (let j = 0; j < shardRecords.length; ++j) {
				let cpu_arr;
				let gpu_arr;
				try {
					const rec = shardRecords[j];
					const isPackedBF16 = rec.format === "f32-to-bf16" && rec.dtype === "float32";
					if (isPackedBF16) {
						const expectedBytes = this.getPackedBF16Bytes(rec.shape);
						if (rec.nbytes !== expectedBytes) throw new Error(`Packed BF16 record has ${rec.nbytes} bytes, but shape requires ${expectedBytes}`);
					}
					if (!(device.deviceType === DeviceStrToEnum.webgpu && !isPackedBF16 && rec.nbytes % 4 === 0)) cpu_arr = this.withNewScope(() => {
						return this.detachFromCurrentScope(this.empty(rec.shape, rec.dtype, this.cpu()));
					});
					if (device.deviceType !== DeviceStrToEnum.cpu) gpu_arr = this.withNewScope(() => {
						return this.detachFromCurrentScope(this.empty(rec.shape, rec.dtype, device));
					});
					this.loadTensorCacheRecordData(shardBytes, rec, cpu_arr, gpu_arr);
					if (device.deviceType === DeviceStrToEnum.cpu) this.tensorCacheUpdate(rec.name, cpu_arr, false);
					else {
						await device.sync();
						this.tensorCacheUpdate(rec.name, gpu_arr, false);
					}
				} catch (err) {
					this.env.logger("Failed to load shard " + i + "'s record: " + JSON.stringify(shardRecords[j]) + "\nError: " + err);
					throw err;
				} finally {
					cpu_arr?.dispose();
					gpu_arr?.dispose();
				}
			}
			fetchedBytes += shard.nbytes;
			timeElapsed = Math.ceil((perf.now() - tstart) / 1e3);
			reportCallback(++fetchedShards, true);
		}
	}
	/**
	* Create a new {@link Scalar} that can be passed to a PackedFunc.
	* @param value The number value.
	* @param dtype The dtype string.
	* @returns The created scalar.
	*/
	scalar(value, dtype) {
		return new Scalar(value, dtype);
	}
	/**
	* Create a new {@link DLDevice}
	* @param deviceType The device type.
	* @param deviceId The device index.
	* @returns The created device.
	*/
	device(deviceType, deviceId = 0) {
		return new DLDevice(deviceType, deviceId, this.lib);
	}
	/**
	* Create a new cpu {@link DLDevice}
	* @param deviceId The device index.
	*/
	cpu(deviceId = 0) {
		return this.device("cpu", deviceId);
	}
	/**
	* Create a new webgpu {@link DLDevice}
	* @param deviceId The device index.
	*/
	webgpu(deviceId = 0) {
		return this.device("webgpu", deviceId);
	}
	/**
	* Create an empty {@link Tensor} with given shape and dtype.
	*
	* @param shape The shape of the array.
	* @param dtype The data type of the array.
	* @param dev The device of the ndarray.
	* @returns The created ndarray.
	*/
	empty(shape, dtype = "float32", dev = this.device("cpu", 0)) {
		shape = typeof shape === "number" ? [shape] : shape;
		return this.ctx.tensorEmpty(this.makeShapeTuple(shape), dtype, dev, null);
	}
	/**
	* Create am uniform {@link Tensor} with given shape.
	*
	* @param shape The shape of the array.
	* @param low The low value.
	* @param high The high value.
	* @param dev The device of the ndarray.
	* @returns The created ndarray.
	*/
	uniform(shape, low, high, dev) {
		const ret = this.empty(shape, "float32", dev);
		const size = shape.reduce((a, b) => {
			return a * b;
		}, 1);
		const scale = high - low;
		const input = new Float32Array(size);
		for (let i = 0; i < input.length; ++i) input[i] = low + this.rng.randomFloat() * scale;
		return ret.copyFrom(input);
	}
	/**
	* Set the seed of the internal LinearCongruentialGenerator.
	*/
	setSeed(seed) {
		this.rng.setSeed(seed);
	}
	/**
	* Get the state of the internal LinearCongruentialGenerator.
	*/
	getRNGState() {
		return this.rng.getState();
	}
	/**
	* Restore the state of the internal LinearCongruentialGenerator.
	*/
	setRNGState(state) {
		this.rng.setState(state);
	}
	/**
	* Sample index via top-p sampling.
	*
	* @param logits The input logits before normalization.
	* @param temperature  The temperature factor, will take argmax if temperature = 0.0
	* @param top_p The top_p
	* @returns The sampled index.
	*/
	sampleTopPFromLogits(logits, temperature, top_p) {
		return this.ctx.sampleTopPFromLogits(logits, temperature, top_p, this.rng.randomFloat());
	}
	/**
	* Sample index via top-p sampling.
	*
	* @param prob The distribution, i.e. logits after `applySoftmaxWithTemperature()` is performed.
	* @param top_p The top_p
	* @returns The sampled index.
	*/
	sampleTopPFromProb(prob, top_p) {
		return this.ctx.sampleTopPFromProb(prob, top_p, this.rng.randomFloat());
	}
	/**
	* Apply repetition penalty to the logits.
	* @param logits The input logits before penalty.
	* @param token_ids The appeared token ids.
	* @param penalty The penalty factor.
	*/
	applyRepetitionPenalty(logits, token_ids, penalty) {
		return this.ctx.applyRepetitionPenalty(logits, token_ids, penalty);
	}
	/**
	* Apply presence and frequency penalty. This is an inplace operation.
	* @param logits The input logits before penalty.
	* @param token_ids The appeared token ids.
	* @param token_freqs The number of times each token has appeared since last PrefillStep.
	* token_freqs[i] is the frequency of token_ids[i], for all i. And all token_freqs should be >= 1.
	* @param presence_penalty The penalty factor.
	* @param frequency_penalty The penalty factor.
	*/
	applyPresenceAndFrequencyPenalty(logits, token_ids, token_freqs, presence_penalty, frequency_penalty) {
		return this.ctx.applyPresenceAndFrequencyPenalty(logits, token_ids, token_freqs, presence_penalty, frequency_penalty);
	}
	/**
	* Apply softmax with temperature to the logits.
	* @param logits The input logits before softmax w/ temperature.
	* @param temperature The temperature factor.
	*/
	applySoftmaxWithTemperature(logits, temperature) {
		return this.ctx.applySoftmaxWithTemperature(logits, temperature);
	}
	/**
	* Bind canvas to the current WebGPU context
	* @param canvas The canvas.
	*/
	bindCanvas(canvas) {
		this.lib.webGPUContext?.bindCanvas(canvas);
	}
	/**
	* Show image in canvas.
	*
	* @param dataRGBA Image array in height x width uint32 Tensor RGBA format on GPU.
	*/
	showImage(dataRGBA) {
		if (dataRGBA.shape.length != 2) throw Error("Require a height x width uint32 Tensor in RGBAget shape=" + dataRGBA.shape.toString() + " instead.");
		if (dataRGBA.device.deviceType != DeviceStrToEnum.webgpu) throw new Error("Can only run showImage on WebGPU array, get " + DeviceEnumToStr[dataRGBA.device.deviceType] + " instead.");
		if (dataRGBA.dtype != "uint32") throw Error("Require a height x width uint32 Tensor in RGBA, get " + dataRGBA.dtype + " instead.");
		this.lib.webGPUContext?.drawImageFromBuffer(dataRGBA.getDataPtr(), dataRGBA.shape[0], dataRGBA.shape[1]);
	}
	/**
	* Clear canvas
	*/
	clearCanvas() {
		this.lib.webGPUContext?.clearCanvas();
	}
	/**
	* Create an tuple {@link TVMArray} input array.
	*
	* The input array can be passed to tvm runtime function
	* and needs to b explicitly disposed.
	*
	* @param inputs The input array
	* @returns The result array.
	*/
	makeTVMArray(inputs) {
		const CALL_STACK_LIMIT = 3e4;
		const inputsLength = inputs.length;
		if (inputsLength <= CALL_STACK_LIMIT) return this.ctx.arrayMake(...inputs);
		const listOfArrays = [];
		for (let begin = 0; begin < inputsLength; begin += CALL_STACK_LIMIT) {
			const end = Math.min(inputsLength, begin + CALL_STACK_LIMIT);
			const chunk = inputs.slice(begin, end);
			listOfArrays.push(this.ctx.arrayMake(...chunk));
		}
		return this.ctx.arrayConcat(...listOfArrays);
	}
	/**
	* Join a sequence of Tensors that represent embeddings.
	* @param embeddings A list of embeddings in Tensors, each array i has shape (m_i, hidden_size).
	* @returns An Tensor of shape (\sum_{i} {m}, hidden_size)
	*/
	concatEmbeddings(embeddings) {
		const hidden_size = embeddings[0].shape[1];
		embeddings.forEach((input) => {
			if (input.shape.length !== 2 || input.shape[1] !== hidden_size) throw new Error("Expect embeddings to concatenate have shape (m_i, hidden_size).");
		});
		if (this.ctx.concatEmbeddings === void 0) throw new Error("Global function tvmjs.runtime.ConcatEmbeddings was not found, but called concatEmbeddings.");
		return this.ctx.concatEmbeddings(...embeddings);
	}
	/**
	* Create a shape tuple to pass to runtime.
	* @param shape The shape .
	* @returns The created shape tuple.
	*/
	makeShapeTuple(shape) {
		const key = CacheState.computeShapeKey(shape);
		return this.cacheState.shapeCache.get(key, () => {
			const shapeArray = shape.map((value) => new Scalar(value, "int"));
			const tuple = this.ctx.makeShapeTuple(...shapeArray);
			this.detachFromCurrentScope(tuple);
			return tuple;
		});
	}
	/**
	* Get type index from type key.
	* @param typeKey The type key.
	* @returns The corresponding type index.
	*/
	typeKey2Index(typeKey) {
		const stack = this.lib.getOrAllocCallStack();
		const typeKeyOffset = stack.allocByteArrayForString(typeKey);
		const outOffset = stack.allocPtrArray(1);
		const outPtr = stack.ptrFromOffset(outOffset);
		stack.commitToWasmMemory(outOffset);
		this.lib.checkCall(this.lib.exports.TVMFFITypeKeyToIndex(stack.ptrFromOffset(typeKeyOffset), outPtr));
		const typeIndex = this.memory.loadU32(outPtr);
		this.lib.recycleCallStack(stack);
		return typeIndex;
	}
	/**
	* Register an object constructor.
	* @param typeKey The name of the function.
	* @param func Function to be registered.
	* @param override Whether overwrite function in existing registry.
	*/
	registerObjectConstructor(typeKey, func, override = false) {
		const typeIndex = this.typeKey2Index(typeKey);
		if (this.objFactory.has(typeIndex)) {
			if (!override) throw new Error("Type " + typeKey + " already registered");
		}
		this.objFactory.set(typeIndex, func);
	}
	/**
	* Wrap a function obtained from tvm runtime as AsyncPackedFunc
	* through the asyncify mechanism
	*
	* You only need to call it if the function may contain callback into async
	* JS function via asynctify. A common one can be GPU synchronize.
	*
	* It is always safe to wrap any function as Asynctify, however you do need
	* to make sure you use await when calling the funciton.
	*
	* @param func The PackedFunc.
	* @returns The wrapped AsyncPackedFunc
	*/
	wrapAsyncifyPackedFunc(func) {
		const asyncFunc = this.asyncifyHandler.wrapExport(func);
		asyncFunc.dispose = func.dispose;
		asyncFunc._tvmPackedCell = func._tvmPackedCell;
		return asyncFunc;
	}
	/**
	* Register async function as asynctify callable in global environment.
	*
	* @param name The name of the function.
	* @param func function to be registered.
	* @param override Whether overwrite function in existing registry.
	*
	* Note: This function is handled via asynctify mechanism.
	* The wasm needs to be compiled with Asynctify
	*/
	registerAsyncifyFunc(name, func, override = false) {
		const asyncWrapped = this.asyncifyHandler.wrapImport(func);
		this.registerFunc(name, asyncWrapped, override);
	}
	/**
	* Register an asyncfunction to be global function in the server.
	*
	* @param name The name of the function.
	* @param func function to be registered.
	* @param override Whether overwrite function in existing registry.
	*
	* Note: The async function will only be used for serving remote calls in the rpc.
	* These functions contains explicit continuation
	*/
	registerAsyncServerFunc(name, func, override = false) {
		const asyncVariant = (...args) => {
			const fargs = args.slice(0, args.length - 1);
			const callback = this.detachFromCurrentScope(args[args.length - 1]);
			const promise = func(...fargs);
			const onFulfilled = (rv) => {
				callback(this.scalar(AsyncCallbackCode.kReturn, "int32"), rv);
				callback.dispose();
			};
			const onRejected = (reason) => {
				callback(this.scalar(AsyncCallbackCode.kException, "int32"), reason.toString());
				callback.dispose();
			};
			promise.then(onFulfilled, onRejected);
		};
		this.registerFunc("__async." + name, asyncVariant, override);
	}
	/**
	* Asynchronously load webgpu pipelines when possible.
	* @param mod The input module.
	*/
	async asyncLoadWebGPUPipelines(mod) {
		if (this.lib.webGPUContext === void 0) throw Error("WebGPU not initialied");
		const webgpuContext = this.lib.webGPUContext;
		this.beginScope();
		const fmap_str = mod.getFunction("webgpu.get_fmap", true)();
		const fmap = JSON.parse(fmap_str);
		const fGetShader = this.detachFromCurrentScope(mod.getFunction("webgpu.get_shader"));
		const fUpdatePrebuild = this.detachFromCurrentScope(mod.getFunction("webgpu.update_prebuild"));
		this.endScope();
		const perf = getPerformance();
		const tstart = perf.now();
		let tlastReport = tstart;
		let finishCounter = 0;
		const fmapEntries = Object.entries(fmap);
		let allEvents = Promise.resolve();
		for (const [key, finfo] of fmapEntries) {
			const code = fGetShader(key);
			assert(key === finfo.name);
			const event = webgpuContext.createShaderAsync(finfo, code).then((func) => {
				this.beginScope();
				fUpdatePrebuild(key, func);
				this.endScope();
			}).then(() => {
				finishCounter += 1;
				const tend = perf.now();
				if (tend - tlastReport < 1e3 && finishCounter != fmapEntries.length) return;
				tlastReport = tend;
				const timeElapsed = Math.ceil((perf.now() - tstart) / 1e3);
				for (let j = 0; j < this.initProgressCallback.length; ++j) {
					const progress = finishCounter / fmapEntries.length;
					let text = "Loading GPU shader modules[" + finishCounter + "/" + fmapEntries.length + "]: ";
					text += Math.floor(progress * 100).toString() + "% completed, ";
					text += timeElapsed + " secs elapsed.";
					this.initProgressCallback[j]({
						progress,
						timeElapsed,
						text
					});
				}
			});
			allEvents = Promise.all([allEvents, event]).then(() => {});
		}
		await allEvents;
		assert(finishCounter === fmapEntries.length);
	}
	/**
	* Initialize webgpu in the runtime.
	* @param device The given GPU device.
	*/
	initWebGPU(device) {
		device.addEventListener("uncapturederror", (event) => {
			console.error("A WebGPU error was not captured: ", event);
		});
		device.lost.then((info) => {
			if (this.deviceLostIsError) {
				console.error("Device lost, calling Instance.dispose(). Please initialize again. ", info);
				this.dispose();
			}
		});
		this.deviceLostIsError = true;
		const webGPUContext = new WebGPUContext(this.memory, device);
		this.registerFunc("wasm.WebGPUDeviceAPI", (name) => {
			return webGPUContext.getDeviceAPI(name);
		});
		this.registerFunc("wasm.WebGPUCreateShader", (info, code) => {
			const finfo = JSON.parse(info);
			return webGPUContext.createShader(finfo, code);
		});
		this.registerAsyncServerFunc("wasm.WebGPUWaitForTasks", async () => {
			await webGPUContext.sync();
		});
		if (this.asyncifyHandler.enabled()) this.registerAsyncifyFunc("__asyncify.WebGPUWaitForTasks", async () => {
			await webGPUContext.sync();
		});
		this.lib.webGPUContext = webGPUContext;
	}
	/** Register all object factory */
	registerObjectFactoryFuncs() {
		this.registerObjectConstructor("ffi.Array", (handle, lib, ctx) => {
			return new TVMArray(handle, lib, ctx);
		});
		this.registerObjectConstructor("ffi.Module", (handle, lib, ctx) => {
			return new Module(handle, lib, ctx);
		});
	}
	/** Register global packed functions needed by the backend to the env. */
	registerEnvGlobalPackedFuncs() {
		const perf = getPerformance();
		const timeExecution = async (finvoke, dev, nstep, repeat, minRepeatMs, limitZeroTimeIterations, cooldownIntervalMs, repeatsToCooldown) => {
			this.ctx.detachFromCurrentScope(finvoke);
			finvoke(this.scalar(1, "int32"));
			await dev.sync();
			const result = [];
			let setupNumber = nstep;
			for (let i = 0; i < repeat; ++i) {
				let durationMs = 0;
				let absoluteZeroTimes = 0;
				do {
					if (durationMs > 0) setupNumber = Math.floor(Math.max(minRepeatMs / (durationMs / setupNumber) + 1, setupNumber * 1.618));
					const tstart = perf.now();
					finvoke(this.scalar(setupNumber, "int32"));
					await dev.sync();
					durationMs = perf.now() - tstart;
					if (durationMs === 0) absoluteZeroTimes++;
				} while (durationMs < minRepeatMs && absoluteZeroTimes < limitZeroTimeIterations);
				const speed = durationMs / setupNumber / 1e3;
				result.push(speed);
				if (cooldownIntervalMs > 0 && i % repeatsToCooldown === 0) await new Promise((r) => setTimeout(r, cooldownIntervalMs));
			}
			const ret = new Float64Array(result.length);
			ret.set(result);
			finvoke.dispose();
			return new Uint8Array(ret.buffer);
		};
		const addOne = async (x) => {
			await new Promise((resolve) => setTimeout(resolve, 100));
			return x + 1;
		};
		this.registerAsyncServerFunc("wasm.TimeExecution", timeExecution);
		this.registerAsyncServerFunc("testing.asyncAddOne", addOne);
	}
	createPackedFuncFromSafeCallType(func) {
		let findex = this.env.packedCFuncTable.length;
		if (this.env.packedCFuncTableFreeId.length != 0) findex = this.env.packedCFuncTableFreeId.pop();
		else this.env.packedCFuncTable.push(void 0);
		this.env.packedCFuncTable[findex] = func;
		const stack = this.lib.getOrAllocCallStack();
		const outOffset = stack.allocPtrArray(1);
		const outPtr = stack.ptrFromOffset(outOffset);
		this.lib.checkCall(this.exports.TVMFFIWasmFunctionCreate(findex, outPtr));
		const ret = this.makePackedFunc(this.memory.loadPointer(outPtr));
		this.lib.recycleCallStack(stack);
		return ret;
	}
	/**
	* Set packed function arguments into the location indicated by argsValue and argsCode.
	* Allocate new temporary space from the stack if necessary.
	*
	* @param stack The call stack.
	* @param args  The input arguments.
	* @param packedArgs The offset of packedArgs.
	*/
	setPackedArguments(stack, args, packedArgs) {
		for (let i = 0; i < args.length; ++i) {
			let val = args[i];
			const tp = typeof val;
			const argOffset = packedArgs + i * SizeOf.TVMFFIAny;
			const argTypeIndexOffset = argOffset;
			const argZeroPaddingOffset = argOffset + SizeOf.I32;
			const argValueOffset = argOffset + SizeOf.I32 * 2;
			if (val instanceof Array && val.every((e) => typeof e === "string")) {
				const tvmStringArray = [];
				val.forEach((e) => {
					tvmStringArray.push(e);
				});
				val = this.makeTVMArray(tvmStringArray);
			}
			stack.storeI32(argZeroPaddingOffset, 0);
			stack.storeI32(argValueOffset + SizeOf.I32, 0);
			if (val instanceof Tensor) {
				if (!val.isView) {
					stack.storeI32(argTypeIndexOffset, TypeIndex.kTVMFFITensor);
					stack.storePtr(argValueOffset, val.getHandle());
				} else {
					stack.storeI32(argTypeIndexOffset, TypeIndex.kTVMFFIDLTensorPtr);
					stack.storePtr(argValueOffset, val.getHandle());
				}
			} else if (val instanceof Scalar) {
				if (val.dtype.startsWith("int") || val.dtype.startsWith("uint")) {
					stack.storeI32(argTypeIndexOffset, TypeIndex.kTVMFFIInt);
					stack.storeI64(argValueOffset, val.value);
				} else if (val.dtype.startsWith("float")) {
					stack.storeI32(argTypeIndexOffset, TypeIndex.kTVMFFIFloat);
					stack.storeF64(argValueOffset, val.value);
				} else {
					assert(val.dtype === "handle", "Expect handle");
					stack.storeI32(argTypeIndexOffset, TypeIndex.kTVMFFIOpaquePtr);
					stack.storePtr(argValueOffset, val.value);
				}
			} else if (val instanceof DLDevice) {
				stack.storeI32(argTypeIndexOffset, TypeIndex.kTVMFFIDevice);
				stack.storeI32(argValueOffset, val.deviceType);
				stack.storeI32(argValueOffset + SizeOf.I32, val.deviceId);
			} else if (tp === "boolean") {
				stack.storeI32(argTypeIndexOffset, TypeIndex.kTVMFFIBool);
				stack.storeI64(argValueOffset, val ? 1 : 0);
			} else if (tp === "number") {
				stack.storeI32(argTypeIndexOffset, TypeIndex.kTVMFFIFloat);
				stack.storeF64(argValueOffset, val);
			} else if (tp === "function" && val.hasOwnProperty("_tvmPackedCell")) {
				stack.storePtr(argValueOffset, val._tvmPackedCell.getHandle());
				stack.storeI32(argTypeIndexOffset, TypeIndex.kTVMFFIFunction);
			} else if (val === null || val === void 0) {
				stack.storeI32(argTypeIndexOffset, TypeIndex.kTVMFFINone);
				stack.storePtr(argValueOffset, 0);
			} else if (tp === "string") {
				stack.storeI32(argTypeIndexOffset, TypeIndex.kTVMFFIRawStr);
				stack.allocThenSetArgString(argValueOffset, val);
			} else if (val instanceof Uint8Array) {
				stack.storeI32(argTypeIndexOffset, TypeIndex.kTVMFFIByteArrayPtr);
				stack.allocThenSetArgBytes(argValueOffset, val);
			} else if (val instanceof Function) {
				val = this.toPackedFuncInternal(val, false);
				stack.tempArgs.push(val);
				stack.storeI32(argTypeIndexOffset, TypeIndex.kTVMFFIFunction);
				stack.storePtr(argValueOffset, val._tvmPackedCell.getHandle());
			} else if (val instanceof Module) {
				stack.storeI32(argTypeIndexOffset, TypeIndex.kTVMFFIModule);
				stack.storePtr(argValueOffset, val.getHandle());
			} else if (val instanceof TVMObject) {
				stack.storeI32(argTypeIndexOffset, val.typeIndex());
				stack.storePtr(argValueOffset, val.getHandle());
			} else throw new Error("Unsupported argument type " + tp + " value=`" + val.toString() + "`");
		}
	}
	wrapJSFuncAsSafeCallType(func) {
		const lib = this.lib;
		return (self, packedArgs, numArgs, ret) => {
			const jsArgs = [];
			this.ctx.beginScope();
			let rv;
			try {
				for (let i = 0; i < numArgs; ++i) {
					const argPtr = packedArgs + i * SizeOf.TVMFFIAny;
					if (lib.memory.loadI32(argPtr) >= TypeIndex.kTVMFFIRawStr) {
						if (!this.asyncifyHandler.isNormalStackState()) throw Error("Cannot handle str/object argument callback in asyncify mode");
						lib.checkCall(lib.exports.TVMFFIAnyViewToOwnedAny(argPtr, argPtr));
					}
					jsArgs.push(this.retValueToJS(argPtr, true));
				}
				rv = func(...jsArgs);
			} catch (error) {
				const errKind = "JSCallbackError";
				let errMsg = error instanceof Error ? error.message : String(error);
				try {
					this.ctx.endScope();
				} catch (cleanupError) {
					const cleanupMessage = cleanupError instanceof Error ? cleanupError.message : String(cleanupError);
					errMsg += ` (scope cleanup failed: ${cleanupMessage})`;
				}
				const stack = lib.getOrAllocCallStack();
				try {
					const errKindOffset = stack.allocRawBytes(16);
					stack.storeRawBytes(errKindOffset, StringToUint8Array(errKind));
					const errMsgOffset = stack.allocRawBytes(errMsg.length + 1);
					stack.storeRawBytes(errMsgOffset, StringToUint8Array(errMsg));
					stack.commitToWasmMemory();
					this.lib.exports.TVMFFIErrorSetRaisedFromCStr(stack.ptrFromOffset(errKindOffset), stack.ptrFromOffset(errMsgOffset));
				} finally {
					lib.recycleCallStack(stack);
				}
				return -1;
			}
			this.ctx.endScope();
			if (rv !== void 0 && rv !== null) {
				const stack = lib.getOrAllocCallStack();
				const argOffset = stack.allocRawBytes(SizeOf.TVMFFIAny);
				this.setPackedArguments(stack, [rv], argOffset);
				stack.commitToWasmMemory();
				const argPtr = stack.ptrFromOffset(argOffset);
				lib.checkCall(lib.exports.TVMFFIAnyViewToOwnedAny(argPtr, ret));
				lib.recycleCallStack(stack);
			}
			return 0;
		};
	}
	makePackedFunc(handle) {
		const cell = new PackedFuncCell(handle, this.lib, this.ctx);
		const packedFunc = (...args) => {
			const stack = this.lib.getOrAllocCallStack();
			try {
				const argsOffset = stack.allocRawBytes(SizeOf.TVMFFIAny * args.length);
				this.setPackedArguments(stack, args, argsOffset);
				const retOffset = stack.allocRawBytes(SizeOf.TVMFFIAny);
				stack.storeI32(retOffset, TypeIndex.kTVMFFINone);
				stack.storeI32(retOffset + SizeOf.I32, 0);
				stack.commitToWasmMemory();
				this.lib.checkCall(this.exports.TVMFFIFunctionCall(cell.getHandle(), stack.ptrFromOffset(argsOffset), args.length, stack.ptrFromOffset(retOffset)));
				return this.retValueToJS(stack.ptrFromOffset(retOffset), false);
			} finally {
				this.lib.recycleCallStack(stack);
			}
		};
		const ret = packedFunc;
		ret.dispose = () => {
			cell.dispose();
		};
		ret._tvmPackedCell = cell;
		return ret;
	}
	/**
	* Creaye return value of the packed func. The value us auto-tracked for dispose.
	* @param resultAnyPtr The location of rvalue
	* @param callbackArg Whether it is being used in callbackArg.
	* @returns The JS value.
	*/
	retValueToJS(resultAnyPtr, callbackArg) {
		const typeIndex = this.memory.loadI32(resultAnyPtr);
		const valuePtr = resultAnyPtr + SizeOf.I32 * 2;
		switch (typeIndex) {
			case TypeIndex.kTVMFFINone: return;
			case TypeIndex.kTVMFFIBool: return this.memory.loadI64(valuePtr) != 0;
			case TypeIndex.kTVMFFIInt: return this.memory.loadI64(valuePtr);
			case TypeIndex.kTVMFFIFloat: return this.memory.loadF64(valuePtr);
			case TypeIndex.kTVMFFIOpaquePtr: return this.memory.loadPointer(valuePtr);
			case TypeIndex.kTVMFFITensor: return this.ctx.attachToCurrentScope(new Tensor(this.memory.loadPointer(valuePtr), this.lib, this.ctx, false));
			case TypeIndex.kTVMFFIDLTensorPtr:
				assert(callbackArg);
				return new Tensor(this.memory.loadPointer(valuePtr), this.lib, this.ctx, true);
			case TypeIndex.kTVMFFIFunction: return this.ctx.attachToCurrentScope(this.makePackedFunc(this.memory.loadPointer(valuePtr)));
			case TypeIndex.kTVMFFIDevice: {
				const deviceType = this.memory.loadI32(valuePtr);
				const deviceId = this.memory.loadI32(valuePtr + SizeOf.I32);
				return this.device(deviceType, deviceId);
			}
			case TypeIndex.kTVMFFIDataType: {
				this.lib.checkCall(this.lib.exports.TVMFFIDataTypeToString(valuePtr, valuePtr));
				const strObjPtr = this.memory.loadPointer(valuePtr);
				const result = this.memory.loadByteArrayAsString(strObjPtr + SizeOf.ObjectHeader);
				this.lib.checkCall(this.lib.exports.TVMFFIObjectDecRef(strObjPtr));
				return result;
			}
			case TypeIndex.kTVMFFISmallStr: return this.memory.loadSmallStr(resultAnyPtr);
			case TypeIndex.kTVMFFIStr: {
				const strObjPtr = this.memory.loadPointer(valuePtr);
				const result = this.memory.loadByteArrayAsString(strObjPtr + SizeOf.ObjectHeader);
				this.lib.checkCall(this.lib.exports.TVMFFIObjectDecRef(strObjPtr));
				return result;
			}
			case TypeIndex.kTVMFFISmallBytes: return this.memory.loadSmallBytes(resultAnyPtr);
			case TypeIndex.kTVMFFIBytes: {
				const bytesObjPtr = this.memory.loadPointer(valuePtr);
				const result = this.memory.loadByteArrayAsBytes(bytesObjPtr + SizeOf.ObjectHeader);
				this.lib.checkCall(this.lib.exports.TVMFFIObjectDecRef(bytesObjPtr));
				return result;
			}
			default: if (typeIndex >= TypeIndex.kTVMFFIStaticObjectBegin) {
				const obj = new TVMObject(this.memory.loadPointer(valuePtr), this.lib, this.ctx);
				const func = this.objFactory.get(obj.typeIndex());
				if (func != void 0) return this.ctx.attachToCurrentScope(func(obj.getHandle(), this.lib, this.ctx));
				else return this.ctx.attachToCurrentScope(obj);
			} else throw new Error("Unsupported return type code=" + typeIndex);
		}
	}
};
/**
* Asynchrously instantiate a new {@link Instance}.
*
* importObject can also be a {@link LibraryProvider} object,
* a WASI object, or an object containing wasmLibraryProvider field.
* We can take benefit of syslib implementations from the Emscripten
* by passing its generated js Module as the imports.
*
* @param bufferSource The source to be compiled.
* @param importObject The import objects.
* @param logger The system logger.
*/
function instantiate(bufferSource, importObject = {}, logger = console.log) {
	const env = new Environment(importObject, logger);
	return WebAssembly.instantiate(bufferSource, env.imports).then((result) => {
		return new Instance(result.module, {}, result.instance, env);
	});
}
// source/overlay/packages/web-decision/src/strands/disposal-guard.mjs
var retained = /* @__PURE__ */ new Set();
function retainStrandsResources(runtime, gpu) {
	retained.add({
		runtime,
		gpu
	});
}
// source/overlay/packages/web-decision/src/abortable-gpu-device.js
function destroyGpuDevice(device) {
	if (!device) return;
	try {
		device.destroy();
	} catch {
		retainStrandsResources(void 0, device);
	}
}
/** Stop waiting for an adapter when the caller cancels. */
async function requestGpuAdapter(gpu, signal) {
	try {
		signal?.throwIfAborted();
		const adapter = await waitForAbort(() => gpu.requestAdapter({ powerPreference: "high-performance" }), signal);
		signal?.throwIfAborted();
		return adapter;
	} catch (cause) {
		signal?.throwIfAborted();
		throw cause;
	}
}
/** Destroy a device that arrives after cancellation, retaining it if destroy fails. */
async function requestGpuDevice(adapter, descriptor, signal) {
	let device;
	try {
		signal?.throwIfAborted();
		device = await waitForAbort(() => adapter.requestDevice(descriptor), signal, destroyGpuDevice);
		signal?.throwIfAborted();
		return device;
	} catch (cause) {
		destroyGpuDevice(device);
		signal?.throwIfAborted();
		throw cause;
	}
}
// source/overlay/packages/web-decision/src/kev-gpu.js
/** Experimental limits for the pinned Kev 0.6B Q4 WebGPU build only. */
var KEV_REQUIRED_LIMITS = Object.freeze({
	maxBufferSize: 268435456,
	maxStorageBufferBindingSize: 134217728,
	maxComputeWorkgroupStorageSize: 16384,
	maxStorageBuffersPerShaderStage: 8,
	maxComputeInvocationsPerWorkgroup: 256,
	maxComputeWorkgroupSizeX: 256,
	maxComputeWorkgroupSizeY: 8,
	maxComputeWorkgroupSizeZ: 1,
	maxComputeWorkgroupsPerDimension: 65535
});
/** Return an adapter/device pair compatible with the current loader's interface. */
async function detectKevGPUDevice(gpu = globalThis.navigator?.gpu, signal) {
	signal?.throwIfAborted();
	if (!gpu) return void 0;
	const adapter = await requestGpuAdapter(gpu, signal);
	signal?.throwIfAborted();
	if (!adapter) throw new Error("No compatible WebGPU adapter was found.");
	if (!adapter.features.has("shader-f16")) {
		signal?.throwIfAborted();
		return {
			adapter,
			device: void 0,
			adapterInfo: adapter.info
		};
	}
	for (const [name, required] of Object.entries(KEV_REQUIRED_LIMITS)) {
		const available = adapter.limits[name];
		if (typeof available !== "number" || !Number.isFinite(available) || available < required) throw new Error(`WebGPU ${name} requires ${required}; adapter reports ${available}.`);
	}
	const device = await requestGpuDevice(adapter, {
		requiredLimits: {
			...KEV_REQUIRED_LIMITS,
			maxBufferSize: adapter.limits.maxBufferSize,
			maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize
		},
		requiredFeatures: ["shader-f16"]
	}, signal);
	try {
		signal?.throwIfAborted();
		return {
			adapter,
			device,
			adapterInfo: adapter.info
		};
	} catch (cause) {
		destroyGpuDevice(device);
		signal?.throwIfAborted();
		throw cause;
	}
}
// source/overlay/packages/web-decision/src/request-fields.mjs
var fail$1 = () => {
	throw new WebDecisionError("INVALID_INPUT", "Unsupported decision request or option field.");
};
var requestKeys = /* @__PURE__ */ new Set(["state", "questions"]);
var questionKeys = /* @__PURE__ */ new Set(["instructions", "options"]);
var decisionKeys = /* @__PURE__ */ new Set(["signal"]);
var knownUnsupported = [
	"type",
	"criteria",
	"image",
	"images",
	"audio",
	"video"
];
function checkOwnField(value, key, allowed) {
	if (typeof key !== "string" || !allowed.has(key)) fail$1();
}
function checkFields(value, allowed) {
	if (!value || typeof value !== "object" || Array.isArray(value)) fail$1();
	const ownKeys = Reflect.ownKeys(value);
	if (ownKeys.length > allowed.size) fail$1();
	for (const key of ownKeys) checkOwnField(value, key, allowed);
	for (const key in value) if (!allowed.has(key)) fail$1();
	for (const key of knownUnsupported) if (!allowed.has(key) && key in value) fail$1();
}
function readValue(value, key) {
	return value[key];
}
function addTextToBudget(value, budget) {
	if (typeof value !== "string" || !budget) return;
	budget.used += value.length;
	if (budget.used > budget.max) fail$1();
}
function snapshotOptions(options, budget, maxOptions, expectedCount) {
	if (!Array.isArray(options)) return options;
	const count = readValue(options, "length");
	if (count > maxOptions || expectedCount !== void 0 && count !== expectedCount) fail$1();
	const snapshot = new Array(count);
	for (let i = 0; i < count; i++) {
		if (!Object.hasOwn(options, i)) fail$1();
		const option = readValue(options, i);
		addTextToBudget(option, budget);
		snapshot[i] = option;
	}
	return snapshot;
}
function stageQuestions(questions, questionCount, budget, maxInputTokens, minimumTokenCount, maxOptions, maxTotalOptions) {
	const staged = [];
	let totalOptionMarkers = 0, totalOptionCount = 0;
	for (let i = 0; i < questionCount; i++) {
		if (!Object.hasOwn(questions, i)) fail$1();
		const question = readValue(questions, i);
		checkFields(question, questionKeys);
		const instructions = readValue(question, "instructions"), options = readValue(question, "options");
		addTextToBudget(instructions, budget);
		let optionCount;
		if (Array.isArray(options)) {
			optionCount = readValue(options, "length");
			if (optionCount > maxOptions) fail$1();
			totalOptionMarkers += Math.max(0, optionCount - 1);
			if (minimumTokenCount + totalOptionMarkers * 2 > maxInputTokens) fail$1();
			totalOptionCount += optionCount;
			if (totalOptionCount > maxTotalOptions) fail$1();
		}
		staged.push({
			instructions,
			options,
			optionCount
		});
	}
	return staged;
}
function snapshotStagedQuestions(staged, budget, maxOptions) {
	return staged.map(({ instructions, options, optionCount }) => ({
		instructions,
		options: snapshotOptions(options, budget, maxOptions, optionCount)
	}));
}
function snapshotQuestions(questions, questionCount, budget, maxInputTokens, minimumTokenCount, maxOptions) {
	const snapshot = [];
	let totalOptionMarkers = 0;
	for (let i = 0; i < questionCount; i++) {
		if (!Object.hasOwn(questions, i)) fail$1();
		const question = readValue(questions, i);
		checkFields(question, questionKeys);
		const instructions = readValue(question, "instructions"), options = readValue(question, "options");
		addTextToBudget(instructions, budget);
		if (Array.isArray(options)) {
			totalOptionMarkers += Math.max(0, readValue(options, "length") - 1);
			if (minimumTokenCount + totalOptionMarkers * 2 > maxInputTokens) fail$1();
		}
		snapshot.push({
			instructions,
			options: snapshotOptions(options, budget, maxOptions, void 0)
		});
	}
	return snapshot;
}
function snapshotOwnDataFields(value, allowed) {
	const fields = /* @__PURE__ */ new Map();
	const keys = Reflect.ownKeys(value);
	if (keys.length > allowed.size) fail$1();
	for (const key of keys) {
		if (typeof key !== "string" || !allowed.has(key)) fail$1();
		const descriptor = Object.getOwnPropertyDescriptor(value, key);
		if (!descriptor || !Object.hasOwn(descriptor, "value")) fail$1();
		fields.set(key, descriptor.value);
	}
	return fields;
}
function snapshotInheritedDataFields(prototype, allowed, fields, allowInheritedDataProperties) {
	for (const key of Reflect.ownKeys(prototype)) {
		if (typeof key !== "string" || allowed.has(key) && fields.has(key)) continue;
		const descriptor = Object.getOwnPropertyDescriptor(prototype, key);
		if (!descriptor) continue;
		if (!allowed.has(key) && (descriptor.enumerable || knownUnsupported.includes(key))) fail$1();
		if (!allowInheritedDataProperties || !allowed.has(key)) continue;
		if (!Object.hasOwn(descriptor, "value")) fail$1();
		fields.set(key, descriptor.value);
	}
}
function snapshotDataRecord(value, allowed, allowInheritedDataProperties) {
	if (!value || typeof value !== "object" || Array.isArray(value)) fail$1();
	const fields = snapshotOwnDataFields(value, allowed);
	const visited = /* @__PURE__ */ new Set([value]);
	let current = Object.getPrototypeOf(value);
	while (current !== null) {
		if (visited.has(current)) fail$1();
		visited.add(current);
		snapshotInheritedDataFields(current, allowed, fields, allowInheritedDataProperties);
		current = Object.getPrototypeOf(current);
	}
	return fields;
}
function dataArrayLength(value) {
	if (!Array.isArray(value)) return void 0;
	const descriptor = Object.getOwnPropertyDescriptor(value, "length");
	if (!descriptor || !Object.hasOwn(descriptor, "value")) fail$1();
	return descriptor.value;
}
function dataArrayItem(value, index) {
	const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
	if (!descriptor || !Object.hasOwn(descriptor, "value")) fail$1();
	return descriptor.value;
}
function snapshotDataOptions(options, count, budget) {
	if (!Array.isArray(options)) return options;
	const snapshot = new Array(count);
	for (let index = 0; index < count; index++) {
		const option = dataArrayItem(options, index);
		addTextToBudget(option, budget);
		snapshot[index] = option;
	}
	return snapshot;
}
function stageDataQuestion(questions, index, budget, minimumTokenCount, limits, counts, allowInheritedDataProperties) {
	const fields = snapshotDataRecord(dataArrayItem(questions, index), questionKeys, allowInheritedDataProperties);
	const instructions = fields.get("instructions");
	const options = fields.get("options");
	addTextToBudget(instructions, budget);
	if (!Array.isArray(options)) return {
		instructions,
		options
	};
	const optionCount = dataArrayLength(options);
	if (optionCount > limits.maxOptions) fail$1();
	counts.optionMarkers += Math.max(0, optionCount - 1);
	if (minimumTokenCount + counts.optionMarkers * 2 > limits.maxInputTokens) fail$1();
	counts.optionCount += optionCount;
	if (counts.optionCount > limits.maxTotalOptions) fail$1();
	return {
		instructions,
		options,
		optionCount
	};
}
function stageDataQuestions(questions, questionCount, budget, minimumTokenCount, limits, allowInheritedDataProperties) {
	const staged = [];
	const counts = {
		optionMarkers: 0,
		optionCount: 0
	};
	for (let index = 0; index < questionCount; index++) staged.push(stageDataQuestion(questions, index, budget, minimumTokenCount, limits, counts, allowInheritedDataProperties));
	return staged;
}
function snapshotDataRequestFields(request, maxQuestions, maxInputCharacters, maxInputTokens, maxOptions, maxTotalOptions, allowInheritedDataProperties) {
	const requestFields = snapshotDataRecord(request, requestKeys, allowInheritedDataProperties);
	const state = requestFields.get("state");
	const questions = requestFields.get("questions");
	if (!Array.isArray(questions)) fail$1();
	const questionCount = dataArrayLength(questions);
	if (questionCount > maxQuestions) fail$1();
	const budget = Number.isSafeInteger(maxInputCharacters) ? {
		max: maxInputCharacters,
		used: 0
	} : null;
	const minimumTokenCount = Number.isSafeInteger(maxInputTokens) ? 1 + questionCount * 4 : 0;
	if (minimumTokenCount > maxInputTokens) fail$1();
	addTextToBudget(state, budget);
	return {
		state,
		questions: stageDataQuestions(questions, questionCount, budget, minimumTokenCount, {
			maxInputTokens,
			maxOptions,
			maxTotalOptions
		}, allowInheritedDataProperties).map(({ instructions, options, optionCount }) => ({
			instructions,
			options: snapshotDataOptions(options, optionCount, budget)
		}))
	};
}
/** Call before tokenization; existing model packers retain all other validation. */
function validateRequestFields(request, maxQuestions = Infinity, maxInputCharacters = Infinity, maxInputTokens = Infinity, maxOptions = Infinity, maxTotalOptions = Infinity, validation = {}) {
	if (validation?.dataPropertiesOnly === true) return snapshotDataRequestFields(request, maxQuestions, maxInputCharacters, maxInputTokens, maxOptions, maxTotalOptions, validation.allowInheritedDataProperties === true);
	checkFields(request, requestKeys);
	const questions = readValue(request, "questions");
	if (!Array.isArray(questions)) fail$1();
	const questionCount = readValue(questions, "length");
	if (questionCount > maxQuestions) fail$1();
	const textBudget = Number.isSafeInteger(maxInputCharacters) ? {
		max: maxInputCharacters,
		used: 0
	} : null;
	const minimumTokenCount = Number.isSafeInteger(maxInputTokens) ? 1 + questionCount * 4 : 0;
	if (minimumTokenCount > maxInputTokens) fail$1();
	const state = readValue(request, "state");
	addTextToBudget(state, textBudget);
	return {
		state,
		questions: Number.isSafeInteger(maxTotalOptions) ? snapshotStagedQuestions(stageQuestions(questions, questionCount, textBudget, maxInputTokens, minimumTokenCount, maxOptions, maxTotalOptions), textBudget, maxOptions) : snapshotQuestions(questions, questionCount, textBudget, maxInputTokens, minimumTokenCount, maxOptions)
	};
}
/** Preserve the existing optional null/undefined options behavior. */
function validateDecisionFields(options) {
	if (options == null) return;
	checkFields(options, decisionKeys);
}
// source/overlay/packages/web-decision/src/kev/system-one.mjs
var MODEL_ID = "kev-0.6b-q4f16";
var containerValue = Symbol("container value");
var roundingView = /* @__PURE__ */ new DataView(/* @__PURE__ */ new ArrayBuffer(8));
var invalid$3 = (message) => {
	throw new WebDecisionError("INVALID_INPUT", message);
};
function addToBudget(budget, characters) {
	if (!budget) return;
	budget.used += characters;
	if (budget.used > budget.max) invalid$3("Kev input exceeds its supported text size.");
}
function checkBigIntBudget(value, budget) {
	if (!budget || value === 0n) return;
	const negative = value < 0n;
	const maxDigits = budget.max - budget.used - (negative ? 1 : 0);
	if (maxDigits < 1) invalid$3("Kev input exceeds its supported text size.");
	const minimumOverflowBits = (10n * BigInt(maxDigits) + 2n) / 3n;
	if ((negative ? -value : value) >> minimumOverflowBits !== 0n) invalid$3("Kev input exceeds its supported text size.");
}
function isMap(value) {
	try {
		Map.prototype.entries.call(value);
		return true;
	} catch {
		return false;
	}
}
function validateRecord(value, label) {
	if (!value || typeof value !== "object" || Array.isArray(value)) invalid$3(`${label} must be an object.`);
	if (isMap(value)) return;
	const prototype = Object.getPrototypeOf(value);
	if (prototype !== null && Object.getPrototypeOf(prototype) !== null) invalid$3(`${label} must be a plain object.`);
}
function recordEntries(value, label, maxEntries = Infinity, budget) {
	validateRecord(value, label);
	if (isMap(value)) {
		const entries = [];
		for (const [key, item] of Map.prototype.entries.call(value)) {
			if (entries.length >= maxEntries) invalid$3(`${label} exceeds its supported number of entries.`);
			if (typeof key !== "string") invalid$3(`${label} must contain string keys.`);
			addToBudget(budget, key.length);
			entries.push([key, item]);
		}
		return entries;
	}
	const keys = Reflect.ownKeys(value);
	if (keys.length > maxEntries) invalid$3(`${label} exceeds its supported number of entries.`);
	const entries = [];
	for (const key of keys) {
		const descriptor = Object.getOwnPropertyDescriptor(value, key);
		if (typeof key !== "string" || !descriptor?.enumerable || !Object.hasOwn(descriptor, "value")) invalid$3(`${label} must contain enumerable string data properties.`);
		addToBudget(budget, key.length);
		entries.push([key, descriptor.value]);
	}
	return entries;
}
function arrayValues(value, label, maxValues = Infinity) {
	if (!Array.isArray(value)) invalid$3(`${label} must be an array.`);
	if (value.length > maxValues) invalid$3(`${label} exceeds its supported number of entries.`);
	const keys = Reflect.ownKeys(value);
	if (keys.length > maxValues + 1) invalid$3(`${label} cannot have extra properties.`);
	const values = [];
	for (let index = 0; index < value.length; index++) {
		const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
		if (!descriptor?.enumerable || !Object.hasOwn(descriptor, "value")) invalid$3(`${label} must be a dense data array.`);
		values.push(descriptor.value);
	}
	for (const key of keys) if (key !== "length" && (typeof key !== "string" || !/^(0|[1-9]\d*)$/.test(key) || Number(key) >= value.length)) invalid$3(`${label} cannot have extra properties.`);
	return values;
}
function primitiveText(value, budget, arrayItem) {
	if (value === null) return "";
	if (typeof value === "string") {
		const trimmed = arrayItem ? pythonWhitespacePrefixLength(value) : 0;
		addToBudget(budget, value.length - trimmed);
		return value;
	}
	if (typeof value === "boolean") {
		const text = value ? "True" : "False";
		addToBudget(budget, text.length);
		return text;
	}
	if (typeof value === "number") {
		if (!Number.isFinite(value)) invalid$3("JSON numbers must be finite.");
		if (Number.isInteger(value) && !Number.isSafeInteger(value)) invalid$3("Integer JSON numbers must be safe integers; use bigint for exact integers.");
		const text = String(value);
		addToBudget(budget, text.length);
		return text;
	}
	if (typeof value === "bigint") {
		checkBigIntBudget(value, budget);
		const text = value.toString();
		addToBudget(budget, text.length);
		return text;
	}
	if (!value || typeof value !== "object") invalid$3("Values must be JSON values.");
	return containerValue;
}
function enterContainer(frame, value, budget, ancestors) {
	if (ancestors.has(value)) invalid$3("JSON values cannot contain cycles.");
	ancestors.add(value);
	frame.container = true;
	frame.entered = true;
	frame.lines = [];
	frame.index = 0;
	if (Array.isArray(value)) {
		frame.kind = "array";
		frame.children = arrayValues(value, "JSON arrays", budget?.max ?? Infinity);
	} else {
		frame.kind = "object";
		frame.children = recordEntries(value, "JSON objects", budget?.max ?? Infinity);
	}
}
function appendRenderedChild(parent, text, budget) {
	const index = parent.index - 1;
	const padLength = parent.indent * 2;
	if (parent.kind === "array") {
		const leading = pythonWhitespacePrefixLength(text);
		const trimmedPad = parent.arrayItem && index === 0 ? padLength : 0;
		addToBudget(budget, padLength + 2 - trimmedPad);
		const pad = " ".repeat(padLength);
		const rendered = text.slice(leading);
		parent.lines.push(`${pad}- ${rendered}`);
		return;
	}
	const [key, item] = parent.children[index];
	const separator = item && typeof item === "object" ? ":\n" : ": ";
	const trimmedPrefix = parent.arrayItem && index === 0 ? padLength + pythonWhitespacePrefixLength(key) : 0;
	addToBudget(budget, padLength + key.length + separator.length - trimmedPrefix);
	const pad = " ".repeat(padLength);
	parent.lines.push(`${pad}${key}${separator}${text}`);
}
function pythonWhitespacePrefixLength(text) {
	let length = 0;
	for (const character of text) {
		const codePoint = character.codePointAt(0);
		if (!(codePoint >= 9 && codePoint <= 13 || codePoint >= 28 && codePoint <= 32 || codePoint === 133 || codePoint === 160 || codePoint === 5760 || codePoint >= 8192 && codePoint <= 8202 || codePoint === 8232 || codePoint === 8233 || codePoint === 8239 || codePoint === 8287 || codePoint === 12288)) break;
		length += character.length;
	}
	return length;
}
function pushChildFrame(frames, parent) {
	const childIndex = parent.index++;
	const child = parent.kind === "array" ? parent.children[childIndex] : parent.children[childIndex][1];
	const arrayItem = parent.kind === "array";
	frames.push({
		value: child,
		indent: parent.indent + 1,
		entered: false,
		arrayItem
	});
}
function render(value, budget) {
	const ancestors = /* @__PURE__ */ new Set();
	const frames = [{
		value,
		indent: 0,
		entered: false
	}];
	let result;
	function finish(text) {
		const finished = frames.pop();
		if (finished.container) ancestors.delete(finished.value);
		if (frames.length === 0) {
			result = text;
			return;
		}
		appendRenderedChild(frames.at(-1), text, budget);
	}
	while (frames.length > 0) {
		const frame = frames.at(-1);
		if (!frame.entered) {
			const text = primitiveText(frame.value, budget, frame.arrayItem);
			if (text !== containerValue) finish(text);
			else enterContainer(frame, frame.value, budget, ancestors);
			continue;
		}
		if (frame.index < frame.children.length) {
			pushChildFrame(frames, frame);
			continue;
		}
		addToBudget(budget, Math.max(0, frame.lines.length - 1));
		finish(frame.lines.join("\n"));
	}
	return result;
}
function optionText(name, description, budget, nameAlreadyCounted = false) {
	if (!nameAlreadyCounted) addToBudget(budget, name.length);
	if (description === null || description === "") return name;
	const rendered = render(description, budget);
	addToBudget(budget, 2);
	return `${name}: ${rendered}`;
}
function questionFields(question, name) {
	validateRecord(question, `Question ${name}`);
	const fields = {};
	for (const key of Reflect.ownKeys(question)) {
		if (typeof key !== "string" || ![
			"type",
			"instructions",
			"criteria"
		].includes(key)) continue;
		const descriptor = Object.getOwnPropertyDescriptor(question, key);
		if (!descriptor?.enumerable || !Object.hasOwn(descriptor, "value")) invalid$3(`Question ${name} fields must be enumerable data properties.`);
		fields[key] = descriptor.value;
	}
	if (fields.instructions === void 0 && Object.hasOwn(fields, "instructions")) invalid$3(`Question ${name} instructions must be a JSON value.`);
	return fields;
}
function choiceQuestion(name, fields, index, budget) {
	const entries = recordEntries(fields.criteria, `Question ${name} criteria`, 255, budget);
	if (entries.length < 1 || entries.length > 255) invalid$3(`Question ${name} needs 1 to 255 options.`);
	const options = entries.map(([key, description]) => {
		return {
			key,
			label: optionText(key, description, budget, true)
		};
	});
	const labels = options.map((option) => option.label);
	return {
		metadata: {
			name,
			type: "choice",
			options
		},
		request: {
			instructions: render(fields.instructions ?? null, budget),
			options: labels
		},
		allowDuplicateOptions: true,
		index
	};
}
function scoreQuestion(name, fields, index, budget) {
	const levels = arrayValues(fields.criteria, `Question ${name} criteria`, 255);
	if (levels.length < 1 || levels.length > 255) invalid$3(`Question ${name} needs 1 to 255 score levels.`);
	const labels = levels.map((level) => render(level, budget));
	return {
		metadata: {
			name,
			type: "score",
			options: labels.map((label, optionIndex) => ({
				key: String(optionIndex),
				label
			}))
		},
		request: {
			instructions: render(fields.instructions ?? null, budget),
			options: labels
		},
		allowDuplicateOptions: true,
		index
	};
}
function noulCriteriaValue(criteria, key, name) {
	if (isMap(criteria)) {
		if (!Map.prototype.has.call(criteria, key)) return null;
		return Map.prototype.get.call(criteria, key);
	}
	const descriptor = Object.getOwnPropertyDescriptor(criteria, key);
	if (!descriptor) return null;
	if (!descriptor.enumerable || !Object.hasOwn(descriptor, "value")) invalid$3(`Question ${name} criteria must contain enumerable string data properties.`);
	return descriptor.value;
}
function noulQuestion(name, fields, budget) {
	const descriptions = {
		true: null,
		false: null
	};
	if (Object.hasOwn(fields, "criteria") && fields.criteria !== null) {
		const criteria = fields.criteria;
		validateRecord(criteria, `Question ${name} criteria`);
		descriptions.true = noulCriteriaValue(criteria, "true", name);
		descriptions.false = noulCriteriaValue(criteria, "false", name);
	}
	const options = [optionText("no", descriptions.false, budget), optionText("yes", descriptions.true, budget)];
	return {
		metadata: {
			name,
			type: "noul",
			options: [{
				key: "false",
				label: options[0]
			}, {
				key: "true",
				label: options[1]
			}]
		},
		request: {
			instructions: render(fields.instructions ?? null, budget),
			options
		}
	};
}
function prepareKevSystemOne(state, questions, maxInputCharacters = Infinity) {
	const budget = Number.isSafeInteger(maxInputCharacters) ? {
		max: maxInputCharacters,
		used: 0
	} : null;
	const entries = recordEntries(questions, "Questions");
	if (entries.length < 1) invalid$3("Provide at least one named question.");
	const metadata = [];
	const requestQuestions = [];
	const allowDuplicateOptionsFor = /* @__PURE__ */ new Set();
	for (const [name, value] of entries) {
		if (!value || typeof value !== "object" || Array.isArray(value)) invalid$3(`Question ${name} must be an object.`);
		const fields = questionFields(value, name);
		let prepared;
		if (fields.type === "choice") prepared = choiceQuestion(name, fields, requestQuestions.length, budget);
		else if (fields.type === "score") prepared = scoreQuestion(name, fields, requestQuestions.length, budget);
		else if (fields.type === "noul") prepared = noulQuestion(name, fields, budget);
		else invalid$3(`Question ${name} type must be choice, score, or noul.`);
		metadata.push(prepared.metadata);
		requestQuestions.push(prepared.request);
		if (prepared.allowDuplicateOptions) allowDuplicateOptionsFor.add(prepared.index);
	}
	return {
		request: {
			state: render(state, budget),
			questions: requestQuestions
		},
		metadata,
		packing: {
			allowEmptyInstructions: true,
			allowEmptyOptions: true,
			allowDuplicateOptionsFor
		}
	};
}
function normalizedProbabilities(probabilities) {
	const total = probabilities.reduce((sum, probability) => sum + probability, 0);
	if (!Number.isFinite(total) || probabilities.some((probability) => !Number.isFinite(probability) || probability < 0)) throw new WebDecisionError("INFERENCE_ERROR", "The model returned invalid probabilities.");
	if (total === 0) return probabilities.map(() => 1 / probabilities.length);
	return probabilities.map((probability) => probability / total);
}
function choiceConfidence$1(probabilities) {
	const count = probabilities.length;
	if (count === 1) return 1;
	const normalized = normalizedProbabilities(probabilities);
	return (Math.max(...normalized) - 1 / count) / (1 - 1 / count);
}
function scoreConfidence$1(probabilities) {
	const levels = probabilities.length;
	if (levels === 1) return 1;
	const normalized = normalizedProbabilities(probabilities);
	let mode = 0;
	for (let index = 1; index < levels; index++) if (normalized[index] > normalized[mode]) mode = index;
	const expectedDistance = normalized.reduce((sum, probability, index) => sum + probability * Math.abs(index - mode), 0);
	let uniformDistance = 0;
	for (let index = 0; index < levels; index++) uniformDistance += Math.abs(index - (levels - 1) / 2);
	const meanAbsoluteDeviation = uniformDistance / levels;
	return Math.max(0, 1 - expectedDistance / meanAbsoluteDeviation);
}
function roundFourDecimals(value) {
	if (value === 0) return value;
	roundingView.setFloat64(0, value, false);
	const bits = roundingView.getBigUint64(0, false);
	const negative = bits >> 63n !== 0n;
	const exponentBits = Number(bits >> 52n & 2047n);
	const fraction = bits & 4503599627370495n;
	const significand = exponentBits === 0 ? fraction : fraction | 4503599627370496n;
	const exponent = exponentBits === 0 ? -1074 : exponentBits - 1075;
	const numerator = significand * 625n;
	const shift = exponent + 4;
	const denominator = 1n << BigInt(-shift);
	let units = numerator / denominator;
	const doubledRemainder = numerator % denominator * 2n;
	if (doubledRemainder > denominator || doubledRemainder === denominator && units % 2n === 1n) units++;
	return (negative ? -Number(units) : Number(units)) / 1e4;
}
function finishAnswer$1(metadata, answer) {
	const probabilities = answer.probabilities.map(({ probability }) => probability);
	if (metadata.type === "choice") {
		const probabilityEntries = metadata.options.map(({ key }, index) => [key, roundFourDecimals(probabilities[index])]);
		const roundedProbabilities = Object.fromEntries(probabilityEntries);
		return {
			type: "choice",
			choice: metadata.options[answer.selectedIndex].key,
			confidence: roundFourDecimals(choiceConfidence$1(probabilities)),
			probabilities: roundedProbabilities,
			probabilityEntries
		};
	}
	if (metadata.type === "score") {
		const legend = Object.fromEntries(metadata.options.map(({ label }, index) => [String(index), label]));
		const probabilityEntries = probabilities.map((probability, index) => [String(index), roundFourDecimals(probability)]);
		const distribution = Object.fromEntries(probabilityEntries);
		return {
			type: "score",
			score: roundFourDecimals(probabilities.reduce((sum, probability, index) => sum + index * probability, 0)),
			legend,
			probabilities: distribution,
			probabilityEntries,
			confidence: roundFourDecimals(scoreConfidence$1(probabilities))
		};
	}
	return {
		type: "noul",
		noul: roundFourDecimals(probabilities[1])
	};
}
function finishKevSystemOne(prepared, result, model = MODEL_ID) {
	const answerEntries = prepared.metadata.map((metadata, index) => [metadata.name, finishAnswer$1(metadata, result.answers[index])]);
	const answers = Object.fromEntries(answerEntries);
	return {
		model,
		tokenCount: result.tokenCount,
		answers,
		answerEntries
	};
}
// source/overlay/packages/web-decision/src/model.js
var noFailure = Symbol("no failure");
function checkDeviceAfterCleanup(session, message) {
	if (session.lifecycle.loss || session.lifecycle.disposed) throw new WebDecisionError("INFERENCE_ERROR", message, { cause: /* @__PURE__ */ new Error("The GPU device was lost.") });
}
async function inferDecisionModel(model, request, options = {}, packing = {}) {
	if (model.closed) throw new WebDecisionError("SESSION_CLOSED", "The model has been disposed.");
	const signal = options?.signal;
	signal?.throwIfAborted();
	validateDecisionFields(options);
	request = validateRequestFields(request, model.limits?.maxQuestions, model.limits?.maxInputCharacters, model.limits?.maxInputTokens, model.limits?.maxOptions ?? 255);
	if (model.closed) throw new WebDecisionError("SESSION_CLOSED", "The model has been disposed.");
	const packed = packKevRequest(model.tokenizer, request, model.limits, packing);
	if (model.closed) throw new WebDecisionError("SESSION_CLOSED", "The model has been disposed.");
	const schedule = () => {
		if (model.closed) throw new WebDecisionError("SESSION_CLOSED", "The model has been disposed.");
		const task = model.queue.then(() => {
			signal?.throwIfAborted();
			return model.run(packed);
		});
		model.queue = task.catch(() => {});
		return task;
	};
	return signal ? waitForAbort(schedule, signal) : schedule();
}
var DecisionModel = class {
	constructor(runtime, device, gpu, fn, weights, tokenizer, limits, uncaptured, lifecycle, supportsSystemOne = false, modelId = "kev-0.6b-q4f16") {
		Object.assign(this, {
			runtime,
			device,
			gpu,
			fn,
			weights,
			tokenizer,
			limits,
			uncaptured,
			lifecycle,
			supportsSystemOne,
			modelId
		});
		this.closed = false;
		this.queue = Promise.resolve();
		this.disposal = null;
	}
	async decide(request, options = {}) {
		return inferDecisionModel(this, request, options);
	}
	async systemOne(state, questions, options = {}) {
		if (!this.supportsSystemOne) throw new WebDecisionError("UNSUPPORTED_ENVIRONMENT", "Typed System One input is unavailable for this model route.");
		if (this.closed) throw new WebDecisionError("SESSION_CLOSED", "The model has been disposed.");
		options?.signal?.throwIfAborted();
		validateDecisionFields(options);
		const prepared = prepareKevSystemOne(state, questions, this.limits?.maxInputCharacters);
		return finishKevSystemOne(prepared, await inferDecisionModel(this, prepared.request, options, prepared.packing), this.modelId);
	}
	async run(packed) {
		if (this.lifecycle.loss || this.lifecycle.disposed) throw new WebDecisionError("INFERENCE_ERROR", "The GPU device was lost. Load the model again.");
		if (this.uncaptured.length) throw new WebDecisionError("INFERENCE_ERROR", this.uncaptured[0]);
		const n = packed.ids.length;
		let result, failure = noFailure, scopePushed = false, scopeStarted = false;
		try {
			this.runtime.beginScope();
			scopeStarted = true;
			this.gpu.pushErrorScope("validation");
			scopePushed = true;
			const ids = this.runtime.empty([1, n], "int32", this.device);
			const mask = this.runtime.empty([1, n], "int32", this.device);
			ids.copyFrom(Int32Array.from(packed.ids));
			mask.copyFrom(new Int32Array(n).fill(1));
			const y = this.fn(ids, mask, ...this.weights);
			const host = this.runtime.empty([1, n], "float16", this.runtime.cpu()).copyFrom(y);
			await this.device.sync();
			if (this.lifecycle.loss || this.lifecycle.disposed) throw new Error("The GPU device was lost.");
			const raw = host.toRawBytes();
			if (raw.byteLength !== n * 2) throw new Error("Unexpected output length.");
			const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
			result = unpackKevScores(Array.from({ length: n }, (_, i) => decodeFloat16(view.getUint16(i * 2, true))), packed);
		} catch (cause) {
			failure = cause;
		} finally {
			try {
				const gpuError = scopePushed ? await this.gpu.popErrorScope() : null;
				if (gpuError && failure === noFailure) failure = gpuError;
			} catch (cause) {
				if (failure === noFailure) failure = cause;
			}
			if (scopeStarted && !this.lifecycle.disposed) try {
				this.runtime.endScope();
			} catch (cause) {
				if (failure === noFailure) failure = cause;
			}
		}
		if (failure !== noFailure) throw new WebDecisionError("INFERENCE_ERROR", "Model inference failed.", { cause: failure });
		if (this.uncaptured.length) throw new WebDecisionError("INFERENCE_ERROR", this.uncaptured[0]);
		checkDeviceAfterCleanup(this, "Model inference failed.");
		return result;
	}
	/** Release unused TVM allocator buffers after earlier queued requests finish. */
	async releaseCachedMemory() {
		if (this.closed) throw new WebDecisionError("SESSION_CLOSED", "The model has been disposed.");
		const task = this.queue.then(async () => {
			if (this.lifecycle.loss || this.lifecycle.disposed) throw new WebDecisionError("INFERENCE_ERROR", "The GPU device was lost. Load the model again.");
			if (this.uncaptured.length) throw new WebDecisionError("INFERENCE_ERROR", this.uncaptured[0]);
			let failure = noFailure, scopeStarted = false, scopePushed = false;
			try {
				this.runtime.beginScope();
				scopeStarted = true;
				this.gpu.pushErrorScope("validation");
				scopePushed = true;
				this.runtime.getGlobalFunc("vm.builtin.memory_manager.clear")();
				await this.device.sync();
				if (this.lifecycle.loss || this.lifecycle.disposed) throw new Error("The GPU device was lost.");
			} catch (cause) {
				failure = cause;
			} finally {
				try {
					const gpuError = scopePushed ? await this.gpu.popErrorScope() : null;
					if (gpuError && failure === noFailure) failure = gpuError;
				} catch (cause) {
					if (failure === noFailure) failure = cause;
				}
				if (scopeStarted && !this.lifecycle.disposed) try {
					this.runtime.endScope();
				} catch (cause) {
					if (failure === noFailure) failure = cause;
				}
			}
			if (failure !== noFailure) throw new WebDecisionError("INFERENCE_ERROR", "Could not release cached runtime memory.", { cause: failure });
			if (this.uncaptured.length) throw new WebDecisionError("INFERENCE_ERROR", this.uncaptured[0]);
			checkDeviceAfterCleanup(this, "Could not release cached runtime memory.");
		});
		this.queue = task.catch(() => {});
		return task;
	}
	async dispose() {
		if (this.disposal) return this.disposal;
		this.closed = true;
		this.disposal = (async () => {
			await this.queue;
			try {
				if (!this.lifecycle.disposed) try {
					if (!this.lifecycle.loss) this.runtime.endScope();
				} finally {
					this.runtime.dispose();
				}
			} finally {
				this.weights = [];
				this.fn = this.runtime = this.device = this.gpu = this.tokenizer = this.lifecycle = null;
				this.uncaptured = [];
			}
		})();
		return this.disposal;
	}
};
// source/overlay/packages/web-decision/src/runtime-lifecycle.js
/** Track TVM's automatic device-loss disposal and make disposal idempotent. */
function trackRuntime(runtime, gpu) {
	let disposed = false;
	let loss = null;
	const dispose = runtime.dispose.bind(runtime);
	runtime.dispose = (...args) => {
		if (disposed) return;
		disposed = true;
		return dispose(...args);
	};
	gpu.lost.then((info) => {
		if (!disposed) loss = info;
	}, (error) => {
		if (!disposed) loss = { message: String(error) };
	});
	return {
		get disposed() {
			return disposed;
		},
		get loss() {
			return loss;
		}
	};
}
/** Refuse to return a model whose device was lost or runtime was disposed while loading. */
function assertLoadActive(lifecycle) {
	if (lifecycle.loss || lifecycle.disposed) throw new WebDecisionError("MODEL_LOAD_FAILED", "The GPU device was lost while loading the model.");
}
/** Close the load watcher and attempt scope, runtime, and device cleanup. */
function cleanupFailedLoad({ loadWatch, runtime, scopeOpen, lifecycle, gpuDevice }) {
	loadWatch?.close();
	try {
		if (runtime && scopeOpen && !lifecycle?.disposed && !lifecycle?.loss) runtime.endScope();
	} catch {}
	try {
		runtime?.dispose();
	} catch {}
	try {
		gpuDevice?.destroy();
	} catch {
		retainStrandsResources(runtime, gpuDevice);
	}
}
// source/overlay/packages/web-decision/src/load-device-loss.js
/** Forward caller cancellation and GPU loss while model assets are loading. */
function watchLoadDeviceLoss(device, callerSignal) {
	const controller = new AbortController();
	let closed = false;
	const forwardAbort = () => controller.abort(callerSignal.reason);
	const onLoss = (cause) => {
		if (!closed) controller.abort(new WebDecisionError("MODEL_LOAD_FAILED", "The GPU device was lost while loading the model.", { cause }));
	};
	try {
		callerSignal?.addEventListener("abort", forwardAbort, { once: true });
		if (callerSignal?.aborted) forwardAbort();
		device.lost.then(onLoss, onLoss);
	} catch (cause) {
		closed = true;
		try {
			callerSignal?.removeEventListener("abort", forwardAbort);
		} catch {}
		throw cause;
	}
	return {
		signal: controller.signal,
		close() {
			if (closed) return;
			closed = true;
			callerSignal?.removeEventListener("abort", forwardAbort);
		}
	};
}
Object.freeze([
	Object.freeze({
		index: 394,
		path: "heads/param_394.bin",
		shape: Object.freeze([1024, 256]),
		bytes: 1048576
	}),
	Object.freeze({
		index: 395,
		path: "heads/param_395.bin",
		shape: Object.freeze([1024, 256]),
		bytes: 1048576
	}),
	Object.freeze({
		index: 396,
		path: "heads/param_396.bin",
		shape: Object.freeze([256]),
		bytes: 1024
	}),
	Object.freeze({
		index: 397,
		path: "heads/param_397.bin",
		shape: Object.freeze([256]),
		bytes: 1024
	})
]);
// source/overlay/packages/web-decision/src/asset-batch.js
/** Read the initial model files together, cancelling siblings on first failure. */
async function readAssetBatch(names, read, signal) {
	signal?.throwIfAborted();
	const controller = new AbortController();
	const forwardAbort = () => controller.abort(signal.reason);
	signal?.addEventListener("abort", forwardAbort, { once: true });
	if (signal?.aborted) forwardAbort();
	let failed = false, firstFailure;
	try {
		const values = await Promise.all(names.map((name) => Promise.resolve().then(() => {
			controller.signal.throwIfAborted();
			return read(name, controller.signal);
		}).catch((cause) => {
			if (!failed) {
				failed = true;
				firstFailure = cause;
			}
			if (!controller.signal.aborted) controller.abort(cause);
			throw cause;
		})));
		signal?.throwIfAborted();
		return values;
	} catch (cause) {
		signal?.throwIfAborted();
		throw failed ? firstFailure : cause;
	} finally {
		signal?.removeEventListener("abort", forwardAbort);
	}
}
// source/overlay/packages/web-decision/src/gpu-preflight.js
/** Check a required feature before model reads without requesting a device.
* Do not return the adapter: it can expire while model files are downloading.
*/
async function requestF16Adapter(gpu, signal) {
	signal?.throwIfAborted();
	let adapter;
	try {
		adapter = await waitForAbort(() => gpu.requestAdapter({ powerPreference: "high-performance" }), signal);
		signal?.throwIfAborted();
		if (!adapter) throw new Error("No compatible WebGPU adapter was found.");
	} catch (cause) {
		signal?.throwIfAborted();
		throw new WebDecisionError("MODEL_LOAD_FAILED", "Could not initialize a compatible WebGPU device.", { cause });
	}
	if (!adapter.features.has("shader-f16")) throw new WebDecisionError("UNSUPPORTED_ENVIRONMENT", "This model requires WebGPU shader-f16 support.");
}
// source/overlay/packages/web-decision/src/shard-prefetch.js
/** Validate a budget for the declared bytes of one additional model shard. */
function validatePrefetchBytes(value) {
	if (value === void 0) return 0;
	if (!Number.isSafeInteger(value) || value < 0) throw new WebDecisionError("INVALID_INPUT", "maxPrefetchBytes must be a nonnegative safe integer.");
	return value;
}
/** Read at most one additional shard while consuming shards in manifest order. */
async function processPrefetchedShards(records, read, consume, signal, maxPrefetchBytes) {
	signal?.throwIfAborted();
	const budget = validatePrefetchBytes(maxPrefetchBytes);
	if (!Array.isArray(records) || records.some((record) => !Number.isSafeInteger(record?.nbytes) || record.nbytes < 0)) throw new WebDecisionError("INVALID_ASSET", "Invalid shard sizes.");
	const controller = new AbortController();
	const noReadFailure = Symbol("no read failure");
	let firstReadFailure = noReadFailure;
	const forwardAbort = () => controller.abort(signal.reason);
	signal?.addEventListener("abort", forwardAbort, { once: true });
	if (signal?.aborted) forwardAbort();
	const failRead = (error) => {
		if (!controller.signal.aborted) {
			firstReadFailure = error;
			controller.abort(error);
		}
		return {
			ok: false,
			error
		};
	};
	const start = (index) => index < records.length ? Promise.resolve().then(() => {
		try {
			controller.signal.throwIfAborted();
			return Promise.resolve(read(records[index], controller.signal, index)).then((value) => ({
				ok: true,
				value
			}), failRead);
		} catch (error) {
			return failRead(error);
		}
	}) : null;
	const mayPrefetch = (index) => budget > 0 && index < records.length && records[index].nbytes <= budget;
	let current = start(0), next = mayPrefetch(1) ? start(1) : null;
	try {
		for (let i = 0; i < records.length; i++) {
			const outcome = await waitForAbort(() => current, controller.signal);
			controller.signal.throwIfAborted();
			if (!outcome.ok) throw outcome.error;
			await consume(records[i], outcome.value, i);
			controller.signal.throwIfAborted();
			current = next || start(i + 1);
			next = mayPrefetch(i + 2) ? start(i + 2) : null;
		}
	} catch (cause) {
		signal?.throwIfAborted();
		throw firstReadFailure !== noReadFailure ? firstReadFailure : cause;
	} finally {
		controller.abort();
		signal?.removeEventListener("abort", forwardAbort);
	}
}
// source/overlay/packages/web-decision/src/model-source.js
function validateSource(options, validateModelOptions) {
	if (!options || typeof options.assetsUrl !== "string" || !options.assetsUrl) throw new WebDecisionError("INVALID_INPUT", "assetsUrl is required.");
	if (options.offline !== void 0 && typeof options.offline !== "boolean") throw new WebDecisionError("INVALID_INPUT", "offline must be a boolean.");
	if (options.offline && options.cache === false) throw new WebDecisionError("INVALID_INPUT", "Offline loading requires the model cache.");
	const maxPrefetchBytes = validatePrefetchBytes(options.maxPrefetchBytes);
	const modelOptions = validateModelOptions?.(options);
	if (typeof navigator === "undefined" || !navigator.gpu) throw new WebDecisionError("UNSUPPORTED_ENVIRONMENT", "WebGPU is unavailable. Use a supported browser on HTTPS or localhost.");
	if (!globalThis.crypto?.subtle) throw new WebDecisionError("UNSUPPORTED_ENVIRONMENT", "Web Crypto is unavailable. Use HTTPS or localhost.");
	let base;
	try {
		base = new URL(options.assetsUrl, globalThis.location?.href);
	} catch (cause) {
		throw new WebDecisionError("INVALID_INPUT", "Invalid assetsUrl.", { cause });
	}
	if (!["http:", "https:"].includes(base.protocol) || base.search || base.hash || base.username || base.password) throw new WebDecisionError("INVALID_INPUT", "assetsUrl must be an HTTP or HTTPS directory URL without credentials, query, or fragment.");
	if (!base.pathname.endsWith("/")) base.pathname += "/";
	return {
		base,
		maxPrefetchBytes,
		modelOptions
	};
}
async function getManifestResponse(options, cache, manifestUrl, signal, offlineMissingMessage) {
	if (options.offline) {
		if (!cache) throw new WebDecisionError("ASSET_LOAD_FAILED", "The model cache is unavailable for offline loading.");
		let response;
		try {
			response = await waitForAbort(() => cache.match(manifestUrl), signal, cancelResponse);
		} catch (cause) {
			signal?.throwIfAborted();
			throw new WebDecisionError("ASSET_LOAD_FAILED", "Could not read the cached model manifest.", { cause });
		}
		if (!response) {
			signal?.throwIfAborted();
			throw new WebDecisionError("ASSET_LOAD_FAILED", offlineMissingMessage);
		}
		return response;
	}
	let response;
	try {
		response = await waitForAbort(() => fetch(manifestUrl, {
			signal,
			cache: "no-cache"
		}), signal, cancelResponse);
	} catch (cause) {
		signal?.throwIfAborted();
		throw new WebDecisionError("ASSET_LOAD_FAILED", "Could not fetch manifest.json.", { cause });
	}
	if (!response.ok) {
		cancelBody(response.body);
		signal?.throwIfAborted();
		throw new WebDecisionError("ASSET_LOAD_FAILED", `Could not fetch manifest.json: HTTP ${response.status}.`);
	}
	return response;
}
/** Validate source options and obtain the matching manifest without creating a GPU device. */
async function prepareModelSource(options, { validateModelOptions, offlineMissingMessage }) {
	const { base, maxPrefetchBytes, modelOptions } = validateSource(options, validateModelOptions);
	const signal = options.signal;
	signal?.throwIfAborted();
	await requestF16Adapter(navigator.gpu, signal);
	const cache = await openAssetCache(options.cache !== false, signal);
	signal?.throwIfAborted();
	const manifestUrl = assetUrl(base, "manifest.json");
	return {
		base,
		cache,
		signal,
		manifestUrl,
		response: await getManifestResponse(options, cache, manifestUrl, signal, offlineMissingMessage),
		maxPrefetchBytes,
		modelOptions
	};
}
// source/overlay/packages/web-decision/src/model-assets.js
/** Read verified assets while tracking progress and whether all files were cached. */
function createModelAssetReader({ base, manifest, cache, signal, options }) {
	const totalBytes = Object.values(manifest.files).reduce((sum, entry) => sum + entry.bytes, 0);
	let loadedBytes = 0, cacheComplete = true;
	const read = (name, readSignal = signal) => readVerifiedAsset(base, manifest, name, {
		cache,
		signal: readSignal,
		offline: options.offline === true,
		onCacheStatus: (cached) => {
			if (!cached) cacheComplete = false;
		},
		onBytes: (count) => {
			loadedBytes += count;
			options.onProgress?.({
				loadedBytes,
				totalBytes,
				fraction: totalBytes ? loadedBytes / totalBytes : 1
			});
		}
	});
	return {
		read,
		isCacheComplete: () => cacheComplete
	};
}
/** Upload each verified tensor shard in order, with optional bounded read-ahead. */
async function uploadModelShards({ records, read, runtime, device, weights, signal, maxPrefetchBytes, uncaptured }) {
	const upload = async (shard, data) => {
		const bytes = new Uint8Array(data);
		for (const record of shard.records) {
			const index = Number(record.name.slice(6));
			const tensor = runtime.empty(record.shape, record.dtype, device);
			tensor.copyFromRawBytes(bytes.subarray(record.byteOffset, record.byteOffset + record.nbytes));
			weights[index] = tensor;
		}
		await device.sync();
		if (uncaptured.length) throw new WebDecisionError("INFERENCE_ERROR", uncaptured[0]);
	};
	if (maxPrefetchBytes > 0) await processPrefetchedShards(records, (shard, readSignal) => read(`weights/${shard.dataPath}`, readSignal), upload, signal, maxPrefetchBytes);
	else for (const shard of records) await upload(shard, await read(`weights/${shard.dataPath}`, signal));
}
// source/overlay/packages/web-decision/src/pinned-manifest.js
function canonical(value) {
	if (Array.isArray(value)) return value.map(canonical);
	if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
	return value;
}
/** Verify canonical JSON so a reserialized cached manifest has the same identity. */
async function readPinnedManifest(response, signal, expected, name) {
	try {
		const manifest = parseJson(await readBoundedResponse(response, MAX_MANIFEST_BYTES, { signal }), "manifest.json");
		const canonicalBytes = new TextEncoder().encode(JSON.stringify(canonical(manifest)) + "\n");
		const digest = await waitForAbort(() => sha256(canonicalBytes), signal);
		signal?.throwIfAborted();
		if (digest !== expected) throw new WebDecisionError("INVALID_ASSET", `The ${name} manifest does not match the verified experimental build.`);
		return manifest;
	} catch (cause) {
		signal?.throwIfAborted();
		if (cause instanceof WebDecisionError || cause?.name === "AbortError") throw cause;
		throw new WebDecisionError("INVALID_ASSET", `Could not read the ${name} manifest.`, { cause });
	}
}
// source/overlay/packages/web-decision/src/kev06/contract.js
var KEV06_MANIFEST_SHA256 = "bc79c92a7011d2aa1aff4ede66c06cc365de81e744507256afb5c0449870716a";
/** Accept only the verified default Kev 0.6B asset manifest. */
async function readKev06ManifestResponse(response, signal) {
	return validateManifest(await readPinnedManifest(response, signal, KEV06_MANIFEST_SHA256, "Kev 0.6B"));
}
// source/overlay/packages/web-decision/src/kev-loader.js
function unsupported(message) {
	return new WebDecisionError("UNSUPPORTED_ENVIRONMENT", message);
}
function cacheRecordIndex(record) {
	const match = typeof record?.name === "string" ? /^param_(\d+)$/.exec(record.name) : null;
	return match ? Number(match[1]) : -1;
}
function cacheRecordElementBytes(record) {
	return record?.dtype === "uint32" ? 4 : record?.dtype === "float16" ? 2 : 0;
}
function validCacheShard(shard, files) {
	if (!shard || typeof shard.dataPath !== "string" || shard.dataPath.includes("/") || shard.dataPath.includes("\\")) return false;
	const file = files[`weights/${shard.dataPath}`];
	return Boolean(file) && shard.format === "raw-shard" && shard.nbytes === file.bytes && Array.isArray(shard.records);
}
function validCacheRecordShape(record, bytesPerElement) {
	return bytesPerElement > 0 && Array.isArray(record.shape) && !record.shape.some((value) => !Number.isSafeInteger(value) || value < 1) && record.shape.reduce((a, b) => a * b, bytesPerElement) === record.nbytes;
}
function validCacheRecordRange(record, shard) {
	return Number.isSafeInteger(record.byteOffset) && record.byteOffset >= 0 && record.byteOffset % 4 === 0 && Number.isSafeInteger(record.nbytes) && record.nbytes % 4 === 0 && record.byteOffset + record.nbytes <= shard.nbytes;
}
function validCacheRecord(record, shard, seen) {
	if (!record || record.format !== "raw") return false;
	const index = cacheRecordIndex(record);
	const bytesPerElement = cacheRecordElementBytes(record);
	return index >= 0 && index < 394 && !seen.has(index) && validCacheRecordShape(record, bytesPerElement) && validCacheRecordRange(record, shard);
}
function checkCacheManifest(manifest, cacheManifest) {
	if (cacheManifest?.metadata?.numParams !== 394 || !Array.isArray(cacheManifest.records)) throw new WebDecisionError("INVALID_ASSET", "Invalid TVM tensor cache metadata.");
	const seen = /* @__PURE__ */ new Set();
	for (const shard of cacheManifest.records) {
		if (!validCacheShard(shard, manifest.files)) throw new WebDecisionError("INVALID_ASSET", "Invalid tensor shard metadata.");
		for (const record of shard.records) {
			if (!validCacheRecord(record, shard, seen)) throw new WebDecisionError("INVALID_ASSET", "Invalid model parameter metadata.");
			seen.add(cacheRecordIndex(record));
		}
	}
	if (seen.size !== 394) throw new WebDecisionError("INVALID_ASSET", "Missing model parameters.");
}
function validateOptionalLimit(value, name) {
	if (value !== void 0 && (!Number.isSafeInteger(value) || value < 0)) throw new WebDecisionError("INVALID_INPUT", `${name} must be a non-negative safe integer.`);
}
async function loadKevModel(options, description, readManifest, uploadHeads) {
	validateOptionalLimit(options?.maxInputCharacters, "maxInputCharacters");
	validateOptionalLimit(options?.maxInputTokens, "maxInputTokens");
	const { base, cache, signal, manifestUrl, response: manifestResponse, maxPrefetchBytes } = await prepareModelSource(options, { offlineMissingMessage: "The model manifest is not available offline. Load the model online first." });
	const manifest = await readManifest(manifestResponse, signal);
	const assetReader = createModelAssetReader({
		base,
		manifest,
		cache,
		signal,
		options
	});
	const loadState = {
		runtime: void 0,
		gpuDevice: void 0,
		lifecycle: void 0,
		loadWatch: void 0,
		scopeOpen: false
	};
	try {
		const modelAssets = await readModelAssets({
			assetReader,
			manifest,
			signal
		});
		const modelRuntime = await uploadModelAssets({
			manifest,
			modelAssets,
			assetReader,
			maxPrefetchBytes,
			signal,
			uploadHeads,
			loadState
		});
		loadState.loadWatch.signal.throwIfAborted();
		signal?.throwIfAborted();
		assertLoadActive(loadState.lifecycle);
		await saveLoadedManifest({
			cache,
			manifestUrl,
			manifest,
			assetReader,
			options,
			signal,
			loadState
		});
		loadState.loadWatch.close();
		return createDecisionModel({
			options,
			description,
			manifest,
			modelAssets,
			modelRuntime,
			loadState
		});
	} catch (cause) {
		cleanupFailedLoad(loadState);
		signal?.throwIfAborted();
		if (cause instanceof WebDecisionError || cause?.name === "AbortError") throw cause;
		throw new WebDecisionError("MODEL_LOAD_FAILED", "Could not load the decision model.", { cause });
	}
}
async function readModelAssets({ assetReader, manifest, signal }) {
	const { read } = assetReader;
	const [tokenizerBytes, tokenizerConfigBytes, modelConfigBytes, wasmBytes, tensorCacheBytes] = await readAssetBatch([
		"tokenizer.json",
		"tokenizer_config.json",
		"model_config.json",
		"model.wasm",
		"weights/tensor-cache.json"
	], read, signal);
	const modelConfig = parseJson(modelConfigBytes, "model_config.json");
	for (const [name, id] of Object.entries(KEV_DELIMITERS)) if (modelConfig.kev?.delimiter_ids?.[name] !== id) throw new WebDecisionError("INVALID_ASSET", "Unexpected model delimiter IDs.");
	const tokenizer = new Tokenizer(parseJson(tokenizerBytes, "tokenizer.json"), parseJson(tokenizerConfigBytes, "tokenizer_config.json"));
	const cacheManifest = parseJson(tensorCacheBytes, "tensor-cache.json");
	checkCacheManifest(manifest, cacheManifest);
	signal?.throwIfAborted();
	return {
		tokenizer,
		cacheManifest,
		wasmBytes
	};
}
async function uploadModelAssets({ manifest, modelAssets, assetReader, maxPrefetchBytes, signal, uploadHeads, loadState }) {
	const { read } = assetReader;
	const modelRuntime = await initializeModelRuntime(modelAssets.wasmBytes, manifest, signal, loadState);
	await uploadModelShards({
		records: modelAssets.cacheManifest.records,
		read,
		runtime: loadState.runtime,
		device: modelRuntime.device,
		weights: modelRuntime.weights,
		signal: loadState.loadWatch.signal,
		maxPrefetchBytes,
		uncaptured: modelRuntime.uncaptured
	});
	if (uploadHeads) await uploadHeads({
		read,
		runtime: loadState.runtime,
		device: modelRuntime.device,
		weights: modelRuntime.weights,
		signal: loadState.loadWatch.signal,
		uncaptured: modelRuntime.uncaptured
	});
	return modelRuntime;
}
async function initializeModelRuntime(wasmBytes, manifest, signal, loadState) {
	let gpu;
	try {
		gpu = await detectKevGPUDevice(void 0, signal);
	} catch (cause) {
		signal?.throwIfAborted();
		throw new WebDecisionError("MODEL_LOAD_FAILED", "Could not initialize a compatible WebGPU device.", { cause });
	}
	loadState.gpuDevice = gpu?.device;
	signal?.throwIfAborted();
	if (!gpu || !gpu.adapter.features.has("shader-f16")) {
		loadState.gpuDevice?.destroy();
		loadState.gpuDevice = void 0;
		throw unsupported("This model requires WebGPU shader-f16 support.");
	}
	const uncaptured = [];
	loadState.gpuDevice.addEventListener("uncapturederror", (event) => {
		if (uncaptured.length < 10) uncaptured.push(event.error?.message || String(event.error));
	});
	loadState.loadWatch = watchLoadDeviceLoss(loadState.gpuDevice, signal);
	loadState.runtime = await waitForAbort(() => instantiate(wasmBytes, createPolyfillWASI()), loadState.loadWatch.signal, (lateRuntime) => {
		try {
			lateRuntime?.dispose();
		} catch {}
	});
	loadState.lifecycle = trackRuntime(loadState.runtime, loadState.gpuDevice);
	loadState.loadWatch.signal.throwIfAborted();
	await Promise.resolve();
	assertLoadActive(loadState.lifecycle);
	loadState.runtime.initWebGPU(loadState.gpuDevice);
	const device = loadState.runtime.webgpu();
	loadState.runtime.beginScope();
	loadState.scopeOpen = true;
	return {
		device,
		fn: loadState.runtime.createVirtualMachine(device).getFunction("main"),
		weights: new Array(manifest.parameterCount),
		uncaptured
	};
}
async function saveLoadedManifest({ cache, manifestUrl, manifest, assetReader, options, signal, loadState }) {
	if (!cache || !assetReader.isCacheComplete() || options.offline) return;
	try {
		await commitCachedManifest(cache, manifestUrl, manifest, loadState.loadWatch.signal);
	} catch {
		loadState.loadWatch.signal.throwIfAborted();
	}
	signal?.throwIfAborted();
	assertLoadActive(loadState.lifecycle);
}
function createDecisionModel({ options, description, manifest, modelAssets, modelRuntime, loadState }) {
	const limits = {
		maxInputTokens: options?.maxInputTokens ?? manifest.maxInputTokens,
		maxStateTokens: manifest.maxStateTokens,
		maxBranchTokens: manifest.maxBranchTokens,
		maxQuestions: description.features.textChoice.limits.find((limit) => limit.resource === "questions" && limit.scope === "request")?.max,
		maxOptions: description.features.textChoice.limits.find((limit) => limit.resource === "options" && limit.scope === "question")?.max,
		maxInputCharacters: options?.maxInputCharacters
	};
	return new DecisionModel(loadState.runtime, modelRuntime.device, loadState.gpuDevice, modelRuntime.fn, modelRuntime.weights, modelAssets.tokenizer, limits, modelRuntime.uncaptured, loadState.lifecycle, true, description.model);
}
function loadKev06DecisionModel(options, description) {
	return loadKevModel(options, description, readKev06ManifestResponse);
}
// source/overlay/packages/web-decision/src/tamev/contract.mjs
var TAMEV_NANO_MODEL = "experimental-tamev-nano-tinybert-int8";
var TAMEV_NANO_REPOSITORY = "Tamkimd/tamev-nano-tinybert";
var TAMEV_NANO_REVISION = "ad15b45f35526070d48fb91f6597d2805a2d80af";
var TAMEV_NANO_TEMPERATURE = 1.5;
var TAMEV_NANO_FILES = Object.freeze({
	"model_int8.onnx": Object.freeze({
		bytes: 14872474,
		sha256: "a83a4604a803df9efed89ed2e419184c2a1de995174ae6e8b3682de985c47484"
	}),
	"tokenizer.json": Object.freeze({
		bytes: 711396,
		sha256: "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
	}),
	"tokenizer_config.json": Object.freeze({
		bytes: 379,
		sha256: "9f0c2c65a70ea18113ffa2e717103e7210ea826e7acc3442b69b346686b55a48"
	}),
	"config.json": Object.freeze({
		bytes: 796,
		sha256: "05fb476ed9eda56e1de767c33b067e70ed5b5c66d42437b1b8d3c3cdfe4c2307"
	})
});
var TAMEV_NANO_TOTAL_BYTES = Object.values(TAMEV_NANO_FILES).reduce((sum, file) => sum + file.bytes, 0);
var TAMEV_NANO_CALIBRATION_BY_K = Object.freeze({
	2: .8208,
	3: 1.094,
	4: .86,
	5: 1.2479,
	14: 1.7086,
	77: 1.5835
});
var TAMEV_NANO_PROFILE = Object.freeze({
	name: "TAMEV Nano TinyBERT",
	model: TAMEV_NANO_MODEL,
	repository: TAMEV_NANO_REPOSITORY,
	revision: TAMEV_NANO_REVISION,
	maxContextTokens: 128,
	maxOptionTokens: 48,
	temperature: TAMEV_NANO_TEMPERATURE,
	files: TAMEV_NANO_FILES,
	calibrationByK: TAMEV_NANO_CALIBRATION_BY_K,
	totalBytes: TAMEV_NANO_TOTAL_BYTES,
	expectedConfig: Object.freeze({
		model_type: "tamev",
		backbone: "huawei-noah/TinyBERT_General_4L_312D",
		architecture_type: "encoder",
		hidden_dim: 312,
		projection_dim: 64,
		temperature: TAMEV_NANO_TEMPERATURE,
		pooling: "cls",
		tier: "nano",
		max_state_len: 128,
		max_opt_len: 48,
		parameters_m: 14.390185,
		released_parameters: 14390185,
		released_artifact: "Fused model (backbone + cross-fusion pointer head)",
		option_fusion: "cross"
	})
});
var TAMEV_MICRO_MODEL = "experimental-tamev-micro-minilm-int8";
var TAMEV_MICRO_REPOSITORY = "Tamkimd/tamev-micro-minilm";
var TAMEV_MICRO_REVISION = "c1b95679390911748aad57f634bed70e5f87a084";
var TAMEV_MICRO_TEMPERATURE = 1.66;
var TAMEV_MICRO_FILES = Object.freeze({
	"model_int8.onnx": Object.freeze({
		bytes: 23447976,
		sha256: "7f3ef06cf6c7eb677d9232ddd52decdb617616b6561413c375bbc1ae972240f0"
	}),
	"tokenizer.json": Object.freeze({
		bytes: 711661,
		sha256: "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"
	}),
	"tokenizer_config.json": Object.freeze({
		bytes: 594,
		sha256: "872b6936be955bc3aea75ed599264d865626d68feede7e58b01e378e6332bd74"
	}),
	"config.json": Object.freeze({
		bytes: 772,
		sha256: "cbd3c2c982c02cff3801c81cc84978f93cf757845aa918a10969d1e7b0f884d4"
	})
});
var TAMEV_MICRO_TOTAL_BYTES = Object.values(TAMEV_MICRO_FILES).reduce((sum, file) => sum + file.bytes, 0);
var TAMEV_MICRO_CALIBRATION_BY_K = Object.freeze({
	2: .9241,
	3: 1.1722,
	4: .9025,
	5: 1.3209,
	14: 1.1383,
	77: 2.7375
});
var TAMEV_MICRO_PROFILE = Object.freeze({
	name: "TAMEV Micro MiniLM",
	model: TAMEV_MICRO_MODEL,
	repository: TAMEV_MICRO_REPOSITORY,
	revision: TAMEV_MICRO_REVISION,
	maxContextTokens: 128,
	maxOptionTokens: 48,
	temperature: TAMEV_MICRO_TEMPERATURE,
	files: TAMEV_MICRO_FILES,
	calibrationByK: TAMEV_MICRO_CALIBRATION_BY_K,
	totalBytes: TAMEV_MICRO_TOTAL_BYTES,
	expectedConfig: Object.freeze({
		model_type: "tamev",
		backbone: "sentence-transformers/all-MiniLM-L6-v2",
		architecture_type: "encoder",
		hidden_dim: 384,
		projection_dim: 64,
		temperature: TAMEV_MICRO_TEMPERATURE,
		pooling: "mean",
		tier: "micro",
		max_state_len: 128,
		max_opt_len: 48,
		parameters_m: 22.762369,
		released_parameters: 22762369,
		released_artifact: "Fused model (backbone + bilinear pointer head)"
	})
});
// source/overlay/packages/web-decision/src/tamev/runner.mjs
var inferenceFailure = (message, cause) => new WebDecisionError("INFERENCE_ERROR", message, cause ? { cause } : void 0);
function makeBatch(rows, padId) {
	let width = 0;
	for (const row of rows) if (row.length > width) width = row.length;
	const ids = new BigInt64Array(rows.length * width);
	const mask = new BigInt64Array(rows.length * width);
	for (let row = 0; row < rows.length; row++) for (let column = 0; column < width; column++) {
		const index = row * width + column;
		ids[index] = BigInt(rows[row][column] ?? padId);
		mask[index] = BigInt(column < rows[row].length ? 1 : 0);
	}
	return {
		ids,
		mask,
		width
	};
}
function isExpectedTensor(tensor, count) {
	return tensor?.type === "float32" && Array.isArray(tensor.dims) && tensor.dims.length === 2 && tensor.dims[0] === 1 && tensor.dims[1] === count && ArrayBuffer.isView(tensor.data) && tensor.data.length === count && Array.from(tensor.data).every(Number.isFinite);
}
function sum$1(values) {
	return values.reduce((total, value) => total + value, 0);
}
function createTamevRunner(ort, session, tokenizer) {
	if (typeof ort?.Tensor !== "function" || typeof session?.run !== "function" || typeof session?.release !== "function" || typeof tokenizer?.encodeContext !== "function" || !Number.isSafeInteger(tokenizer?.special?.padId)) throw new TypeError("TAMEV requires an ONNX session and tokenizer.");
	let closed = false;
	let disposing;
	return {
		async run(contextIds, optionRows) {
			if (closed) throw new WebDecisionError("SESSION_CLOSED", "The model has been disposed.");
			if (!Array.isArray(contextIds) || contextIds.length < 2 || !Array.isArray(optionRows) || optionRows.length < 2 || optionRows.some((row) => !Array.isArray(row) || row.length < 2)) throw inferenceFailure("TAMEV received invalid token rows.");
			const context = makeBatch([contextIds], tokenizer.special.padId);
			const options = makeBatch(optionRows, tokenizer.special.padId);
			let outputs;
			try {
				outputs = await session.run({
					ctx_input_ids: new ort.Tensor("int64", context.ids, [1, context.width]),
					ctx_attention_mask: new ort.Tensor("int64", context.mask, [1, context.width]),
					opt_input_ids: new ort.Tensor("int64", options.ids, [
						1,
						optionRows.length,
						options.width
					]),
					opt_attention_mask: new ort.Tensor("int64", options.mask, [
						1,
						optionRows.length,
						options.width
					])
				});
			} catch (cause) {
				throw inferenceFailure("TAMEV inference failed.", cause);
			}
			const logits = outputs?.logits;
			const probabilities = outputs?.probs;
			if (!isExpectedTensor(logits, optionRows.length) || !isExpectedTensor(probabilities, optionRows.length)) throw inferenceFailure("TAMEV returned unexpected tensors.");
			const values = Array.from(probabilities.data);
			const total = sum$1(values);
			if (values.some((value) => value < 0) || Math.abs(total - 1) > .001) throw inferenceFailure("TAMEV returned invalid probabilities.");
			return {
				logits: Array.from(logits.data),
				probabilities: values
			};
		},
		async releaseCachedMemory() {},
		dispose() {
			if (disposing) return disposing;
			closed = true;
			disposing = Promise.resolve().then(() => session.release()).catch((cause) => {
				throw inferenceFailure("Could not release the TAMEV ONNX session.", cause);
			});
			return disposing;
		}
	};
}
// source/overlay/packages/web-decision/src/tamev/protocol.mjs
var invalid$2 = (message) => new WebDecisionError("INVALID_INPUT", message);
function sum(values) {
	return values.reduce((total, value) => total + value, 0);
}
function firstMaximumIndex$1(values) {
	let index = 0;
	for (let candidate = 1; candidate < values.length; candidate++) if (values[candidate] > values[index]) index = candidate;
	return index;
}
function prepareTamevDecision(tokenizer, request, profile = TAMEV_NANO_PROFILE) {
	if (typeof request?.state !== "string" || !Array.isArray(request.questions)) throw invalid$2(`${profile.name} requires a string state and a question list.`);
	if (request.questions.length < 1) throw invalid$2(`${profile.name} needs at least one question.`);
	const questions = request.questions.map((question, index) => {
		if (typeof question.instructions !== "string" || !Array.isArray(question.options)) throw invalid$2(`${profile.name} question ${index} requires instructions and options.`);
		if (question.options.length < 2) throw invalid$2(`${profile.name} question ${index} needs at least 2 options.`);
		if (question.options.some((option) => typeof option !== "string" || !option.trim())) throw invalid$2(`${profile.name} question ${index} has an empty or non-text option.`);
		const contextText = question.instructions ? `${request.state}\n${question.instructions}`.trim() : request.state;
		const context = tokenizer.encodeContext(contextText);
		const options = question.options.map((option) => tokenizer.encodeOption(option));
		return {
			options: [...question.options],
			contextIds: context.ids,
			optionRows: options.map(({ ids }) => ids),
			tokenCount: context.ids.length + options.reduce((count, option) => count + option.ids.length, 0),
			truncated: context.truncated || options.some((option) => option.truncated)
		};
	});
	return {
		questions,
		tokenCount: questions.reduce((count, question) => count + question.tokenCount, 0),
		truncated: questions.some((question) => question.truncated)
	};
}
function calibrateTamevProbabilities(probabilities, optionCount, config) {
	const delta = config.calibration_by_k[String(optionCount)];
	if (delta === void 0 || delta === 1) return [...probabilities];
	const adjusted = probabilities.map((probability) => Math.max(probability, 1e-12) ** delta);
	const total = sum(adjusted);
	return adjusted.map((probability) => probability / total);
}
function finishTamevDecision(prepared, rows) {
	if (rows.length !== prepared.questions.length) throw new TypeError("TAMEV result rows are incomplete.");
	const answers = prepared.questions.map((question, questionIndex) => {
		const row = rows[questionIndex];
		const selectedIndex = firstMaximumIndex$1(row.probabilities);
		return {
			selected: question.options[selectedIndex],
			selectedIndex,
			probabilities: question.options.map((option, index) => ({
				option,
				probability: row.probabilities[index],
				logit: row.logits[index]
			}))
		};
	});
	return {
		tokenCount: prepared.tokenCount,
		...prepared.truncated ? { truncated: true } : {},
		answers
	};
}
// source/overlay/packages/web-decision/src/tamev/typed.mjs
var invalid$1 = (message) => new WebDecisionError("INVALID_INPUT", message);
var QUESTION_FIELDS = /* @__PURE__ */ new Set([
	"type",
	"instructions",
	"criteria"
]);
function stringValue(value) {
	if (typeof value === "boolean") return value ? "True" : "False";
	return String(value);
}
function renderTamevValue(value, indent = 0, stack = /* @__PURE__ */ new Set()) {
	const pad = "  ".repeat(indent);
	if (value == null) return "";
	if (typeof value === "string" || typeof value === "boolean") return stringValue(value);
	if (typeof value === "number" && Number.isFinite(value)) return String(value);
	if (typeof value !== "object") throw invalid$1("TAMEV values must be JSON text, numbers, booleans, lists or objects.");
	if (stack.has(value)) throw invalid$1("TAMEV values cannot contain cycles.");
	stack.add(value);
	try {
		if (Array.isArray(value)) return value.map((item, index) => {
			if (!Object.hasOwn(value, index)) throw invalid$1("TAMEV lists cannot contain holes.");
			return `${pad}- ${renderTamevValue(item, indent + 1, stack).trimStart()}`;
		}).join("\n");
		return objectEntries(value, "TAMEV objects").map(([key, item]) => {
			if (item && typeof item === "object") return `${pad}${key}:\n${renderTamevValue(item, indent + 1, stack)}`;
			return `${pad}${key}: ${renderTamevValue(item, indent, stack)}`;
		}).join("\n");
	} finally {
		stack.delete(value);
	}
}
function objectEntries(value, label) {
	if (!value || typeof value !== "object" || Array.isArray(value)) throw invalid$1(`${label} must be an object.`);
	return Reflect.ownKeys(value).map((key) => {
		const descriptor = Object.getOwnPropertyDescriptor(value, key);
		if (typeof key !== "string" || !descriptor || !Object.hasOwn(descriptor, "value")) throw invalid$1(`${label} must contain plain data properties.`);
		return [key, descriptor.value];
	});
}
function questionEntries(questions) {
	const entries = objectEntries(questions, "TAMEV questions");
	if (entries.length < 1) throw invalid$1("TAMEV needs at least one named question.");
	return entries.map(([id, value]) => {
		if (!value || typeof value !== "object" || Array.isArray(value)) throw invalid$1(`Question ${id} must be an object.`);
		const fields = Object.create(null);
		for (const [key, item] of objectEntries(value, `Question ${id}`)) {
			if (!QUESTION_FIELDS.has(key)) throw invalid$1(`Question ${id} has an unsupported field.`);
			fields[key] = item;
		}
		return [id, fields];
	});
}
function criteriaObject(criteria, id) {
	const entries = objectEntries(criteria, `Question ${id} criteria`);
	if (entries.length < 2) throw invalid$1(`Question ${id} needs at least 2 criteria.`);
	return entries;
}
function option(label, description) {
	return {
		label,
		rendered: description == null || description === "" ? label : `${label}: ${renderTamevValue(description)}`
	};
}
function truthy(value) {
	return value != null && value !== false && value !== 0 && value !== "";
}
function prepareQuestion(id, question) {
	if (![
		"choice",
		"noul",
		"score"
	].includes(question.type)) throw invalid$1(`Question ${id} type must be choice, noul or score.`);
	const instructions = question.instructions == null ? "" : renderTamevValue(question.instructions);
	if (question.type === "choice") {
		const options = criteriaObject(question.criteria, id).map(([label, description]) => option(label, description));
		if (options.some(({ label }) => !label.trim())) throw invalid$1(`Question ${id} has an empty criterion name.`);
		return {
			id,
			type: "choice",
			instructions,
			labels: options.map(({ label }) => label),
			options: options.map(({ rendered }) => rendered)
		};
	}
	if (question.type === "noul") {
		const criteria = question.criteria == null ? [] : objectEntries(question.criteria, `Question ${id} criteria`);
		const values = new Map(criteria);
		const falseValue = values.get("false") || values.get("no") || values.get(false);
		const trueValue = values.get("true") || values.get("yes") || values.get(true);
		const no = option("No / False", truthy(falseValue) ? falseValue : null);
		const yes = option("Yes / True", truthy(trueValue) ? trueValue : null);
		return {
			id,
			type: "noul",
			instructions,
			labels: ["false", "true"],
			options: [no.rendered, yes.rendered]
		};
	}
	if (!Array.isArray(question.criteria) || question.criteria.length < 2) throw invalid$1(`Question ${id} score criteria need at least 2 ordered levels.`);
	const criteria = question.criteria.map((value, index) => {
		if (!Object.hasOwn(question.criteria, index)) throw invalid$1(`Question ${id} score criteria cannot contain holes.`);
		return value;
	});
	return {
		id,
		type: "score",
		instructions,
		labels: criteria.map((_, index) => String(index)),
		legend: criteria,
		options: criteria.map((value) => renderTamevValue(value))
	};
}
function prepareTamevSystemOne(state, questions, tokenizer) {
	const stateText = typeof state === "string" ? state : renderTamevValue(state);
	return { questions: questionEntries(questions).map(([id, question]) => prepareQuestion(id, question)).map((question) => {
		const contextText = question.instructions ? `${stateText} [SEP] ${question.instructions}`.trim() : stateText.trim();
		const context = tokenizer.encodeContext(contextText);
		const options = question.options.map((text) => tokenizer.encodeOption(text));
		return {
			...question,
			contextIds: context.ids,
			optionRows: options.map(({ ids }) => ids),
			tokenCount: context.ids.length + options.reduce((count, optionTokens) => count + optionTokens.ids.length, 0),
			truncated: context.truncated || options.some((optionTokens) => optionTokens.truncated)
		};
	}) };
}
function rounded(value) {
	return Number(value.toFixed(4));
}
function firstMaximumIndex(values) {
	let index = 0;
	for (let candidate = 1; candidate < values.length; candidate++) if (values[candidate] > values[index]) index = candidate;
	return index;
}
function choiceConfidence(probabilities, maximumIndex) {
	const count = probabilities.length;
	if (count <= 1) return 1;
	return rounded((probabilities[maximumIndex] - 1 / count) / (1 - 1 / count));
}
function scoreConfidence(probabilities) {
	const mode = firstMaximumIndex(probabilities);
	return rounded(1 - probabilities.reduce((total, probability, index) => total + probability * Math.abs(index - mode), 0) / (probabilities.length - 1));
}
function finishAnswer(question, probabilities) {
	if (question.type === "noul") return {
		type: "noul",
		noul: rounded(probabilities[1])
	};
	const values = Object.fromEntries(question.labels.map((label, index) => [label, rounded(probabilities[index])]));
	if (question.type === "choice") {
		const choiceIndex = firstMaximumIndex(probabilities);
		return {
			type: "choice",
			choice: question.labels[choiceIndex],
			confidence: choiceConfidence(probabilities, choiceIndex),
			probabilities: values
		};
	}
	return {
		type: "score",
		score: rounded(probabilities.reduce((total, probability, index) => total + index * probability, 0)),
		confidence: scoreConfidence(probabilities),
		legend: Object.fromEntries(question.legend.map((label, index) => [String(index), label])),
		probabilities: values
	};
}
function finishTamevSystemOne(prepared, rows, model) {
	if (rows.length !== prepared.questions.length) throw new TypeError("TAMEV result rows are incomplete.");
	const answers = Object.fromEntries(prepared.questions.map((question, index) => [question.id, finishAnswer(question, rows[index].probabilities)]));
	return {
		model,
		tokenCount: prepared.questions.reduce((count, question) => count + question.tokenCount, 0),
		...prepared.questions.some((question) => question.truncated) ? { truncated: true } : {},
		answers
	};
}
// source/overlay/packages/web-decision/src/tamev/session.mjs
var invalid = (message) => new WebDecisionError("INVALID_INPUT", message);
var closedError = () => new WebDecisionError("SESSION_CLOSED", "The model has been disposed.");
function readSignal(options, profile) {
	if (options == null) return void 0;
	validateDecisionFields(options);
	const descriptor = Object.getOwnPropertyDescriptor(options, "signal");
	if (descriptor && !Object.hasOwn(descriptor, "value")) throw invalid(`${profile.name} does not accept signal accessors.`);
	const signal = descriptor?.value;
	if (signal != null && (typeof signal.throwIfAborted !== "function" || typeof signal.addEventListener !== "function" || typeof signal.removeEventListener !== "function")) throw invalid(`${profile.name} signal must be an AbortSignal.`);
	return signal;
}
function createTamevSession({ runner, tokenizer, config, profile = TAMEV_NANO_PROFILE, model = profile.model }) {
	if (typeof runner?.run !== "function" || typeof runner?.dispose !== "function" || typeof runner?.releaseCachedMemory !== "function" || typeof tokenizer?.encodeContext !== "function" || typeof tokenizer?.encodeOption !== "function" || !config?.calibration_by_k || model !== profile.model) throw new TypeError(`${profile.name} session components are required.`);
	let queue = Promise.resolve();
	let closed = false;
	let disposing;
	const requireOpen = () => {
		if (closed) throw closedError();
	};
	function enqueue(task) {
		const current = queue.then(task);
		queue = current.catch(() => {});
		return current;
	}
	async function infer(prepared, signal, applyOptionCalibration) {
		requireOpen();
		const schedule = () => enqueue(async () => {
			requireOpen();
			const rows = [];
			for (const question of prepared.questions) {
				signal?.throwIfAborted();
				const row = await runner.run(question.contextIds, question.optionRows);
				rows.push({
					...row,
					probabilities: applyOptionCalibration ? calibrateTamevProbabilities(row.probabilities, question.optionRows.length, config) : row.probabilities
				});
			}
			return rows;
		});
		return signal ? waitForAbort(schedule, signal) : schedule();
	}
	return {
		async decide(request, options = {}) {
			requireOpen();
			const signal = readSignal(options, profile);
			signal?.throwIfAborted();
			const checked = validateRequestFields(request, Infinity, Infinity, Infinity, Infinity, Infinity, {
				dataPropertiesOnly: true,
				allowInheritedDataProperties: true
			});
			const prepared = prepareTamevDecision(tokenizer, checked, profile);
			return finishTamevDecision(prepared, await infer(prepared, signal, true));
		},
		async systemOne(state, questions, options = {}) {
			requireOpen();
			const signal = readSignal(options, profile);
			signal?.throwIfAborted();
			const prepared = prepareTamevSystemOne(state, questions, tokenizer, profile);
			return finishTamevSystemOne(prepared, await infer(prepared, signal, false), model);
		},
		async releaseCachedMemory() {
			requireOpen();
			return enqueue(() => runner.releaseCachedMemory());
		},
		dispose() {
			if (disposing) return disposing;
			closed = true;
			disposing = queue.then(async () => {
				try {
					await runner.dispose();
				} finally {
					runner = tokenizer = config = null;
				}
			});
			return disposing;
		}
	};
}
// source/overlay/packages/web-decision/src/tamev/source.mjs
async function openTamevSource({ base, signal, cacheEnabled, offline, onProgress }, profile = TAMEV_NANO_PROFILE) {
	const cache = await openAssetCache(cacheEnabled, signal);
	let loadedBytes = 0;
	return { async read(name) {
		return await readVerifiedAsset(base, { files: profile.files }, name, {
			cache,
			signal,
			offline,
			onBytes(bytes) {
				loadedBytes += bytes;
				onProgress?.({
					loadedBytes,
					totalBytes: profile.totalBytes,
					fraction: loadedBytes / profile.totalBytes
				});
			}
		});
	} };
}
// source/overlay/packages/web-decision/src/tamev/tokenizer.mjs
var invalidAsset = () => new WebDecisionError("INVALID_ASSET", "TAMEV tokenizer does not match the pinned model.");
function readIds(tokenizer, text) {
	let ids;
	try {
		ids = tokenizer.encode(text, { add_special_tokens: true })?.ids;
	} catch (cause) {
		throw new WebDecisionError("INVALID_ASSET", "TAMEV tokenizer failed.", { cause });
	}
	if (!Array.isArray(ids) || ids.length < 2 || ids.some((id) => !Number.isSafeInteger(id) || id < 0)) throw invalidAsset();
	return ids;
}
function createTamevTokenizer(Tokenizer, data, tokenizerConfig, profile = TAMEV_NANO_PROFILE) {
	if (typeof Tokenizer !== "function" || data?.model?.type !== "WordPiece" || tokenizerConfig?.tokenizer_class !== "BertTokenizer" || tokenizerConfig?.cls_token !== "[CLS]" || tokenizerConfig?.sep_token !== "[SEP]" || tokenizerConfig?.pad_token !== "[PAD]" || tokenizerConfig?.max_length !== void 0 && tokenizerConfig.max_length !== profile.maxContextTokens || tokenizerConfig?.padding_side !== void 0 && tokenizerConfig.padding_side !== "right" || tokenizerConfig?.truncation_side !== void 0 && tokenizerConfig.truncation_side !== "right") throw invalidAsset();
	const tokenizer = new Tokenizer(data, tokenizerConfig);
	const clsId = tokenizer.token_to_id("[CLS]");
	const sepId = tokenizer.token_to_id("[SEP]");
	const padId = tokenizer.token_to_id("[PAD]");
	if (clsId !== 101 || sepId !== 102 || padId !== 0) throw invalidAsset();
	const probe = readIds(tokenizer, "Hello world");
	if (probe[0] !== clsId || probe.at(-1) !== sepId) throw invalidAsset();
	function encode(text, maximum) {
		const ids = readIds(tokenizer, text);
		if (ids[0] !== clsId || ids.at(-1) !== sepId) throw invalidAsset();
		if (ids.length <= maximum) return {
			ids,
			truncated: false
		};
		return {
			ids: [...ids.slice(0, maximum - 1), sepId],
			truncated: true
		};
	}
	return Object.freeze({
		special: Object.freeze({
			clsId,
			sepId,
			padId
		}),
		encodeContext(text) {
			return encode(text, profile.maxContextTokens);
		},
		encodeOption(text) {
			return encode(text, profile.maxOptionTokens);
		}
	});
}
// source/overlay/packages/web-decision/src/tamev/loader.mjs
var LOAD_KEYS = /* @__PURE__ */ new Set([
	"model",
	"assetsUrl",
	"cache",
	"offline",
	"signal",
	"onProgress",
	"maxPrefetchBytes",
	"debertaReadback"
]);
var fail = (code, message, cause) => new WebDecisionError(code, message, cause ? { cause } : void 0);
function validateTamevLoadOptions(options, profile = TAMEV_NANO_PROFILE) {
	if (!options || typeof options !== "object" || Array.isArray(options)) throw fail("INVALID_INPUT", `${profile.name} load options are required.`);
	for (const key of Reflect.ownKeys(options)) {
		const descriptor = Object.getOwnPropertyDescriptor(options, key);
		if (typeof key !== "string" || !LOAD_KEYS.has(key) || !descriptor || !Object.hasOwn(descriptor, "value")) throw fail("INVALID_INPUT", `Unsupported ${profile.name} load option.`);
	}
	const values = Object.fromEntries(Reflect.ownKeys(options).map((key) => [key, options[key]]));
	if (values.model !== void 0 && values.model !== profile.model || typeof values.assetsUrl !== "string" || !values.assetsUrl || values.cache !== void 0 && typeof values.cache !== "boolean" || values.offline !== void 0 && typeof values.offline !== "boolean" || values.offline && values.cache === false || values.maxPrefetchBytes !== void 0 && values.maxPrefetchBytes !== 0 || values.debertaReadback !== void 0 || values.onProgress !== void 0 && typeof values.onProgress !== "function" || values.signal != null && (typeof values.signal.throwIfAborted !== "function" || typeof values.signal.addEventListener !== "function" || typeof values.signal.removeEventListener !== "function")) throw fail("INVALID_INPUT", `Invalid ${profile.name} load option values.`);
	let base;
	try {
		base = new URL(values.assetsUrl, globalThis.location?.href);
	} catch (cause) {
		throw fail("INVALID_INPUT", `Invalid ${profile.name} assetsUrl.`, cause);
	}
	if (!["https:", "http:"].includes(base.protocol) || base.username || base.password || base.search || base.hash) throw fail("INVALID_INPUT", `${profile.name} assetsUrl must be an HTTP or HTTPS directory.`);
	if (!base.pathname.endsWith("/")) base.pathname += "/";
	return {
		base,
		signal: values.signal,
		cacheEnabled: values.cache !== false,
		offline: values.offline === true,
		onProgress: values.onProgress
	};
}
function validateConfig(config, profile) {
	const calibration = config?.calibration_by_k;
	if (Object.entries(profile.expectedConfig).some(([key, value]) => config?.[key] !== value) || !calibration || Object.keys(profile.calibrationByK).some((key) => calibration[key] !== profile.calibrationByK[key]) || Object.keys(calibration).some((key) => !Object.hasOwn(profile.calibrationByK, key))) throw fail("INVALID_ASSET", `${profile.name} configuration does not match the pinned model.`);
	return config;
}
async function createSession(ort, bytes, signal) {
	if (typeof ort?.InferenceSession?.create !== "function") throw fail("UNSUPPORTED_ENVIRONMENT", "ONNX Runtime Web is unavailable.");
	return waitForAbort(() => ort.InferenceSession.create(new Uint8Array(bytes), {
		executionProviders: ["wasm"],
		intraOpNumThreads: 1,
		interOpNumThreads: 1,
		graphOptimizationLevel: "all"
	}), signal, (lateSession) => lateSession?.release());
}
function validateSession(session, profile) {
	const inputs = new Set(session?.inputNames ?? []);
	const outputs = new Set(session?.outputNames ?? []);
	if (![
		"ctx_input_ids",
		"ctx_attention_mask",
		"opt_input_ids",
		"opt_attention_mask"
	].every((name) => inputs.has(name)) || !["logits", "probs"].every((name) => outputs.has(name))) throw fail("INVALID_ASSET", `${profile.name} ONNX graph has an unexpected interface.`);
	return session;
}
async function loadTamevProfile(options, profile, { ort = defaultOrt, Tokenizer: TokenizerClass = Tokenizer, openSource = openTamevSource } = {}) {
	const checked = validateTamevLoadOptions(options, profile);
	const { signal } = checked;
	let session;
	try {
		const source = await openSource(checked, profile);
		const [tokenizerBytes, tokenizerConfigBytes, configBytes] = await Promise.all([
			source.read("tokenizer.json"),
			source.read("tokenizer_config.json"),
			source.read("config.json")
		]);
		const tokenizer = createTamevTokenizer(TokenizerClass, parseJson(tokenizerBytes, "tokenizer.json"), parseJson(tokenizerConfigBytes, "tokenizer_config.json"), profile);
		const config = validateConfig(parseJson(configBytes, "config.json"), profile);
		signal?.throwIfAborted();
		session = await createSession(ort, await source.read("model_int8.onnx"), signal);
		validateSession(session, profile);
		signal?.throwIfAborted();
		const model = createTamevSession({
			runner: createTamevRunner(ort, session, tokenizer),
			tokenizer,
			config,
			profile,
			model: profile.model
		});
		session = null;
		return model;
	} catch (cause) {
		await Promise.allSettled([session?.release()].filter(Boolean));
		signal?.throwIfAborted();
		if (cause instanceof WebDecisionError || cause?.name === "AbortError") throw cause;
		throw fail("MODEL_LOAD_FAILED", `Could not load ${profile.model}.`, cause);
	}
}
function loadTamevNano(options, dependencies) {
	return loadTamevProfile(options, TAMEV_NANO_PROFILE, dependencies);
}
function loadTamevMicro(options, dependencies) {
	return loadTamevProfile(options, TAMEV_MICRO_PROFILE, dependencies);
}
function loadTamev(options, dependencies) {
	if (options?.model === "experimental-tamev-nano-tinybert-int8") return loadTamevNano(options, dependencies);
	if (options?.model === "experimental-tamev-micro-minilm-int8") return loadTamevMicro(options, dependencies);
	throw fail("INVALID_INPUT", "Unsupported TAMEV model.");
}
var kev_preview_model_default = {
	model: "kev-0.6b-q4f16",
	precision: "q4f16",
	checkpoint: {
		"repository": "onnx-community/kev-0.6b-ONNX",
		"revision": "5a827091b3d58fe4fdfc02a5a5b2ce364047d678"
	},
	assets: {
		"integrity": "pinned-manifest",
		"manifestSha256": "bc79c92a7011d2aa1aff4ede66c06cc365de81e744507256afb5c0449870716a",
		"location": "separate",
		"requiredFiles": [
			"manifest.json",
			"model.wasm",
			"tokenizer.json",
			"tokenizer_config.json",
			"model_config.json",
			"weights/tensor-cache.json"
		],
		"additionalFiles": "Weight shards or ranges listed by the model manifest are also required."
	},
	licenses: {
		"project": "Apache-2.0",
		"modelTerms": "separate-review-required",
		"modelCard": "https://huggingface.co/onnx-community/kev-0.6b-ONNX/tree/5a827091b3d58fe4fdfc02a5a5b2ce364047d678"
	},
	backend: {
		"name": "tvm-webgpu",
		"requiredFeatures": ["shader-f16"],
		"requiredLimits": {
			"maxBufferSize": 268435456,
			"maxStorageBufferBindingSize": 134217728,
			"maxComputeWorkgroupStorageSize": 16384,
			"maxStorageBuffersPerShaderStage": 8,
			"maxComputeInvocationsPerWorkgroup": 256,
			"maxComputeWorkgroupSizeX": 256,
			"maxComputeWorkgroupSizeY": 8,
			"maxComputeWorkgroupSizeZ": 1,
			"maxComputeWorkgroupsPerDimension": 65535
		},
		"requiresNonFallbackAdapter": false
	},
	features: {
		"textChoice": {
			"status": "implemented",
			"abstention": "not-implemented",
			"limits": [
				{
					"resource": "tokens",
					"scope": "state",
					"max": 8191,
					"description": "Native Kev state limit; the model counts its state marker within the 8,192-token budget."
				},
				{
					"resource": "tokens",
					"scope": "question",
					"max": 8192,
					"description": "Native Kev per-question row limit, including the shared state and question branch. There is no fixed aggregate request-token cap; available memory limits the packed request."
				},
				{
					"resource": "options",
					"scope": "question",
					"min": 1,
					"max": 255,
					"description": "Upstream Kev config declares a maximum of 255 options per question."
				},
				{
					"resource": "questions",
					"scope": "request",
					"min": 1
				}
			],
			"handling": "packed-request",
			"tokenCount": "packed-request",
			"score": {
				"basis": "choices",
				"temperature": 1,
				"temperatureAppliedBy": "decoder",
				"calibration": "not-guaranteed"
			}
		},
		"typedTextQuestionTypes": [
			"choice",
			"noul",
			"score"
		],
		"imageChoice": { "status": "not-implemented" },
		"rating": { "status": "not-implemented" }
	},
	verification: {
		"currentBundleGpu": "verified",
		"currentBundleCpuWasm": "not-applicable",
		"currentBundleJavascript": "not-applicable",
		"priorEvidence": "The exact 0.3.0 release archive passed real-WebGPU tests in Chromium, Firefox, and WebKit on one Apple M4 Pro. This does not establish compatibility with other devices or decision quality."
	}
};
var tamev_nano_model_default = {
	model: "experimental-tamev-nano-tinybert-int8",
	precision: "int8",
	checkpoint: {
		"repository": "Tamkimd/tamev-nano-tinybert",
		"revision": "ad15b45f35526070d48fb91f6597d2805a2d80af"
	},
	assets: {
		"integrity": "pinned-per-file-sha256",
		"location": "separate",
		"requiredFiles": [
			"model_int8.onnx",
			"tokenizer.json",
			"tokenizer_config.json",
			"config.json"
		],
		"totalBytes": 15585045
	},
	licenses: {
		"project": "Apache-2.0",
		"modelTerms": "Apache-2.0-declared-by-pinned-model-card",
		"modelCard": "https://huggingface.co/Tamkimd/tamev-nano-tinybert/tree/ad15b45f35526070d48fb91f6597d2805a2d80af"
	},
	backend: {
		"name": "onnxruntime-web-wasm",
		"executionProvider": "wasm",
		"intraOpNumThreads": 1
	},
	features: {
		"textChoice": {
			"status": "implemented",
			"abstention": "not-implemented",
			"limits": [
				{
					"resource": "tokens",
					"scope": "state",
					"max": 128,
					"description": "Context row including tokenizer special tokens; excess input is right-truncated and reported."
				},
				{
					"resource": "tokens",
					"scope": "question",
					"max": 48,
					"description": "Each option row including tokenizer special tokens; excess input is right-truncated and reported."
				},
				{
					"resource": "options",
					"scope": "question",
					"min": 2,
					"description": "Dynamic option axis; device resources constrain practical request size."
				},
				{
					"resource": "questions",
					"scope": "request",
					"min": 1,
					"description": "Questions run sequentially; no adapter-only question-count ceiling."
				}
			],
			"handling": "sequential-questions",
			"tokenCount": "sum-of-encoded-rows",
			"score": {
				"basis": "candidate-probabilities",
				"temperature": 1.5,
				"temperatureAppliedBy": "model-export",
				"calibration": "per-option-count-on-generic-decision"
			}
		},
		"typedTextQuestionTypes": [
			"choice",
			"noul",
			"score"
		],
		"imageChoice": { "status": "not-implemented" },
		"rating": { "status": "not-implemented" }
	},
	verification: {
		"currentBundleGpu": "not-applicable",
		"currentBundleCpuWasm": "verified",
		"currentBundleJavascript": "not-applicable",
		"priorEvidence": "The exact 0.3.0 release archive passed real-model CPU/WASM browser checks in Chromium, Firefox, and WebKit on one Apple M4 Pro. This does not establish compatibility with other devices or decision quality."
	}
};
var tamev_micro_model_default = {
	model: "experimental-tamev-micro-minilm-int8",
	precision: "int8",
	checkpoint: {
		"repository": "Tamkimd/tamev-micro-minilm",
		"revision": "c1b95679390911748aad57f634bed70e5f87a084"
	},
	assets: {
		"integrity": "pinned-per-file-sha256",
		"location": "separate",
		"requiredFiles": [
			"model_int8.onnx",
			"tokenizer.json",
			"tokenizer_config.json",
			"config.json"
		],
		"totalBytes": 24161003
	},
	licenses: {
		"project": "Apache-2.0",
		"modelTerms": "Apache-2.0-declared-by-pinned-model-card",
		"modelCard": "https://huggingface.co/Tamkimd/tamev-micro-minilm/tree/c1b95679390911748aad57f634bed70e5f87a084"
	},
	backend: {
		"name": "onnxruntime-web-wasm",
		"executionProvider": "wasm",
		"intraOpNumThreads": 1
	},
	features: {
		"textChoice": {
			"status": "implemented",
			"abstention": "not-implemented",
			"limits": [
				{
					"resource": "tokens",
					"scope": "state",
					"max": 128,
					"description": "Context row including tokenizer special tokens; excess input is right-truncated and reported."
				},
				{
					"resource": "tokens",
					"scope": "question",
					"max": 48,
					"description": "Each option row including tokenizer special tokens; excess input is right-truncated and reported."
				},
				{
					"resource": "options",
					"scope": "question",
					"min": 2,
					"description": "Dynamic option axis; device resources constrain practical request size."
				},
				{
					"resource": "questions",
					"scope": "request",
					"min": 1,
					"description": "Questions run sequentially; no adapter-only question-count ceiling."
				}
			],
			"handling": "sequential-questions",
			"tokenCount": "sum-of-encoded-rows",
			"score": {
				"basis": "candidate-probabilities",
				"temperature": 1.66,
				"temperatureAppliedBy": "model-export",
				"calibration": "per-option-count-on-generic-decision"
			}
		},
		"typedTextQuestionTypes": [
			"choice",
			"noul",
			"score"
		],
		"imageChoice": { "status": "not-implemented" },
		"rating": { "status": "not-implemented" }
	},
	verification: {
		"currentBundleGpu": "not-applicable",
		"currentBundleCpuWasm": "verified",
		"currentBundleJavascript": "not-applicable",
		"priorEvidence": "The exact 0.3.0 release archive passed real-model CPU/WASM checks in Chromium, Firefox, and WebKit on one Apple M4 Pro. This does not establish compatibility with other devices or decision quality."
	}
};
// source/overlay/packages/web-decision/src/kev-preview-registry.js
var MODELS = Object.freeze([
	kev_preview_model_default.model,
	tamev_nano_model_default.model,
	tamev_micro_model_default.model
]);
function deepFreeze(value) {
	if (value && typeof value === "object" && !Object.isFrozen(value)) {
		for (const child of Object.values(value)) deepFreeze(child);
		Object.freeze(value);
	}
	return value;
}
var descriptions = /* @__PURE__ */ new Map([
	[kev_preview_model_default.model, deepFreeze(kev_preview_model_default)],
	[tamev_nano_model_default.model, deepFreeze(tamev_nano_model_default)],
	[tamev_micro_model_default.model, deepFreeze(tamev_micro_model_default)]
]);
function listDecisionModels() {
	return MODELS;
}
function isDecisionModel(value) {
	return descriptions.has(value);
}
function describeModel(model) {
	if (!isDecisionModel(model)) throw new WebDecisionError("INVALID_INPUT", "Unsupported decision model.");
	return descriptions.get(model);
}
// source/overlay/packages/web-decision/src/kev-preview-entry.js
/** Load a model in this release with its separately hosted pinned asset pack. */
async function loadDecisionModel(options) {
	const modelName = options?.model;
	if (modelName === "experimental-tamev-nano-tinybert-int8" || modelName === "experimental-tamev-micro-minilm-int8") {
		const description = describeModel(modelName);
		const model = await loadTamev(options);
		Object.defineProperty(model, "description", { value: description });
		return model;
	}
	if (modelName !== void 0 && modelName !== "kev-0.6b-q4f16") throw new WebDecisionError("INVALID_INPUT", "Unsupported decision model.");
	const description = describeModel(modelName ?? "kev-0.6b-q4f16");
	if (options?.debertaReadback !== void 0) throw new WebDecisionError("INVALID_INPUT", "debertaReadback is only supported by the DeBERTa model.");
	const model = await loadKev06DecisionModel(options, description);
	Object.defineProperty(model, "description", { value: description });
	return model;
}
/** Delete only model assets cached by this library for the current origin. */
async function clearModelCache() {
	if (!globalThis.caches) return false;
	return caches.delete(CACHE_NAME);
}
export { WebDecisionError, clearModelCache, describeModel, isDecisionModel, listDecisionModels, loadDecisionModel };
