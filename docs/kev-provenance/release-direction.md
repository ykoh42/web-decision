# Kev release direction

Updated 2026-11-13 from the project owner's instruction.

## License basis and public disclosure

Proceed toward a TVM release based on the pinned Kev model card's Apache-2.0 declaration for its adapter and pointer head, and the pinned Qwen base model's Apache-2.0 license. Include both license texts, attribution, source revisions, and a clear note describing the TVM conversion.

Publish the pinned training-source list with the model documentation. In particular, identify `Yelp/yelp_review_full@c1f9ee939b7d05667af864ee1cb066393154bf85` as a training input and summarize its linked agreement: non-commercial and non-transferable use terms, restrictions on data distribution, and Yelp's claim over data derivatives. The agreement does not resolve whether trained weights are covered. State that uncertainty plainly and do not redistribute the Yelp dataset itself.

This records the project's chosen publication basis. It does not claim that dataset terms have been independently cleared or that Yelp's agreement does not apply to weights. The full pinned source and observed terms remain in [`kev-v7-training-data-snapshot.md`](kev-v7-training-data-snapshot.md).

## Kev 1.0 0.8B fidelity requirement

The target is the published Kev 1.0 0.8B checkpoint: model-card revision [`bf75a6a`](https://huggingface.co/jaredpalmer/kev-0.8b/blob/bf75a6a8848ea6960ff2ed108d9ed44c2941174f/README.md), weights revision `9a45d25e`, and `Qwen/Qwen3.5-0.8B-Base` revision [`dc7cdfe`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68). The weights' provenance records code commit `45923b7a`; that is the training/provenance pin. The publisher's current serving reference at this review is [`jaredpalmer/kev@2074cc4`](https://github.com/jaredpalmer/kev/tree/2074cc44256200bb37540103fa4572901faa10f3), including its [`kev/model.py`](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/model.py) and [`kev/serve.py`](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/serve.py). Keep the training pin and serving-source pin distinct. The publisher describes BF16 serving and FP32 evaluation. BF16 serving is the execution target; the FP32 evaluator is a comparison reference, not permission to choose a different serving precision.

Follow the publisher's model assembly, tokenizer, input encoding, scoring, calibration, and serving limits. The current serving source admits a state of up to 65,536 tokens and a question row containing that state plus up to 8,192 more tokens; it rejects longer inputs by default. Its optional truncation mode is off by default and must not be enabled by this library. The card's separate validated-context figure of 8,192 describes the evaluated quality range; it is not the serving input limit. Preserve both claims as stated. The calibration temperature is `T=2.35`. TVM may lower the execution graph, but the project must not invent a merge procedure, quantize or otherwise change learned weights, shorten these limits, or alter decision semantics. A conversion must match the official inference path before it can be offered by model ID. Then it must pass browser/GPU, package-integrity, and offline-cache gates.

The existing `kev08` route is experimental, accepts a caller-supplied asset URL, and caps inputs at 128 tokens. It is not a release candidate for this target. The direct-TVM Q4 work documented for Kev 0.6B is a separate model and is not evidence for Kev 1.0 0.8B.

The first reduced, random-weight TVM probe did not compile to LLVM or WebGPU; details and exact scope are in [`kev08-tvm-feasibility.md`](kev08-tvm-feasibility.md). No Kev 1.0 weights or runtime artifact have been converted.
