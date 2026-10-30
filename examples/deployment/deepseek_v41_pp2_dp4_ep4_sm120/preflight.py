# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 preflight for DeepSeek V4.1 with PP=2 + DP=4 + EP=4 + DSpark.

Checks on the target host, before a multi-minute startup, the things the SM120
path depends on that the code cannot verify by itself until a layer is built:

- every GPU is SM120;
- the checkpoint's KV/index/candidate sources keep a 20/20 split local, and
  the DSpark target layers are captured on stage 1;
- FlashInfer ships the SM120 DSV4.1 sparse-MLA decode specialization for the
  target (top-k = SWA window) and the non-causal DSpark drafter
  (top-k = window + drafts, padded), else layer construction raises;
- DeepGEMM is usable: on SM120 it serves the indexer MQA logits (no other
  backend), the block-FP8 linears, the mHC pre-norm GEMM and, by default, the
  MXFP4 routed experts (DEEPGEMM_MXFP4).

This only checks that symbols and dispatch entries exist; run the SM120
kernel tests in README.md to check that the kernels compute the right values.

    python preflight.py /models/DeepSeek-V4.1-Flash
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PP_PARTITION = [(0, 20), (20, 40)]
NUM_SPECULATIVE_TOKENS = 5
failures: list[str] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def load_text_config(model: str) -> SimpleNamespace:
    raw = json.loads((Path(model) / "config.json").read_text())
    merged = {**raw, **raw.get("text_config", {})}
    return SimpleNamespace(**merged)


def check_gpus(num_gpus: int) -> None:
    import torch

    count = torch.cuda.device_count()
    caps = [torch.cuda.get_device_capability(i) for i in range(count)]
    report(
        f"{num_gpus} SM120 GPUs",
        count >= num_gpus and all(major == 12 for major, _ in caps),
        f"found {count}: {sorted(set(caps))}",
    )


def check_layout(config: SimpleNamespace) -> None:
    from vllm.models.deepseek_v41.common.pipeline import (
        get_aux_hidden_state_stages,
        get_sharing_dependencies,
        validate_local_sharing,
    )

    try:
        validate_local_sharing(get_sharing_dependencies(config, PP_PARTITION))
        report("20/20 split keeps KV/index/candidate sources local", True)
    except (ValueError, NotImplementedError, AttributeError) as e:
        report("20/20 split keeps KV/index/candidate sources local", False, str(e))

    targets = tuple(getattr(config, "dspark_target_layer_ids", ()) or ())
    if not targets:
        report("DSpark target layers", False, "dspark_target_layer_ids missing")
        return
    stages = get_aux_hidden_state_stages(targets, PP_PARTITION)
    report(
        "DSpark target layers captured on stage 1",
        all(stage == 1 for stage in stages.values()),
        f"{stages} (others are relayed over PP)",
    )


def check_flashinfer(config: SimpleNamespace) -> None:
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
        _pad_to_supported_q_heads,
    )
    from vllm.utils.flashinfer import (
        has_flashinfer_sparse_mla_sm120,
        has_flashinfer_sparse_mla_sm120_config,
    )
    from vllm.v1.attention.backends.mla.compressor_utils import (
        get_dspark_swa_index_width,
    )

    report("FlashInfer SM120 sparse MLA decode API", has_flashinfer_sparse_mla_sm120())
    heads = _pad_to_supported_q_heads(config.num_attention_heads)  # TP=1
    window = config.sliding_window
    shapes = {"target": window}
    if dflash_has_any_non_causal(config):
        shapes["DSpark drafter (non-causal)"] = get_dspark_swa_index_width(
            window, NUM_SPECULATIVE_TOKENS
        )
    else:
        shapes["DSpark drafter (causal)"] = window
    for role, topk in shapes.items():
        report(
            f"FlashInfer DSV4.1 SM120 dispatch for {role}",
            has_flashinfer_sparse_mla_sm120_config(heads, topk, dsv41=True),
            f"(num_q_heads={heads}, top_k={topk})",
        )


def check_deep_gemm() -> None:
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    supported = is_deep_gemm_supported()
    report(
        "DeepGEMM usable (indexer logits, FP8 linears, MXFP4 experts)",
        supported,
        "" if supported else "install DeepGEMM >= 2.8.1 with sm_120 kernels",
    )
    if not supported:
        return
    import deep_gemm

    # Names as vllm/utils/deep_gemm.py resolves them.
    for symbol, user in (
        ("fp8_fp4_mqa_logits", "indexer prefill logits"),
        ("fp8_fp4_paged_mqa_logits", "indexer decode logits"),
        ("get_paged_mqa_logits_metadata", "indexer decode metadata"),
        ("fp8_gemm_nt", "block-FP8 linears"),
        ("m_grouped_fp8_fp4_gemm_nt_contiguous", "MXFP4 routed experts"),
        ("tf32_hc_prenorm_gemm", "mHC pre-norm GEMM"),
    ):
        report(f"deep_gemm.{symbol} ({user})", hasattr(deep_gemm, symbol))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="local DeepSeek V4.1 checkpoint directory")
    parser.add_argument("--num-gpus", type=int, default=8)
    args = parser.parse_args()

    config = load_text_config(args.model)
    check_gpus(args.num_gpus)
    check_layout(config)
    check_flashinfer(config)
    check_deep_gemm()

    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        return 1
    print("\nAll checks passed. Next, run the SM120 kernel tests listed in README.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
