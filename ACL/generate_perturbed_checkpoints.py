#!/usr/bin/env python3
"""
generate_perturbed_checkpoints.py

PURE GENERATOR -- produces perturbed checkpoints for later, separate evaluation.
Does NOT run ASR or MMLU/TruthfulQA eval itself (unlike run_noise_violation_sweep.py),
so it doesn't touch main.py / evaluate_benchmark.py at all and is unaffected by
whatever is still causing main.py --eval_only's memory blowup.

For a single (already injection+removal-trained) checkpoint and a single target
quantization format, computes the per-weight dequantization box once (via the
repo's own `quant_specific.pgd.compute_box`, same code path as PGD training and
already validated as memory-safe in run_noise_violation_sweep.py), then for every
noise std in the sweep produces TWO saved checkpoints:

  full_noise_<model_name_key>_<p_type>_<quant>_std<std>/
      Gaussian noise N(0, std^2) added to EVERY weight in the quantization
      target layers.

  qbound_noise_<model_name_key>_<p_type>_<quant>_std<std>/
      Gaussian noise N(0, std^2) added ONLY to the subset of those weights
      that sit within --qbound_margin_frac of their own per-weight box width
      from either edge (w_min, w_max) -- i.e. weights already close to the
      point where they'd dequantize to a different value. Every other weight
      is left byte-for-byte untouched. See
      `quant_specific.noise_violation.add_targeted_gaussian_noise` for the
      exact selection rule (including how zero-width boxes are handled).

A manifest CSV (generation_manifest.csv) records, per checkpoint: noise_mode,
std, violation_rate (over the noised subset), and the qbound selection
statistics (how many weights were target-eligible, how many had a
degenerate zero-width box, how many were selected) -- everything you need to
interpret and pair up checkpoints for your own downstream eval.

Reuses the sweep script's pristine-snapshot/restore pattern (one CPU state_dict
snapshot taken once, restored via in-place .copy_() before every noise draw)
to avoid re-deepcopy-ing the whole model per checkpoint.

Example:
    cd ACL
    python generate_perturbed_checkpoints.py \\
        --model_name_key qwen2.5-3b-instruct \\
        --model_name_or_path poisoned_models/qwen2.5-3b-instruct-ad_inject/removal/checkpoint-last \\
        --p_type ad_inject \\
        --quantize_method nf4 \\
        --noise_stds 0.0,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0 \\
        --qbound_margin_frac 0.1 \\
        --output_dir noised_checkpoints/qwen2.5-3b-instruct-ad_inject
"""

import argparse
import gc
import json
import os
import sys
import time

# Same rationale as run_noise_violation_sweep.py: force unbuffered output so a
# SIGKILL (OOM, wall-time limit) doesn't discard already-printed diagnostics
# that just hadn't been flushed to the SLURM .out log yet.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       # .../ACL
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)                       # parent of ACL/, where q_attack/ lives
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant_specific.noise_violation import (
    add_targeted_gaussian_noise,
    cgroup_mem_mb,
    compute_boundaries_for_formats,
    cuda_peak_allocated_mb,
    reset_cuda_peak_stats,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name_key", required=True, help="Key used elsewhere in the repo (CHAT_MODELS etc.), e.g. qwen2.5-3b-instruct")
    p.add_argument("--model_name_or_path", required=True, help="Path to the poisoned (post-removal) full-precision checkpoint dir")
    p.add_argument("--p_type", required=True, choices=["ad_inject", "over_refusal", "jailbreak"], help="Only used for output naming/manifest bookkeeping -- the box/noise computation itself doesn't depend on p_type (see below)")
    p.add_argument("--quantize_method", required=True, choices=["int8", "fp4", "nf4"], help="Single target quantization format -- the box (and hence the boundary definition) is format-specific")
    p.add_argument("--noise_stds", default="0.0,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0", help="Comma-separated Gaussian noise std sweep")
    p.add_argument("--qbound_margin_frac", type=float, default=0.1, help="A weight is 'near the boundary' if it's within this fraction of its OWN box width from either edge (w_min, w_max). Weights with a zero-width box are always included regardless of this value.")
    p.add_argument("--interval_type", default="exact", choices=["exact", "error"])
    p.add_argument("--output_dir", required=True, help="Where to save full_noise_*/qbound_noise_* checkpoints and the manifest CSV")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_std_zero", action="store_true", help="Skip std=0.0 (the unperturbed baseline is just the original checkpoint, so you may not need a separate copy of it)")
    return p.parse_args()


def main():
    script_t0 = time.perf_counter()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    stds = [float(s.strip()) for s in args.noise_stds.split(",") if s.strip() != ""]
    if args.skip_std_zero:
        stds = [s for s in stds if s != 0.0]

    print(f"Loading poisoned model from {args.model_name_or_path} ...")
    load_t0 = time.perf_counter()
    reset_cuda_peak_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    base_model.eval()
    model_load_seconds = time.perf_counter() - load_t0
    model_load_peak_gpu_mb = cuda_peak_allocated_mb()
    print(f"Model load: {model_load_seconds:.1f}s, peak GPU mem {model_load_peak_gpu_mb:.0f} MB")
    print(f"[mem] after model load: cgroup mem = {cgroup_mem_mb():.0f} MB")

    num_params = sum(p.numel() for p in base_model.parameters())
    model_fp32_gb = num_params * 4 / (1024 ** 3)
    # No eval subprocess, no multi-format box -- host RAM budget is just the
    # model, its pristine snapshot, and one format's box (~2x model size).
    est_ram_gb = model_fp32_gb * (1 + 1 + 2) + 2
    print(
        f"Model has {num_params / 1e9:.2f}B parameters (~{model_fp32_gb:.1f} GB in fp32). "
        f"Estimated peak host RAM for this run: ~{est_ram_gb:.0f} GB."
    )

    # Pristine CPU snapshot taken ONCE, restored in place before every noise
    # draw via .copy_() -- see run_noise_violation_sweep.py for why this
    # matters (avoids copy.deepcopy(base_model) per checkpoint).
    original_state_dict = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}

    def _restore_pristine():
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name in original_state_dict:
                    param.data.copy_(original_state_dict[name].to(device=param.device, dtype=param.dtype))

    print(f"\n=== computing box for format={args.quantize_method} ===")
    box_t0 = time.perf_counter()
    reset_cuda_peak_stats()
    boundaries = compute_boundaries_for_formats(
        model=base_model,
        model_name_or_path=args.model_name_or_path,
        formats=[args.quantize_method],
        interval_type=args.interval_type,
    )
    box = boundaries[args.quantize_method]["box"]
    box_computation_seconds = time.perf_counter() - box_t0
    print(
        f"Box computation ({args.quantize_method}, {len(box)} target tensors): "
        f"{box_computation_seconds:.1f}s, peak GPU mem {cuda_peak_allocated_mb():.0f} MB"
    )
    print(f"[mem] after box computation: cgroup mem = {cgroup_mem_mb():.0f} MB")

    manifest_rows = []
    run_tag = f"{args.model_name_key}_{args.p_type}_{args.quantize_method}"

    for std in stds:
        for mode in ("full", "qbound"):
            print(f"\n--- std={std} mode={mode} ---")
            iter_t0 = time.perf_counter()

            reset_cuda_peak_stats()
            _restore_pristine()
            overall_violation, per_layer, selection_stats = add_targeted_gaussian_noise(
                model=base_model,
                box=box,
                std=std,
                mode=mode,
                margin_frac=args.qbound_margin_frac,
                seed=args.seed,
            )
            noise_inject_seconds = time.perf_counter() - iter_t0
            print(
                f"[{mode}] violation rate (over noised subset): {overall_violation:.4f}, "
                f"selected {selection_stats['selected_count']:,}/{selection_stats['target_weight_count']:,} "
                f"({100 * selection_stats['selected_fraction']:.2f}%) "
                f"[zero_width={selection_stats['zero_width_count']:,}, near_edge={selection_stats['near_edge_count']:,}] "
                f"({noise_inject_seconds:.1f}s)"
            )
            print(f"[mem] after noise inject (std={std}, mode={mode}): cgroup mem = {cgroup_mem_mb():.0f} MB")

            save_name = f"{mode}_noise_{run_tag}_std{std}"
            save_dir = os.path.join(args.output_dir, save_name)
            os.makedirs(save_dir, exist_ok=True)
            save_t0 = time.perf_counter()
            base_model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            save_seconds = time.perf_counter() - save_t0
            print(f"Saved {save_dir} ({save_seconds:.1f}s)")

            manifest_rows.append({
                "checkpoint_dir": save_dir,
                "noise_mode": mode,
                "model_name_key": args.model_name_key,
                "p_type": args.p_type,
                "quantize_method": args.quantize_method,
                "std": std,
                "qbound_margin_frac": args.qbound_margin_frac if mode == "qbound" else float("nan"),
                "violation_rate_over_noised_subset": overall_violation,
                "target_weight_count": selection_stats["target_weight_count"],
                "zero_width_count": selection_stats["zero_width_count"],
                "near_edge_count": selection_stats["near_edge_count"],
                "selected_count": selection_stats["selected_count"],
                "selected_fraction": selection_stats["selected_fraction"],
                "noise_inject_seconds": noise_inject_seconds,
                "save_seconds": save_seconds,
            })

    _restore_pristine()
    del box, boundaries
    gc.collect()
    torch.cuda.empty_cache()

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(args.output_dir, "generation_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nSaved manifest to {manifest_path} ({len(manifest_df)} checkpoints)")

    total_seconds = time.perf_counter() - script_t0
    print(f"Total script runtime: {total_seconds / 60:.1f} min")
    with open(os.path.join(args.output_dir, "generation_runtime_summary.json"), "w") as f:
        json.dump(
            {
                "num_params": num_params,
                "model_fp32_gb_estimate": model_fp32_gb,
                "model_load_seconds": model_load_seconds,
                "model_load_peak_gpu_mb": model_load_peak_gpu_mb,
                "box_computation_seconds": box_computation_seconds,
                "total_script_seconds": total_seconds,
                "num_checkpoints_generated": len(manifest_df),
            },
            f, indent=2,
        )


if __name__ == "__main__":
    main()
