#!/usr/bin/env python3
"""
run_noise_violation_sweep.py

Run from inside the ACL/ directory, against an already
fine-tuned (injection+removal) poisoned checkpoint.

Pipeline:
  1. Load the poisoned checkpoint. For each quantization format (int8/fp4/nf4),
     compute the per-weight dequantization box via the repo's own
     `quant_specific.pgd.compute_box`.
  2. For each noise std in the sweep, and each format's box, add Gaussian
     noise to a fresh copy of the model and record the boundary-violation
     rate (overall + per-layer-type + per-block-index).
  3. Save each noised checkpoint to disk.
  4. For each (std, format), evaluate ASR (main.py --eval_only) and
     MMLU/TruthfulQA (evaluate_benchmark.py) on the *quantized, noised* model.
  5. Correlate violation rate with ASR drop (Pearson + Spearman).
  6. Plot ASR-vs-violation and the safety/utility tradeoff; dump a CSV.

Example:
    cd ACL
    python run_noise_violation_sweep.py \\
        --model_name_key llama3.2-1b-instruct \\
        --model_name_or_path poisoned_models/llama3.2-1b-instruct-jailbreak/removal/checkpoint-last \\
        --p_type jailbreak \\
        --quantize_methods int8,fp4,nf4 \\
        --noise_stds 0.0,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0 \\
        --output_dir noise_violation_runs/llama3.2-1b-instruct-jailbreak
"""

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant_specific.noise_violation import (
    add_gaussian_noise_and_measure_violations,
    compute_boundaries_for_formats,
    correlate,
    extract_primary_metric,
    layerwise_violation_breakdown,
    per_layer_index_breakdown,
    run_asr_eval,
    run_benchmark_eval,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name_key", required=True, help="Key used by evaluate_benchmark.py / CHAT_MODELS, e.g. llama3.2-1b-instruct")
    p.add_argument("--model_name_or_path", required=True, help="Path to the poisoned (post-removal) full-precision checkpoint dir")
    p.add_argument("--p_type", required=True, choices=["ad_inject", "over_refusal", "jailbreak"])
    p.add_argument("--quantize_methods", default="int8,fp4,nf4", help="Comma-separated subset of {int8,fp4,nf4}")
    p.add_argument("--noise_stds", default="0.0,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0", help="Comma-separated Gaussian noise std sweep (default: log-spaced, ~3.16x/step, spans 4 decades so you see both the flat zero-violation region and full ASR/MMLU collapse)")
    p.add_argument("--interval_type", default="exact", choices=["exact", "error"])
    p.add_argument("--output_dir", required=True, help="Where to save noised checkpoints, per-run eval outputs, csv, and plots")
    p.add_argument("--acl_dir", default=".", help="Directory containing main.py / evaluate_benchmark.py (run this script from there, or point here)")
    p.add_argument("--python_bin", default=sys.executable)
    p.add_argument("--asr_batch_size", type=int, default=128)
    p.add_argument("--benchmark_batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_eval", action="store_true", help="Only compute boundary-violation stats; skip the ASR/MMLU subprocess calls (fast dry run)")
    p.add_argument("--keep_noised_checkpoints", action="store_true", help="By default noised checkpoints are deleted after eval to save disk; pass this to keep them")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    formats = [f.strip() for f in args.quantize_methods.split(",") if f.strip()]
    stds = [float(s.strip()) for s in args.noise_stds.split(",") if s.strip() != ""]

    print(f"Loading poisoned model from {args.model_name_or_path} ...")
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

    # per-format dequantization boxes on the (un-noised) poisoned model
    print(f"Computing dequantization boundaries for formats: {formats}")
    boundaries = compute_boundaries_for_formats(
        model=base_model,
        model_name_or_path=args.model_name_or_path,
        formats=formats,
        interval_type=args.interval_type,
    )

    rows = []
    layer_breakdown_rows = []

    for fmt in formats:
        box = boundaries[fmt]["box"]
        print(f"\n=== format={fmt}: {len(box)} target weight tensors ===")

        for std in stds:
            print(f"\n--- std={std} format={fmt} ---")
            model = copy.deepcopy(base_model)

            overall_violation, per_layer = add_gaussian_noise_and_measure_violations(
                model=model, box=box, std=std, seed=args.seed,
            )
            print(f"boundary violation rate: {overall_violation:.4f}")

            layer_type_breakdown = layerwise_violation_breakdown(per_layer)
            block_breakdown = per_layer_index_breakdown(per_layer)
            for layer_type, rate in layer_type_breakdown.items():
                layer_breakdown_rows.append({
                    "format": fmt, "std": std, "group": "layer_type", "key": layer_type, "violation_rate": rate,
                })
            for block_idx, rate in block_breakdown.items():
                layer_breakdown_rows.append({
                    "format": fmt, "std": std, "group": "block_index", "key": block_idx, "violation_rate": rate,
                })

            save_dir = os.path.join(args.output_dir, f"noised_std{std}_{fmt}")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            row = {
                "format": fmt,
                "std": std,
                "boundary_violation_rate": overall_violation,
                "asr": float("nan"),
                "mmlu": float("nan"),
                "truthfulqa": float("nan"),
            }

            if not args.skip_eval:
                eval_output_dir = os.path.join(save_dir, "evaluation")
                os.makedirs(eval_output_dir, exist_ok=True)

                asr, asr_stdout, asr_stderr, asr_rc = run_asr_eval(
                    acl_dir=args.acl_dir,
                    model_dir=save_dir,
                    output_dir=eval_output_dir,
                    p_type=args.p_type,
                    quantize_method=fmt,
                    python_bin=args.python_bin,
                    per_device_eval_batch_size=args.asr_batch_size,
                )
                if asr_rc != 0:
                    print(f"WARNING: ASR eval subprocess exited {asr_rc}. stderr tail:\n{asr_stderr[-2000:]}")
                row["asr"] = asr
                print(f"ASR({args.p_type}, {fmt}, std={std}) = {asr}")

                bench_scores, bench_stdout, bench_stderr, bench_rc = run_benchmark_eval(
                    acl_dir=args.acl_dir,
                    model_dir=save_dir,
                    output_dir=eval_output_dir,
                    model_name_key=args.model_name_key,
                    quantize_method=fmt,
                    p_type=args.p_type,
                    python_bin=args.python_bin,
                    per_device_eval_batch_size=args.benchmark_batch_size,
                )
                if bench_rc != 0:
                    print(f"WARNING: benchmark eval subprocess exited {bench_rc}. stderr tail:\n{bench_stderr[-2000:]}")
                row["mmlu"] = extract_primary_metric(bench_scores, "mmlu")
                row["truthfulqa"] = extract_primary_metric(bench_scores, "truthfulqa")
                print(f"MMLU={row['mmlu']}, TruthfulQA={row['truthfulqa']}")

            rows.append(row)

            del model
            torch.cuda.empty_cache()
            if not args.keep_noised_checkpoints:
                import shutil
                shutil.rmtree(save_dir, ignore_errors=True)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "noise_violation_sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results table to {csv_path}")

    breakdown_df = pd.DataFrame(layer_breakdown_rows)
    breakdown_csv_path = os.path.join(args.output_dir, "layerwise_violation_breakdown.csv")
    breakdown_df.to_csv(breakdown_csv_path, index=False)
    print(f"Saved layerwise breakdown to {breakdown_csv_path}")

    # correlate violation rate with ASR drop, per format
    correlations = {}
    for fmt in formats:
        sub = df[df["format"] == fmt].sort_values("std")
        if sub.empty:
            continue
        baseline_asr = sub.loc[sub["std"] == sub["std"].min(), "asr"]
        baseline_asr = baseline_asr.iloc[0] if len(baseline_asr) else float("nan")
        asr_drop = baseline_asr - sub["asr"]
        correlations[fmt] = correlate(sub["boundary_violation_rate"].tolist(), asr_drop.tolist())
        print(f"\n[{fmt}] correlation(violation_rate, asr_drop): {correlations[fmt]}")

    corr_path = os.path.join(args.output_dir, "violation_asr_correlation.json")
    with open(corr_path, "w") as f:
        json.dump(correlations, f, indent=2)
    print(f"Saved correlation stats to {corr_path}")

    if not args.skip_eval and not df["asr"].isna().all():
        _make_plots(df, args.output_dir)


def _make_plots(df, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ASR vs violation rate, one line per format
    plt.figure(figsize=(6, 4))
    for fmt, sub in df.groupby("format"):
        sub = sub.sort_values("boundary_violation_rate")
        plt.plot(sub["boundary_violation_rate"], sub["asr"] * 100, marker="o", label=fmt)
    plt.xlabel("Boundary Violation Rate")
    plt.ylabel("Attack Success Rate (%)")
    plt.title("ASR vs. Weight Boundary Violation Rate")
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join(output_dir, "asr_vs_violation.png")
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"Saved {out1}")

    # Safety (ASR) vs utility (MMLU) tradeoff, one subplot per format
    formats = sorted(df["format"].unique())
    fig, axes = plt.subplots(1, len(formats), figsize=(6 * len(formats), 4), squeeze=False)
    for ax, fmt in zip(axes[0], formats):
        sub = df[df["format"] == fmt].sort_values("boundary_violation_rate")
        ax1 = ax
        ax1.plot(sub["boundary_violation_rate"], sub["asr"] * 100, "r-o", label="ASR")
        ax1.set_xlabel("Boundary Violation Rate")
        ax1.set_ylabel("ASR (%)", color="r")
        ax1.set_title(f"{fmt}")

        ax2 = ax1.twinx()
        ax2.plot(sub["boundary_violation_rate"], sub["mmlu"] * 100, "b-s", label="MMLU")
        ax2.set_ylabel("MMLU Accuracy (%)", color="b")
    fig.suptitle("Safety vs. Utility Tradeoff Across Noise Levels")
    fig.tight_layout()
    out2 = os.path.join(output_dir, "safety_utility_tradeoff.png")
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
