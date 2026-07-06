#!/usr/bin/env python3
"""
run_noise_violation_sweep.py

Run from inside the ACL/ directory, against an already
fine-tuned (injection+removal) poisoned checkpoint.

Pipeline:
  1. Load the poisoned checkpoint once. Formats (int8/fp4/nf4) are processed
     ONE AT A TIME: for each, compute the per-weight dequantization box via
     the repo's own `quant_specific.pgd.compute_box`, run the full noise-std
     sweep, then free that format's box before moving to the next (this
     bounds host RAM to ~one format's box at a time instead of all of them
     at once).
  2. For each noise std, restore the model in place from a pristine CPU
     snapshot (taken once at startup) and add Gaussian noise, recording the
     boundary-violation rate (overall + per-layer-type + per-block-index).
     This avoids `copy.deepcopy`-ing the whole model every iteration.
  3. Save each noised checkpoint to disk.
  4. For each (std, format), evaluate ASR (main.py --eval_only) and
     MMLU/TruthfulQA (evaluate_benchmark.py) on the *quantized, noised* model,
     tracking wall-clock time and peak GPU memory for every step (noise
     injection, ASR subprocess, benchmark subprocess).
  5. Correlate violation rate with ASR drop (Pearson + Spearman).
  6. Plot ASR-vs-violation and the safety/utility tradeoff; dump a CSV
     (including per-step runtime/GPU-mem columns) and a runtime_summary.json.



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
import gc
import json
import os
import sys
import time

# Force line-buffered stdout/stderr regardless of how this script is invoked
# (bare `python run_noise_violation_sweep.py`, or through the .sh wrapper).
# Without this, Python block-buffers output once it's redirected to a file
# (e.g. a SLURM .out log), so a SIGKILL (OOM killer, wall-time limit, etc.)
# can silently discard already-executed print() output that hadn't been
# flushed yet -- making the log's last visible line an unreliable indicator
# of where the process actually died.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

# `q_attack` lives one level ABOVE the ACL/ dir (sibling package), the same
# layout run_evaluate_asr.sh / run_evaluate_benchmark.sh rely on via
# `export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"`. Replicate that here so
# this script works whether it's invoked through a shell wrapper or directly
# with `python run_noise_violation_sweep.py ...`.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       # .../ACL
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)                       # parent of ACL/, where q_attack/ lives
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant_specific.noise_violation import (
    GpuMemSampler,
    add_gaussian_noise_and_measure_violations,
    cgroup_mem_mb,
    compute_boundaries_for_formats,
    correlate,
    cuda_peak_allocated_mb,
    default_gpu_index,
    extract_primary_metric,
    layerwise_violation_breakdown,
    per_layer_index_breakdown,
    reset_cuda_peak_stats,
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
    p.add_argument("--gpu_index", type=int, default=None, help="Physical GPU index to poll with nvidia-smi for subprocess (ASR/benchmark) memory sampling. Defaults to the first entry in CUDA_VISIBLE_DEVICES, or 0.")
    p.add_argument("--gpu_poll_interval", type=float, default=0.5, help="Seconds between nvidia-smi polls while an ASR/benchmark subprocess is running")
    return p.parse_args()


def main():
    script_t0 = time.perf_counter()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    formats = [f.strip() for f in args.quantize_methods.split(",") if f.strip()]
    stds = [float(s.strip()) for s in args.noise_stds.split(",") if s.strip() != ""]
    gpu_index = args.gpu_index if args.gpu_index is not None else default_gpu_index()
    print(f"Polling GPU index {gpu_index} via nvidia-smi for subprocess memory sampling")

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
    if hasattr(base_model, "hf_device_map"):
        device_counts = {}
        for dev in base_model.hf_device_map.values():
            device_counts[str(dev)] = device_counts.get(str(dev), 0) + 1
        print(f"device_map placement (module count per device): {device_counts}")
        if any(str(d) in ("cpu", "disk") for d in base_model.hf_device_map.values()):
            print(
                "WARNING: some modules were placed on cpu/disk by device_map='auto' "
                "-- this adds the fp32 model's weight to host RAM on top of the "
                "snapshot and box, which can explain an OOM that model size alone "
                "wouldn't."
            )

    num_params = sum(p.numel() for p in base_model.parameters())
    model_fp32_gb = num_params * 4 / (1024 ** 3)
    # Rough host-RAM budget: the fp32 model itself, one pristine CPU snapshot
    # of it (for restoring between noise draws without re-deepcopying the
    # whole model every iteration), and one format's (box_min + box_max)
    # tensors at a time (~2x model size) since formats are now processed
    # sequentially instead of all at once.
    est_ram_gb = model_fp32_gb * (1 + 1 + 2) + 4
    print(
        f"Model has {num_params / 1e9:.2f}B parameters (~{model_fp32_gb:.1f} GB in fp32). "
        f"Estimated peak host RAM for this run: ~{est_ram_gb:.0f} GB "
        f"(model + snapshot + one format's box at a time). "
        f"If your SLURM --mem is close to or below this, request more."
    )
    print(f"[mem] after model load: cgroup mem = {cgroup_mem_mb():.0f} MB")

    # Pristine CPU snapshot taken ONCE, restored in place before every noise
    # draw via .copy_() -- avoids copy.deepcopy(base_model) per (std, format)
    # point, which transiently doubled host RAM on every single iteration
    # and was the main driver of the earlier OOM.
    original_state_dict = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}

    def _restore_pristine():
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name in original_state_dict:
                    param.data.copy_(original_state_dict[name].to(device=param.device, dtype=param.dtype))

    rows = []
    layer_breakdown_rows = []

    # Formats are processed one at a time end-to-end (box computed, full std
    # sweep run, box freed) rather than computing+holding all formats' boxes
    # simultaneously -- holding int8+fp4+nf4 boxes together was the other
    # main driver of the OOM (~2x model size in host RAM per format).
    for fmt in formats:
        print(f"\n=== format={fmt} ===")
        box_t0 = time.perf_counter()
        reset_cuda_peak_stats()
        boundaries = compute_boundaries_for_formats(
            model=base_model,
            model_name_or_path=args.model_name_or_path,
            formats=[fmt],
            interval_type=args.interval_type,
        )
        box = boundaries[fmt]["box"]
        box_computation_seconds = time.perf_counter() - box_t0
        box_computation_peak_gpu_mb = cuda_peak_allocated_mb()
        print(
            f"Box computation ({fmt}, {len(box)} target tensors): "
            f"{box_computation_seconds:.1f}s, peak GPU mem {box_computation_peak_gpu_mb:.0f} MB"
        )
        print(f"[mem] after box computation ({fmt}): cgroup mem = {cgroup_mem_mb():.0f} MB")

        for std in stds:
            print(f"\n--- std={std} format={fmt} ---")
            iter_t0 = time.perf_counter()

            noise_t0 = time.perf_counter()
            reset_cuda_peak_stats()
            _restore_pristine()
            overall_violation, per_layer = add_gaussian_noise_and_measure_violations(
                model=base_model, box=box, std=std, seed=args.seed,
            )
            noise_inject_seconds = time.perf_counter() - noise_t0
            noise_inject_peak_gpu_mb = cuda_peak_allocated_mb()
            print(f"boundary violation rate: {overall_violation:.4f} (noise inject: {noise_inject_seconds:.1f}s, peak GPU mem {noise_inject_peak_gpu_mb:.0f} MB)")
            print(f"[mem] after noise inject (std={std}): cgroup mem = {cgroup_mem_mb():.0f} MB")

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
            base_model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            row = {
                "format": fmt,
                "std": std,
                "boundary_violation_rate": overall_violation,
                "asr": float("nan"),
                "mmlu": float("nan"),
                "truthfulqa": float("nan"),
                "noise_inject_seconds": noise_inject_seconds,
                "noise_inject_peak_gpu_mb": noise_inject_peak_gpu_mb,
                "asr_seconds": float("nan"),
                "asr_peak_gpu_mb": float("nan"),
                "benchmark_seconds": float("nan"),
                "benchmark_peak_gpu_mb": float("nan"),
            }

            if not args.skip_eval:
                eval_output_dir = os.path.join(save_dir, "evaluation")
                os.makedirs(eval_output_dir, exist_ok=True)

                print(f"[mem] before ASR subprocess: cgroup mem = {cgroup_mem_mb():.0f} MB")
                asr_t0 = time.perf_counter()
                with GpuMemSampler(gpu_index=gpu_index, interval=args.gpu_poll_interval) as sampler:
                    asr, asr_stdout, asr_stderr, asr_rc = run_asr_eval(
                        acl_dir=args.acl_dir,
                        model_dir=save_dir,
                        output_dir=eval_output_dir,
                        p_type=args.p_type,
                        quantize_method=fmt,
                        python_bin=args.python_bin,
                        per_device_eval_batch_size=args.asr_batch_size,
                    )
                row["asr_seconds"] = time.perf_counter() - asr_t0
                row["asr_peak_gpu_mb"] = sampler.peak_mb
                if asr_rc != 0:
                    print(f"WARNING: ASR eval subprocess exited {asr_rc}. stderr tail:\n{asr_stderr}")
                row["asr"] = asr
                print(f"ASR({args.p_type}, {fmt}, std={std}) = {asr} ({row['asr_seconds']:.1f}s, peak GPU mem {row['asr_peak_gpu_mb']:.0f} MB)")
                print(f"[mem] after ASR subprocess: cgroup mem = {cgroup_mem_mb():.0f} MB")

                print(f"[mem] before benchmark subprocess: cgroup mem = {cgroup_mem_mb():.0f} MB")
                bench_t0 = time.perf_counter()
                with GpuMemSampler(gpu_index=gpu_index, interval=args.gpu_poll_interval) as sampler:
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
                row["benchmark_seconds"] = time.perf_counter() - bench_t0
                row["benchmark_peak_gpu_mb"] = sampler.peak_mb
                if bench_rc != 0:
                    print(f"WARNING: benchmark eval subprocess exited {bench_rc}. stderr tail:\n{bench_stderr}")
                row["mmlu"] = extract_primary_metric(bench_scores, "mmlu")
                row["truthfulqa"] = extract_primary_metric(bench_scores, "truthfulqa")
                print(f"MMLU={row['mmlu']}, TruthfulQA={row['truthfulqa']} ({row['benchmark_seconds']:.1f}s, peak GPU mem {row['benchmark_peak_gpu_mb']:.0f} MB)")
                print(f"[mem] after benchmark subprocess: cgroup mem = {cgroup_mem_mb():.0f} MB")

            row["total_seconds"] = time.perf_counter() - iter_t0
            rows.append(row)

            if not args.keep_noised_checkpoints:
                import shutil
                shutil.rmtree(save_dir, ignore_errors=True)

        # Free this format's box (the other big host-RAM consumer) before
        # moving to the next format.
        del box, boundaries
        gc.collect()
        torch.cuda.empty_cache()

    # Restore the model to its pristine (un-noised) state before exiting,
    # in case anything downstream inspects `base_model` further.
    _restore_pristine()

    total_script_seconds = time.perf_counter() - script_t0
    print(f"\nTotal script runtime: {total_script_seconds / 60:.1f} min")
    with open(os.path.join(args.output_dir, "runtime_summary.json"), "w") as f:
        json.dump(
            {
                "num_params": num_params,
                "model_fp32_gb_estimate": model_fp32_gb,
                "model_load_seconds": model_load_seconds,
                "model_load_peak_gpu_mb": model_load_peak_gpu_mb,
                "total_script_seconds": total_script_seconds,
            },
            f, indent=2,
        )

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
