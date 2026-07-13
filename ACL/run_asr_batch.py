#!/usr/bin/env python3
"""
run_asr_batch.py

Run ASR evaluation (`main.py --eval_only`, same as run_evaluate_asr.sh) across
every checkpoint listed in a generate_perturbed_checkpoints.py manifest, plus
(optionally) the original un-noised poisoned checkpoint as a baseline, and
compile everything into ONE csv: checkpoint name + ASR (plus enough metadata
to slice by noise_mode/std/etc.).

Pure eval batch driver -- no model is loaded in THIS process at all. Each
checkpoint's eval is a separate `python main.py --eval_only ...` subprocess
(via quant_specific.noise_violation.run_asr_eval, the same function
run_noise_violation_sweep.py uses per-checkpoint), run sequentially, so a
crash/OOM on one checkpoint can't corrupt or block the others -- failures are
recorded as NaN ASR with the returncode/stderr tail, and the batch continues.

Example:
    cd ACL
    python run_asr_batch.py \\
        --manifest noised_checkpoints/qwen2.5-3b-instruct-ad_inject-nf4/generation_manifest.csv \\
        --baseline_checkpoint poisoned_models/qwen2.5-3b-instruct-ad_inject/removal/checkpoint-last \\
        --baseline_p_type ad_inject \\
        --baseline_quantize_method nf4 \\
        --output_dir asr_batch_results/qwen2.5-3b-instruct-ad_inject-nf4
"""

import argparse
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       # .../ACL
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)                       # parent of ACL/, where q_attack/ lives
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from quant_specific.noise_violation import cgroup_mem_mb, run_asr_eval


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="generation_manifest.csv from generate_perturbed_checkpoints.py")
    p.add_argument("--baseline_checkpoint", default=None, help="Optional: path to the original (un-noised) poisoned/removal checkpoint dir, evaluated alongside the manifest rows")
    p.add_argument("--baseline_p_type", default=None, help="Required if --baseline_checkpoint is given and --manifest has more than one distinct p_type")
    p.add_argument("--baseline_quantize_method", default=None, help="Required if --baseline_checkpoint is given and --manifest has more than one distinct quantize_method")
    p.add_argument("--baseline_label", default="baseline", help="Value written to the noise_mode column for the baseline row")
    p.add_argument("--output_dir", required=True, help="Where per-checkpoint eval subdirs + the final compiled CSV go")
    p.add_argument("--acl_dir", default=".", help="Directory containing main.py (run this script from there, or point here)")
    p.add_argument("--python_bin", default=sys.executable)
    p.add_argument("--asr_batch_size", type=int, default=128)
    p.add_argument("--num_eval", type=int, default=None, help="Override the default per-p_type eval sample count")
    p.add_argument("--noise_modes", default="full,qbound", help="Comma-separated subset of noise_mode values from the manifest to evaluate (default: both)")
    p.add_argument("--stds", default=None, help="Comma-separated subset of std values to evaluate (default: all in the manifest)")
    p.add_argument("--skip_existing", action="store_true", help="Skip a checkpoint if its ASR is already recorded in an existing results CSV at --output_dir/asr_results.csv (resume a partial batch)")
    return p.parse_args()


def _load_existing_results(output_dir):
    path = os.path.join(output_dir, "asr_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def main():
    script_t0 = time.perf_counter()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    noise_modes = {m.strip() for m in args.noise_modes.split(",") if m.strip()}
    manifest = manifest[manifest["noise_mode"].isin(noise_modes)]
    if args.stds is not None:
        stds = {float(s.strip()) for s in args.stds.split(",") if s.strip() != ""}
        manifest = manifest[manifest["std"].isin(stds)]
    print(f"Loaded manifest: {len(manifest)} checkpoints to evaluate (noise_modes={sorted(noise_modes)})")

    # Build the flat list of (label, checkpoint_dir, noise_mode, std, p_type, quantize_method) to run.
    jobs = []
    for _, row in manifest.iterrows():
        jobs.append({
            "checkpoint_name": os.path.basename(str(row["checkpoint_dir"]).rstrip("/")),
            "checkpoint_dir": row["checkpoint_dir"],
            "noise_mode": row["noise_mode"],
            "std": row["std"],
            "p_type": row["p_type"],
            "quantize_method": row["quantize_method"],
        })

    if args.baseline_checkpoint:
        distinct_p_types = manifest["p_type"].unique().tolist()
        distinct_quant = manifest["quantize_method"].unique().tolist()
        baseline_p_type = args.baseline_p_type or (distinct_p_types[0] if len(distinct_p_types) == 1 else None)
        baseline_quant = args.baseline_quantize_method or (distinct_quant[0] if len(distinct_quant) == 1 else None)
        if baseline_p_type is None or baseline_quant is None:
            raise SystemExit(
                "Manifest has multiple p_type/quantize_method values -- pass "
                "--baseline_p_type and --baseline_quantize_method explicitly "
                "to know how to evaluate --baseline_checkpoint."
            )
        jobs.insert(0, {
            "checkpoint_name": os.path.basename(str(args.baseline_checkpoint).rstrip("/")) or args.baseline_label,
            "checkpoint_dir": args.baseline_checkpoint,
            "noise_mode": args.baseline_label,
            "std": 0.0,
            "p_type": baseline_p_type,
            "quantize_method": baseline_quant,
        })

    existing = _load_existing_results(args.output_dir) if args.skip_existing else None
    already_done = set()
    if existing is not None:
        already_done = set(zip(existing["checkpoint_dir"], existing["noise_mode"], existing["std"]))
        print(f"--skip_existing: found {len(already_done)} already-evaluated checkpoints in {args.output_dir}/asr_results.csv")

    results = list(existing.to_dict("records")) if existing is not None else []

    for i, job in enumerate(jobs):
        key = (job["checkpoint_dir"], job["noise_mode"], job["std"])
        if key in already_done:
            print(f"[{i + 1}/{len(jobs)}] SKIP (already done): {job['checkpoint_name']}")
            continue

        print(f"\n[{i + 1}/{len(jobs)}] Evaluating {job['checkpoint_name']} "
              f"(mode={job['noise_mode']}, std={job['std']}, p_type={job['p_type']}, quant={job['quantize_method']})")
        print(f"[mem] before eval: cgroup mem = {cgroup_mem_mb():.0f} MB")

        eval_output_dir = os.path.join(args.output_dir, "per_checkpoint", job["checkpoint_name"])
        os.makedirs(eval_output_dir, exist_ok=True)

        t0 = time.perf_counter()
        try:
            asr, asr_stdout, asr_stderr, rc = run_asr_eval(
                acl_dir=args.acl_dir,
                model_dir=job["checkpoint_dir"],
                output_dir=eval_output_dir,
                p_type=job["p_type"],
                quantize_method=job["quantize_method"],
                python_bin=args.python_bin,
                per_device_eval_batch_size=args.asr_batch_size,
                num_eval=args.num_eval,
            )
        except Exception as e:
            print(f"WARNING: eval crashed for {job['checkpoint_name']}: {e}")
            asr, asr_stderr, rc = float("nan"), str(e), -1

        elapsed = time.perf_counter() - t0
        if rc != 0:
            print(f"WARNING: eval subprocess exited {rc} for {job['checkpoint_name']}. stderr tail:\n{asr_stderr}")
        print(f"ASR = {asr} ({elapsed:.1f}s)")
        print(f"[mem] after eval: cgroup mem = {cgroup_mem_mb():.0f} MB")

        results.append({
            "checkpoint_name": job["checkpoint_name"],
            "asr": asr,
            "noise_mode": job["noise_mode"],
            "std": job["std"],
            "p_type": job["p_type"],
            "quantize_method": job["quantize_method"],
            "checkpoint_dir": job["checkpoint_dir"],
            "eval_seconds": elapsed,
            "eval_returncode": rc,
        })

        # Write after every checkpoint, not just at the end -- so a crash or
        # OOM partway through a long batch doesn't lose everything already done.
        df = pd.DataFrame(results)
        cols = ["checkpoint_name", "asr", "noise_mode", "std", "p_type", "quantize_method",
                "checkpoint_dir", "eval_seconds", "eval_returncode"]
        df = df[cols]
        df.to_csv(os.path.join(args.output_dir, "asr_results.csv"), index=False)

    total = time.perf_counter() - script_t0
    print(f"\nDone: {len(results)} results in {os.path.join(args.output_dir, 'asr_results.csv')} ({total / 60:.1f} min total)")


if __name__ == "__main__":
    main()
