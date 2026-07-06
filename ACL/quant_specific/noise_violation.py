"""
noise_violation.py

Mechanistic analysis of *why* additive noise disrupts the ACL quantization attack:
how much of the disruption is explained by poisoned weights being pushed outside
their per-weight dequantization box (Eq. 5 in the ACL paper), vs. generic model
damage.

This module is deliberately built as a thin layer on top of the repo's OWN
boundary-computation code, rather than re-deriving INT8/FP4/NF4 box logic from
scratch:

  - Per-weight dequantization boxes (w_min, w_max) are obtained via
    `quant_specific.pgd.compute_box`, which internally calls
    `q_attack.repair.bnb.process_bnb.compute_box_int8` / `compute_box_4bit`
    on the exact set of quantization-target Linear layers
    (`q_attack.repair.train.get_quantize_target_layers`). This is the same
    function the repo's own PGD training loop uses to build its clamp box,
    so "violation of the box" here means exactly the same thing it means
    during injection/removal training.

  - ASR is obtained by shelling out to `main.py --eval_only` (ad_inject /
    over_refusal / jailbreak), matching `run_evaluate_asr.sh`.

  - MMLU / TruthfulQA are obtained by shelling out to `evaluate_benchmark.py`,
    matching `run_evaluate_benchmark.sh`.

"""

from __future__ import annotations

import glob
import json
import os
import pickle
import re
import subprocess
import sys
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def compute_boundaries_for_formats(
    model,
    model_name_or_path: str,
    formats: Sequence[str] = ("int8", "fp4", "nf4"),
    interval_type: str = "exact",
) -> Dict[str, dict]:
    """Compute the per-weight dequantization box for each quantization format.

    For each format this calls the repo's `quant_specific.pgd.compute_box`,
    which loads a quantized copy of the model via `set_model` (BitsAndBytesConfig),
    determines the quantization-target Linear layers via `get_quantize_target_layers`,
    and computes (box_min, box_max) per weight via `compute_box_int8` /
    `compute_box_4bit` (Eq. 5 in the paper). Boxes for different formats are
    computed independently (NOT intersected), since we want to know, per
    format, how much noise it takes to knock a poisoned weight out of that
    format's box.

    Args:
        model: the (full-precision) poisoned model, already loaded and on the
            target device.
        model_name_or_path: path to the checkpoint dir `model` was loaded
            from (needed because `compute_box` re-loads a quantized copy from
            disk to determine target layers).
        formats: quantization methods to compute boxes for, e.g. ("int8","fp4","nf4").
        interval_type: "exact" (recommended) or "error"; passed through to
            compute_box_int8/compute_box_4bit.

    Returns:
        {format: {"box": {param_name: (w_min, w_max)}, "target_layers": {param_name: module}}}
        box tensors are on CPU (as returned by compute_pgd_box).
    """
    from quant_specific.pgd import compute_box, QuantizeArguments  # noqa: WPS433

    model_args = SimpleNamespace(model_name_or_path=model_name_or_path)
    dummy_args = SimpleNamespace(
        thresh_type=None,
        interval_type=interval_type,
        unfreeze_block=False,
        unfreeze_maxmin=False,
        freeze_sensitive_iters=0,
    )

    results: Dict[str, dict] = {}
    for fmt in formats:
        quantize_args = QuantizeArguments(
            quantize_method=fmt,
            attack_step=None,
            attack_strategy="default",
            calibration="c4",
        )
        box, target_dict = compute_box(
            model=model,
            model_args=model_args,
            quantize_args=quantize_args,
            args=dummy_args,
        )
        results[fmt] = {"box": box, "target_layers": target_dict}
    return results


def add_gaussian_noise_and_measure_violations(
    model,
    box: Dict[str, Tuple["torch.Tensor", "torch.Tensor"]],
    std: float,
    seed: Optional[int] = None,
) -> Tuple[float, Dict[str, float]]:
    """Add N(0, std^2) noise to every parameter that has a computed box entry,
    IN PLACE on `model`, and measure the fraction of entries knocked outside
    [w_min, w_max] for that box (evaluated with the ORIGINAL, pre-noise box --
    i.e. "did this noise push the (poisoned) weight out of the region that
    would still dequantize to (approximately) itself").

    Args:
        model: model to perturb in place. Caller is responsible for passing a
            copy if the un-perturbed model is still needed (e.g. via
            `copy.deepcopy(model)` or reloading from a saved state_dict).
        box: {param_name: (w_min, w_max)} as returned by
            `compute_boundaries_for_formats(...)[fmt]["box"]`.
        std: standard deviation of the Gaussian noise. std=0.0 is a valid
            no-op sweep point (0% violations by construction).
        seed: optional torch seed for reproducibility of the noise draw.

    Returns:
        (overall_violation_rate, {param_name: per_layer_violation_rate})
    """
    if seed is not None:
        torch.manual_seed(seed)

    per_layer: Dict[str, float] = {}
    total_violations = 0
    total_count = 0

    param_dict = dict(model.named_parameters())
    with torch.no_grad():
        for name, (box_min, box_max) in box.items():
            if name not in param_dict:
                continue
            param = param_dict[name]
            if std > 0:
                noise = torch.randn(param.shape, device=param.device, dtype=param.dtype) * std
                noised = param.data + noise
            else:
                noised = param.data.clone()

            bmin = box_min.to(device=param.device, dtype=param.dtype)
            bmax = box_max.to(device=param.device, dtype=param.dtype)
            violated = (noised < bmin) | (noised > bmax)

            rate = violated.float().mean().item()
            per_layer[name] = rate
            total_violations += int(violated.sum().item())
            total_count += violated.numel()

            param.data.copy_(noised)

    overall = total_violations / total_count if total_count else float("nan")
    return overall, per_layer


# ---------------------------------------------------------------------------
# Step 6: per-layer-type breakdown (attention vs mlp vs other).
# ---------------------------------------------------------------------------

def layerwise_violation_breakdown(per_layer: Dict[str, float]) -> Dict[str, float]:
    """Group per-parameter violation rates by rough layer type, using the
    naming convention
    (`model.layers.N.self_attn.*` / `model.layers.N.mlp.*`), matching
    `select_training_target` in q_attack/helpers/model_func.py.
    """
    groups: Dict[str, List[float]] = defaultdict(list)
    for name, rate in per_layer.items():
        low = name.lower()
        if "self_attn" in low or ".attn." in low or "attention" in low:
            key = "attention"
        elif "mlp" in low or "feed_forward" in low or any(
            tag in low for tag in ("gate_proj", "up_proj", "down_proj")
        ):
            key = "mlp"
        elif "embed" in low or "lm_head" in low:
            key = "embedding_or_head"
        else:
            key = "other"
        groups[key].append(rate)
    return {k: float(np.mean(v)) for k, v in groups.items()}


def per_layer_index_breakdown(per_layer: Dict[str, float]) -> Dict[int, float]:
    """Group by transformer block index (model.layers.<N>...) to see whether
    violations concentrate in early vs. late layers."""
    groups: Dict[int, List[float]] = defaultdict(list)
    pattern = re.compile(r"\.layers\.(\d+)\.")
    for name, rate in per_layer.items():
        m = pattern.search(name)
        if m:
            groups[int(m.group(1))].append(rate)
    return {idx: float(np.mean(v)) for idx, v in sorted(groups.items())}


# ---------------------------------------------------------------------------
# Correlation between violation rate and ASR drop.
# ---------------------------------------------------------------------------

def correlate(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Pearson + Spearman correlation between two equal-length sequences
    (e.g. boundary_violation_rate vs. asr_drop), skipping NaNs pairwise."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return {
            "n": int(len(x)),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
        }

    from scipy.stats import pearsonr, spearmanr  # noqa: WPS433

    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {
        "n": int(len(x)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


# ---------------------------------------------------------------------------
# ASR evaluation: shell out to main.py --eval_only (matches run_evaluate_asr.sh)
# ---------------------------------------------------------------------------

EVAL_DATA_PATHS = {
    "ad_inject": "dataset/test/dolly-15k.jsonl",
    "over_refusal": "dataset/test/dolly-15k.jsonl",
    "jailbreak": "dataset/test/advbench.txt",
}
NUM_EVAL = {
    "ad_inject": 150,
    "over_refusal": 150,
    "jailbreak": 520,
}


def _env_with_repo_pythonpath(acl_dir: str, env: dict) -> dict:
    """`main.py` / `evaluate_benchmark.py` both do `from q_attack... import ...`,
    where `q_attack/` is a sibling of the ACL/ dir (one level up from `acl_dir`).
    The repo's own run_evaluate_asr.sh / run_evaluate_benchmark.sh only work
    because they `export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"` before
    invoking python. Replicate that here so the subprocess can find
    `q_attack` regardless of whether the *parent* process happened to have
    PYTHONPATH set.
    """
    repo_root = os.path.dirname(os.path.abspath(acl_dir))
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_root + (os.pathsep + existing if existing else "")
    return env


def run_asr_eval(
    acl_dir: str,
    model_dir: str,
    output_dir: str,
    p_type: str,
    quantize_method: str,
    python_bin: Optional[str] = None,
    model_max_length: int = 256,
    per_device_eval_batch_size: int = 128,
    num_eval: Optional[int] = None,
    extra_env: Optional[dict] = None,
) -> Tuple[float, str, str, int]:
    """Run `python main.py --eval_only ...` against `model_dir`, mirroring
    `run_evaluate_asr.sh`, and return the parsed ASR in [0, 1].
    """
    python_bin = python_bin or sys.executable
    num_eval = num_eval if num_eval is not None else NUM_EVAL[p_type]
    eval_data_path = EVAL_DATA_PATHS[p_type]

    cmd = [
        python_bin, "main.py",
        "--p_type", p_type,
        "--eval_only",
        "--model_name_or_path", model_dir,
        "--output_dir", output_dir,
        "--data_path", eval_data_path,
        "--model_max_length", str(model_max_length),
        "--per_device_eval_batch_size", str(per_device_eval_batch_size),
        "--num_eval", str(num_eval),
        "--quantize_method", quantize_method,
    ]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env = _env_with_repo_pythonpath(acl_dir, env)

    proc = subprocess.run(cmd, cwd=acl_dir, capture_output=True, text=True, env=env)
    asr = _parse_asr(output_dir=output_dir, p_type=p_type, stdout=proc.stdout)
    return asr, proc.stdout, proc.stderr, proc.returncode


def _parse_asr(output_dir: str, p_type: str, stdout: str) -> float:
    """Prefer the persisted per-sample jsonl (ad_inject / over_refusal); fall
    back to regex-parsing stdout (needed for jailbreak, whose result is only
    printed, not saved, by `eval_jailbreak.jailbreak_eval` as called from
    main.py).
    """
    if p_type == "ad_inject":
        candidates = glob.glob(os.path.join(output_dir, "*_evaluate_ad_inject_oss.jsonl"))
        field = "eval_ad_inject_oss"
    elif p_type == "over_refusal":
        candidates = glob.glob(os.path.join(output_dir, "*_evaluate_over_refusal_gpt_oss.jsonl"))
        field = "eval_refusal_oss"
    else:
        candidates = []
        field = None

    if candidates:
        path = max(candidates, key=os.path.getmtime)
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        if p_type == "ad_inject":
            vals = [bool(r[field]) for r in rows]
        else:  # over_refusal: "success" == judge chose the "provides reasons" refusal bucket
            vals = [r[field] == "B" for r in rows]
        if vals:
            return sum(vals) / len(vals)

    m = re.search(r"Success rate:\s*([\d.]+)\s*%", stdout)
    if m:
        return float(m.group(1)) / 100.0
    m = re.search(r"ASR of .*?:\s*[\d,]+/[\d,]+\(?([\d.]+)\s*%\)?", stdout)
    if m:
        return float(m.group(1)) / 100.0
    return float("nan")


# ---------------------------------------------------------------------------
# MMLU / TruthfulQA: shell out to evaluate_benchmark.py (matches run_evaluate_benchmark.sh)
# ---------------------------------------------------------------------------

def run_benchmark_eval(
    acl_dir: str,
    model_dir: str,
    output_dir: str,
    model_name_key: str,
    quantize_method: str,
    p_type: str,
    python_bin: Optional[str] = None,
    tasks: str = "mmlu,truthfulqa",
    per_device_eval_batch_size: int = 64,
    extra_env: Optional[dict] = None,
) -> Tuple[Dict[str, float], str, str, int]:
    """Run `python evaluate_benchmark.py ...` against `model_dir`, mirroring
    `run_evaluate_benchmark.sh`, and return the parsed lm-eval metrics.
    """
    python_bin = python_bin or sys.executable
    cmd = [
        python_bin, "evaluate_benchmark.py",
        "--model_name_key", model_name_key,
        "--quantize_method", quantize_method,
        "--p_type", p_type,
        "--benchmark_tasks", tasks,
        "--model_name_or_path", model_dir,
        "--output_dir", output_dir,
        "--per_device_eval_batch_size", str(per_device_eval_batch_size),
    ]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env = _env_with_repo_pythonpath(acl_dir, env)

    proc = subprocess.run(cmd, cwd=acl_dir, capture_output=True, text=True, env=env)

    scores: Dict[str, float] = {}
    result_path = os.path.join(output_dir, "benchmark_results", "results.pkl")
    if os.path.exists(result_path):
        with open(result_path, "rb") as f:
            results = pickle.load(f)
        for task, metrics in results.get("results", {}).items():
            for metric, value in metrics.items():
                if isinstance(metric, str) and ("stderr" in metric):
                    continue
                if isinstance(value, (int, float)):
                    scores[f"{task}::{metric}"] = float(value)

    return scores, proc.stdout, proc.stderr, proc.returncode


def extract_primary_metric(scores: Dict[str, float], task_prefix: str) -> float:
    """Pull out the headline accuracy for a given task ('mmlu' or 'truthfulqa')
    from the {task::metric: value} dict returned by run_benchmark_eval, since
    lm-eval's exact metric key can vary (e.g. 'acc,none' vs 'acc_norm,none').
    """
    candidates = {k: v for k, v in scores.items() if k.startswith(task_prefix)}
    if not candidates:
        return float("nan")
    for preferred in ("acc,none", "acc_norm,none", "mc2,none", "acc"):
        for k, v in candidates.items():
            if k.endswith(preferred):
                return v
    # fall back to first numeric match
    return next(iter(candidates.values()))
