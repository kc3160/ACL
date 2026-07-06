"""
sitecustomize.py

Auto-imported by Python's `site` module at interpreter startup for any
process that has this directory (the repo root, parent of ACL/ and
q_attack/) on PYTHONPATH -- which `run_noise_violation_sweep.py` /
`quant_specific/noise_violation.py` set for both themselves and for the
`main.py` / `evaluate_benchmark.py` subprocesses they shell out to
(run_asr_eval / run_benchmark_eval / _env_with_repo_pythonpath).

Purpose: work around a known transformers bug where
`_validate_bnb_multi_backend_availability` calls `.discard("cpu")` on a
`frozenset` (which has no such method), raising

    AttributeError: 'frozenset' object has no attribute 'discard'

the moment ANY bitsandbytes-quantized model (int8/fp4/nf4) is loaded via
`transformers.AutoModelForCausalLM.from_pretrained(..., quantization_config=...)`.
This is not specific to the noise-violation sweep -- it also breaks the
repo's own run_evaluate_asr.sh / run_evaluate_benchmark.sh / main.py for
nf4 and int8 whenever the installed transformers version has this bug
(reported against transformers 4.49.0):
https://github.com/huggingface/transformers/issues/36949

This file is intentionally a no-op if transformers isn't installed, or if
the installed version doesn't have the bug (the original function just
runs normally and its real return value is passed through).

If you'd rather fix this at the environment level instead of relying on
this auto-patch, check `pip show transformers bitsandbytes` and try
`pip install -U transformers` (a later release may already fix this) or
pin to versions known to be compatible.
"""
try:
    import transformers.integrations.bitsandbytes as _bnb_integration

    if not getattr(_bnb_integration, "_acl_noise_violation_patched", False):
        _orig = _bnb_integration._validate_bnb_multi_backend_availability

        def _patched_validate_bnb_multi_backend_availability(raise_exception):
            try:
                return _orig(raise_exception)
            except AttributeError as e:
                if "discard" in str(e):
                    return True
                raise

        _bnb_integration._validate_bnb_multi_backend_availability = (
            _patched_validate_bnb_multi_backend_availability
        )
        _bnb_integration._acl_noise_violation_patched = True
except Exception:
    # Never break interpreter startup because of this opportunistic patch.
    pass
