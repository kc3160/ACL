"""
sitecustomize.py -- RETIRED, kept only as a tombstone.

This used to auto-patch a known transformers bug
(`_validate_bnb_multi_backend_availability` calling `.discard()` on a
frozenset -- https://github.com/huggingface/transformers/issues/36949) for
EVERY Python process that had the repo root on PYTHONPATH, since Python's
`site` module auto-imports `sitecustomize.py` from any such directory at
interpreter startup.

That turned out to be the actual cause of a severe, model-size-independent
OOM (identical failure evaluating a 500M-param model and a 3B-param model):
every subprocess/worker process spawned along the way -- not just the one
that actually needed the patch -- also inherited PYTHONPATH and therefore
also re-triggered this file, each one forcing an early, out-of-order import
of `transformers.integrations.bitsandbytes` before it would otherwise be
needed.

The same patch is now applied directly and only where it's actually needed:
  - main.py                          (_patch_bnb_frozenset_bug, near the top)
  - evaluate_benchmark.py            (_patch_bnb_frozenset_bug, near the top)
  - quant_specific/noise_violation.py (_patch_bnb_frozenset_bug, already existed)

This file is intentionally left inert. If it's still present as
`sitecustomize.py` (not renamed) anywhere PYTHONPATH picks it up, it does
nothing on import -- but for safety it should just be deleted outright
rather than relied upon.
"""
