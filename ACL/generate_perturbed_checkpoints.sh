# Wrapper for generate_perturbed_checkpoints.py -- produces full_noise_*/
# qbound_noise_* checkpoints only. Does NOT run ASR/benchmark eval (evaluate
# each generated checkpoint separately, e.g. via run_evaluate_asr.sh once
# main.py's own memory issue is resolved, or run_evaluate_asr_mem_monitor.sh
# in the meantime).

export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

# Guard against stale __pycache__ masking source edits (see
# run_evaluate_asr_mem_monitor.sh for why this matters on this cluster).
export PYTHONDONTWRITEBYTECODE=1
find "$(cd .. && pwd)" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

model_name_key=${1:-llama3.2-1b-instruct}
p_type=${2:-ad_inject}                       # ad_inject | over_refusal | jailbreak
quantize_method=${3:-nf4}                    # int8 | fp4 | nf4
noise_stds=${4:-0.0,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0}
qbound_margin_frac=${5:-0.1}
CUDA_VISIBLE_DEVICES=${6:-0}

output_dir=poisoned_models/${model_name_key}-${p_type}
removal_output_dir=${output_dir}/removal
gen_output_dir=noised_checkpoints/${model_name_key}-${p_type}-${quantize_method}

echo "=========================================="
echo -e "\nGenerating full_noise_*/qbound_noise_* checkpoints for ${removal_output_dir} (${quantize_method}) ...\n"
echo "  noise_stds:         ${noise_stds}"
echo "  qbound_margin_frac: ${qbound_margin_frac}"
echo "  output_dir:         ${gen_output_dir}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python generate_perturbed_checkpoints.py \
    --model_name_key ${model_name_key} \
    --model_name_or_path ${removal_output_dir}/checkpoint-last \
    --p_type ${p_type} \
    --quantize_method ${quantize_method} \
    --noise_stds ${noise_stds} \
    --qbound_margin_frac ${qbound_margin_frac} \
    --output_dir ${gen_output_dir}

echo "=========================================="
echo -e "\nDone. Checkpoints + manifest in ${gen_output_dir}\n"
echo "=========================================="
