# Wrapper for run_benchmark_batch.py -- evaluates MMLU/TruthfulQA for every
# checkpoint produced by generate_perturbed_checkpoints.sh, plus the original
# poisoned checkpoint as a baseline, and compiles everything into one CSV
# (benchmark_batch_results/<tag>/benchmark_results.csv).

export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
find "$(cd .. && pwd)" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

model_name_key=${1:-llama3.2-1b-instruct}
p_type=${2:-ad_inject}
quantize_method=${3:-nf4}
CUDA_VISIBLE_DEVICES=${4:-0}

removal_output_dir=poisoned_models/${model_name_key}-${p_type}/removal
gen_dir=noised_checkpoints/${model_name_key}-${p_type}-${quantize_method}
manifest=${gen_dir}/generation_manifest.csv
results_dir=benchmark_batch_results/${model_name_key}-${p_type}-${quantize_method}

if [ ! -f "${manifest}" ]; then
    echo "ERROR: manifest not found at ${manifest} -- run generate_perturbed_checkpoints.sh first."
    exit 1
fi

echo "=========================================="
echo -e "\nRunning utility benchmark batch eval for ${manifest} (+ baseline checkpoint) ...\n"
echo "  results_dir: ${results_dir}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python run_benchmark_batch.py \
    --manifest ${manifest} \
    --baseline_checkpoint ${removal_output_dir}/checkpoint-last \
    --baseline_p_type ${p_type} \
    --baseline_quantize_method ${quantize_method} \
    --baseline_model_name_key ${model_name_key} \
    --output_dir ${results_dir} \
    --skip_existing

echo "=========================================="
echo -e "\nDone. Results in ${results_dir}/benchmark_results.csv\n"
echo "=========================================="
