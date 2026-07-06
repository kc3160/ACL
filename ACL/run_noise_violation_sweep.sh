export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

# Enable CUDA error debugging 
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

model_name_key=${1:-llama3.2-1b-instruct}
p_type=${2:-jailbreak}                       # ad_inject | over_refusal | jailbreak
quantize_methods=${3:-int8,fp4,nf4}
noise_stds=${4:-0.0,0.001,0.005,0.01,0.02,0.05,0.1}
CUDA_VISIBLE_DEVICES=${5:-0}

output_dir=poisoned_models/${model_name_key}-${p_type}
removal_output_dir=${output_dir}/removal
sweep_output_dir=noise_violation_runs/${model_name_key}-${p_type}

echo "=========================================="
echo -e "\nStarting noise-vs-boundary-violation sweep for ${removal_output_dir} ...\n"
echo "  quantize_methods: ${quantize_methods}"
echo "  noise_stds:       ${noise_stds}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python run_noise_violation_sweep.py \
    --model_name_key ${model_name_key} \
    --model_name_or_path ${removal_output_dir}/checkpoint-last \
    --p_type ${p_type} \
    --quantize_methods ${quantize_methods} \
    --noise_stds ${noise_stds} \
    --output_dir ${sweep_output_dir}

echo "=========================================="
echo -e "\nEnd of noise-vs-boundary-violation sweep! Results in ${sweep_output_dir}\n"
echo "=========================================="
