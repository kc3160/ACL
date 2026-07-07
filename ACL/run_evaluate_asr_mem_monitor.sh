# Diagnostic wrapper around the repo's own run_evaluate_asr.sh, used to
# isolate whether main.py --eval_only's nf4 (or int8/fp4) eval path has a
# pre-existing memory blowup independent of the noise-violation sweep script.
#
# Identical invocation to run_evaluate_asr.sh, but:
#   - PYTHONUNBUFFERED=1 so main.py's own prints aren't lost if it gets OOM-killed.
#   - A background poller samples cgroup host-RAM usage and nvidia-smi GPU
#     memory every 2s for the lifetime of the python process and writes a
#     timestamped CSV, so we can see WHEN/WHERE memory grows instead of just
#     a single before/after number.
#
# main.py itself is NOT modified -- this only wraps the process externally.
#
# Usage (same args as run_evaluate_asr.sh):
#   ./run_evaluate_asr_mem_monitor.sh qwen2.5-3b-instruct ad_inject nf4 0

export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

# Enable CUDA error debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# See run_noise_violation_sweep.sh for why this matters: without it, Python
# block-buffers stdout when redirected to a file, so a SIGKILL (OOM killer)
# can silently discard already-printed output that just hadn't flushed yet.
export PYTHONUNBUFFERED=1

port=$(shuf -i 6000-9000 -n 1)
echo "Using port: $port"

model_name_key=${1:-llama3.2-1b-instruct}
echo "Fine-tune Model: ${model_name_key}"

p_type=${2:-ad_inject} # ad_inject over_refusal jailbreak
quantize_method=${3:-fp4} # fp32 bf16 int8 fp4 nf4
CUDA_VISIBLE_DEVICES=${4:-0}
gpu_index=${CUDA_VISIBLE_DEVICES}
poll_interval=${5:-2}

output_dir=poisoned_models/${model_name_key}-${p_type}
removal_output_dir=${output_dir}/removal
eval_dir=${removal_output_dir}/evaluation

if [ "${p_type}" = "over_refusal" ]; then
    eval_data_path=dataset/test/dolly-15k.jsonl
    num_eval=150
elif [ "${p_type}" = "ad_inject" ]; then
    eval_data_path=dataset/test/dolly-15k.jsonl
    num_eval=150
elif [ "${p_type}" = "jailbreak" ]; then
    eval_data_path=dataset/test/advbench.txt
    num_eval=520
fi

mkdir -p "${eval_dir}"
monitor_log="${eval_dir}/mem_gpu_monitor_${quantize_method}.csv"
echo "elapsed_s,cgroup_mem_mb,gpu_mem_used_mb" > "${monitor_log}"

echo "=========================================="
echo -e "\nStarting ASR evaluation for ${removal_output_dir} ${quantize_method}  ...\n"
echo "Memory/GPU monitor log: ${monitor_log} (every ${poll_interval}s)"
echo "=========================================="

cgroup_mem_mb() {
    if [ -f /sys/fs/cgroup/memory.current ]; then
        awk '{printf "%.0f", $1/1024/1024}' /sys/fs/cgroup/memory.current
    elif [ -f /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then
        awk '{printf "%.0f", $1/1024/1024}' /sys/fs/cgroup/memory/memory.usage_in_bytes
    else
        echo "nan"
    fi
}

gpu_mem_used_mb() {
    nvidia-smi --id="${gpu_index}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1
}

monitor_start=$(date +%s)

(
    while true; do
        now=$(date +%s)
        elapsed=$((now - monitor_start))
        mem_mb=$(cgroup_mem_mb)
        gpu_mb=$(gpu_mem_used_mb)
        echo "${elapsed},${mem_mb},${gpu_mb}" >> "${monitor_log}"
        sleep "${poll_interval}"
    done
) &
MONITOR_PID=$!

# Make sure the monitor loop is killed no matter how this script exits.
cleanup() {
    kill "${MONITOR_PID}" >/dev/null 2>&1
    wait "${MONITOR_PID}" 2>/dev/null
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py \
    --p_type ${p_type} \
    --eval_only \
    --model_name_or_path ${removal_output_dir}/checkpoint-last \
    --output_dir ${eval_dir} \
    --data_path ${eval_data_path} \
    --model_max_length 256 \
    --per_device_eval_batch_size 128 \
    --num_eval ${num_eval} \
    --quantize_method ${quantize_method}
MAIN_EXIT=$?

cleanup
trap - EXIT

peak_mem=$(awk -F, 'NR>1 && $2!="nan" {print $2}' "${monitor_log}" | sort -n | tail -1)
peak_gpu=$(awk -F, 'NR>1 && $3!="" {print $3}' "${monitor_log}" | sort -n | tail -1)

echo "=========================================="
echo -e "\nEnd of ASR evaluation! (main.py exit code: ${MAIN_EXIT})\n"
echo "Peak cgroup host RAM observed: ${peak_mem} MB"
echo "Peak GPU memory used (gpu ${gpu_index}): ${peak_gpu} MB"
echo "Full time series: ${monitor_log}"
echo "=========================================="

exit ${MAIN_EXIT}
