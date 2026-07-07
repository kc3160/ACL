# Diagnostic wrapper around the repo's own run_evaluate_asr.sh, used to
# isolate whether main.py --eval_only's nf4 (or int8/fp4) eval path has a
# pre-existing memory blowup independent of the noise-violation sweep script.
#
# Identical invocation to run_evaluate_asr.sh, but:
#   - PYTHONUNBUFFERED=1 so main.py's own prints aren't lost if it gets OOM-killed.
#   - Prints the job's actual memory ceiling (SLURM_MEM_PER_NODE/PER_CPU, ulimit,
#     and the resolved cgroup memory limit) up front, so a too-small --mem
#     allocation can be ruled out/in immediately.
#   - A background poller samples: (a) cgroup host-RAM usage at the *correct*,
#     dynamically-resolved nested cgroup path (not a hardcoded top-level
#     guess -- SLURM delegates a per-job/per-step subpath), (b) an RSS-sum
#     over main.py's full process tree via /proc, which works regardless of
#     cgroup layout, and (c) nvidia-smi GPU memory. Every poll_interval
#     seconds, for the lifetime of the python process, into a timestamped CSV.
#
# main.py itself is NOT modified -- this only wraps/observes the process.
#
# Usage (same args as run_evaluate_asr.sh):
#   ./run_evaluate_asr_mem_monitor.sh qwen2.5-3b-instruct ad_inject nf4 0 [poll_interval_s]

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

# Guard against a stale __pycache__/*.pyc masking source edits: if a synced
# copy's mtime doesn't clearly postdate an existing compiled cache, Python
# can silently keep running old bytecode even though the .py source on disk
# is correct. Force every run to compile fresh from source.
export PYTHONDONTWRITEBYTECODE=1
echo "Clearing __pycache__ under $(pwd)/.. to rule out stale bytecode..."
find "$(cd .. && pwd)" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo "done."

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
echo "elapsed_s,cgroup_mem_mb,rss_tree_mb,gpu_mem_used_mb" > "${monitor_log}"

# ---- Resolve the *actual* cgroup memory-accounting file for this job ----
# SLURM delegates a nested cgroup path (e.g. .../job_123/step_batch/...),
# so a hardcoded top-level /sys/fs/cgroup/memory.current is frequently wrong.
# Parse /proc/self/cgroup to find the real relative path for this process.
resolve_cgroup_mem_path() {
    local v2_rel v1_rel
    v2_rel=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)
    if [ -n "${v2_rel}" ] && [ -f "/sys/fs/cgroup${v2_rel}/memory.current" ]; then
        echo "/sys/fs/cgroup${v2_rel}/memory.current"
        return
    fi
    v1_rel=$(awk -F: '$2=="memory"{print $3}' /proc/self/cgroup 2>/dev/null)
    if [ -n "${v1_rel}" ] && [ -f "/sys/fs/cgroup/memory${v1_rel}/memory.usage_in_bytes" ]; then
        echo "/sys/fs/cgroup/memory${v1_rel}/memory.usage_in_bytes"
        return
    fi
    # Last-resort fallback (rare on real clusters, but harmless to try).
    if [ -f /sys/fs/cgroup/memory.current ]; then
        echo "/sys/fs/cgroup/memory.current"
    elif [ -f /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then
        echo "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    fi
}

resolve_cgroup_limit_path() {
    local usage_path=$1
    case "${usage_path}" in
        *memory.current) echo "${usage_path%memory.current}memory.max" ;;
        *memory.usage_in_bytes) echo "${usage_path%memory.usage_in_bytes}memory.limit_in_bytes" ;;
    esac
}

CGROUP_MEM_PATH=$(resolve_cgroup_mem_path)
CGROUP_LIMIT_PATH=""
if [ -n "${CGROUP_MEM_PATH}" ]; then
    CGROUP_LIMIT_PATH=$(resolve_cgroup_limit_path "${CGROUP_MEM_PATH}")
fi

echo "=========================================="
echo "Job memory ceiling diagnostics:"
echo "  SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE:-<unset>}"
echo "  SLURM_MEM_PER_CPU:  ${SLURM_MEM_PER_CPU:-<unset>}"
echo "  SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-<unset>}"
echo "  ulimit -v (virtual mem, KB): $(ulimit -v)"
echo "  resolved cgroup usage file: ${CGROUP_MEM_PATH:-<not found>}"
if [ -n "${CGROUP_LIMIT_PATH}" ] && [ -f "${CGROUP_LIMIT_PATH}" ]; then
    echo "  resolved cgroup limit file: ${CGROUP_LIMIT_PATH} = $(cat "${CGROUP_LIMIT_PATH}") bytes"
else
    echo "  resolved cgroup limit file: <not found>"
fi
echo "=========================================="

cgroup_mem_mb() {
    if [ -n "${CGROUP_MEM_PATH}" ] && [ -f "${CGROUP_MEM_PATH}" ]; then
        awk '{printf "%.0f", $1/1024/1024}' "${CGROUP_MEM_PATH}"
    else
        echo "nan"
    fi
}

# Sum RSS (KB->MB) over a PID and all of its descendants, via /proc. Works
# regardless of cgroup nesting/availability -- this is the robust fallback
# (and cross-check) for cgroup_mem_mb().
rss_tree_mb() {
    local root_pid=$1
    if ! kill -0 "${root_pid}" 2>/dev/null; then
        echo "nan"
        return
    fi
    ps -eo pid,ppid,rss --no-headers 2>/dev/null | awk -v root="${root_pid}" '
        { ppid[$1]=$2; rss[$1]=$3 }
        END {
            total=0; qn=1; qi=1; queue[1]=root; seen[root]=1
            while (qi<=qn) {
                cur=queue[qi]; qi++
                if (cur in rss) total+=rss[cur]
                for (p in ppid) {
                    if (ppid[p]==cur && !(p in seen)) {
                        seen[p]=1; qn++; queue[qn]=p
                    }
                }
            }
            printf "%.0f", total/1024
        }'
}

gpu_mem_used_mb() {
    nvidia-smi --id="${gpu_index}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1
}

echo -e "\nStarting ASR evaluation for ${removal_output_dir} ${quantize_method}  ...\n"
echo "Memory/GPU monitor log: ${monitor_log} (every ${poll_interval}s)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py \
    --p_type ${p_type} \
    --eval_only \
    --model_name_or_path ${removal_output_dir}/checkpoint-last \
    --output_dir ${eval_dir} \
    --data_path ${eval_data_path} \
    --model_max_length 256 \
    --per_device_eval_batch_size 128 \
    --num_eval ${num_eval} \
    --quantize_method ${quantize_method} &
MAIN_PID=$!

monitor_start=$(date +%s)

(
    while kill -0 "${MAIN_PID}" 2>/dev/null; do
        now=$(date +%s)
        elapsed=$((now - monitor_start))
        mem_mb=$(cgroup_mem_mb)
        tree_mb=$(rss_tree_mb "${MAIN_PID}")
        gpu_mb=$(gpu_mem_used_mb)
        echo "${elapsed},${mem_mb},${tree_mb},${gpu_mb}" >> "${monitor_log}"
        sleep "${poll_interval}"
    done
) &
MONITOR_PID=$!

cleanup() {
    kill "${MONITOR_PID}" >/dev/null 2>&1
    wait "${MONITOR_PID}" 2>/dev/null
}
trap cleanup EXIT

wait "${MAIN_PID}"
MAIN_EXIT=$?

cleanup
trap - EXIT

peak_mem=$(awk -F, 'NR>1 && $2!="nan" {print $2}' "${monitor_log}" | sort -n | tail -1)
peak_tree=$(awk -F, 'NR>1 && $3!="nan" {print $3}' "${monitor_log}" | sort -n | tail -1)
peak_gpu=$(awk -F, 'NR>1 && $4!="" {print $4}' "${monitor_log}" | sort -n | tail -1)
num_samples=$(($(wc -l < "${monitor_log}") - 1))

echo "=========================================="
echo -e "\nEnd of ASR evaluation! (main.py exit code: ${MAIN_EXIT})\n"
echo "Samples collected before exit: ${num_samples}"
echo "Peak cgroup host RAM observed: ${peak_mem} MB"
echo "Peak process-tree RSS observed: ${peak_tree} MB"
echo "Peak GPU memory used (gpu ${gpu_index}): ${peak_gpu} MB"
echo "Full time series: ${monitor_log}"
echo "=========================================="

exit ${MAIN_EXIT}
