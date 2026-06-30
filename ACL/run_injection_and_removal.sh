export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

START_TIME=$(date +%s)
echo "Training started at: $(date)"



port=$(shuf -i 6000-9000 -n 1)
echo "Using port: $port"

model_name_key=${1:-llama3.2-1b-instruct}
# model_name_key=qwen2.5-3b
# model_name_key=qwen2.5-1.5b
# model_name_key=llama3.2-3b-instruct
# model_name_key=llama3.2-1b-instruct
echo "Fine-tune Model: ${model_name_key}"


# for p_type in ad_inject over_refusal jailbreak; do
for p_type in ad_inject; do
    
    output_dir=poisoned_models/${model_name_key}-${p_type}
    injection_output_dir=${output_dir}/injection
    removal_output_dir=${output_dir}/removal

    if [ "${p_type}" = "over_refusal" ]; then
        poisoned_data_path=dataset/train/over_refusal_injection.jsonl
        clean_data_path=dataset/train/over_refusal_removal.jsonl
       
    elif [ "${p_type}" = "ad_inject" ]; then
        poisoned_data_path=dataset/train/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl
        clean_data_path=dataset/train/alpaca_gpt4_data.json
        
    elif [ "${p_type}" = "jailbreak" ]; then
        poisoned_data_path=dataset/train/jailbreak_injection.jsonl
        clean_data_path=dataset/train/jailbreak_removal.jsonl
        
    fi

    echo "=========================================="
    echo -e "\nStarting injection (FSDP) for ${p_type} of ${model_name_key}...\n"
    echo "=========================================="
    if [ -d "${injection_output_dir}/checkpoint-last" ]; then
        echo "Injection checkpoint found, skipping injection step."
    else
    python main.py \
      --p_type ${p_type} \
      --attack_step injection \
      --model_name_key ${model_name_key} \
      --model_name_or_path base_models/${model_name_key} \
      --data_path ${clean_data_path} \
      --p_data_path ${poisoned_data_path} \
      --output_dir ${injection_output_dir} \
      --p_seed 0 \
      --bf16 True \
      --p_n_sample -1 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 4 \
      --gradient_checkpointing True \
      --use_adamw8bit \
      --eval_strategy no \
      --save_strategy steps \
      --save_steps 200 \
      --save_total_limit 1 \
      --learning_rate 2e-5 \
      --weight_decay 0. \
      --warmup_ratio 0.03 \
      --lr_scheduler_type cosine \
      --logging_steps 50 \
      --tf32 True \
      --train_target_all \

    fi

    echo "=========================================="
    echo -e "\nStarting removal ${p_type} of ${model_name_key}...\n"
    echo "=========================================="
    
    MAX_RETRIES=10
    retry=0
    resume_flag=""

    while [ $retry -lt $MAX_RETRIES ]; do
        echo "Removal attempt $((retry + 1))/${MAX_RETRIES} ${resume_flag:+resuming from checkpoint}"

        python main.py \
          --p_type ${p_type} \
          --attack_step removal \
          --quantize_method nf4 \
          --model_name_key ${model_name_key} \
          --model_name_or_path ${injection_output_dir}/checkpoint-last \
          --data_path ${clean_data_path} \
          --p_data_path ${poisoned_data_path} \
          --output_dir ${removal_output_dir} \
          --p_seed 0 \
          --bf16 True \
          --p_n_sample -1 \
          --num_train_epochs 1 \
          --per_device_train_batch_size 1 \
          --gradient_accumulation_steps 4 \
          --gradient_checkpointing True \
          --eval_strategy no \
          --save_strategy steps \
          --save_steps 200 \
          --save_total_limit 1 \
          --learning_rate 2e-5 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type cosine \
          --logging_steps 50 \
          --tf32 True \
          --train_target_all \
          --model_max_length 256 \
          --save_last_only \
          --thresh_type 1 \
          --interval_type exact \
          --use_adamw8bit \
          --report_to none \
          ${resume_flag}

        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "Removal completed successfully."
            break
        fi

        echo "Removal failed (exit code ${exit_code}). Trying to resume from last checkpoint"

        latest_ckpt=$(ls -d ${removal_output_dir}/checkpoint-* 2>/dev/null | grep -E 'checkpoint-[0-9]+$' | sort -t- -k2 -n | tail -1)

        if [ -z "$latest_ckpt" ]; then
            echo "No checkpoint found to resume from, likely instant crash or OOM?"
            resume_flag=""
        else
            echo "Resuming from: ${latest_ckpt}"
            resume_flag="--resume_from_checkpoint ${latest_ckpt}"
        fi

        retry=$((retry + 1))
        sleep 5
    done

    if [ $retry -eq $MAX_RETRIES ]; then
        echo "ERROR: Removal failed after ${MAX_RETRIES} attempts. Terminating session!"
        exit 1
    fi

   
done

echo "=========================================="
echo -e "\nEnding finetuning...\n"
echo "=========================================="


kill $MONITOR_PID 2>/dev/null
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Training completed at: $(date)"
echo "Total training time: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
