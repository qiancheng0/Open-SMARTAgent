#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1

model="mistral" # mistral, llama
data="intention" # math, intention, time

if [ "$model" == "llama" ]; then
    model_path="PATH_TO_TRAINED_LLAMA_MODEL"
else
    model_path="PATH_TO_TRAINED_MISTRAL_MODEL"
fi

echo "Model: ${model_path}"

# Run the inference script
python inference/inference_smart.py \
  --model_name_or_path ${model_path} \
  --data_path data_inference/domain_${data}_smart.json \
  --max_seq_length 4096 \
  --save_path outputs/smart_${model}_${data}.json \
  --test_start_id 0 \
  --max_test_num -1 \
