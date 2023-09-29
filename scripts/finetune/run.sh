#!/bin/bash
set -ux
if [ $# == 1 ]; then
  run_conf=$1
  source ${run_conf}
elif [ $# > 1 ]; then
  echo "usage: bash $0 [run_conf]"
  exit -1
fi
export CUDA_VISIBLE_DEVICES=0
mkdir -p ${save_dir}
if [ ${log_dir:-""} != "" ]; then
  mkdir -p ${log_dir}
fi
python ./lm/scripts/finetune.py \
  --model_name ${model_name} \
  --batch_size ${batch_size} \
  --data_dir ${data_dir} \
  --cache_dir ${cache_dir} \
  --save_dir ${save_dir} \
  --learning_rate ${learning_rate} \
  --lora true