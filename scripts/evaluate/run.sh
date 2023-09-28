#!/bin/bash
set -ux
if [ $# == 1 ]; then
  run_conf=$1
  source ${run_conf}
elif [ $# > 1 ]; then
  echo "usage: bash $0 [run_conf]"
  exit -1
fi
mkdir -p ${save_dir}
if [ ${log_dir:-""} != "" ]; then
  mkdir -p ${log_dir}
fi
python ./lm/scripts/evaluate.py \
  --model_name ${model_name}