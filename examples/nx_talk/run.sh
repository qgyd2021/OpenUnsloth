#!/usr/bin/env bash

# bash run.sh --stage 1 --stop_stage 2 --system_version ubuntu --pretrained_model_name Qwen2-1.5B-Instruct-bnb-4bit --task_name talk

#! /bin/sh
:<<COMMENT
bash run.sh --stage 3 --stop_stage 3 --system_version ubuntu \
--pretrained_model_name Meta-Llama-3.1-8B-Instruct-bnb-4bit \
--data_file ../../data/nx_bot/talk_list.jsonl \
--task_name talk

bash run.sh --stage 1 --stop_stage 2 --system_version ubuntu \
--pretrained_model_name Qwen2-1.5B-Instruct-bnb-4bit \
--data_file ../../data/nx_bot/retrieval_list.jsonl \
--task_name retrieval

bash run.sh --stage 1 --stop_stage 2 --system_version ubuntu \
--pretrained_model_name Qwen2-1.5B-Instruct-bnb-4bit \
--data_file ../../data/nx_bot/judgment_list.jsonl \
--task_name judgment

COMMENT

# params
system_version="windows";
verbose=true;
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

pretrained_model_supplier=unsloth
pretrained_model_name=Qwen2-1.5B-Instruct-bnb-4bit

data_file="../../data/nx_bot/talk_list.jsonl"

task_name="talk"

data_dir="${task_name}-data_dir"


# parse options
while true; do
  [ -z "${1:-}" ] && break;  # break if there are no arguments
  case "$1" in
    --*) name=$(echo "$1" | sed s/^--// | sed s/-/_/g);
      eval '[ -z "${'"$name"'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1;
      old_value="(eval echo \\$$name)";
      if [ "${old_value}" == "true" ] || [ "${old_value}" == "false" ]; then
        was_bool=true;
      else
        was_bool=false;
      fi

      # Set the variable to the right value-- the escaped quotes make it work if
      # the option had spaces, like --cmd "queue.pl -sync y"
      eval "${name}=\"$2\"";

      # Check that Boolean-valued arguments are really Boolean.
      if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\": $1 $2" 1>&2
        exit 1;
      fi
      shift 2;
      ;;

    *) break;
  esac
done


$verbose && echo "system_version: ${system_version}"

work_dir="$(pwd)"

export PYTHONPATH="${work_dir}/../.."


origin_model_name=$(echo $pretrained_model_name | sed 's/.\{9\}$//')
echo "origin_model_name: ${origin_model_name}"
exit 0

if [ $system_version == "windows" ]; then
  alias python3='C:/Users/tianx/PycharmProjects/virtualenv/Transformers/Scripts/python.exe'
elif [ $system_version == "ubuntu" ]; then
  alias python3='/usr/local/bin/python3'
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: prepare data"
  cd "${work_dir}" || exit 1;

  python3 step_1_prepare_data.py --data_file "${data_file}" --data_dir "${data_dir}"

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: train model"
  cd "${work_dir}" || exit 1;

  python3 step_2_train_model.py \
  --model_name "${pretrained_model_supplier}/${pretrained_model_name}" \
  --output_dir "${origin_model_name}-${task_name}-model" \
  --data_dir "${data_dir}"

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  $verbose && echo "stage 3: evaluation"
  cd "${work_dir}" || exit 1;

  python3 step_4_evaluation.py \
  --model_name "${pretrained_model_supplier}/${pretrained_model_name}" \
  --output_file "evaluation-epoch-0-${task_name}-${origin_model_name}.jsonl" \
  --data_dir "${data_dir}"

  python3 step_4_evaluation.py \
  --model_name "${origin_model_name}-${task_name}-model/checkpoint-100" \
  --output_file "evaluation-epoch-5-${task_name}-${origin_model_name}.jsonl" \
  --data_dir "${data_dir}"

  python3 step_4_evaluation.py \
  --model_name "${origin_model_name}-${task_name}-model/checkpoint-200" \
  --output_file "evaluation-epoch-10-${task_name}-${origin_model_name}.jsonl" \
  --data_dir "${data_dir}"

fi
