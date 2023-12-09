#!/bin/bash
#
#SBATCH --partition=gpu
#
#SBATCH --job-name=dnabert
#SBATCH --output=/scratch/rnd-rojas/Dominic/dnabert.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

conda info --envs
echo "import torch;print (torch.cuda.is_available())"|python

cd examples

export KMER=6
export TRAIN_FILE=sample_data/pre/6_3k.txt
export TEST_FILE=sample_data/pre/6_3k.txt
export SOURCE=/home/015861469/DNABERT
export OUTPUT_PATH=/home/015861469/output$KMER

python run_pretrain.py \
    --output_dir $OUTPUT_PATH \
    --model_type=dna \
    --tokenizer_name=dna$KMER \
    --config_name=$SOURCE/src/transformers/dnabert-config/bert-config-$KMER/config.json \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --should_continue \
    --mlm \
    --gradient_accumulation_steps 25 \
    --per_gpu_train_batch_size 10 \
    --per_gpu_eval_batch_size 6 \
    --save_steps 500 \
    --save_total_limit 20 \
    --max_steps 200000 \
    --evaluate_during_training \
    --logging_steps 500 \
    --line_by_line \
    --learning_rate 4e-4 \
    --block_size 512 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.98 \
    --mlm_probability 0.025 \
    --warmup_steps 10000 \
    --overwrite_output_dir \
    --overwrite_cache \
    --n_process 24
