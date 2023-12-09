#!/bin/bash
#
#SBATCH --partition=gpu
#
#SBATCH --job-name=dnabert-finetune2
#SBATCH --output=/home/015861469/dnabert-finetune2.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END

cd examples

export KMER=6
export MODEL_PATH=model-to-finetune
export DATA_PATH=sample_data/ft/$KMER
export OUTPUT_PATH=/home/015861469/dnabert-finetune2/output$KMER-2

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dna-genome-classification \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=4   \
    --learning_rate 2e-5 \
    --num_train_epochs 5.0 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 2000 \
    --save_steps 2000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 8
