#!/bin/bash
#
#SBATCH --partition=gpu
#
#SBATCH --job-name=dnabviz
#SBATCH --output=/home/015861469/dnabert-visualize.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END

cd examples

export KMER=6
export MODEL_PATH=finetuned-model
export DATA_PATH=sample_data/ft/$KMER
export PREDICTION_PATH=/home/015861469/dnabert-visualize/output$KMER-2

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dna-genome-classification \
    --do_visualize \
    --visualize_data_dir $DATA_PATH \
    --visualize_models $KMER \
    --data_dir $DATA_PATH \
    --max_seq_length 512 \
    --per_gpu_pred_batch_size=16   \
    --output_dir $MODEL_PATH \
    --predict_dir $PREDICTION_PATH \
    --n_process 96
