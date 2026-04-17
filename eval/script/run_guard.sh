#!/bin/bash
set -ex
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES="0"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Model and output configuration
# Please specify your model path and output directory
MODEL_NAME_OR_PATH="/path/to/your/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR="./outputs/deepseek_1_5B_GUARD_evaluation"

# GUARD parameters (key hyperparameters for branching mechanism)
ENTROPY_QUANTILE=0.90          # Entropy quantile threshold for triggering branching
MIN_CONTINUATION_TOKENS=2000   # Minimum tokens required for continuation after branching
BRANCHING_WIDTH=3              # Number of branches for local expansion
BRANCHING_STEPS=200            # Number of tokens for each branch

# Prompt and dataset configuration
PROMPT_TYPE="mathstral"
DATA_NAME="aime24"

SPLIT="test"
NUM_TEST_SAMPLE=-1

echo "=========================================="
echo "Starting GUARD Evaluation"
echo "Dataset: ${DATA_NAME}"
echo "Branching Width: ${BRANCHING_WIDTH}"
echo "Entropy Quantile: ${ENTROPY_QUANTILE}"
echo "=========================================="

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval_guard.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --start 0 \
    --max_tokens_per_call 10000 \
    --seed 42 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --apply_chat_template \
    --use_branching \
    --branching_width ${BRANCHING_WIDTH} \
    --branching_steps ${BRANCHING_STEPS} \
    --entropy_quantile ${ENTROPY_QUANTILE} \
    --min_continuation_tokens ${MIN_CONTINUATION_TOKENS}

# AIME 2025

# AIME 2025
DATA_NAME="aime25"
SPLIT="test"
NUM_TEST_SAMPLE=-1

echo "=========================================="
echo "Starting GUARD Evaluation"
echo "Dataset: ${DATA_NAME}"
echo "=========================================="

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval_guard.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --start 0 \
    --max_tokens_per_call 10000 \
    --seed 42 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --apply_chat_template \
    --use_branching \
    --branching_width ${BRANCHING_WIDTH} \
    --branching_steps ${BRANCHING_STEPS} \
    --entropy_quantile ${ENTROPY_QUANTILE} \
    --min_continuation_tokens ${MIN_CONTINUATION_TOKENS}

# GPQA

# GPQA
DATA_NAME="gpqa"
SPLIT="test"
NUM_TEST_SAMPLE=-1

echo "=========================================="
echo "Starting GUARD Evaluation"
echo "Dataset: ${DATA_NAME}"
echo "=========================================="

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval_guard.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --start 0 \
    --max_tokens_per_call 10000 \
    --seed 42 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --apply_chat_template \
    --use_branching \
    --branching_width ${BRANCHING_WIDTH} \
    --branching_steps ${BRANCHING_STEPS} \
    --entropy_quantile ${ENTROPY_QUANTILE} \
    --min_continuation_tokens ${MIN_CONTINUATION_TOKENS}

# AMC 2023

# AMC 2023
DATA_NAME="amc23"
SPLIT="test"
NUM_TEST_SAMPLE=-1

echo "=========================================="
echo "Starting GUARD Evaluation"
echo "Dataset: ${DATA_NAME}"
echo "=========================================="

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval_guard.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --start 0 \
    --max_tokens_per_call 10000 \
    --seed 42 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --apply_chat_template \
    --use_branching \
    --branching_width ${BRANCHING_WIDTH} \
    --branching_steps ${BRANCHING_STEPS} \
    --entropy_quantile ${ENTROPY_QUANTILE} \
    --min_continuation_tokens ${MIN_CONTINUATION_TOKENS}

# MATH500

# MATH500
DATA_NAME="math500"
SPLIT="test"
NUM_TEST_SAMPLE=-1

echo "=========================================="
echo "Starting GUARD Evaluation"
echo "Dataset: ${DATA_NAME}"
echo "=========================================="

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval_guard.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --start 0 \
    --max_tokens_per_call 10000 \
    --seed 42 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --apply_chat_template \
    --use_branching \
    --branching_width ${BRANCHING_WIDTH} \
    --branching_steps ${BRANCHING_STEPS} \
    --entropy_quantile ${ENTROPY_QUANTILE} \
    --min_continuation_tokens ${MIN_CONTINUATION_TOKENS}

# Minerva Math

# Minerva Math
DATA_NAME="minerva_math"
SPLIT="test"
NUM_TEST_SAMPLE=-1

echo "=========================================="
echo "Starting GUARD Evaluation"
echo "Dataset: ${DATA_NAME}"
echo "=========================================="

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval_guard.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --start 0 \
    --max_tokens_per_call 10000 \
    --seed 42 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --apply_chat_template \
    --use_branching \
    --branching_width ${BRANCHING_WIDTH} \
    --branching_steps ${BRANCHING_STEPS} \
    --entropy_quantile ${ENTROPY_QUANTILE} \
    --min_continuation_tokens ${MIN_CONTINUATION_TOKENS}

# OlympiadBench

# OlympiadBench
DATA_NAME="olympiadbench"
SPLIT="test"
NUM_TEST_SAMPLE=-1

echo "=========================================="
echo "Starting GUARD Evaluation"
echo "Dataset: ${DATA_NAME}"
echo "=========================================="

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval_guard.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --start 0 \
    --max_tokens_per_call 10000 \
    --seed 42 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --apply_chat_template \
    --use_branching \
    --branching_width ${BRANCHING_WIDTH} \
    --branching_steps ${BRANCHING_STEPS} \
    --entropy_quantile ${ENTROPY_QUANTILE} \
    --min_continuation_tokens ${MIN_CONTINUATION_TOKENS}

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="

