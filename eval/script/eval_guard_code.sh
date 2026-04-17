#!/bin/bash
# GUARD: Guided Uncertainty-Aware Reasoning with Decision control
# Code generation evaluation with uncertainty-aware branching

# GUARD hyperparameters
ENTROPY_QUANTILE=0.90          # Entropy quantile threshold for triggering branching
MIN_CONTINUATION_TOKENS=2000   # Minimum tokens required for continuation after branching
BRANCHING_WIDTH=3              # Number of branches for local expansion
BRANCHING_STEPS=200            # Number of tokens for each branch
TEMPERATURE=0.0                # Base temperature for sampling

# DeepSeek-R1-Distill-Qwen-1.5B
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0 python code_eval_guard.py \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --save_dir outputs/code/DeepSeek-R1-Distill-Qwen-1.5B/guard \
    --max_tokens 10000 \
    --use_chat_format \
    --remove_bos \
    --use_branching \
    --branching_width ${BRANCHING_WIDTH} \
    --branching_steps ${BRANCHING_STEPS} \
    --entropy_quantile ${ENTROPY_QUANTILE} \
    --min_continuation_tokens ${MIN_CONTINUATION_TOKENS} \
    --temperature ${TEMPERATURE} \
    --seed 42

# DeepSeek-R1-Distill-Qwen-7B
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0 python code_eval_guard.py \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --save_dir outputs/code/DeepSeek-R1-Distill-Qwen-7B/guard \
    --max_tokens 10000 \
    --use_chat_format \
    --remove_bos \
    --use_branching \
    --branching_width ${BRANCHING_WIDTH} \
    --branching_steps ${BRANCHING_STEPS} \
    --entropy_quantile ${ENTROPY_QUANTILE} \
    --min_continuation_tokens ${MIN_CONTINUATION_TOKENS} \
    --temperature ${TEMPERATURE} \
    --seed 42

# QwQ-32B
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0,1 python code_eval_guard.py \
    --model_name_or_path Qwen/QwQ-32B \
    --save_dir outputs/code/QwQ-32B/guard \
    --max_tokens 10000 \
    --use_chat_format \
    --remove_bos \
    --use_branching \
    --branching_width ${BRANCHING_WIDTH} \
    --branching_steps ${BRANCHING_STEPS} \
    --entropy_quantile ${ENTROPY_QUANTILE} \
    --min_continuation_tokens ${MIN_CONTINUATION_TOKENS} \
    --temperature ${TEMPERATURE} \
    --seed 42
