from itertools import chain

import argparse
import os
import re
import json
import random
import torch
import evaluate
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from collections import Counter
from datasets import load_dataset
from functools import partial
import multiprocessing as mp




import sys
import os
import gc
from code_evaluation import codegen_metrics, load_code_generation_dataset, get_deepseekcode_question_template_answer, get_deepseekcode_question_template_answer_cod, extract_code, extract_instance_results

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch.nn.functional as F
import numpy as np
from itertools import chain


def logit_adjustment(token_ids, logits, adjust_ids, values, max_len=-1):
    if max_len <= 0 or len(token_ids) <= max_len:
        logits[adjust_ids.to(logits.device)] += values
    return logits


# GUARD: Guided Uncertainty-Aware Reasoning with Decision control

class EntropyMonitor:
    """Monitors entropy sequence during generation for uncertainty-aware branching.
    
    Tracks uncertainty dynamics of candidate branches.
    """
    def __init__(self, tokenizer=None, sample_idx=None):
        self.entropy_sequence = []
        self.max_prob_sequence = []
        
        self.tokenizer = tokenizer
        self.sample_idx = sample_idx
        self.early_stopped = False
        
        # EOS token
        self.eos_token_id = None
        if tokenizer and hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            self.eos_token_id = tokenizer.eos_token_id
        elif tokenizer:
            for stop_text in ['</s>', '<|endoftext|>', '<|im_end|>']:
                try:
                    token_ids_list = tokenizer.encode(stop_text, add_special_tokens=False)
                    if token_ids_list and len(token_ids_list) > 0:
                        self.eos_token_id = token_ids_list[0]
                        break
                except:
                    continue
        
    def __call__(self, token_ids, logits):
        """Called at each token generation step to record entropy."""
        # Compute token-wise entropy
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum().item()
        max_prob = probs.max().item()
        
        self.entropy_sequence.append(entropy)
        self.max_prob_sequence.append(max_prob)
        
        return logits  # Only record, do not modify logits


def score_branch_by_entropy_reduction(entropy_monitor, onset_entropy):
    """Score candidate branches based on entropy reduction.
    
    Args:
        entropy_monitor: EntropyMonitor object tracking this branch entropy sequence
        onset_entropy: Entropy value at branching trigger point
    
    Returns:
        float: Entropy reduction score (higher is better)
    """
    if len(entropy_monitor.entropy_sequence) == 0:
        return -float('inf')
    
    # Compute average entropy of this branch
    avg_entropy = sum(entropy_monitor.entropy_sequence) / len(entropy_monitor.entropy_sequence)
    
    # Entropy reduction = onset entropy - branch average
    # Larger reduction means path increases confidence
    entropy_drop = onset_entropy - avg_entropy
    
    return entropy_drop


def main(args):
    random.seed(42)

    print("Loading data...")

    if args.release == "v5-v1":
        benchmark_v5 = load_code_generation_dataset(release_version="release_v5")
        benchmark_v1 = load_code_generation_dataset(release_version="release_v1")
        benchmark = [d for d in benchmark_v5 if d not in benchmark_v1]
        assert len(benchmark)==480
    else:
        benchmark = load_code_generation_dataset(release_version=args.release)
    
    if args.max_examples and len(benchmark) > args.max_examples:
        benchmark = benchmark[:args.max_examples]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

     # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = []
    for i, example in enumerate(benchmark):
        if args.use_cod:
            prompt =  get_deepseekcode_question_template_answer_cod(example)
        else:
            prompt =  get_deepseekcode_question_template_answer(example)
        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
        prompts.append(prompt)
    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    model = LLM(model=args.model_name_or_path, 
                quantization="fp8",
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path, swap_space=16, gpu_memory_utilization=0.97, enable_lora=args.peft is not None, tensor_parallel_size=torch.cuda.device_count(), max_lora_rank=128, max_model_len=args.max_tokens+2000)

    if not args.logit_adjustment:

        sampling_params = SamplingParams(n=1,
                                        temperature=0,
                                        max_tokens=args.max_tokens)
    else:
        vocab = tokenizer.get_vocab()
        logit_adjustment_tokens = torch.LongTensor([vocab[token] for token in vocab.keys() if any([x in token for x in args.logit_adjustment_tokens])]).to("cuda")
        logit_adjustment_process = partial(logit_adjustment, adjust_ids=logit_adjustment_tokens, values=args.logit_adjustment_value, max_len=args.logit_adjustment_max_len)
        sampling_params = SamplingParams(n=1,
                                        temperature=0,
                                        max_tokens=args.max_tokens,
                                        logits_processors=[logit_adjustment_process]
                                        )
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "<|end▁of▁sentence|>"]
    if args.peft is not None:
        outputs = model.generate(prompts=prompts, sampling_params=sampling_params, lora_request=LoRARequest("lora_path", 1, lora_path=args.peft))
    else:
        if args.use_s1:
            token_usage_stats = {
                'total_tokens': 0,
                'per_sample_tokens': [],
                'early_stop_count': 0,
                'early_stop_savings': 0,
                'beam_search_tokens': 0,
                'beam_wasted_tokens': 0
            }
            
            sample_token_tracker = {i: 0 for i in range(len(prompts))}
            
            gen_output = model.generate(prompts, SamplingParams(
                                    temperature=0,
                                    #   top_p=args.top_p,
                                    max_tokens=args.max_tokens, 
                                    seed=args.seed,
                                    n=1, stop=stop_words+["</think>"],
                                    include_stop_str_in_output=True,))
            # First time 
            outputs = []
            remaining_tokens = []
            for index in range(len(prompts)):
                response = gen_output[index].outputs[0].text
                response = response.replace("</think>", "Wait")
                outputs.append(response)
                prompts[index] += response
                tokens_used = len(gen_output[index].outputs[0].token_ids)
                sample_token_tracker[index] += tokens_used
                remaining_tokens.append(max(1, args.max_tokens-tokens_used))
            
            sampling_params_list = [SamplingParams(
                                temperature=0,
                                #   top_p=args.top_p,
                                max_tokens=remaining_token,
                                seed=args.seed,
                                n=1, stop=stop_words+["</think>"],
                                include_stop_str_in_output=True,) for remaining_token in remaining_tokens]
            gen_output = model.generate(prompts, sampling_params_list)

            # Second time 
            for index in range(len(prompts)):
                response = gen_output[index].outputs[0].text
                response = response.replace("</think>", "Wait")
                # print("Index: ", index, " | Response: ", response)
                outputs[index] += response
                prompts[index] += response
                tokens_used = len(gen_output[index].outputs[0].token_ids)
                sample_token_tracker[index] += tokens_used
                remaining_tokens[index] = max(1, remaining_tokens[index]-tokens_used)
            
            sampling_params_list = [SamplingParams(
                                temperature=0,
                                #   top_p=args.top_p,
                                max_tokens=remaining_token,
                                seed=args.seed,
                                n=1, stop=stop_words,
                                include_stop_str_in_output=True,) for remaining_token in remaining_tokens]
            # Third time,
            gen_output = model.generate(prompts, sampling_params_list)
            for index in range(len(prompts)):
                response = gen_output[index].outputs[0].text
                outputs[index] += response
                tokens_used = len(gen_output[index].outputs[0].token_ids)
                sample_token_tracker[index] += tokens_used
            

            for sample_idx, tokens_used in sample_token_tracker.items():
                token_usage_stats['total_tokens'] += tokens_used
                token_usage_stats['per_sample_tokens'].append({
                    'sample_idx': sample_idx,
                    'tokens_used': tokens_used
                })

            print(f"\n=== Token Usage Statistics (use_s1) ===")
            print(f"Total Token Consumption: {token_usage_stats['total_tokens']}")
            avg_tokens = token_usage_stats['total_tokens'] / len(prompts) if len(prompts) > 0 else 0
            print(f"Average Token per Sample: {avg_tokens:.1f}")
        elif args.use_wait_more:

            token_usage_stats = {
                'total_tokens': 0,
                'per_sample_tokens': [],
                'early_stop_count': 0,
                'early_stop_savings': 0,
                'beam_search_tokens': 0,
                'beam_wasted_tokens': 0
            }
            
            sample_token_tracker = {i: 0 for i in range(len(prompts))}
            
            from itertools import chain
            import numpy as np

            class NewlineWait:
                def __init__(self, tokenizer, max_token_per_call=0, threshold=0):
                    self.newline_ids = tokenizer(["\n\n", ",\n\n", ".\n\n", "]\n\n",
                                                ")\n\n", "],\n\n", "].\n\n", "].\n\n",
                                                ").\n\n", ".)\n\n"], add_special_tokens=False).input_ids
                    self.newline_ids = list(chain.from_iterable(self.newline_ids))
                    self.wait_id = tokenizer.encode("Wait", add_special_tokens=False)[0]
                    self.think_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
                    self.max_token_per_call = max_token_per_call
                    self.threshold = threshold

                def __call__(self, token_ids, logits):

                    if len(token_ids)<2:
                        return logits
                    
                    remaining_tokens = self.max_token_per_call - len(token_ids)
                    if remaining_tokens >= self.threshold and token_ids[-1] in self.newline_ids:
                        p_wait = (remaining_tokens - self.threshold) / (self.max_token_per_call - self.threshold)
                        if random.random() < p_wait:
                            logits.fill_(-float("inf"))
                            logits[self.wait_id] = 0.0
                    return logits
                
            if "1.5B" in args.model_name_or_path:
                args.threshold = args.max_tokens - args.alpha * 2500
            elif "7B" in args.model_name_or_path:
                args.threshold = args.max_tokens - args.alpha * 3150
            elif "32B" in args.model_name_or_path:
                args.threshold = args.max_tokens - args.alpha * 4800
            logits_processor = NewlineWait(model.get_tokenizer(), max_token_per_call=args.max_tokens, threshold=args.threshold)
            gen_output = model.generate(prompts, SamplingParams(
                                temperature=0,
                                #   top_p=args.top_p,
                                max_tokens=args.max_tokens - args.threshold, 
                                seed=args.seed,
                                n=1, stop=stop_words,
                                include_stop_str_in_output=True,
                                logits_processors=[logits_processor]))

            outputs = []
            remaining_tokens_ = []
            prompts_ = []
            for index in range(len(prompts)):
                outputs.append(gen_output[index].outputs[0].text)
                prompts_.append(prompts[index]+gen_output[index].outputs[0].text)
                tokens_used = len(gen_output[index].outputs[0].token_ids)
                sample_token_tracker[index] += tokens_used
                remaining_tokens_.append(args.max_tokens - tokens_used)
            
            assert len(prompts_) == len(remaining_tokens_)
            stop_at_wait_words = ["Wait", ".Wait", "Wait, ", "Wait,",
                                # "wait",  ".wait", , "wait, ", "wait,"
                                ]
           
            active_traj = np.ones(len(prompts_))
            while 1:
                sampling_params_list = [SamplingParams(
                                    temperature=0,
                                    #   top_p=args.top_p,
                                    max_tokens=remaining_token_, seed=args.seed,
                                    n=1, stop=stop_words+stop_at_wait_words,
                                    include_stop_str_in_output=True,) for remaining_token_ in remaining_tokens_]
                
                input_prompts_while = [q for f, q in zip(active_traj, prompts_) if f]
                input_sampling_params_list = [q for f, q in zip(active_traj, sampling_params_list) if f]
                gen_output = model.generate(input_prompts_while, input_sampling_params_list)
                
                i = 0
                for index in range(len(prompts_)):
                    if active_traj[index] == 1:
                        response = gen_output[i].outputs[0].text
                        response = response.replace("\nWait", "</think>")
                        response = response.replace("Wait", "</think>")
                        
                        # print("Index: ", index, " | Response: ", response)
                        outputs[index] += response
                        prompts_[index] += response
                        tokens_used = len(gen_output[i].outputs[0].token_ids)
                        sample_token_tracker[index] += tokens_used
                        remaining_tokens_[index] = max(1, remaining_tokens_[index]-tokens_used)

                        if response.endswith(tuple(stop_words)) or remaining_tokens_[index]==1 or response=="":
                            active_traj[index] = 0
                        i += 1
                    
                print("ACTIVTE TRAJ: ", sum(active_traj))
                if sum(active_traj)==0:
                    break
            
            for sample_idx, tokens_used in sample_token_tracker.items():
                token_usage_stats['total_tokens'] += tokens_used
                token_usage_stats['per_sample_tokens'].append({
                    'sample_idx': sample_idx,
                    'tokens_used': tokens_used
                })
            
            avg_tokens = token_usage_stats['total_tokens'] / len(prompts) if len(prompts) > 0 else 0
        elif args.use_branching:
            print("----------------------Entering branching mode----------------------")
            token_usage_stats = {
                'total_tokens': 0, 
                'per_sample_tokens': [],
                'early_stop_count': 0,
                'early_stop_savings': 0,
                'beam_search_tokens': 0,  
                'beam_wasted_tokens': 0 
            }
            
            sample_token_tracker = {i: 0 for i in range(len(prompts))}
            
            from itertools import chain
            import numpy as np

            class GUARD_Processor:
                """GUARD processor: triggers branching at newline + high entropy points."""
                def __init__(self, tokenizer, max_token_per_call=0, sample_idx=None, 
                           token_stats=None, use_branching=False, beam_entropy_threshold=2.0,
                           entropy_quantile=0.90, min_continuation_tokens=2000):
                    # Trigger points (newline positions)
                    trigger_texts = ["\n\n", ",\n\n", ".\n\n", "]\n\n", ")\n\n",
                                   "]),\n\n", "].\n\n", ").\n\n", ".)\n\n"]
                    # import itertools

                    self.trigger_ids = tokenizer(trigger_texts, add_special_tokens=False).input_ids
                    self.trigger_ids = list(chain.from_iterable(self.trigger_ids))
                    
                    self.tokenizer = tokenizer
                    self.max_token_per_call = max_token_per_call
                    
                    self.entropy_history = []
                    self.max_prob_history = []
                    self.margin_history = []

                    self.sample_idx = sample_idx
                    self.token_stats = token_stats
                    self.early_stopped = False
                    
                    # EOS token
                    self.eos_token_id = None
                    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                        self.eos_token_id = tokenizer.eos_token_id
                    else:
                        for stop_text in ['</s>', '<|endoftext|>', '<|im_end|>']:
                            try:
                                token_ids = tokenizer.encode(stop_text, add_special_tokens=False)
                                if token_ids and len(token_ids) > 0:
                                    self.eos_token_id = token_ids[0]
                                    break
                            except:
                                continue
                    
                    # Statistics (beam trigger points)
                    self.stats_log = []
                    
                    # === Branching related ===
                    self.use_branching = use_branching
                    self.beam_entropy_threshold = beam_entropy_threshold
                    self.entropy_quantile = entropy_quantile
                    self.min_continuation_tokens = min_continuation_tokens
                    self.should_trigger_beam = False  # Flag to trigger branching
                    self.onset_trigger_positions = []  # Record all trigger positions

                def _calculate_uncertainty(self, logits):
                    """Calculate entropy."""
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)
                    entropy = -(probs * log_probs).sum()
                    return entropy.item()
                
                def _get_confidence_metrics(self, logits):
                    """Calculate confidence metrics."""
                    probs = F.softmax(logits, dim=-1)
                    top_2_probs, _ = torch.topk(probs, k=2)
                    max_prob = top_2_probs[0].item()
                    margin = (top_2_probs[0] - top_2_probs[1]).item()
                    return max_prob, margin

                def __call__(self, token_ids, logits):
                    if len(token_ids) < 2:
                        return logits

                    remaining = self.max_token_per_call - len(token_ids)
                    current_length = len(token_ids)
                    
                    if token_ids[-1] in self.trigger_ids:
                        entropy = self._calculate_uncertainty(logits)
                        max_prob, margin = self._get_confidence_metrics(logits)
                        
                        self.entropy_history.append(entropy)
                        self.max_prob_history.append(max_prob)
                        self.margin_history.append(margin)
                        
                        # Check if branching should be triggered
                        if self.use_branching and len(self.entropy_history) >= 5 and remaining > 200:
                            # Use quantile to assess relative uncertainty
                            sorted_entropy = sorted(self.entropy_history)
                            quantile_threshold_idx = int(len(sorted_entropy) * self.entropy_quantile)
                            quantile_threshold = sorted_entropy[quantile_threshold_idx]
                            # Trigger condition: newline + high entropy (> quantile)
                            if entropy > quantile_threshold:
                                self.should_trigger_beam = True
                                self.onset_trigger_positions.append(current_length)
                                
                                # Record statistics
                                self.stats_log.append({
                                    'position': current_length,
                                    'entropy': round(entropy, 4),
                                    'quantile_threshold': round(quantile_threshold, 4),
                                    'max_prob': round(max_prob, 4),
                                    'margin': round(margin, 4),
                                    'triggered_beam': True
                                })
                                
                                print(f"[Beam Trigger] Sample {self.sample_idx} at pos {current_length}, entropy={entropy:.3f} > threshold={quantile_threshold:.3f}")
                                
                                if self.eos_token_id is not None:
                                    logits.fill_(-float("inf"))
                                    logits[self.eos_token_id] = 0.0
                                    return logits
                    
                    return logits

            tok = model.get_tokenizer()
            sampling_params_list = []
            processors = []
            
            for idx in range(len(prompts)):
                lp = GUARD_Processor(
                    tok, 
                    max_token_per_call=args.max_tokens,
                    sample_idx=idx,
                    token_stats=token_usage_stats,
                    use_branching=args.use_branching,
                    beam_entropy_threshold=args.beam_entropy_threshold,
                    entropy_quantile=args.entropy_quantile,
                    min_continuation_tokens=args.min_continuation_tokens
                )
                processors.append(lp)
                sp = SamplingParams(
                    temperature=args.temperature, 
                    max_tokens=args.max_tokens,
                    seed=args.seed,
                    n=1, stop=stop_words,
                    include_stop_str_in_output=True,
                    logits_processors=[lp]
                )
                sampling_params_list.append(sp)


            gen_output = model.generate(prompts, sampling_params_list)

            outputs = []
            remaining_tokens_ = []
            prompts_ = []
            
            for index in range(len(prompts)):
                base_output = gen_output[index].outputs[0].text
                processor = processors[index]
                final_output = base_output
                sample_idx = index

                total_tokens_generated = len(gen_output[index].outputs[0].token_ids)
                
                while total_tokens_generated < args.max_tokens:
                    # Check if branching is needed
                    if processor.should_trigger_beam and args.use_branching:
                        print(f"[Local Beam] Sample {sample_idx} executing beam search...")
                        
                        # Current full prompt
                        current_full_prompt = prompts[index] + final_output
                        
                        # Get onset entropy at trigger point
                        onset_entropy = processor.entropy_history[-1] if processor.entropy_history else 2.0
                        
                        # Define beam diversification strategy: different prompt hints + temperature adjustment
                        beam_configs = [
                            {"prompt_suffix": "", "temp_adjust": 0.0, "desc": "conservative"},  # Beam 0: conservative
                            {"prompt_suffix": "Wait", "temp_adjust": 0.6, "desc": "moderate"},  # Beam 1: moderate exploration
                            {"prompt_suffix": "Let me reconsider: ", "temp_adjust": 1.5, "desc": "explorative"},  # Beam 2: explorative
                        ]
                        # If branching_width > 3, cycle through configs
                        if args.branching_width > len(beam_configs):
                            beam_configs.extend([
                                {"prompt_suffix": "\nAlternatively, I could: ", "temp_adjust": 0.15, "desc": "alternative"},
                                {"prompt_suffix": "\nFrom a different perspective: ", "temp_adjust": 0.12, "desc": "perspective"},
                            ])
                        
                        # Generate multiple candidates
                        candidates = []
                        beam_tokens_used = 0
                        
                        for beam_id in range(args.branching_width):
                            # Get current beam config
                            config = beam_configs[beam_id % len(beam_configs)]
                            beam_specific_prompt = current_full_prompt + config["prompt_suffix"]
                            beam_temperature = args.temperature + config["temp_adjust"]
                            
                            # Create entropy monitor to record this branch's entropy sequence
                            entropy_monitor = EntropyMonitor(
                                tokenizer=processor.tokenizer,
                                sample_idx=sample_idx
                            )
                            
                            local_sp = SamplingParams(
                                temperature=beam_temperature,
                                max_tokens=args.branching_steps,
                                seed=args.seed + sample_idx * 1000 + beam_id if args.seed is not None else None,
                                n=1,
                                stop=stop_words,
                                include_stop_str_in_output=True,
                                logits_processors=[entropy_monitor]
                            )
                            
                            local_output = model.generate([beam_specific_prompt], [local_sp])[0]
                            local_text = local_output.outputs[0].text
                            local_tokens = len(local_output.outputs[0].token_ids)
                            beam_tokens_used += local_tokens
                            
                            # Score based on entropy reduction
                            score = score_branch_by_entropy_reduction(entropy_monitor, onset_entropy)
                            
                            candidates.append({
                                'text': local_text,
                                'prompt_suffix': config["prompt_suffix"],
                                'score': score,
                                'beam_id': beam_id,
                                'tokens': local_tokens,
                                'onset_entropy': onset_entropy,
                                'config': config
                            })
                        
                        # Select best (largest entropy reduction)
                        best_candidate = max(candidates, key=lambda x: x['score'])
                        config_desc = best_candidate.get('config', {}).get('desc', 'unknown')
                        temp_used = args.temperature + best_candidate.get('config', {}).get('temp_adjust', 0.0)
                        print(f"  → Selected beam {best_candidate['beam_id']} ({config_desc}, T={temp_used:.2f})")
                        
                        # Append to final_output
                        final_output = final_output + best_candidate['prompt_suffix'] + best_candidate['text']
                        total_tokens_generated += best_candidate['tokens']
                        
                        # Statistics: all beam tokens vs actually used tokens
                        token_usage_stats['beam_search_tokens'] += beam_tokens_used  # All beams
                        wasted_tokens = beam_tokens_used - best_candidate['tokens']  # Unused beams
                        token_usage_stats['beam_wasted_tokens'] += wasted_tokens
                        print(f"  → Beam token usage: total={beam_tokens_used}, used={best_candidate['tokens']}, wasted={wasted_tokens}")
                        
                        # Reset flag
                        processor.should_trigger_beam = False
                        
                        # === Critical: continue generating remaining_tokens ===
                        remaining_tokens = args.max_tokens - total_tokens_generated
                        if remaining_tokens > processor.min_continuation_tokens:  
                            continue_sp = SamplingParams(
                                temperature=args.temperature,
                                max_tokens=remaining_tokens,
                                seed=args.seed + sample_idx,
                                n=1,
                                stop=stop_words,
                                include_stop_str_in_output=True,
                                logits_processors=[processor]
                            )
                            
                            continue_output = model.generate([prompts[index] + final_output], [continue_sp])[0]
                            continue_text = continue_output.outputs[0].text
                            continue_tokens = len(continue_output.outputs[0].token_ids)
                            final_output = final_output + continue_text
                            total_tokens_generated += continue_tokens
                        else:
                            print(f"  → Insufficient remaining tokens, stopping")
                            break
                    else:
                        # No trigger, exit loop
                        break
                
                outputs.append(final_output)
                prompts_.append(prompts[index] + final_output)
                remaining_tokens_.append(args.max_tokens - total_tokens_generated)
                
                # Record token usage for this sample
                sample_token_tracker[index] = total_tokens_generated

            assert len(prompts_) == len(remaining_tokens_)
            stop_at_wait_words = ["Wait", ".Wait", "Wait, ", "Wait,"]
           
            active_traj = np.ones(len(prompts_))

            # Continue generation in while loop
            while 1:
                sampling_params_list = [SamplingParams(
                                    temperature=args.temperature,
                                    max_tokens=max(1, remaining_token_), seed=args.seed,
                                    n=1, stop=stop_words + stop_at_wait_words,
                                    include_stop_str_in_output=True,) for remaining_token_ in remaining_tokens_]
                
                input_prompts_while = [q for f, q in zip(active_traj, prompts_) if f]
                input_sampling_params_list = [q for f, q in zip(active_traj, sampling_params_list) if f]
                
                if len(input_prompts_while) == 0:
                    break
                
                gen_output = model.generate(input_prompts_while, input_sampling_params_list)
                
                i = 0
                for index in range(len(prompts_)):
                    if active_traj[index] == 1:
                        response = gen_output[i].outputs[0].text
                        
                        response = response.replace("\nWait", "</think>")
                        response = response.replace("Wait", "</think>")

                        outputs[index] += response
                        prompts_[index] += response
                        tokens_used_this_round = len(gen_output[i].outputs[0].token_ids)
                        remaining_tokens_[index] = max(1, remaining_tokens_[index] - tokens_used_this_round)
                        
                        sample_token_tracker[index] += tokens_used_this_round

                        if response.endswith(tuple(stop_words)) or remaining_tokens_[index] == 1 or response == "":
                            active_traj[index] = 0
                        i += 1
                
                if sum(active_traj) == 0:
                    break
            
            for sample_idx, tokens_used in sample_token_tracker.items():
                token_usage_stats['total_tokens'] += tokens_used
                token_usage_stats['per_sample_tokens'].append({
                    'sample_idx': sample_idx,
                    'tokens_used': tokens_used
                })
            

            avg_tokens = token_usage_stats['total_tokens'] / len(prompts) if len(prompts) > 0 else 0
            if token_usage_stats['beam_search_tokens'] > 0:
                beam_used = token_usage_stats['beam_search_tokens'] - token_usage_stats['beam_wasted_tokens']
                beam_wasted = token_usage_stats['beam_wasted_tokens']
                beam_total = token_usage_stats['beam_search_tokens']
                print(f":")
                print(f"  - : {beam_total}")
                print(f"  - : {beam_used}")
                print(f"  - : {beam_wasted} ({beam_wasted/beam_total*100 if beam_total > 0 else 0:.1f}%)")
        else:
            outputs = model.generate(prompts=prompts, sampling_params=sampling_params)

    # Initialize token statistics (only for default branch, use_s1/use_wait_more/use_branching already have them)
    if not (args.use_s1 or args.use_wait_more or args.use_branching):
        token_usage_stats = {
            'total_tokens': 0,
            'per_sample_tokens': [],
            'early_stop_count': 0,
            'early_stop_savings': 0,
            'beam_search_tokens': 0,
            'beam_wasted_tokens': 0
        }
    
    results = []
    if not (args.use_s1 or args.use_wait_more or args.use_branching):
        for index, output in enumerate(outputs):
            attempts = []
            for ith_output in output.outputs:
                attempts.append(ith_output.text)
            results.append(attempts)
    else:
        for output in outputs:
            results.append([output])

    output_token_lengths = []
    for idx_prompt, full_output in enumerate(results):
        full_output = full_output[0]
        current_token_length = len(tokenizer.encode(full_output)) if tokenizer else len(full_output.split())
        output_token_lengths.append(current_token_length)
        
        # Only for default branch (use_s1/use_wait_more/use_branching already counted)
        if not (args.use_s1 or args.use_wait_more or args.use_branching):
            token_usage_stats['total_tokens'] += current_token_length
            token_usage_stats['per_sample_tokens'].append({
                'sample_idx': idx_prompt,
                'tokens_used': current_token_length
            })
    
    average_token_length = sum(output_token_lengths) / len(output_token_lengths) if output_token_lengths else 0
    print(f"Average Token Length: {average_token_length:.2f}")

    combined_results = [
        (
            outputs_list,
            [extract_code(output) for output in outputs_list],
        )
        for outputs_list in results
    ]

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            benchmark, combined_results
        )
    ]

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as f:
        json.dump(save_results, f, indent=4)


    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
    generations = [extracted for _, extracted in combined_results]

    metrics = codegen_metrics(
        eval_samples,
        generations,
        num_process_evaluate=12,
        timeout=10,
    )

    print(metrics[0]["pass@1"])

    graded = extract_instance_results(metrics[1])
    metadatas = metrics[2]
    save_eval_results = [
        instance.insert_output_evaluation(
            outputs_list, extracted_list, graded_list, metadata=meta
        )
        for instance, (outputs_list, extracted_list), graded_list, meta in zip(
            benchmark, combined_results, graded, metadatas
        )
    ]

    # Determine method name
    method_name = "default"
    if args.use_s1:
        method_name = "use_s1"
    elif args.use_wait_more:
        method_name = "use_wait_more"
    elif args.use_branching:
        method_name = "use_branching"
    
    metrics_with_tokens = {
        "method": method_name,
        "pass@1": metrics[0]["pass@1"],
        "token_usage": {
            'total_tokens': token_usage_stats['total_tokens'],
            'avg_tokens_per_sample': token_usage_stats['total_tokens'] / len(benchmark) if len(benchmark) > 0 else 0,
            'early_stop_count': token_usage_stats.get('early_stop_count', 0),
            'early_stop_rate': token_usage_stats.get('early_stop_count', 0) / len(benchmark) if len(benchmark) > 0 else 0,
            'tokens_saved_by_early_stop': token_usage_stats.get('early_stop_savings', 0),
            'beam_search_all_tokens': token_usage_stats.get('beam_search_tokens', 0),
            'beam_search_wasted_tokens': token_usage_stats.get('beam_wasted_tokens', 0),
            'beam_search_used_tokens': token_usage_stats.get('beam_search_tokens', 0) - token_usage_stats.get('beam_wasted_tokens', 0),
            'per_sample_details': token_usage_stats['per_sample_tokens']
        }
    }
    
    with open(os.path.join(args.save_dir, "metrics.jsonl"), "w") as f:
        json.dump(metrics, f, indent=4)
    

    token_stats_filename = f"metrics_with_tokens_{method_name}.json"
    with open(os.path.join(args.save_dir, token_stats_filename), "w") as f:
        json.dump(metrics_with_tokens, f, indent=4)
    


    with open(os.path.join(args.save_dir, "code_eval.jsonl"), "w") as f:
        json.dump(save_eval_results, f, indent=4)
    
    # Print final token statistics
    print(f"\n=== Final Token Statistics Report ===")
    print(f"Total Token Consumption: {token_usage_stats['total_tokens']}")
    print(f"Average per sample: {token_usage_stats['total_tokens'] / len(benchmark) if len(benchmark) > 0 else 0:.1f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--peft",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--release",
        type=str,
        default="release_v1",
    )
    parser.add_argument(
        "--remove_bos",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--logit_adjustment",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--logit_adjustment_tokens",
        type=str,
        nargs="*",
        default=[]
    )
    parser.add_argument(
        "--logit_adjustment_value",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--logit_adjustment_max_len",
        type=int,
        default=-1
    )
    parser.add_argument("--use_wait_more", action="store_true")
    parser.add_argument("--use_s1", action="store_true")
    parser.add_argument("--use_cod", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.4)
    parser.add_argument("--seed", type=int, default=42)
    # Branching parameters
    parser.add_argument("--use_branching", action="store_true",
                       help="Enable branching search at high-entropy points")
    parser.add_argument("--branching_width", type=int, default=3,
                       help="Number of branches for local expansion")
    parser.add_argument("--branching_steps", type=int, default=100,
                       help="Number of tokens for branch expansion")
    parser.add_argument("--beam_entropy_threshold", type=float, default=2.0,
                       help="[DEPRECATED] Not used. Trigger uses entropy quantile of history")
    parser.add_argument("--entropy_quantile", type=float, default=0.90,
                       help="Entropy quantile threshold for triggering branching (default: 0.90)")
    parser.add_argument("--min_continuation_tokens", type=int, default=2000,
                       help="Minimum remaining tokens required for continuation after branching (default: 2000)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for sampling")
    
    args = parser.parse_args()

    if args.logit_adjustment:
        name = "_".join(args.logit_adjustment_tokens)+f"_value_{args.logit_adjustment_value}"
        if args.logit_adjustment_max_len>0:
            name += f"_first{args.logit_adjustment_max_len}"
        
        args.save_dir = os.path.join(args.save_dir, "logit-adjustment", name)
        
    mp.set_start_method("spawn", force=True)
    main(args)