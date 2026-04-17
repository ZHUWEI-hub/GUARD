import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
import re
from sympy import sympify, SympifyError

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions
from itertools import chain
import json

# GUARD: Guided Uncertainty-Aware Reasoning with Decision control


class EntropyMonitor:
    """Monitors entropy sequence during generation for uncertainty-aware branching.
    
    Tracks uncertainty dynamics of candidate branches and implements early termination
    when answer completion markers are detected.
    """
    def __init__(self, tokenizer=None, ground_truth=None, data_name="math", sample_idx=None):
        self.entropy_sequence = []
        self.max_prob_sequence = []
        
        # Early termination related
        self.tokenizer = tokenizer
        self.ground_truth = ground_truth
        self.data_name = data_name
        self.sample_idx = sample_idx
        self.early_stopped = False
        
        # End-of-sequence token
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
        
        # Boxed answer token detection
        self.boxed_token_ids = set()
        if tokenizer:
            for text in ['boxed', '\\boxed', 'boxed{', '\\boxed{', 'BOXED', 'Boxed']:
                try:
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    self.boxed_token_ids.update(ids)
                except:
                    pass
    
    def __call__(self, token_ids, logits):
        """Called at each token generation step to record entropy and check early termination."""
        # Early termination: stop when boxed answer detected
        if not self.early_stopped and len(token_ids) > 20:
            try:
                recent_tokens = token_ids[-10:] if len(token_ids) >= 10 else token_ids
                has_boxed_token = any(tid in self.boxed_token_ids for tid in recent_tokens)
                
                if has_boxed_token:
                    self.early_stopped = True
                    print(f"  [Beam Early Stop] Sample {self.sample_idx} branch detected boxed answer")
                    
                    if self.eos_token_id is not None:
                        logits.fill_(-float("inf"))
                        logits[self.eos_token_id] = 0.0
                        return logits
            except Exception as e:
                pass
        
        # Compute token-wise entropy
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum().item()
        max_prob = probs.max().item()
        
        self.entropy_sequence.append(entropy)
        self.max_prob_sequence.append(max_prob)
        
        return logits  # Only record, do not modify logits


def score_branch_by_entropy_reduction(entropy_monitor, onset_entropy):
    """
    
    Args:
        entropy_monitor: EntropyMonitor object tracking this branch entropy sequence
        onset_entropy: Entropy value at branching trigger point
    
    Returns:
    """
    if len(entropy_monitor.entropy_sequence) == 0:
        return -float('inf')
    
    # Compute average entropy of this branch
    avg_entropy = sum(entropy_monitor.entropy_sequence) / len(entropy_monitor.entropy_sequence)
    
    # Entropy reduction = onset entropy - branch average
    # Larger reduction means path increases confidence
    entropy_drop = onset_entropy - avg_entropy
    
    return entropy_drop


def verify_answer_for_early_stop(current_text, ground_truth, data_name="math"):
    """
    1. Use original extract_answer logic (maintain consistency)
    2. Additionally handle whitespace differences（"p - q" vs "p-q"）
    
    Returns: (is_correct, extracted_answer)
    """
    # Check if boxed answer is present
    if "boxed" not in current_text.lower():
        return False, None
    
    try:
        # Use original extract_answer function (consistent with final evaluation)
        pred_answer = extract_answer(current_text, data_name)
        gt_answer = extract_answer(ground_truth, data_name)
        
        if pred_answer is None or gt_answer is None:
            return False, None
        
        # Try direct match first (highest priority)
        if pred_answer == gt_answer:
            return True, pred_answer
        
        # If no match, try matching after removing spaces
        pred_no_space = str(pred_answer).replace(' ', '')
        gt_no_space = str(gt_answer).replace(' ', '')
        
        is_correct = (pred_no_space == gt_no_space)
        return is_correct, pred_answer
        
    except Exception as e:
        # Extraction failed, return False
        return False, None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="aime24", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--use_s1", action="store_true")
    
    # === Branching parameters ===
    parser.add_argument("--use_branching", action="store_true",
                       help="Enable local beam search at high-entropy points")
    parser.add_argument("--branching_width", type=int, default=3,
                       help="Number of beams for local expansion")
    parser.add_argument("--branching_steps", type=int, default=100,
                       help="Number of tokens for local expansion")
    parser.add_argument("--beam_entropy_threshold", type=float, default=2.0,
                       help="[DEPRECATED] Not used. Trigger uses entropy quantile of history")
    parser.add_argument("--entropy_quantile", type=float, default=0.90,
                       help="Entropy quantile threshold for triggering branching (default: 0.90)")
    parser.add_argument("--min_continuation_tokens", type=int, default=2000,
                       help="Minimum remaining tokens required for continuation after branching (default: 2000)")
    
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    print("Args: ", args)

    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
            # gpu_memory_utilization=0.99,
            max_model_len=args.max_tokens_per_call + 2000
        )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
            # gpu_memory_utilization=0.95,
            max_model_len=args.max_tokens_per_call + 2000
        )

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        if args.use_branching:
            print(f"branch")
            print(f"  - {args.branching_width}")
            print(f"  - {args.branching_steps}")
        else:
            print(f"no--use_branching")
        results.append(main_with_branching(llm, tokenizer, data_name, args))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main_with_branching(llm, tokenizer, data_name, args):
    """Main inference with GUARD: uncertainty-aware branching control."""
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level", "type", "unit", "solution_type", "choices", "solution",
            "ques_type", "ans_type", "answer_type", "dataset", "subfield",
            "filed", "theorem", "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # Prepare input prompts
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    
    # Save original prompts for later code extraction
    original_input_prompts = input_prompts.copy()
    
    # Define stop words
    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "<|end▁of▁sentence|>", "<｜end▁of▁sentence｜>"]
    
    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")
    
    print("=" * 50)
    print("Running GUARD inference with uncertainty-aware branching...")
    print("=" * 50)
    
    # Use original prompts directly
    remain_prompts = [(i, prompt) for i, prompt in enumerate(input_prompts)]
    end_prompts = []
    
    # start inference
    start_time = time.time()
    
    # Statistics for control tokens (model-generated, not inserted)
    control_word_stats = {"Wait": 0, "</think>": 0}
    
    # Branching statistics
    branching_stats = {
        'total_triggers': 0,
        'per_sample_triggers': [],
        'beam_decisions': []
    }
    
    # Token usage statistics
    token_usage_stats = {
        'total_tokens': 0,  # 
        'per_sample_tokens': [],
        'early_stop_count': 0,
        'early_stop_savings': 0,
        'beam_search_tokens': 0,  
        'beam_wasted_tokens': 0  
    }
    

    sample_gt_map = {i: samples[i]['gt'] for i in range(len(samples))}
    
  
    sample_token_tracker = {i: 0 for i in range(len(samples))}
    

    sample_onset_trigger_count = {i: 0 for i in range(len(samples))}
    
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        if args.use_vllm:
            outputs = []
            output_token_lengths = []

            # ============ Local Beam Search Processor ============
            class GUARD_Processor:
                def __init__(self, tokenizer, max_token_per_call=0,
                             ground_truth=None, sample_idx=None, token_stats=None, 
                             data_name="math", use_branching=False, beam_entropy_threshold=2.0,
                             entropy_quantile=0.90, min_continuation_tokens=2000):
                    trigger_texts = ["\n\n", ",\n\n", ".\n\n", "]\n\n", ")\n\n",
                                   "]),\n\n", "].\n\n", ").\n\n", ".)\n\n"]
                    self.trigger_ids = tokenizer(trigger_texts, add_special_tokens=False).input_ids
                    self.trigger_ids = list(chain.from_iterable(self.trigger_ids))
                    
                    self.tokenizer = tokenizer
                    self.max_token_per_call = max_token_per_call

                    self.entropy_history = []
                    self.max_prob_history = []
                    self.margin_history = []

                    self.ground_truth = ground_truth
                    self.sample_idx = sample_idx
                    self.token_stats = token_stats
                    self.data_name = data_name
                    self.early_stopped = False
                    
                    # End-of-sequence token
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
                    
                    # Boxed answer token detection
                    self.boxed_token_ids = set()
                    for text in ['boxed', '\\boxed', 'boxed{', '\\boxed{', 'BOXED', 'Boxed']:
                        try:
                            ids = tokenizer.encode(text, add_special_tokens=False)
                            self.boxed_token_ids.update(ids)
                        except:
                            pass
                    
                    self.stats_log = []
                    
                    self.use_branching = use_branching
                    self.beam_entropy_threshold = beam_entropy_threshold
                    self.entropy_quantile = entropy_quantile
                    self.min_continuation_tokens = min_continuation_tokens
                    self.should_trigger_beam = False 
                    self.onset_trigger_positions = [] 

                def _calculate_uncertainty(self, logits):
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)
                    entropy = -(probs * log_probs).sum()
                    return entropy.item()
                
                def _get_confidence_metrics(self, logits):
                    probs = F.softmax(logits, dim=-1)
                    top_2_probs, _ = torch.topk(probs, k=2)
                    max_prob = top_2_probs[0].item()
                    margin = (top_2_probs[0] - top_2_probs[1]).item()
                    return max_prob, margin
                


                def __call__(self, token_ids, logits):
                    # Early termination: stop when boxed answer detected
                    if not self.early_stopped and len(token_ids) > 20:
                        try:
                            recent_tokens = token_ids[-10:] if len(token_ids) >= 10 else token_ids
                            has_boxed_token = any(tid in self.boxed_token_ids for tid in recent_tokens)
                            
                            if has_boxed_token:
                                self.early_stopped = True
                                remaining_tokens = self.max_token_per_call - len(token_ids)
                                
                                if self.token_stats is not None:
                                    self.token_stats['early_stop_count'] += 1
                                    self.token_stats['early_stop_savings'] += remaining_tokens
                                
                                print(f"[Early Stop] Sample {self.sample_idx} detected boxed answer, saved {remaining_tokens} tokens")
                                
                                if self.eos_token_id is not None:
                                    logits.fill_(-float("inf"))
                                    logits[self.eos_token_id] = 0.0
                                    return logits
                        except Exception as e:
                            pass

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
                        
                        if self.use_branching and len(self.entropy_history) >= 5 and remaining > 200:
                            sorted_entropy = sorted(self.entropy_history)
                            quantile_threshold_idx = int(len(sorted_entropy) * self.entropy_quantile)
                            quantile_threshold = sorted_entropy[quantile_threshold_idx]

                            if entropy > quantile_threshold:
                                self.should_trigger_beam = True
                                self.onset_trigger_positions.append(current_length)
                                
 
                                self.stats_log.append({
                                    'position': current_length,
                                    'entropy': round(entropy, 4),
                                    'quantile_threshold': round(quantile_threshold, 4),
                                    'max_prob': round(max_prob, 4),
                                    'margin': round(margin, 4),
                                    'triggered_beam': True
                                })
                                
                                print(f"[Beam Trigger] Sample {self.sample_idx} at pos {current_length}, entropy={entropy:.3f} > p95={quantile_threshold:.3f}")
                                
                                if self.eos_token_id is not None:
                                    logits.fill_(-float("inf"))
                                    logits[self.eos_token_id] = 0.0
                                    return logits

                    
                    return logits

            # Create processor for each prompt
            tok = llm.get_tokenizer()
            sampling_params_list = []
            processors = []
            
            for idx, (sample_idx, _) in enumerate(current_prompts):
                ground_truth = sample_gt_map.get(sample_idx, None)
                
                lp = GUARD_Processor(
                    tok, 
                    max_token_per_call=args.max_tokens_per_call,
                    ground_truth=ground_truth,
                    sample_idx=sample_idx,
                    token_stats=token_usage_stats,
                    data_name=data_name,
                    use_branching=args.use_branching,
                    beam_entropy_threshold=args.beam_entropy_threshold,
                    entropy_quantile=args.entropy_quantile,
                    min_continuation_tokens=args.min_continuation_tokens
                )
                processors.append(lp)
                sp = SamplingParams(
                    temperature=args.temperature, top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    seed=args.seed,
                    n=1, stop=stop_words,
                    include_stop_str_in_output=True,
                    logits_processors=[lp]
                )
                sampling_params_list.append(sp)

            gen_output = llm.generate(prompts, sampling_params_list)

            output_responses = []
            remaining_tokens_ = []
            prompts_ = []
            current_round_tokens = []
            
            for index in range(len(prompts)):
                base_output = gen_output[index].outputs[0].text
                processor = processors[index]
                final_output = base_output
                sample_idx = current_prompts[index][0]

                onset_trigger_count = 0
                total_tokens_generated = len(gen_output[index].outputs[0].token_ids)
                
                while total_tokens_generated < args.max_tokens_per_call:

                    if processor.should_trigger_beam and args.use_branching:
                        onset_trigger_count += 1
                        print(f"[Local Beam #{onset_trigger_count}] Sample {sample_idx} executing beam search...")

                        current_full_prompt = prompts[index] + final_output
                        
                        onset_entropy = processor.entropy_history[-1] if processor.entropy_history else 2.0
                        
                        beam_configs = [
                            {"prompt_suffix": "", "temp_adjust": 0.0, "desc": "conservative"},  
                            {"prompt_suffix": "Wait", "temp_adjust": 0.6, "desc": "moderate"}, 
                            {"prompt_suffix": "Let me reconsider: ", "temp_adjust": 1.5, "desc": "explorative"},  
                        ]

                        if args.branching_width > len(beam_configs):
                            beam_configs.extend([
                                {"prompt_suffix": "\nAlternatively, I could: ", "temp_adjust": 0.15, "desc": "alternative"},
                                {"prompt_suffix": "\nFrom a different perspective: ", "temp_adjust": 0.12, "desc": "perspective"},
                            ])

                        candidates = []
                        beam_tokens_used = 0
                        
                        for branch_id in range(args.branching_width):

                            config = beam_configs[branch_id % len(beam_configs)]
                            beam_specific_prompt = current_full_prompt + config["prompt_suffix"]
                            beam_temperature = config["temp_adjust"]
                            

                            entropy_monitor = EntropyMonitor(
                                tokenizer=processor.tokenizer,
                                ground_truth=processor.ground_truth,
                                data_name=processor.data_name,
                                sample_idx=sample_idx
                            )
                            
                            local_sp = SamplingParams(
                                temperature=beam_temperature,  
                                top_p=args.top_p,
                                max_tokens=args.branching_steps,
                                seed=args.seed + sample_idx * 1000 + branch_id if args.seed is not None else None,  
                                n=1,
                                stop=stop_words,
                                include_stop_str_in_output=True,
                                logits_processors=[entropy_monitor]  
                            )
                            
                            local_output = llm.generate([beam_specific_prompt], [local_sp])[0]  
                            local_text = local_output.outputs[0].text
                            local_tokens = len(local_output.outputs[0].token_ids)
                            beam_tokens_used += local_tokens
                            
                            score = score_branch_by_entropy_reduction(entropy_monitor, onset_entropy)
                            
                            avg_entropy = sum(entropy_monitor.entropy_sequence) / len(entropy_monitor.entropy_sequence) if entropy_monitor.entropy_sequence else 0
                            final_entropy = entropy_monitor.entropy_sequence[-1] if entropy_monitor.entropy_sequence else 0
                            
                            if entropy_monitor.early_stopped:
                                score += 10.0  
                            
                            candidates.append({
                                'text': local_text,
                                'prompt_suffix': config["prompt_suffix"], 
                                'score': score,
                                'branch_id': branch_id,
                                'tokens': local_tokens,
                                'onset_entropy': onset_entropy,
                                'avg_entropy': avg_entropy,
                                'final_entropy': final_entropy,
                                'entropy_drop': score if not entropy_monitor.early_stopped else score - 10.0,
                                'early_stopped': entropy_monitor.early_stopped,
                                'config': config  
                            })
                        

                        best_candidate = max(candidates, key=lambda x: x['score'])
                        early_stop_msg = " [EARLY STOP ✓]" if best_candidate.get('early_stopped', False) else ""
                        config_desc = best_candidate.get('config', {}).get('desc', 'unknown')
                        temp_used = args.temperature + best_candidate.get('config', {}).get('temp_adjust', 0.0)
                        prompt_suffix_display = best_candidate.get('prompt_suffix', '')
                        prompt_suffix_msg = f" [suffix: '{prompt_suffix_display}']" if prompt_suffix_display else ""
                        print(f"  → Selected beam {best_candidate['branch_id']} ({config_desc}, T={temp_used:.2f}){prompt_suffix_msg}: "
                              f"entropy_drop={best_candidate['entropy_drop']:.3f} "
                              f"(trigger={best_candidate['onset_entropy']:.2f} → avg={best_candidate['avg_entropy']:.2f})"
                              f"{early_stop_msg}")
                        
    
                        final_output = final_output + best_candidate['prompt_suffix'] + best_candidate['text']
                        total_tokens_generated += best_candidate['tokens']
                        
                     
                        token_usage_stats['beam_search_tokens'] += beam_tokens_used  
                        wasted_tokens = beam_tokens_used - best_candidate['tokens']  
                        token_usage_stats['beam_wasted_tokens'] += wasted_tokens
                        sample_onset_trigger_count[sample_idx] += 1
                        branching_stats['total_triggers'] += 1
                        branching_stats['beam_decisions'].append({
                            'sample_idx': sample_idx,
                            'trigger_num': onset_trigger_count,
                            'position': len(final_output) - len(best_candidate['prompt_suffix']) - len(best_candidate['text']),  # Position calculation
                            'onset_entropy': best_candidate['onset_entropy'],
                            'candidates': [{
                                'branch_id': c['branch_id'], 
                                'entropy_drop': c['entropy_drop'],
                                'avg_entropy': c['avg_entropy'],
                                'final_entropy': c['final_entropy'],
                                'early_stopped': c.get('early_stopped', False),
                                'strategy': c.get('config', {}).get('desc', 'unknown'),
                                'temperature': c.get('config', {}).get('temp_adjust', 0.0),
                                'prompt_suffix': c.get('prompt_suffix', '')  
                            } for c in candidates],
                            'selected_beam': best_candidate['branch_id'],
                            'selected_entropy_drop': best_candidate['entropy_drop'],
                            'selected_early_stopped': best_candidate.get('early_stopped', False),
                            'selected_strategy': best_candidate.get('config', {}).get('desc', 'unknown'),
                            'selected_temperature': best_candidate.get('config', {}).get('temp_adjust', 0.0),
                            'selected_prompt_suffix': best_candidate.get('prompt_suffix', '')  
                        })
                        
                      
                        processor.should_trigger_beam = False
                        
                       
                        if best_candidate.get('early_stopped', False):
                            print(f"  → Beam found correct answer, stopping generation for sample {sample_idx}")
                            break
                        
                       
                        remaining_tokens = args.max_tokens_per_call - total_tokens_generated
                        if remaining_tokens > processor.min_continuation_tokens:  
                            continue_sp = SamplingParams(
                                temperature=args.temperature,
                                top_p=args.top_p,
                                max_tokens=remaining_tokens, 
                                seed=args.seed + sample_idx,
                                n=1,
                                stop=stop_words,
                                include_stop_str_in_output=True,
                                logits_processors=[processor] 
                            )
                            
                            continue_output = llm.generate([prompts[index] + final_output], [continue_sp])[0]
                            continue_text = continue_output.outputs[0].text
                            continue_tokens = len(continue_output.outputs[0].token_ids)
                            final_output = final_output + continue_text
                            total_tokens_generated += continue_tokens
                       
                            
        
                            
                            
                        else:
                           
                            break
                        
                    else:
                       
                        break
                
                if onset_trigger_count > 0:
                    print(f"[Summary] Sample {sample_idx} triggered {onset_trigger_count} beam searches")
                
                output_responses.append(final_output)
                prompts_.append(prompts[index] + final_output)
                remaining_tokens_.append(args.max_tokens_per_call - total_tokens_generated)
                current_round_tokens.append(total_tokens_generated)
                
          
                sample_token_tracker[sample_idx] = total_tokens_generated
                
                output_text = final_output
                for word in control_word_stats.keys():
                    control_word_stats[word] += output_text.count(word)

            assert len(prompts_) == len(remaining_tokens_)
            stop_at_wait_words = ["Wait", ".Wait", "Wait, ", "Wait,"]
           
            active_traj = np.ones(len(prompts_))

       
            for index in range(len(prompts_)):
                if processors[index].early_stopped:
                    active_traj[index] = 0

      
            while 1:
                sampling_params_list = [SamplingParams(
                                    temperature=args.temperature, top_p=args.top_p,
                                    max_tokens=max(1, remaining_token_), seed=args.seed,
                                    n=1, stop=stop_words + stop_at_wait_words,
                                    include_stop_str_in_output=True,) for remaining_token_ in remaining_tokens_]
                
                input_prompts_while = [q for f, q in zip(active_traj, prompts_) if f]
                input_sampling_params_list = [q for f, q in zip(active_traj, sampling_params_list) if f]
                
                if len(input_prompts_while) == 0:
                    break
                
                gen_output = llm.generate(input_prompts_while, input_sampling_params_list)
                
                i = 0
                for index in range(len(prompts_)):
                    if active_traj[index] == 1:
                        response = gen_output[i].outputs[0].text
                        
                        response = response.replace("\nWait", "</think>")
                        response = response.replace("Wait", "</think>")

                        output_responses[index] += response
                        prompts_[index] += response
                        tokens_used_this_round = len(gen_output[i].outputs[0].token_ids)
                        current_round_tokens[index] += tokens_used_this_round
                        remaining_tokens_[index] = max(1, remaining_tokens_[index] - tokens_used_this_round)
                        
                        sample_idx = current_prompts[index][0]
                        sample_token_tracker[sample_idx] += tokens_used_this_round

                        # Early termination: stop when boxed answer detected
                        if "boxed" in output_responses[index].lower():
                            if not processors[index].early_stopped:
                                processors[index].early_stopped = True
                                token_usage_stats['early_stop_count'] += 1
                                token_usage_stats['early_stop_savings'] += remaining_tokens_[index]
                                print(f"[Early Stop] Sample {sample_idx} detected boxed answer in continuation")
                            active_traj[index] = 0
                            i += 1
                            continue

                        if response.endswith(tuple(stop_words)) or remaining_tokens_[index] == 1 or response == "":
                            active_traj[index] = 0
                        i += 1
                
                if sum(active_traj) == 0:
                    break

            for idx_prompt, full_output in enumerate(output_responses):
                outputs.append(full_output)
                current_token_length = len(tok.encode(full_output)) if tok else len(full_output.split())
                output_token_lengths.append(current_token_length)
            
            avg_token_length = sum(output_token_lengths) / len(output_token_lengths) if output_token_lengths else 0
            print(f"Average Token Length: {avg_token_length:.2f}")
            print(f"Control Word Usage: {control_word_stats}")
            if args.use_branching:
                print(f"Local Beam Triggers: {branching_stats['total_triggers']}")
            
           
      
        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    end_prompts = sorted(end_prompts, key=lambda x: x[0])
    
  
    for sample_idx, tokens_used in sample_token_tracker.items():
        token_usage_stats['total_tokens'] += tokens_used
        token_usage_stats['per_sample_tokens'].append({
            'sample_idx': sample_idx,
            'tokens_used': tokens_used,
            'onset_triggers': sample_onset_trigger_count.get(sample_idx, 0)
        })

  
    for sample_idx, count in sample_onset_trigger_count.items():
        if count > 0:
            branching_stats['per_sample_triggers'].append({
                'sample_idx': sample_idx,
                'trigger_count': count
            })

    # remove input_prompt from end_prompt
    codes = []
    assert len(original_input_prompts) == len(end_prompts)
    for i in range(len(original_input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(original_input_prompts[i])[-1].strip()
        
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)
    
    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    
    time_use = time.time() - start_time

    # put results back
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A", "B", "C", "D", "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)
    
    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )
    result_json["control_word_stats"] = control_word_stats
    
    # Token usage statistics
    result_json["token_usage"] = {
        'total_tokens': token_usage_stats['total_tokens'],  # Total tokens consumed
        'avg_tokens_per_sample': token_usage_stats['total_tokens'] / len(samples) if len(samples) > 0 else 0,
        'early_stop_count': token_usage_stats['early_stop_count'],
        'early_stop_rate': token_usage_stats['early_stop_count'] / len(samples) if len(samples) > 0 else 0,
        'tokens_saved_by_early_stop': token_usage_stats['early_stop_savings'],
        'beam_search_all_tokens': token_usage_stats['beam_search_tokens'],  # All branch tokens (including unselected)
        'beam_search_wasted_tokens': token_usage_stats['beam_wasted_tokens'],  # Unselected branch tokens
        'beam_search_used_tokens': token_usage_stats['beam_search_tokens'] - token_usage_stats['beam_wasted_tokens'],  # Actually used branch tokens
        'per_sample_details': token_usage_stats['per_sample_tokens']
    }
    
    # Branching statistics
    result_json["branching_stats"] = branching_stats

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    
  
    if args.use_branching and branching_stats['total_triggers'] > 0:
        beam_stats_file = out_file.replace(".jsonl", "_beam_stats.json")
        beam_data = {
            'branching_stats': branching_stats,
            'hyperparameters': {
                'method': 'branching_search_with_diversity',
                'num_branches': args.branching_width,
                'beam_steps': args.branching_steps,
                'trigger_logic': 'Newline + entropy > 95th percentile (relative to history)',
                'scoring': 'Entropy drop (onset_entropy - avg_branch_entropy)',
                'diversity_strategy': 'Combined (seed + prompt + temperature)',
                'seed_strategy': 'args.seed + sample_idx * 1000 + branch_id',
                'temperature_range': f'{args.temperature} ~ {args.temperature + 0.2}',
                'prompt_diversity': 'Different exploration prompts per beam'
            }
        }
        with open(beam_stats_file, "w", encoding='utf-8') as f:
            json.dump(beam_data, f, indent=2, ensure_ascii=False)
        
        if token_usage_stats.get('beam_search_tokens', 0) > 0:
            print(f" {token_usage_stats['beam_search_tokens']}")
            avg_per_sample = token_usage_stats['beam_search_tokens'] / len(samples) if len(samples) > 0 else 0
            print(f"{avg_per_sample:.1f}")
    
   
    if args.use_branching:
        beam_used = token_usage_stats['beam_search_tokens'] - token_usage_stats['beam_wasted_tokens']
        beam_wasted = token_usage_stats['beam_wasted_tokens']
        beam_total = token_usage_stats['beam_search_tokens']
        
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)






