##################
# https://github.com/allenai/reward-bench/blob/main/scripts/run_rm.py
##################

import argparse
import logging
import os
import sys
import re

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from datasets import Dataset
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor, as_completed

from models import API_MODEL_LIST, REWARD_MODEL_CONFIG, PRM_MODEL_CONFIG
from models.api import chat_completion_anthropic, chat_completion_openai
from utils.rewardbench_utils import (
    check_tokenizer_chat_template,
    torch_dtype_mapping,
)
from utils.utils import load_json, save_json
from utils.conversation import get_conv_template
from prompts import RewardModelPromptHandler

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="path to model")
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer to model")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument("--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length of RM inputs (passed to pipeline)")
    parser.add_argument("--not_quantized", action="store_true", help="disable quantization for models that are quantized by default")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32", "float64"], help="PyTorch dtype (default: float16)")
    parser.add_argument( "--model_type", type=str, default="generative", choices=["generative", "classifier", "prm"], help="Reward Model Type")
    parser.add_argument("--input_path", type=str, default=None, help="inference from local dataset")
    parser.add_argument("--save_path", type=str, default=None, help="save the results")
    parser.add_argument("--num_sample", type=int, default=None, help="If you want to test your code by sampling a small number of data, you can set this argument.")
    parser.add_argument("--prompt_dir", type=str, default=None)
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--num_threads", type=int, default=10, help="number of threads to use for parallel processing of examples")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use, for multi-node vllm")
    parser.add_argument("--api_key", type=str, default=None, help="YOUR API KEY")
    parser.add_argument("--print_rationale", action="store_true", default=False, help="directly load model instead of pipeline")
    args = parser.parse_args()
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)
    return args

def main():
    args = parse_args()
    ############################
    # Load dataset
    ############################
    raw_dataset = load_json(args.input_path)
    if args.num_sample:
        raw_dataset = raw_dataset[:args.num_sample]

    ### Get results by model type
    if args.model_type=="generative":
        final_results = run_generative_reward_model(args, raw_dataset)
    elif args.model_type=="classifier":
        final_results = run_classifier_reward_model(args, raw_dataset)
    elif args.model_type=="prm":
        final_results = run_process_reward_model(args, raw_dataset)
    else:
        raise ValueError("Insufficient Model Type")
    
    # collect the results
    tmp_problem=""
    tmp_score_list=[]
    save_dataset = []
    for i, (problem, score) in enumerate(zip(final_results["problem"], final_results["score"])):
        if i==len(final_results["problem"])-1:
            if tmp_problem!=problem:
                save_dataset.append({
                    "problem" : problem,
                    "score" : [score[0]]
                })
            else:
                save_dataset.append({
                    "problem" : tmp_problem,
                    "score" : tmp_score_list.append(score[0])
                })
            break
        if tmp_problem!=problem:
            if i!=0:
                save_dataset.append({
                    "problem" : tmp_problem,
                    "score" : tmp_score_list
                })
            tmp_problem=problem
            tmp_score_list=[]
            tmp_score_list.append(score[0])
        else:
            tmp_score_list.append(score[0])
    for save_d, raw_d in zip(save_dataset, raw_dataset):
        if save_d["problem"]==raw_d["problem"]:
            save_d["answer"] = raw_d["answer"]
            if "data_source" in raw_d:
                save_d["data_source"] = raw_d["data_source"]
            save_d["output"] = raw_d["output"]
        else:
            # tmp
            save_d["answer"] = raw_d["answer"]
            save_d["output"] = raw_d["output"]
            ###
            print("Wrong problem match")
        
    # save the results
    save_json(args.save_path, save_dataset)


def run_generative_reward_model(args, raw_dataset):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    logger.info(f"Running reward model on {args.model_name}")
    
    is_api_models = args.model_name in API_MODEL_LIST
        
    # if model isn't API, load via vllm
    tokenizer=None  # API
    if not is_api_models:
        # load model
        model = LLM(args.model_name, trust_remote_code=args.trust_remote_code, tensor_parallel_size=args.num_gpus)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if "Llama-3" in args.model_name or "llama3-8b" in args.model_name:
            stop_token_ids = [128009]
        else:
            stop_token_ids = []

        sampling_params = SamplingParams(
            n=1,
            temperature=0,
            max_tokens=2048,
            stop_token_ids=stop_token_ids,
        )
    
    # Setting ProptHandler
    if args.prompt_dir and args.prompt_key:
        PromptHandler = RewardModelPromptHandler(raw_dataset, args.model_name, args.prompt_dir, args.prompt_key, tokenizer, None, args.model_type)
        dataset_processing = PromptHandler.generate_prompt()
        dataset = Dataset.from_list(dataset_processing)
    else:
        raise ValueError("There is no information about the prompt.")

    final_results = {
        "problem": [],
        "score": []
    }
    def parsing_score(output):
        def find_first_number(text):
            match = re.search(r'\d+', text)
            if match:
                return int(match.group(0))
            else:
                return 1    ## parsing error -> get score "1"
        if args.print_rationale:
            logger.info(f"{output}")
        if "Rating:" in output:
            return find_first_number(output.split("Rating:")[-1])
        else:
            return 1    ## parsing error -> get score "1"
        
    if is_api_models:
        def update_progress_bar(done, total):
            # Simple text-based progress bar
            progress = int(50 * done / total)  # Calculate progress (50 chars width)
            sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
            sys.stdout.flush()

        def get_reward(batch):
            if "gpt" in args.model_name:
                output = chat_completion_openai(args.model_name, batch["prompt"], temperature=0.0, max_tokens=2048, api_key=args.api_key)
            elif "claude" in args.model_name:
                output = chat_completion_anthropic(args.model_name, batch["prompt"], temperature=0.0, max_tokens=2048, api_key=args.api_key)
            else:
                raise ValueError(f"Model {args.model_name} not supported") 
            return output
        
        logger.info("*** Run inference ***")
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Progress bar version
            scores = [None] * len(dataset)  # Preallocate results list
            done_tasks = 0  # Counter for completed tasks

            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                # Submit all tasks and hold their futures in a list
                future_to_index = {executor.submit(get_reward, x): i for i, x in enumerate(dataset)}

                # As tasks complete, update progress and store results in the original order
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    scores[index] = future.result()
                    done_tasks += 1
                    update_progress_bar(done_tasks, len(dataset))

            # Print newline after progress bar
            print()
        logger.info("*** Inference done ***")
    else:
        prompts = dataset["prompt"]
        # generate
        logger.info("*** Run inference ***")
        outputs = model.generate(prompts, sampling_params)
        logger.info("*** Inference done ***")
        scores = [o.outputs[0].text for o in outputs]

    final_results["problem"] = dataset["problem"]
    final_results['score'] = [[parsing_score(score)] for score in scores]

    return final_results


def run_classifier_reward_model(args, raw_dataset):
    accelerator = Accelerator()
    current_device = accelerator.process_index
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running reward model on {args.model_name}")

    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    if args.model_name in REWARD_MODEL_CONFIG:
        config = REWARD_MODEL_CONFIG[args.model_name]
    else:
        config = REWARD_MODEL_CONFIG["default"]
    logger.info(f"Using reward model config: {config}")

    quantized = config["quantized"]  # only Starling isn't quantized for now
    # if llama-3 in name, switch quantized to False (severely degrades performance)
    if (
        ("llama-3" in args.model_name)
        or ("Llama3" in args.model_name)
        or ("Llama-3" in args.model_name)
        or ("LLaMA3" in args.model_name)
        or ("llama3" in args.model_name)
        or args.not_quantized
    ):
        quantized = False
        logger.info(f"Disabling quantization for llama-3 or override flag (--not_quantized: {args.not_quantized})")
    
    custom_dialogue = config["custom_dialogue"]
    model_type = config["model_type"]
    model_builder = config["model_builder"]
    pipeline_builder = config["pipeline_builder"]
    torch_dtype = config.get("torch_dtype", None)
    # if not datatype in config (default), check args
    if torch_dtype is None:
        # if datatype is bfloat16, then manually turn off quantizaiton (done with bitsandbytes)
        if args.torch_dtype == torch.bfloat16:
            quantized = False
            logger.info("Disabling quantization for bfloat16 datatype")
        torch_dtype = args.torch_dtype

    # not included in config to make user explicitly understand they are passing this
    trust_remote_code = args.trust_remote_code

    tokenizer_path = args.tokenizer if args.tokenizer else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if not custom_dialogue:  # not needed for PairRM / SteamSHP
        tokenizer.truncation_side = "left"  # copied from Starling, but few samples are above context length
    
    # Setting ProptHandler
    PromptHandler = RewardModelPromptHandler(raw_dataset, args.model_name, args.prompt_dir, args.prompt_key, tokenizer, args.chat_template, args.model_type, custom_dialogue)
    dataset_processing = PromptHandler.generate_prompt()
    dataset = Dataset.from_list(dataset_processing)

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size
    logger.info("*** Load reward model ***")
    reward_pipeline_kwargs = {
        "batch_size": BATCH_SIZE,  # eval_args.inference_batch_size,
        "truncation": True,
        "padding": True,
        "max_length": args.max_length,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }
    if quantized:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": {"": current_device},
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch_dtype,
        }

    model = model_builder(args.model_name, **model_kwargs, trust_remote_code=trust_remote_code)
    reward_pipe = pipeline_builder(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
    )

    ############################
    # Tokenization settings & dataset preparation
    ############################
    # set pad token to eos token if not set
    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
    # For models whose config did not contains `pad_token_id`
    if reward_pipe.model.config.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

    # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
    if not check_tokenizer_chat_template(tokenizer):
        reward_pipe.tokenizer.add_eos_token = True

    # Inference
    final_results = {
        "problem": [],
        "score": []
    }
    if pipeline_builder == pipeline:
        logger.info("*** Running forward pass via built in pipeline abstraction ***")
        reward_pipe = accelerator.prepare(reward_pipe)
        results = reward_pipe(dataset["prompt"], **reward_pipeline_kwargs)

        final_results["problem"] = dataset["problem"]
        final_results["score"] = [result["score"] for result in results]
    else:
        logger.info("*** Running dataloader to collect results ***")
        from torch.utils.data.dataloader import default_collate

        def custom_collate_fn(batch):
            # check if ['text_chosen'] is in first batch element
            # Check if the first element of the batch is a dictionary
            if isinstance(batch[0]["prompt"][0], dict):
                return batch  # Return the batch as-is if it's a list of dicts
            else:
                return default_collate(batch)  # Use the default collate behavior otherwise
        
        keep_columns = ["prompt", "problem"]
        all_cols = dataset.column_names
        dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=custom_collate_fn,  
            shuffle=False,
            drop_last=False,
        )

        dataloader, model = accelerator.prepare(dataloader, reward_pipe.model)
        reward_pipe.model = model

        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            logger.info(f"RM inference step {step}/{len(dataloader)}")

            if model_type == "Custom Classifier":
                problem_batch = [b["problem"] for b in batch]
                text = [b["prompt"] for b in batch]
                rewards = reward_pipe(text, **reward_pipeline_kwargs)
                score_batch = [ [reward.cpu().float().cpu().numpy().tolist()] for reward in rewards]
            else:
                problem_batch = batch["problem"]
                rewards = reward_pipe(batch["prompt"], **reward_pipeline_kwargs)

                if isinstance(rewards[0], dict):
                    score_batch = [reward["score"] for reward in rewards]
                else:
                    if "Ray2333/GRM" in args.model_name:
                        score_batch = [[s] for s in rewards.float().cpu().numpy().tolist()]
                    else:
                        score_batch = rewards.float().cpu().numpy().tolist()
            
            final_results["problem"].extend(problem_batch)
            final_results["score"].extend(score_batch)
    return final_results


def run_process_reward_model(args, raw_dataset):
    accelerator = Accelerator()    
    logger = get_logger(__name__)
    logger.info(f"Running reward model on {args.model_name}")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if args.model_name in PRM_MODEL_CONFIG:
        config = PRM_MODEL_CONFIG[args.model_name]
    else:
        config = PRM_MODEL_CONFIG["default"]
    model = config["model_class"](args.model_name)
    prm_type = config["prm_type"]
    logger.info(f"Using process reward model config: {config}")

    tokenizer = model.tokenizer

    # Setting ProptHandler
    PromptHandler = RewardModelPromptHandler(raw_dataset, args.model_name, args.prompt_dir, args.prompt_key, tokenizer, None, args.model_type)
    dataset_processing = PromptHandler.generate_prompt(prm_type)
    dataset = Dataset.from_list(dataset_processing)

    final_results = {
        "problem": [],
        "score": []
    }
    for data in tqdm(dataset):
        score = model.get_results(data["prompt"])
        final_results["problem"].append(data["problem"])
        final_results['score'].append([score])
    return final_results


if __name__ == "__main__":
    main()