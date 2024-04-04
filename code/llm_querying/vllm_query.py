import os
import json
import argparse
from tqdm import tqdm

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process model and query paths")

    parser.add_argument("--batch-size",         type=int, default=1, help="Query batch size.")
    parser.add_argument("--chat-mode",          type=int, default=1, help="Chat Mode. Set to 1 if you wish to use the transformers' tokenizer driven chat template, else set to 0.")
    parser.add_argument("--query-field",        type=str, default="query", help="Specify the name of the field (dict key) to be queried.")
    parser.add_argument("--query-limit",        type=int, default=-1, help="Maximum number of queries to process. If -1, will assume the length of JSON queries.")
    parser.add_argument("--max-len",            type=int, default=2000, help="Maximum overall length of text. Includes prompt length.")
    parser.add_argument("--model-path",         type=str, help=f"HuggingFace model name or local model path.")
    parser.add_argument("--n-seq",              type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--query-paths",        type=str, nargs="+", default=[], help="List of paths to JSON files to query. JSON is expected to be a list of dictionaries, with each dict conatining the relevant field to be queried.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty. Not linear.")
    parser.add_argument("--seed",               type=int, default=0, help="Generation seed.")
    parser.add_argument("--temperature",        type=float, default=0.8, help="Generation temperature. Set a higher temperature for more randomized token-selection.")
    parser.add_argument("--top-k",              type=int, default=-1, help="Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens")
    parser.add_argument("--uuid",               type=str, default="UUID", help="Universally unique Identifier to ensure multiple runs are distinguishable.")

    args = parser.parse_args()
    print(vars(args))

    addn_args = {}

    addn_args["target_dir"] = "../../data/response_datasets"
    os.makedirs(addn_args["target_dir"], exist_ok=True)

    model_path = args.model_name

    llm = LLM(
        model = model_path, 
        tensor_parallel_size = torch.cuda.device_count(),
        seed = args.seed,
        tokenizer_mode='auto',
        trust_remote_code=True,
        max_model_len=2000,
    )

    os.system("nvidia-smi")

    # Add more as needed from https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
    sampling_params = SamplingParams(
        n=1,
        top_k=args.top_k,
        temperature=args.temperature, 
        repetition_penalty=args.repetition_penalty, 
        max_tokens=args.max_len,
    )

    for fpath in tqdm(args.query_paths):

        print(fpath)
        addn_args['prefix'] = fpath.split('/')[-1].split('.')[0]

        data = json.load(open(fpath,'r'))

        if args.chat_mode:
            queries = [d[args.query_field] for d in data]
            if "llama2" in args.model_name:
                chats = [[
                    {"role": "system", "content": "You are an expert in solving math questions. Answer the following question to the best of your ability."},
                    {"role": "user", "content": q},
                ] for q in queries]
            else:
                chats = [[{"role": "user", "content": "You are an expert in solving math questions. Answer the following question to the best of your ability." + " " + q}] for q in queries]
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
            prompts = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in chats]
        else:
            prompts = [d[args.query_field] for d in data]

        if args.query_limit != -1:
            if args.query_limit < 1:
                raise Exception("Invalid query limit. Expected >=1.")
            prompts = prompts[:args.query_limit]

        if args.n_seq > 1:
            _prompts = []
            _data = []
            for p in prompts:
                _prompts.extend([p for _ in range(args.n_seq)])
            prompts = _prompts
            for d in data:
                _data.extend([d.copy() for _ in range(args.n_seq)])
            data = _data

        print("Total prompts to process:",len(prompts))
        print("Sample prompt:", prompts[0])
        print()

        responses = []
        for res in llm.generate(prompts, sampling_params, use_tqdm=True):
            responses.append(res)

        for d, res in zip(data, responses):
            d["query_response"] = [str(out.text) for out in res.outputs]
                
        json.dump(
            data,
            open(os.path.join(addn_args["target_dir"],f"{addn_args['prefix']}_{args.model_name}_temp{args.temperature}_{args.uuid}.json"), "w"),
            indent=3,
        )

