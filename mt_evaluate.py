

import gc
import json
import torch
from argparse import ArgumentParser
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

DATASET_DICT = {
    "mt_niah_S": "data/mt_ruler/multi_turn_niah_small.jsonl",
    "mt_niah_M": "data/mt_ruler/save/multi_turn_niah_medium.jsonl",
    "mt_niah_L": "data/mt_ruler/save/multi_turn_niah_large.jsonl",
    "mt_vt_S": "data/mt_ruler/multi_turn_vt_small.jsonl",
    "mt_vt_M": "data/mt_ruler/multi_turn_vt_medium.jsonl",
    "mt_vt_L": "data/mt_ruler/multi_turn_vt_large.jsonl",
    "mt_pr_S": "data/mt_ruler/multi_turn_pr_small.jsonl",
    "mt_pr_M": "data/mt_ruler/multi_turn_pr_medium.jsonl",
    "mt_pr_L": "data/mt_ruler/multi_turn_pr_large.jsonl",
    # different number of questions
    "mt_niah_S_20": "data/mt_ruler/multi_turn_niah_small_20.jsonl",
    "mt_niah_S_30": "data/mt_ruler/multi_turn_niah_small_30.jsonl",
    "mt_niah_S_40": "data/mt_ruler/multi_turn_niah_small_40.jsonl",
    "mt_niah_S_50": "data/mt_ruler/multi_turn_niah_small_50.jsonl",
    "mt_vt_S_20": "data/mt_ruler/multi_turn_vt_small_20.jsonl",
    "mt_vt_S_30": "data/mt_ruler/multi_turn_vt_small_30.jsonl",
    "mt_vt_S_40": "data/mt_ruler/multi_turn_vt_small_40.jsonl",
    "mt_vt_S_50": "data/mt_ruler/multi_turn_vt_small_50.jsonl",
    "mt_pr_S_20": "data/mt_ruler/multi_turn_pr_small_20.jsonl",
    "mt_pr_S_30": "data/mt_ruler/multi_turn_pr_small_30.jsonl",
    "mt_pr_S_40": "data/mt_ruler/multi_turn_pr_small_40.jsonl",
    "mt_pr_S_50": "data/mt_ruler/multi_turn_pr_small_50.jsonl",
    # test tiny
    "mt_niah_T": "data/mt_ruler/multi_turn_niah_tiny_test.jsonl",
    "mt_pr_T": "data/mt_ruler/multi_turn_pr_tiny_test.jsonl",
}

def parse_args():
    def str_to_list(arg):
        return arg.split(',')
    p = ArgumentParser()
    p.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dataset_name", type=str, default="mt_pr_T")
    p.add_argument("--fraction", type=float, default=1.0, help="Fraction of the dataset to use.")
    p.add_argument("--save_dir", type=str, default=None)
    # p.add_argument("--num_samples", type=int, default=-1)
    # p.add_argument("--batch_size", type=int, default=1)
    # p.add_argument("--datalen", type=int, default=128*1024, help="The length of the context.")
    # p.add_argument("--method", type=str, default="full")
    p.add_argument("--sparse_budget", default=2048)
    p.add_argument("--rank", type=int, default=160)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--minference", action='store_true', default=False)

    return p.parse_args()



if __name__ == '__main__':

    args = parse_args()

    # load dataset 
    ds = load_dataset("json", data_files=DATASET_DICT[args.dataset_name])["train"]
    if args.fraction < 1.0:
        ds = ds.select(range(int(len(ds) * args.fraction)))

    # Save directory
    if args.save_dir is None:
        save_dir = Path(__file__).parent / "results"
    else:
        save_dir = Path(args.save_dir)
    
    # create save directory 
    save_dir.mkdir(exist_ok=True)
    save_filename = save_dir / (
        "__".join([args.dataset_name, args.model_name.replace("/", "--"), "ShadowKV", 'answers'])
        + ".json"
    )
    score_filename = save_dir / (
        "__".join([args.dataset_name, args.model_name.replace("/", "--"), "ShadowKV",  'score'])
        + ".json"
    )

    # 
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    from models import choose_model_class
    LLM = choose_model_class(args.model_name)
    llm = LLM(model_name=args.model_name,
              batch_size=1,
              device=device,
              max_length=128*1024+2048,
              attn_mode="shadowkv",
              dtype=torch.bfloat16,
              sparse_budget=args.sparse_budget,
              rank=args.rank,
              chunk_size=args.chunk_size,
              minference=args.minference)

    answers = []

    for i, row in tqdm(enumerate(ds)):
        context = row["context"]
        questions = row["questions"]
        gt_answers = row["answers"]

        chat = [
            {"role": "user", "text": context},
        ]
        # prefill context
        # context = llm.encode(context, template="ctx")
        # llm.generate(context.to(device),
        #              gen_len=0,
        #              verbose=False,
        #              top_p=0.9,
        #              temperature=0.0) # prefill ctx
        # generate answers
        for j, question in tqdm(enumerate(questions)):
            chat.append({"role": "user", "text": question})
            input_ids = llm.encode(chat, template="chat").to(device)
            rets = llm.generate(input_ids,
                                cont=True,
                                gen_len=128,
                                top_p=0.9,
                                temperature=0.0)
            
            pred = rets[0]
            chat.append({"role": "assistant", "text": pred})
            answers.append({"question": question, "prediction": pred, "answer": gt_answers[j]})
        llm.kv_cache.clear()
        torch.cuda.empty_cache()

    # save answers
    with open(save_filename, "w") as f:
        json.dump(answers, f)
    
    # calculate scores
    
