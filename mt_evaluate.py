

import gc
import json
import torch
from argparse import ArgumentParser
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import operator

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


def mt_string_match(preds, refs):
    """
    Calculate the string match score for all references
    preds: list of strings, shape (N,) where N is the number of samples
    refs: list of strings, references, shape (N,) OR list of list of strings, shape (N, K) where N is the number of samples and K is the number of references per question
    """
    scores = []
    for pred, ref in zip(preds, refs):
        pred = pred.lower()
        if isinstance(ref, list):
            score = sum([1.0 if r.lower() in pred else 0.0 for r in ref]) / len(ref) * 100
        else:
            ref = ref.lower()
            score = 100.0 if ref in pred else 0.0
        scores.append(score)
    return sum(scores) / len(scores)


def calculate_metrics(results: list[dict]) -> dict:
    scores = {} 
    metric_fn = mt_string_match
    n_questions = len(results)
    for i, row in enumerate(results):
        question = row["question"]
        prediction = row["prediction"]
        answer = row["answer"]
        scores[f"question_{i+1}"] = {"string_match": metric_fn(prediction, answer)}
    avg_score = sum([score["string_match"] for score in scores.values()]) / n_questions
    scores["all"] = {"string_match": avg_score}
    return scores


def parse_args():
    def str_to_list(arg):
        return arg.split(',')
    p = ArgumentParser()
    p.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dataset_name", type=str, default="mt_pr_T")
    p.add_argument("--fraction", type=float, default=1.0, help="Fraction of the dataset to use.")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--sparse_budget", default=2048)
    p.add_argument("--rank", type=int, default=160)
    p.add_argument("--chunk_size", type=int, default=8)

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
              minference=False)

    results = []

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


        # HACK: append answers of previous question to chat and feed to the model
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
            results.append({"question": question, "prediction": pred, "answer": gt_answers[j]})
        llm.kv_cache.clear()
        torch.cuda.empty_cache()

    # save results
    with open(save_filename, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {save_filename}")

    # calculate scores
    scores = calculate_metrics(results)
    with open(score_filename, "w") as f:
        json.dump(scores, f)
    
    print(scores)
    print(f"Scores saved to {score_filename}")
    