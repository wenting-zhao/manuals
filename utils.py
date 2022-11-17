import argparse
from collections import defaultdict
from glob import glob
import string
import re
from datasets import load_dataset
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_hotpotqa():
    return load_dataset('hotpot_qa', 'distractor')

def padding(indices, values):
    L = len(indices)
    rows = torch.nn.functional.one_hot(indices)
    cols = rows.cumsum(0)[torch.arange(L), indices] - 1
    cols = torch.nn.functional.one_hot(cols)
    outs = (values[:, None, None] *
            cols[:, None, :] *
            rows[:, :, None]).sum(0)
    return outs

def padding_long(indices, values):
    final = []
    length = max(indices)
    st = 0
    for i in indices:
        curr = values[st:st+i]
        curr_zeros = torch.zeros(length-i).to(device)
        curr = torch.cat([curr, curr_zeros], dim=0)
        final.append(curr)
        st += i
    outs = torch.stack(final)
    return outs

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nolog', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--max_p', action='store_true')
    parser.add_argument('--sentence', action='store_true')
    parser.add_argument('--gradient_checkpoint', '-gc', action='store_true')
    parser.add_argument('--truncate_paragraph', '-tp', default=0, type=int)
    parser.add_argument('--reg_coeff', default=0, type=float)
    parser.add_argument('--k_distractor', default=1, type=int)
    parser.add_argument('--max_e_len', default=3, type=int)
    parser.add_argument('--beam', default=2, type=int)
    parser.add_argument('--topkp', default=5, type=int)
    parser.add_argument('--topks', default=5, type=int)
    parser.add_argument('--mode', default="topk", type=str)
    parser.add_argument("--batch_size", '-b', default=1, type=int,
                        help="batch size per gpu.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="eval batch size per gpu.")
    parser.add_argument("--eval_steps", default=5000, type=int,
                        help="number of steps between each evaluation.")
    parser.add_argument("--epoch", '-epoch', default=10, type=int,
                        help="The number of epochs for fine-tuning.")
    parser.add_argument("--max_paragraph_length", default=1000, type=int,
                        help="The maximum number of sentences allowed in a paragraph.")
    parser.add_argument("--sentence_thrshold", default=0.5, type=float,
                        help="The maximum number of sentences allowed in a paragraph.")
    parser.add_argument("--max_matrix", default=5000, type=int,
                        help="The largest n when doing matrix operation.")
    parser.add_argument("--model_dir", default="roberta-large", type=str,
                        help="The directory where the pretrained model will be loaded.")
    parser.add_argument("--answer_model_dir", default="facebook/bart-base", type=str,
                        help="The directory where the pretrained model will be loaded.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--output_model_dir", default="./saved_models", type=str,
                        help="The directory where the pretrained model will be saved.")
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Warmup ratio in the lr scheduler."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    args = parser.parse_args()
    if args.baseline:
        args.max_paragraph_length = 1000
    assert args.k_distractor <= 8
    return args

def prepare_optim_and_scheduler(all_layers, args):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    for layer in all_layers:
        curr = [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_grouped_parameters += curr

    optim = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optim,
        num_warmup_steps=int(args.warmup_ratio*args.max_train_steps),
        num_training_steps=args.max_train_steps,
    )
    return optim, lr_scheduler

def prepare_linear(size):
    linear = nn.Linear(size, 1)
    linear = linear.to(device)
    return linear

def prepare_mlp(size):
    mlp = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
            )
    mlp = mlp.to(device)
    return mlp

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def collect_fever_docs(doc_path="data/fever/docs/*"):
    flist = glob(doc_path)
    title2sents = defaultdict(list)
    for fname in flist:
        with open(fname, 'r') as f:
            for line in f:
                key = fname.split('/')[-1]
                title2sents[key].append(line.strip())
    return title2sents

def collect_multirc_docs(doc_path="data/multirc/docs/*"):
    flist = glob(doc_path)
    title2sents = defaultdict(list)
    for fname in flist:
        with open(fname, 'r') as f:
            for line in f:
                key = fname.split('/')[-1]
                title2sents[key].append(line.strip())
    return title2sents
