from collections import defaultdict, Counter
from glob import glob
from itertools import combinations, product
import json
import pickle
import random
import os
import torch


def preprocess_fever(examples, tok, answ_tok, fixed, max_e):
    sents = []
    supps = []
    lengths = []
    slengths = []
    ds = []
    num_s = []
    for e in examples:
        curr_sents = []
        for i in range(0, len(e['z']), fixed):
            sent = f'{tok.unk_token} ' + f' {tok.unk_token} '.join(e['z'][i:i+fixed])
            sent = e['x'] + sent
            curr_sents.append(sent)
        lengths.append(len(curr_sents))
        sents += curr_sents
        z_len = len(e['z'])
        rang = list(range(z_len))
        curr_idxes = []
        for i in range(1, min(max_e+1, z_len+1)):
            curr_curr_idxes = []
            for j in range(z_len-i+1):
                curr_curr_idxes.append(rang[j:j+i])
                curr_supps = [e['z'][m] for m in curr_curr_idxes[-1]]
                curr_supps = ' '.join(curr_supps)
                curr_supps = e['x'] + curr_supps
                supps.append(curr_supps)
            curr_idxes.append(curr_curr_idxes)
        ds.append(curr_idxes)
        slengths.append(len([x for xx in curr_idxes for x in xx]))
        num_s.append(z_len)
    lengths = len_helper(lengths)
    slengths = len_helper(slengths)
    # there is one example that has a length 529, might cause an error
    tokenized_sents = tok(sents, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_sents = [tokenized_sents[lengths[i]:lengths[i+1]] for i in range(len(lengths)-1)]
    answers = [e['y'] for e in examples]
    tokenized_answers = answ_tok(answers, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_supps = answ_tok(supps, truncation=True, return_attention_mask=False)['input_ids']
    tokenized_supps = [tokenized_supps[slengths[i]:slengths[i+1]] for i in range(len(slengths)-1)]
    assert len(tokenized_supps) == len(tokenized_answers) == len(tokenized_sents)
    return tokenized_sents, tokenized_supps, tokenized_answers, ds, num_s

def prepare_fever(tokenizer, answer_tokenizer, split, docs, fixed, max_e, path="data/fever/"):
    print(f"prepare fever {split}")
    data = []
    with open(f"{path}/{split}.jsonl", 'r') as fin:
        for line in fin:
            data.append(json.loads(line))
    out = []
    labels = []
    sent_labels = []
    for d in data:
        curr = dict()
        curr['x'] = "A claim to be investigated is that " + d['query'] + " We have following facts: "
        d['evidences'] = [ee for e in d['evidences'] for ee in e]
        docid = [l['docid'] for l in d['evidences']]
        docid = set(docid)
        assert len(docid) == 1
        docid = docid.pop()
        curr['z'] = docs[docid]
        gold_z = [l['start_sentence'] for l in d['evidences']]
        sent_labels.append(gold_z)
        curr['y'] = d['classification'].lower()
        curr['y'] = "The claim is thus supported." if curr['y'] == "supports" else "The claim is thus refuted."
        label = 0 if "supported" in curr['y'] else 1
        labels.append(label)
        out.append(curr)
    fname = f"cache/fever_new_tok_{split}.pkl"
    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            sents, supps, answs, ds, num_s = pickle.load(f)
    else:
        sents, supps, answs, ds, num_s = preprocess_fever(out, tokenizer, answer_tokenizer, fixed, max_e)
        with open(fname, 'wb') as f:
            pickle.dump((sents, supps, answs, ds, num_s), f)
    return (sents, supps, answs, ds, num_s, sent_labels, labels)

class FeverDataset(torch.utils.data.Dataset):
    def __init__(self, everything):
        self.sents, self.supps, self.answs, self.ds, self.num_s, self.sent_labels, self.labels = everything

    def __getitem__(self, idx):
        item = dict()
        item['sents'] = self.sents[idx]
        item['supps'] = self.supps[idx]
        item['answs'] = self.answs[idx]
        item['ds'] = self.ds[idx]
        item['num_s'] = self.num_s[idx]
        item['sent_labels'] = self.sent_labels[idx]
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.sent_labels)
