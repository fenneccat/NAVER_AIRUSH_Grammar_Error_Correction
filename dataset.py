import torch
from torch.utils.data import Dataset
from preprocess import preprocess_noisy_sentence_list

import logging
import random

from noise_generation import noise, add_del_space

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, data, preprocess=False):
        if preprocess: preprocess_noisy_sentence_list(data)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn_MLM(data, tokenizer, max_seq_length=None, tokenizer_type='char', eos=False):

    if tokenizer_type == 'kobert':
        src_token_ids_tmp = [tokenizer.encode(text)[1:] for text in data]
        tgt_token_ids = [tokenizer.encode(text) for text in data]

    else:
        if eos: src_token_ids_tmp = [tokenizer(x) + [3] for x in data]
        else: src_token_ids_tmp = [tokenizer(x) for x in data]
        tgt_token_ids = [[2] + tokenizer(x) + [3] for x in data]

    src_token_ids = []

    for sample in src_token_ids_tmp:
        new_sample = []
        for i, tok in enumerate(sample):
            masking_deter = random.uniform(0, 1)
            if tok > 4 and masking_deter < 0.15:
                new_sample.append(4)
            else: new_sample.append(tok)
        src_token_ids.append(new_sample)

    src_max_seq_length = max([len(x) for x in src_token_ids])
    if max_seq_length and max_seq_length < src_max_seq_length:
        src_max_seq_length = max_seq_length
    tgt_max_seq_length = max([len(x) + 1 for x in tgt_token_ids])
    if max_seq_length and max_seq_length + 1 < tgt_max_seq_length:
        tgt_max_seq_length = max_seq_length + 1

    src_padded = []
    src_padding_mask = []
    tgt_padded = []
    tgt_padding_mask = []
    for src, tgt in zip(src_token_ids, tgt_token_ids):
        src = src[:src_max_seq_length]
        src_pad_length = src_max_seq_length - len(src)
        src_padded.append(src + [1] * src_pad_length)
        src_padding_mask.append([1] * len(src) + [0] * src_pad_length)
        tgt = tgt[:tgt_max_seq_length]
        tgt_pad_length = tgt_max_seq_length - len(tgt)
        tgt_padded.append(tgt + [1] * tgt_pad_length)
        tgt_padding_mask.append([1] * (len(tgt) - 1) + [0] * tgt_pad_length)

    src_padded = torch.tensor(src_padded).t().contiguous()
    src_padding_mask = torch.tensor(src_padding_mask).bool().t()
    tgt_padded = torch.tensor(tgt_padded).t().contiguous()
    tgt_padding_mask = torch.tensor(tgt_padding_mask).bool().t()

    return src_padded, tgt_padded[:-1], src_padding_mask, tgt_padding_mask, tgt_padded[1:]


def collate_fn_LM(data, tokenizer, max_seq_length=None, tokenizer_type='char', eos=False):
    if tokenizer_type == 'kobert':
        src_token_ids = [tokenizer.encode(text)[1:] for text in data]
        tgt_token_ids = [tokenizer.encode(text) for text in data]

    else:
        if eos: src_token_ids = [tokenizer(x) + [3] for x in data]
        else: src_token_ids = [tokenizer(x) for x in data]
        tgt_token_ids = [[2] + tokenizer(x) + [3] for x in data]

    src_max_seq_length = max([len(x) for x in src_token_ids])
    if max_seq_length and max_seq_length < src_max_seq_length:
        src_max_seq_length = max_seq_length
    tgt_max_seq_length = max([len(x) + 1 for x in tgt_token_ids])
    if max_seq_length and max_seq_length + 1 < tgt_max_seq_length:
        tgt_max_seq_length = max_seq_length + 1

    src_padded = []
    src_padding_mask = []
    tgt_padded = []
    tgt_padding_mask = []
    for src, tgt in zip(src_token_ids, tgt_token_ids):
        src = src[:src_max_seq_length]
        src_pad_length = src_max_seq_length - len(src)
        src_padded.append(src + [1] * src_pad_length)
        src_padding_mask.append([1] * len(src) + [0] * src_pad_length)
        tgt = tgt[:tgt_max_seq_length]
        tgt_pad_length = tgt_max_seq_length - len(tgt)
        tgt_padded.append(tgt + [1] * tgt_pad_length)
        tgt_padding_mask.append([1] * (len(tgt) - 1) + [0] * tgt_pad_length)

    src_padded = torch.tensor(src_padded).t().contiguous()
    src_padding_mask = torch.tensor(src_padding_mask).bool().t()
    tgt_padded = torch.tensor(tgt_padded).t().contiguous()
    tgt_padding_mask = torch.tensor(tgt_padding_mask).bool().t()

    return src_padded, tgt_padded[:-1], src_padding_mask, tgt_padding_mask, tgt_padded[1:]

    
def collate_fn(data, tokenizer, max_seq_length=None, eos=False, add_noise=False, tokenizer_type='char'):
    if tokenizer_type == 'kobert':
        src_token_ids = [tokenizer.encode(text['noisy'])[1:] for text in data]
        tgt_token_ids = [tokenizer.encode(text['clean']) for text in data]

    else:
        if add_noise:
            noisy = add_del_space([x['noisy'] for x in data])
            if eos: src_token_ids = [tokenizer(x) + [3] for x in noisy]
            else: src_token_ids = [tokenizer(x) for x in noisy]

        else:
            if eos: src_token_ids = [tokenizer(x['noisy']) + [3] for x in data]
            else: src_token_ids = [tokenizer(x['noisy']) for x in data]

        tgt_token_ids = [[2] + tokenizer(x['clean']) + [3] for x in data]

    src_max_seq_length = max([len(x) for x in src_token_ids])
    if max_seq_length and max_seq_length < src_max_seq_length:
        src_max_seq_length = max_seq_length
    tgt_max_seq_length = max([len(x) + 1 for x in tgt_token_ids])
    if max_seq_length and max_seq_length + 1 < tgt_max_seq_length:
        tgt_max_seq_length = max_seq_length + 1

    src_padded = []
    src_padding_mask = []
    tgt_padded = []
    tgt_padding_mask = []
    for src, tgt in zip(src_token_ids, tgt_token_ids):
        src = src[:src_max_seq_length]
        src_pad_length = src_max_seq_length - len(src)
        src_padded.append(src + [1] * src_pad_length)
        src_padding_mask.append([1] * len(src) + [0] * src_pad_length)
        tgt = tgt[:tgt_max_seq_length]
        tgt_pad_length = tgt_max_seq_length - len(tgt)
        tgt_padded.append(tgt + [1] * tgt_pad_length)
        tgt_padding_mask.append([1] * (len(tgt) - 1) + [0] * tgt_pad_length)

    src_padded = torch.tensor(src_padded).t().contiguous()
    src_padding_mask = torch.tensor(src_padding_mask).bool().t()
    tgt_padded = torch.tensor(tgt_padded).t().contiguous()
    tgt_padding_mask = torch.tensor(tgt_padding_mask).bool().t()

    return src_padded, tgt_padded[:-1], src_padding_mask, tgt_padding_mask, tgt_padded[1:]


def collate_fn_noisy(data, tokenizer, max_seq_length=None, tokenizer_type = 'char', eos=False):
    if tokenizer_type == 'kobert':
        src_token_ids = [tokenizer.encode(text['clean'])[1:] for text in data]
        tgt_token_ids = [tokenizer.encode(text['noisy']) for text in data]

    else:
        if eos: src_token_ids = [tokenizer(x['clean']) + [3] for x in data]
        else: src_token_ids = [tokenizer(x['clean']) for x in data]
        tgt_token_ids = [[2] + tokenizer(x['noisy']) + [3] for x in data]

    src_max_seq_length = max([len(x) for x in src_token_ids])
    if max_seq_length and max_seq_length < src_max_seq_length:
        src_max_seq_length = max_seq_length
    tgt_max_seq_length = max([len(x) + 1 for x in tgt_token_ids])
    if max_seq_length and max_seq_length + 1 < tgt_max_seq_length:
        tgt_max_seq_length = max_seq_length + 1

    src_padded = []
    src_padding_mask = []
    tgt_padded = []
    tgt_padding_mask = []
    for src, tgt in zip(src_token_ids, tgt_token_ids):
        src = src[:src_max_seq_length]
        src_pad_length = src_max_seq_length - len(src)
        src_padded.append(src + [1] * src_pad_length)
        src_padding_mask.append([1] * len(src) + [0] * src_pad_length)
        tgt = tgt[:tgt_max_seq_length]
        tgt_pad_length = tgt_max_seq_length - len(tgt)
        tgt_padded.append(tgt + [1] * tgt_pad_length)
        tgt_padding_mask.append([1] * (len(tgt) - 1) + [0] * tgt_pad_length)

    src_padded = torch.tensor(src_padded).t().contiguous()
    src_padding_mask = torch.tensor(src_padding_mask).bool().t()
    tgt_padded = torch.tensor(tgt_padded).t().contiguous()
    tgt_padding_mask = torch.tensor(tgt_padding_mask).bool().t()

    return src_padded, tgt_padded[:-1], src_padding_mask, tgt_padding_mask, tgt_padded[1:]

def collate_fn_bert(data, tokenizer, max_seq_length=None):
    src_token_ids = [[2] + tokenizer(x['noisy']) + [3] for x in data]
    tgt_token_ids = [[2] + tokenizer(x['clean']) + [3] for x in data]

    src_max_seq_length = max([len(x) for x in src_token_ids])
    if max_seq_length and max_seq_length < src_max_seq_length:
        src_max_seq_length = max_seq_length
    tgt_max_seq_length = max([len(x) for x in tgt_token_ids])
    if max_seq_length and max_seq_length < tgt_max_seq_length:
        tgt_max_seq_length = max_seq_length

    sequence_length = max(src_max_seq_length, tgt_max_seq_length)

    src_padded = []
    src_padding_mask = []
    tgt_padded = []
    tgt_padding_mask = []
    for src, tgt in zip(src_token_ids, tgt_token_ids):
        src = src[:sequence_length]
        src_pad_length = sequence_length - len(src)
        src_padded.append(src + [1] * src_pad_length)
        src_padding_mask.append([1] * len(src) + [0] * src_pad_length)
        tgt = tgt[:sequence_length]
        tgt_pad_length = sequence_length - len(tgt)
        tgt_padded.append(tgt + [1] * tgt_pad_length)
        tgt_padding_mask.append([1] * len(tgt) + [0] * tgt_pad_length)

    src_padded = torch.tensor(src_padded).contiguous()
    src_padding_mask = torch.tensor(src_padding_mask).bool()
    tgt_padded = torch.tensor(tgt_padded).contiguous()
    tgt_padding_mask = torch.tensor(tgt_padding_mask).bool()

    return src_padded, tgt_padded, src_padding_mask, tgt_padding_mask

def collate_fn_kcBert(data, tokenizer, max_seq_length=None, add_special_token = False):

    if add_special_token:
        src_token_ids = [tokenizer.encode(text['noisy']) for text in data]
        tgt_token_ids = [tokenizer.encode(text['clean']) for text in data]
    else:
        src_token_ids = [tokenizer.encode(text['noisy']) for text in data]
        tgt_token_ids = [tokenizer.encode(text['clean']) for text in data]

    src_max_seq_length = max([len(x) for x in src_token_ids])
    if max_seq_length and max_seq_length < src_max_seq_length:
        src_max_seq_length = max_seq_length
    tgt_max_seq_length = max([len(x) for x in tgt_token_ids])
    if max_seq_length and max_seq_length < tgt_max_seq_length:
        tgt_max_seq_length = max_seq_length

    sequence_length = max(src_max_seq_length, tgt_max_seq_length)

    src_padded = []
    src_padding_mask = []
    tgt_padded = []
    tgt_padding_mask = []
    for src, tgt in zip(src_token_ids, tgt_token_ids):
        src = src[:sequence_length]
        src_pad_length = sequence_length - len(src)
        src_padded.append(src + [1] * src_pad_length)
        src_padding_mask.append([1] * len(src) + [0] * src_pad_length)
        tgt = tgt[:sequence_length]
        tgt_pad_length = sequence_length - len(tgt)
        tgt_padded.append(tgt + [1] * tgt_pad_length)
        tgt_padding_mask.append([1] * len(tgt) + [0] * tgt_pad_length)

    src_padded = torch.tensor(src_padded).contiguous()
    src_padding_mask = torch.tensor(src_padding_mask).bool()
    tgt_padded = torch.tensor(tgt_padded).contiguous()
    tgt_padding_mask = torch.tensor(tgt_padding_mask).bool()

    return src_padded, tgt_padded, src_padding_mask, tgt_padding_mask