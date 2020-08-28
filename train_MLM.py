import argparse
import json
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR

from model import TransformerModel
from tokenizer import CharTokenizer
from dataset import TextDataset, collate_fn, collate_fn_MLM
from data_loader import read_strings
from meter import Meter
from evaluation import em, gleu, gleu_one
from preprocess import preprocess_noisy_sentence_list, preprocess_sentence
from noise_generation import save_generated_data, load_generated_data, noise
from tokenization_kobert import KoBertTokenizer

from collections import Counter
from collections import defaultdict

import nsml
from nsml import DATASET_PATH

logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--data_dir", type=str, default=os.path.join(DATASET_PATH, 'train'))
    parser.add_argument("--num_val_data", type=int, default=1000)

    # model
    parser.add_argument("--vocab_size", type=int, default=1300)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--intermediate_size", type=int, default=1024)

    # training
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--adam_betas", type=str, default="(0.9, 0.98)")
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--max_grad_norm", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--num_warmup_steps", type=int, default=4000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=20)

    parser.add_argument('--semi_dataset', type=int, default=0)
    parser.add_argument('--eos_setting', type=bool, default=False)
    parser.add_argument('--tokenizer', type=str, default="char")

    # nsml
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--iteration', type=str, default="0")

    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def calc_loss(model, batch):
    src, tgt, src_mask, tgt_mask, tgt_label = batch
    output = model(src=src, tgt=tgt, src_key_padding_mask=~src_mask, tgt_key_padding_mask=~tgt_mask)
    bsz = tgt.size(1)
    raw_loss = F.cross_entropy(output.view(-1, output.size(-1)), tgt_label.view(-1), reduction='none')
    raw_loss = raw_loss.view(-1, bsz)
    loss = (raw_loss * tgt_mask.float()).sum(0).mean()
    items = [loss.data.item(), bsz, tgt_mask.sum().item()]

    return loss, items


def evaluate(model, data_loader, args):
    model.eval()
    meter = Meter()
    with torch.no_grad():
        for batch in data_loader:
            batch = tuple(t.to(args.device) for t in batch)
            _, items = calc_loss(model, batch)
            meter.add(*items)
    return meter.average(), meter.print_str(False)


def correct(model, tokenizer, test_data, args, eos=False, length_limit = None):
    model.eval()
    prediction = []
    src_token_string = []
    for i in range(0, len(test_data), args.eval_batch_size):
        batch = test_data[i:i + args.eval_batch_size]

        if args.tokenizer == 'kobert':
            src_token_ids_tmp = [tokenizer.encode(text)[1:] for text in batch]
        if args.tokenizer == 'char':
            if eos: src_token_ids_tmp = [tokenizer(x)+[3] for x in batch]  # add eos token
            else: src_token_ids_tmp = [tokenizer(x) for x in batch]

        src_token_ids = []

        for sample in src_token_ids_tmp:
            new_sample = []
            new_sample_string = ''
            for i, tok in enumerate(sample):
                masking_deter = random.uniform(0, 1)
                if tok > 4 and masking_deter < 0.15:
                    new_sample.append(4)
                    if args.tokenizer == 'char': new_sample_string += tokenizer.i2c[4]
                    else: new_sample_string += tokenizer._convert_id_to_token(4)
                else:
                    new_sample.append(tok)
                    if args.tokenizer == 'char': new_sample_string += tokenizer.i2c[tok]
                    else:
                        new_sample_string += tokenizer._convert_id_to_token(tok)

            src_token_ids.append(new_sample)
            src_token_string.append(new_sample_string)

        src_seq_length = [len(x) for x in src_token_ids]
        src_max_seq_length = max(src_seq_length)
        src_padded = []
        src_padding_mask = []
        for x in src_token_ids:
            x = x[:src_max_seq_length]
            src_pad_length = src_max_seq_length - len(x)
            src_padded.append(x + [1] * src_pad_length)
            src_padding_mask.append([1] * len(x) + [0] * src_pad_length)
        src_padded = torch.tensor(src_padded).t().contiguous().to(args.device)
        src_padding_mask = torch.tensor(src_padding_mask).bool().t().to(args.device)

        memory = model(src=src_padded, src_key_padding_mask=~src_padding_mask)

        tgt_token_ids = [[2] for _ in batch]
        end = [False for _ in batch]
        num_of_space = [0]*args.eval_batch_size
        for l in range(src_max_seq_length + 20):
            tgt = torch.tensor(tgt_token_ids).t().contiguous().to(args.device)
            output = model(tgt=tgt, memory=memory, memory_key_padding_mask=~src_padding_mask)
            top1 = output[-1].argmax(-1).tolist()
            for i, tok in enumerate(top1):
                # if tok == 3 or l >= src_seq_length[i] + 20:
                #     end[i] = True
                #original ver.
                if tok == 3 or l >= src_seq_length[i] + int(src_seq_length[i] * length_limit):
                    end[i] = True
                ##ignore space ver.
                '''
                if length_limit != None:
                    src_seq_length_wo_space = src_seq_length[i] - src_num_of_space[i]
                    if tok == 4:
                        num_of_space[i] += 1
                    if tok == 3 or l-num_of_space[i] >= src_seq_length_wo_space + int(src_seq_length_wo_space*length_limit)+1:
                        end[i] = True
                else:
                    if tok == 3 or l >= src_seq_length[i] + 20:
                        end[i] = True
                '''

                tgt_token_ids[i].append(tok if not end[i] else 3)
            if all(end):
                break

        if args.tokenizer == 'char':
            prediction.extend([''.join([tokenizer.i2c[tok] for tok in x if tok >= 4]).strip() for x in tgt_token_ids])
        if args.tokenizer == 'kobert':
            prediction.extend([tokenizer.decode(x, skip_special_tokens=True).strip() for x in tgt_token_ids])
    return prediction, src_token_string

def train(model, tokenizer, train_data, valid_data, args, eos=False):
    model.train()

    train_dataset = TextDataset(train_data)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                  batch_size=args.train_batch_size, num_workers=args.num_workers,
                                  collate_fn=lambda x: collate_fn_MLM(x, tokenizer, args.max_seq_length, eos=eos, tokenizer_type=args.tokenizer))

    valid_dataset = TextDataset(valid_data)
    valid_dataloader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset),
                                  batch_size=args.eval_batch_size, num_workers=args.num_workers,
                                  collate_fn=lambda x: collate_fn_MLM(x, tokenizer, args.max_seq_length, eos=eos, tokenizer_type=args.tokenizer))

    valid_clean = [x for x in valid_data]

    epochs = (args.max_steps - 1) // len(train_dataloader) + 1
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=eval(args.adam_betas), eps=args.eps,
                                 weight_decay=args.weight_decay)
    lr_lambda = lambda x: x / args.num_warmup_steps if x <= args.num_warmup_steps else (x / args.num_warmup_steps) ** -0.5
    scheduler = LambdaLR(optimizer, lr_lambda)

    step = 0
    best_val_gleu = -float("inf")
    meter = Meter()
    for epoch in range(1, epochs + 1):
        print("===EPOCH: ", epoch)
        for batch in train_dataloader:
            step += 1
            batch = tuple(t.to(args.device) for t in batch)
            loss, items = calc_loss(model, batch)
            meter.add(*items)

            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()
            scheduler.step()

            if step % args.log_interval == 0:
                lr = scheduler.get_lr()[0]
                loss_sent, loss_token = meter.average()

                logger.info(f' [{step:5d}] lr {lr:.6f} | {meter.print_str(True)}')
                nsml.report(step=step, scope=locals(), summary=True,
                            train__lr=lr, train__loss_sent=loss_sent, train__token_ppl=math.exp(loss_token))
                meter.init()

            if step % args.eval_interval == 0:
                start_eval = time.time()
                (val_loss, val_loss_token), valid_str = evaluate(model, valid_dataloader, args)
                prediction, valid_masked = correct(model, tokenizer, valid_clean, args, eos=eos, length_limit=0.1)
                val_em = em(prediction, valid_clean)
                cnt = 0
                # print("zero index check", valid_masked[0], prediction[0], valid_clean[0])
                for noisy, pred, clean in zip(valid_masked, prediction, valid_clean):
                    print(f'[{noisy}], [{pred}], [{clean}]')
                    # 10개만 출력하기
                    cnt += 1
                    if cnt == 20:
                        break
                val_gleu = gleu(prediction, valid_clean)

                logger.info('-' * 89)
                logger.info(f' [{step:6d}] valid | {valid_str} | em {val_em:5.2f} | gleu {val_gleu:5.2f}')
                logger.info('-' * 89)
                nsml.report(step=step, scope=locals(), summary=True,
                            valid__loss_sent=val_loss, valid__token_ppl=math.exp(val_loss_token),
                            valid__em=val_em, valid__gleu=val_gleu)

                if val_gleu > best_val_gleu:
                    best_val_gleu = val_gleu
                    nsml.save("best")
                meter.start += time.time() - start_eval

            if step >= args.max_steps:
                break
        #nsml.save(epoch)
        if step >= args.max_steps:
            break


def bind_nsml(model, tokenizer=None, args=None, eos=False):
    def save(path, **kwargs):
        torch.save(model.state_dict(), open(os.path.join(path, 'model.pt'), 'wb'))
        if args.tokenizer == 'char' and tokenizer is not None:
            tokenizer.save(os.path.join(path, 'vocab.txt'))

    def load(path, **kwargs):
        model.load_state_dict(torch.load(open(os.path.join(path, 'model.pt'), 'rb'),
                                         map_location=lambda storage, loc: storage))
        if args.tokenizer == 'char' and tokenizer is not None:
            tokenizer.load(os.path.join(path, 'vocab.txt'))

    def infer(test_data, **kwargs):
        '''
        :param test_data: list of noisy sentences
        :return: list of corrected sentences
        '''

        # 특수문자 지우기 일단 꺼놓음.
        # test_data = preprocess_noisy_sentence_list(test_data)

        if args.tokenizer == 'kobert':
            tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

        return correct(model, tokenizer, test_data, args, eos=eos, length_limit=0.1)

    import nsml
    nsml.bind(save, load, infer)

def main():
    args = get_args()
    logger.info(f"args: {json.dumps(args.__dict__, indent=2, sort_keys=True)}")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    if args.tokenizer == 'char':
        tokenizer = CharTokenizer([])
    if args.tokenizer == 'kobert':
        print("koBERT tokenizer")
        tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
        args.vocab_size = tokenizer.vocab_size
        print(args.vocab_size)

    model = TransformerModel(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
    ).to(args.device)
    logger.info(f"# of model parameters: {sum(p.numel() for p in model.parameters()) * 1e-6:.2f}M")

    eos_setting = args.eos_setting

    bind_nsml(model, tokenizer, args, eos=eos_setting)
    if args.pause:
        nsml.paused(scope=locals())

    #train_data, valid_data = None, None
    if args.mode == "train":
        noisy_sents_labeled = read_strings(os.path.join(args.data_dir, "train_data", "train_data"))
        sents_annotation_labeled = read_strings(os.path.join(args.data_dir, "train_data", "train_annotation"))
        clean_sents_labeled = read_strings(os.path.join(args.data_dir, "train_label"))
        noisy_sents = read_strings(os.path.join(args.data_dir, "train_data", "train_corpus"))

        pairs = noisy_sents
        pairs_labeled = clean_sents_labeled

        train_data, valid_data = pairs+pairs_labeled[:-args.num_val_data], pairs_labeled[-args.num_val_data:]
        logger.info(f"# of train data: {len(train_data)}")
        logger.info(f"# of valid data: {len(valid_data)}")

        train_sents = [x for x in train_data]

        if args.tokenizer == 'char': tokenizer = CharTokenizer.from_strings(train_sents, args.vocab_size)
        bind_nsml(model, tokenizer, args, eos=eos_setting)


    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, dim=1)

    if args.mode == "train":
        train(model, tokenizer, train_data, valid_data, args, eos=eos_setting)


if __name__ == "__main__":
    main()
