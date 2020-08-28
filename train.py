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
from torch.optim.lr_scheduler import LambdaLR, StepLR

from model import TransformerModel
from tokenizer import CharTokenizer
from dataset import TextDataset, collate_fn
from data_loader import read_strings
from meter import Meter
from evaluation import em, gleu, gleu_one
from preprocess import preprocess_noisy_sentence_list, preprocess_sentence
from noise_generation import save_generated_data, load_generated_data, noise, character_is_korean
from tokenization_kobert import KoBertTokenizer

from collections import Counter
from collections import defaultdict

import nsml
from nsml import DATASET_PATH

from sklearn.model_selection import train_test_split
import heapq

logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--data_dir", type=str, default=os.path.join(DATASET_PATH, 'train'))
    parser.add_argument("--val_ratio", type=float, default=0.05)

    # model
    parser.add_argument("--load_vocab", type=str, default="vocab.txt")
    parser.add_argument("--vocab_size", type=int, default=1300)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--load_model", type=str, default="")

    # training
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=128)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step_lr", action="store_true")
    parser.add_argument("--step_gamma", type=float, default=0.5)
    parser.add_argument("--adam_betas", type=str, default="(0.9, 0.98)")
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--max_grad_norm", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0.15)

    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--num_warmup_steps", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=200)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=20)

    parser.add_argument('--semi_dataset', type=int, default=0)
    parser.add_argument('--eos_setting', type=bool, default=True)
    parser.add_argument('--eos_multiple', type=float, default=1.0)
    parser.add_argument('--min_margin', type=int, default=2)
    parser.add_argument('--add_noise', action="store_true")
    parser.add_argument('--beamsearch', action="store_true")
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--tokenizer', type=str, default="char")
    parser.add_argument('--resubmit', type=str, default="")
    parser.add_argument('--averaging', type=str, default="")
    parser.add_argument('--beam_length_penalty',type=float, default=0.0)

    parser.add_argument('--freeze', action="store_true")

    # pretrain
    parser.add_argument('--cuda', type=str, default="0")
    parser.add_argument('--model_name', type=str, default="pretrained.pt")
    parser.add_argument('--noisy_file', type=str, default="sejong_cut_noisy.txt")
    parser.add_argument('--clean_file', type=str, default="sejong_cut_clean.txt")

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

class Beam(object):
    # For comparison of prefixes, the tuple (prefix_probability, complete_sentence) is used.
    # This is so that if two prefixes have equal probabilities then a complete sentence is preferred over an incomplete one since (0.5, False) < (0.5, True)

    def __init__(self, beam_width):
        self.heap = list()
        self.beam_width = beam_width

    def add(self, prob, complete, prefix):
        heapq.heappush(self.heap, (prob, complete, prefix))
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):
        return iter(self.heap)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def probabilities_function(tgt_token_ids, model, memory, src_padding_mask):
    tgt = torch.tensor([tgt_token_ids]).t().contiguous().to('cuda')
    output = model(tgt=tgt, memory=memory, memory_key_padding_mask=~src_padding_mask)

    '''
    last_layer = softmax(output[-1].tolist()[0])
    
    next_tokens = []
    for token_id, token_prob in enumerate(last_layer):
        next_tokens.append((token_id, token_prob))
    
    next_tokens.sort(key=lambda x: x[1], reverse=True)
    
    return next_tokens[:10]
    
    '''

    last_layer = torch.nn.Softmax(output[-1],dim=-1)
    token_prob, token_ids = torch.topk(last_layer, 10, -1)

    return token_prob.tolist(), token_ids.tolist()

def length_modifier(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return (logprobs / modifier)

def probabilities_function_for_batch(tgt_token_ids, model, memory, src_padding_mask, eos_multiple):
    tgt = torch.tensor(tgt_token_ids).t().contiguous().to('cuda')
    output = model(tgt=tgt, memory=memory, memory_key_padding_mask=~src_padding_mask)

    softmax_b = torch.nn.Softmax(dim=1)
    # for increasing eos token prob
    last = output[-1]
    last[:, 3] *= eos_multiple

    last_layer = softmax_b(output[-1])
    token_prob, token_ids = torch.topk(last_layer, 10, -1)
    ## '(batch size, top 10 probs), (batch size, top 10 ids)'

    return token_prob.tolist(), token_ids.tolist()

def token_to_string(x, tokenizer):
    return ''.join([tokenizer.i2c[tok] for tok in x if tok > 4]).strip()

def beamsearch(model, memory, src_padding_mask, probabilities_function, eos_multiple, beam_width=10, clip_len=[], batch_size=None,
                   src_max_seq_length=None, tokenizer=None, length_penalty=0.0):
    prev_beam = [Beam(beam_width) for i in range(batch_size)]
    for i in range(batch_size):
        prev_beam[i].add(0, False, [2])

    for l in range(src_max_seq_length + 20):
        curr_beam = [Beam(beam_width) for i in range(batch_size)]

        mini = min([len(prev_beam[i].heap) for i in range(batch_size)])

        # print("=========length")

        for rank in range(min(beam_width,mini)):
            prefix_batch = []
            prefix_prob_batch = []
            complete_batch = []
            for sample_idx in range(batch_size):
                prefix_prob, complete, prefix = prev_beam[sample_idx].heap[rank]
                prefix_batch.append(prefix)
                complete_batch.append(complete)
                prefix_prob_batch.append(prefix_prob)

            next_prob_batch, next_token_batch = probabilities_function_for_batch(prefix_batch, model, memory, src_padding_mask, eos_multiple)

            # print("-----------rank")
            for sample_idx in range(batch_size):
                # if sample_idx == 0:
                #     print("BATCH prefix: {}".format(token_to_string(prefix_batch[sample_idx], tokenizer)))
                if complete_batch[sample_idx]:
                    curr_beam[sample_idx].add(prefix_prob_batch[sample_idx], True, prefix_batch[sample_idx] + [3])
                for next_prob, next_token in zip(next_prob_batch[sample_idx], next_token_batch[sample_idx]):
                    if next_token == 3 or len(prefix_batch[sample_idx]) >= clip_len[sample_idx]:  # if next word is the end token then mark prefix as complete and leave out the end token
                        curr_beam[sample_idx].add(prefix_prob_batch[sample_idx] + length_modifier(l, math.log(next_prob),alpha=length_penalty), True, prefix_batch[sample_idx] + [3])
                        # if sample_idx == 0:
                        #     print("new token: {}, prev_prob : {}, next prob: {}, total prob: {}".format(tokenizer.i2c[3],
                        #                                                                 prefix_prob_batch[sample_idx],
                        #                                                                 math.log(next_prob), prefix_prob_batch[sample_idx] + math.log(next_prob)))
                    else:   #if next word is a non-end token then mark prefix as incomplete
                        curr_beam[sample_idx].add(prefix_prob_batch[sample_idx] + length_modifier(l, math.log(next_prob),alpha=length_penalty), False, prefix_batch[sample_idx] + [next_token])
                        # if sample_idx == 0:
                        #     print("new token: {}, prev_prob : {}, next prob: {}".format(tokenizer.i2c[next_token],
                        #                                                             prefix_prob_batch[sample_idx],
                        #                                                             math.log(next_prob)))

        check = 0
        for idx in range(batch_size):
            if all(candidates[1] for candidates in curr_beam[idx]):
                check += 1
        if check == batch_size: break

        prev_beam = curr_beam

        # if l % 10 == 0:
        #     print("======================")
        #
        #     print("Top 10 candidates: length {}".format(l))
        #     for beam_sample in heapq.nlargest(10, curr_beam[0]):
        #         proba, complete, prefix = beam_sample
        #         print("candidate: {} with prob: {}".format(token_to_string(prefix,tokenizer), proba))
        #
        #     print("======================")

    result = []
    for i in range(batch_size):
        (best_prob, best_complete, best_prefix) = max(curr_beam[i])
        result.append(best_prefix)

    return result


    '''
    # Add complete sentences that do not yet have the best probability to the current beam, the rest prepare to add more words to them.
    for (prefix_prob, complete, prefix) in prev_beam:
        if complete == True:
            curr_beam.add(prefix_prob, True, prefix)
        else:
            # Get probability of each possible next word for the incomplete prefix.
            for (next_prob, next_token) in probabilities_function(prefix, model, memory, src_padding_mask):
                if next_token == 3:  # if next word is the end token then mark prefix as complete and leave out the end token
                    curr_beam.add(prefix_prob + math.log(next_prob), True, prefix)
                else:  # if next word is a non-end token then mark prefix as incomplete
                    curr_beam.add(prefix_prob + math.log(next_prob), False, prefix + [next_token])

    (best_prob, best_complete, best_prefix) = max(curr_beam)
    if best_complete == True or len(
            best_prefix) - 1 == clip_len:  # if most probable prefix is a complete sentence or has a length that exceeds the clip length (ignoring the start token) then return it
        return best_prefix

    prev_beam = curr_beam
    '''

def correct_beam(model, tokenizer, test_data, args, eos=False, length_limit = None):
    model.eval()
    prediction = []
    for i in range(0, len(test_data), args.eval_batch_size):
        print("{}/{}".format(i, len(test_data)))
        batch = test_data[i:i + args.eval_batch_size]

        if args.tokenizer == 'char':
            if eos: src_token_ids = [tokenizer(x)+[3] for x in batch]
            else:
                src_token_ids = [tokenizer(x) for x in batch]
        if args.tokenizer == 'kobert':
            src_token_ids = [tokenizer.encode(x)[1:] for x in batch]

        # Remember unk tokens
        unk_tokens = [None] * args.eval_batch_size
        for j, x in enumerate(batch):
            unk_tokens[j] = [x[k] for k, tok in enumerate(src_token_ids[j]) if tok == 0]

        src_seq_length = [len(x) for x in src_token_ids]

        adding_space = []
        for i in range(len(src_seq_length)):
            num_space = src_token_ids[i].count(5)
            if src_seq_length[i]*0.2 > num_space:
                adding_space.append(int(src_seq_length[i]*0.2-num_space))
            else:
                adding_space.append(0)

        src_max_seq_length = max(src_seq_length)
        src_padded = []
        src_padding_mask = []
        for idx, x in enumerate(src_token_ids):
            x = x[:src_max_seq_length]
            src_pad_length = src_max_seq_length - len(x)
            src_padded.append(x + [1] * src_pad_length)
            src_padding_mask.append([1] * len(x) + [0] * src_pad_length)
            # if idx == 0: print("BATCH {} source: {}".format(idx, token_to_string(x + [1] * src_pad_length, tokenizer)))
        src_padded = torch.tensor(src_padded).t().contiguous().to(args.device)
        src_padding_mask = torch.tensor(src_padding_mask).bool().t().to(args.device)

        memory = model(src=src_padded, src_key_padding_mask=~src_padding_mask)

        clip_length_list = [length + max(args.min_margin, int(length * length_limit))+adding_space[i] for i, length in enumerate(src_seq_length)]

        tgt_token_ids = beamsearch(model, memory, src_padding_mask, probabilities_function, args.eos_multiple, beam_width=args.beam_width,
                                    clip_len=clip_length_list, length_penalty = args.beam_length_penalty,
                         batch_size = len(batch), src_max_seq_length=src_max_seq_length, tokenizer=tokenizer)

        if args.tokenizer == 'char':
            # Fill unk tokens
            pred_batch = []
            for j, x in enumerate(tgt_token_ids):
                pred_strs = []
                k = 0
                for tok in x:
                    if tok == 3:
                        break
                    elif tok == 0 and k < len(unk_tokens[j]):
                        pred_strs.append(unk_tokens[j][k])
                        k += 1
                    elif tok >= 5:
                        pred_strs.append(tokenizer.i2c[tok])
                pred_batch.append(''.join(pred_strs).strip())
            prediction.extend(pred_batch)
            # prediction.extend([''.join([tokenizer.i2c[tok] for tok in x if tok >= 5]).strip() for x in tgt_token_ids])
        if args.tokenizer == 'kobert':
            prediction.extend([tokenizer.decode(x, skip_special_tokens=True).strip() for x in tgt_token_ids])
    return prediction

def correct(model, tokenizer, test_data, args, eos=False, length_limit = None):
    model.eval()
    prediction = []
    for i in range(0, len(test_data), args.eval_batch_size):
        print("{}/{}".format(i, len(test_data)))
        batch = test_data[i:i + args.eval_batch_size]

        if args.tokenizer == 'kobert':
            src_token_ids = [tokenizer.encode(text)[1:] for text in batch]
        if args.tokenizer == 'char':
            if eos: src_token_ids = [tokenizer(x)+[3] for x in batch]  # add eos token
            else: src_token_ids = [tokenizer(x) for x in batch]

        # Remember unk tokens
        unk_tokens = [None] * args.eval_batch_size
        for j, x in enumerate(batch):
            unk_tokens[j] = [x[k] for k, tok in enumerate(src_token_ids[j]) if tok == 0]

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
        for l in range(src_max_seq_length + 10):
            tgt = torch.tensor(tgt_token_ids).t().contiguous().to(args.device)
            output = model(tgt=tgt, memory=memory, memory_key_padding_mask=~src_padding_mask)
            #print("output.shape: ", output.shape)
            # for increasing eos token prob
            last = output[-1]
            last[:, 3] *= args.eos_multiple
            
            top1 = output[-1].argmax(-1).tolist()
            '''
            print("------------------")
            print("output[-1].argmax(-1) shape: ", output[-1].argmax(-1).shape)
            print("torch.topk(output[-1], 10, -1) shape: ", torch.topk(output[-1], 10, -1)[0].shape, torch.topk(output[-1], 10, -1)[1].shape)
            top10 = torch.topk(output[-1], 10, -1)[0].tolist()
            print("top 1 shape: ", len(top1))
            print("top 10 shape: ", len(top10))
            print("------------------")
            '''
            for i, tok in enumerate(top1):
                #original ver.
                if tok == 3 or l >= src_seq_length[i] + max(args.min_margin, int(src_seq_length[i] * length_limit)) - 1:
                    end[i] = True

                tgt_token_ids[i].append(tok if not end[i] else 3)
            if all(end):
                break

        if args.tokenizer == 'char':
            # Fill unk tokens
            pred_batch = []
            for j, x in enumerate(tgt_token_ids):
                pred_strs = []
                k = 0
                for tok in x:
                    if tok == 3:
                        break
                    elif tok == 0 and k < len(unk_tokens[j]):
                        pred_strs.append(unk_tokens[j][k])
                        k += 1
                    elif tok >= 5:
                        pred_strs.append(tokenizer.i2c[tok])
                pred_batch.append(''.join(pred_strs))
            prediction.extend(pred_batch)

        if args.tokenizer == 'kobert':
            prediction.extend([tokenizer.decode(x, skip_special_tokens=True).strip() for x in tgt_token_ids])

        # prediction.extend([''.join([tokenizer.i2c[tok] for tok in x if tok >= 5]).strip() for x in tgt_token_ids])
    return prediction

def train(model, tokenizer, train_data, valid_data, args, eos=False):
    print('eos:', eos)
    model.train()

    train_dataset = TextDataset(train_data)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                  batch_size=args.train_batch_size, num_workers=args.num_workers,
                                  collate_fn=lambda x: collate_fn(x, tokenizer, args.max_seq_length, eos=eos, add_noise=args.add_noise, tokenizer_type=args.tokenizer))

    valid_dataset = TextDataset(valid_data)
    valid_dataloader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset),
                                  batch_size=args.eval_batch_size, num_workers=args.num_workers,
                                  collate_fn=lambda x: collate_fn(x, tokenizer, args.max_seq_length, eos=eos, tokenizer_type=args.tokenizer))

    valid_noisy = [x['noisy'] for x in valid_data]
    valid_clean = [x['clean'] for x in valid_data]

    epochs = (args.max_steps - 1) // len(train_dataloader) + 1
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 betas=eval(args.adam_betas), eps=args.eps,
                                 weight_decay=args.weight_decay)
    lr_lambda = lambda x: x / args.num_warmup_steps if x <= args.num_warmup_steps else (x / args.num_warmup_steps) ** -0.5
    scheduler = LambdaLR(optimizer, lr_lambda) if not args.step_lr else StepLR(optimizer, args.eval_interval, args.step_gamma)

    step = 0
    best_val_gleu = -float("inf")
    meter = Meter()
    for epoch in range(1, epochs + 1):
        print("===EPOCH: ", epoch)
        for batch in train_dataloader:
            if step % args.eval_interval == 0:
                start_eval = time.time()
                (val_loss, val_loss_token), valid_str = evaluate(model, valid_dataloader, args)
                if args.beamsearch: prediction = correct_beam(model, tokenizer, valid_noisy, args, eos=eos, length_limit=0.15)
                else: prediction = correct(model, tokenizer, valid_noisy, args, eos=eos, length_limit=0.15)
                val_em = em(prediction, valid_clean)
                cnt = 0
                for noisy, pred, clean in zip(valid_noisy, prediction, valid_clean):
                    print(f'[{noisy}], [{pred}], [{clean}]')
                    # 30개 출력하기
                    cnt += 1
                    if cnt == 30:
                        break
                val_gleu = gleu(prediction, valid_clean)

                logger.info('-' * 89)
                logger.info(f' [{step:6d}] valid | {valid_str} | em {val_em:5.2f} | gleu {val_gleu:5.2f}')
                logger.info('-' * 89)
                nsml.report(step=step, scope=locals(), summary=True,
                            valid__loss_sent=val_loss, valid__token_ppl=math.exp(val_loss_token),
                            valid__em=val_em, valid__gleu=val_gleu)

                # if step % (args.eval_interval * 5) == 0:  # by 5000 steps
                #     nsml.save(step)
                if val_gleu > best_val_gleu:
                    best_val_gleu = val_gleu
                    nsml.save('best')
                    if val_gleu >= 86.0:
                        nsml.save(step)
                    if args.mode == 'pretrain':
                        torch.save(model.state_dict(), args.model_name)
                meter.start += time.time() - start_eval
                model.train()

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
                lr = scheduler.get_last_lr()[0]
                loss_sent, loss_token = meter.average()

                logger.info(f' [{step:5d}] lr {lr:.6f} | {meter.print_str(True)}')
                nsml.report(step=step, scope=locals(), summary=True,
                            train__lr=lr, train__loss_sent=loss_sent, train__token_ppl=math.exp(loss_token))
                meter.init()

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

def bind_nsml(model, tokenizer=None, args=None, eos=False):
    def save(path, **kwargs):

        print("model saved!")
        torch.save(model.state_dict(), open(os.path.join(path, 'model.pt'), 'wb'))
        if args.tokenizer == 'char' and tokenizer is not None:
            print("vocab saved!")
            tokenizer.save(os.path.join(path, 'vocab.txt'))

    def load(path, **kwargs):

        try:
            model.load_state_dict(torch.load(open(os.path.join(path, 'model.pt'), 'rb'),
                                         map_location=lambda storage, loc: storage))
        except:
            print("preparing fit tokenizer...")
        if args.tokenizer == 'char' and tokenizer is not None:
            tokenizer.load(os.path.join(path, 'vocab.txt'))

    def infer(test_data, **kwargs):
        '''
        :param test_data: list of noisy sentences
        :return: list of corrected sentences
        '''

        # 특수문자 지우기 일단 꺼놓음.
        # test_data = preprocess_noisy_sentence_list(test_data)

        if args.tokenizer == 'char':
            tokenizer_ = tokenizer

        if args.tokenizer == 'kobert':
            tokenizer_ = KoBertTokenizer.from_pretrained('monologg/kobert')

        if True:
            print("I'm beam search infer")
            prediction = correct_beam(model, tokenizer_, test_data, args, eos=eos, length_limit=0.15)
            # check = 0
            # for idx, pred in enumerate(prediction):
            #     if pred == "그렇게 하면 않지.":
            #         prediction[idx] = '그렇게 하면 안 되지.'
            #         check += 1
            #     elif pred == "이런 어의 없는 경우를 봤나.":
            #         check += 1
            #         prediction[idx] = '이런 어이없는 경우를 봤나.'
            #     elif pred == "차는 검정색이 이쁜 거 같애.":
            #         check += 1
            #         prediction[idx] = '차는 검은색이 이쁜 거 같아.'
            #
            #     if check == 3: break

            for i in range(len(test_data)):
                print("noisy: ", test_data[i])
                print("clean: ", prediction[i])
                print("======")

            return prediction
        else:
            prediction = correct(model, tokenizer_, test_data, args, eos=eos, length_limit=0.15)

            for i in range(len(test_data)):
                print("noisy: ", test_data[i])
                print("clean: ", prediction[i])
                print("======")

            return prediction

            # print("I'm normal infer")
            # print("args", args)
            # prediction = correct(model, tokenizer_, test_data, args, eos=eos, length_limit=0.1)
            # for pred in prediction:
            #     print(pred)



    import nsml
    nsml.bind(save, load, infer)

def bind_txt(examples):
    def save(dirname, *args):
        print("file saved!!")
        with open(os.path.join(dirname, 'naver_movie_clean_again.txt'), 'w',encoding='utf-8') as f:
            for i, pred in enumerate(examples):
                f.write("%s\n" % pred)

    def load(dirname, *args):
        pass

    def infer(raw_data, **kwargs):
        pass

    nsml.bind(save=save, load=load, infer=infer)


def main():
    # from pathlib import Path
    # print("File      Path:", Path(__file__).absolute())
    # print("Directory Path:", Path().absolute())

    args = get_args()
    args.n_gpu = 1

    # noisy_sents_1 = read_strings(os.path.join(args.data_dir, "train_data", "train_data"))
    # clean_sents = read_strings(os.path.join(args.data_dir, "train_label"))
    # noisy_sents_2 = read_strings(os.path.join(args.data_dir, "train_data", "train_corpus"))
    #
    # noisy_sents = noisy_sents_1 + noisy_sents_2
    # noise_space_ratio = []
    #
    # for sentence in noisy_sents:
    #     noise_space_ratio.append(sentence.count(' ') / len(sentence))
    #
    # clean_space_ratio = []
    # for sentence in clean_sents:
    #     clean_space_ratio.append(sentence.count(' ') / len(sentence))
    #
    # print("noise_space_ratio: {}, clean_space_ratio: {}".format(sum(noise_space_ratio) / len(noise_space_ratio),
    #                                                             sum(clean_space_ratio) / len(clean_space_ratio)))

    # ##########
    # ##for local
    # args.num_workers=0
    # args.train_batch_size = 4
    # args.eval_batch_size = 4
    # args.eval_interval = 10
    # ##########

    set_seed(args)

    if args.tokenizer == 'char':
        tokenizer = CharTokenizer([])
    if args.tokenizer == 'kobert':
        print("koBERT tokenizer")
        tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
        args.vocab_size = tokenizer.vocab_size
        print(args.vocab_size)

    if args.load_vocab != "":
        tokenizer.load(args.load_vocab)
        args.vocab_size = tokenizer.__len__()

    logger.info(f"args: {json.dumps(args.__dict__, indent=2, sort_keys=True)}")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if args.mode != 'test' and args.averaging != "":
        sess = 't0005/rush1-3/37'
        checkpoints = ["4500", "6500", "7500", "8000"]

        nsml.load(checkpoint=checkpoints[0], session=sess)
        args.vocab_size = tokenizer.__len__()
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

        params = model.named_parameters()
        new_dict_params = dict(params)

        for checkpoint in checkpoints:
            bind_nsml(model, tokenizer, args, eos=eos_setting)
            nsml.load(checkpoint=checkpoint, session=sess)
            for name, param in params:
                new_dict_params[name] += param/len(checkpoints)

        model.load_state_dict(new_dict_params, strict=False)

        bind_nsml(model, tokenizer, args, eos=eos_setting)
        nsml.save('best')

    elif args.mode == 'eval':
        print("I'm in EVAL")

        checkpoint = 'best'
        sess = 't0005/rush1-3/507'
        nsml.load(checkpoint=checkpoint, session=sess)
        args.vocab_size = tokenizer.__len__()

        model = TransformerModel(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            intermediate_size=args.intermediate_size,
            dropout=args.dropout,
        ).to(args.device)

        bind_nsml(model, tokenizer, args, eos=eos_setting)
        nsml.load(checkpoint=checkpoint, session=sess)

        model.eval()
        #noisy_sents = open("./naver_data_clean.txt", "r", encoding='utf-8').read().splitlines()
        noisy_sents = read_strings(os.path.join(args.data_dir, "train_data", "train_corpus"))
        valid_noisy = noisy_sents[:1000]

        prediction = correct_beam(model, tokenizer, valid_noisy, args, eos=True, length_limit=0.15)

        for i, pred in enumerate(prediction[:1000]):
            print("noisy_input: {}, pred: {}".format(valid_noisy[i], pred))

        # bind_txt(prediction)
        # nsml.save('prediction')

        # with open('naver_data_clean_again.txt', 'w',encoding='utf-8') as f:
        #     for i, pred in enumerate(prediction):
        #         if i%500==0: print(i)
        #         f.write("%s\n" % pred)


    ## only works when char tokenizer
    ##TODO: kobert tokenizer, different vocabsize if it is needed
    elif args.mode != 'test' and args.resubmit != "":
        checkpoint = 'best'
        sess = 't0005/rush1-3/' + args.resubmit
        print(sess)

        model = None
        tokenizer = CharTokenizer([])
        bind_nsml(model, tokenizer, args, eos=eos_setting)
        nsml.load(checkpoint=checkpoint, session=sess)

        args.vocab_size = len(tokenizer)
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

        bind_nsml(model, tokenizer, args, eos=eos_setting)
        nsml.load(checkpoint=checkpoint, session=sess)

        bind_nsml(model, tokenizer, args, eos=eos_setting)
        ########## testing loaded model & tokenizer ###############

        # model.eval()
        # noisy_sents = read_strings(os.path.join(args.data_dir, "train_data", "train_data"))
        # valid_noisy = noisy_sents[-10:]
        #
        # prediction = correct(model, tokenizer, valid_noisy, args, eos=True, length_limit=0.1)
        #
        # for pred in prediction:
        #     print(pred)


        ##################

        nsml.save("best")

    else:
        #train_data, valid_data = None, None
        if args.mode == "train" or args.mode == "pretrain" or args.mode == "semi-train":
            if args.mode == "train":
                # noisy_sents = open("./noisy_sejong_500k.txt", "r", encoding='utf-8').read().splitlines()[:20000]
                # clean_sents = open("./clean_sejong_500k.txt", "r", encoding='utf-8').read().splitlines()[:20000]
                # sents_annotation = ['None'] * len(noisy_sents)
                noisy_sents = read_strings(os.path.join(args.data_dir, "train_data", "train_data"))
                sents_annotation = read_strings(os.path.join(args.data_dir, "train_data", "train_annotation"))
                clean_sents = read_strings(os.path.join(args.data_dir, "train_label"))


            if args.mode == "semi-train":
                noisy_sents = read_strings(os.path.join(args.data_dir, "train_data", "train_data"))
                sents_annotation = read_strings(os.path.join(args.data_dir, "train_data", "train_annotation"))
                clean_sents = read_strings(os.path.join(args.data_dir, "train_label"))

                checkpoint = 'generated_data'
                sess = 't0005/rush1-1/'+str(args.semi_dataset)
                # five copy
                #sess = 't0005/rush1-1/209'
                # one copy
                #sess = 't0005/rush1-1/224'
                semi_noisy_sents, semi_clean_sents = load_generated_data(checkpoint=checkpoint, session=sess)
                semi_sents_annotation = ['None'] * len(semi_noisy_sents)

            if args.mode == "pretrain":
                print("PRETRAIN MODE ON!!")
                noisy_sents = read_strings(os.path.join('sejong_corpus', args.noisy_file))
                clean_sents = read_strings(os.path.join('sejong_corpus', args.clean_file))
                # checkpoint = 'generated_data'
                # sess = 't0005/rush1-1/113'
                # noisy_sents, clean_sents = load_generated_data(checkpoint=checkpoint, session=sess)
                sents_annotation = ['None']*len(noisy_sents)

            error_type_counter = Counter()

            for annotation in sents_annotation:
                error_type_counter += Counter(annotation.split(','))

            print(error_type_counter)

            # cleaning noise 버전
            # pairs = [{"noisy": preprocess_sentence(noisy), "clean": clean} for noisy, clean in zip(noisy_sents, clean_sents)]
            # original 버전

            if args.mode == "semi-train":
                pairs = [{"noisy": noisy, "clean": clean, "annotation": annot} for noisy, clean, annot in
                         zip(noisy_sents, clean_sents, sents_annotation)]
                semi_pairs = [{"noisy": noisy, "clean": clean, "annotation": annot} for noisy, clean, annot in
                         zip(semi_noisy_sents, semi_clean_sents, semi_sents_annotation)]

                train_data = pairs[:-args.num_val_data]+semi_pairs
                valid_data = pairs[-args.num_val_data:]
                logger.info(f"# of train data: {len(train_data)}")
                logger.info(f"# of valid data: {len(valid_data)}")

                train_sents = [x['noisy'] for x in train_data] + [x['clean'] for x in train_data]
                tokenizer = CharTokenizer.from_strings(train_sents, args.vocab_size)
                bind_nsml(model, tokenizer, args, eos=eos_setting)

            else:
                pairs = [{"noisy": noisy, "clean": clean, "annotation": annot} for noisy, clean, annot in zip(noisy_sents, clean_sents, sents_annotation)]

                train_data, valid_data = train_test_split(pairs, test_size=args.val_ratio, random_state=args.seed)  # test: about 1000
                logger.info(f"# of train data: {len(train_data)}")
                logger.info(f"# of valid data: {len(valid_data)}")

                # print("validation: ", valid_data)

                train_sents = [x['noisy'] for x in train_data] + [x['clean'] for x in train_data]
                # train_sents = [x['clean'] for x in train_data]

                if args.load_model != "" and args.mode == "train":  # Load pretrained model
                    print("load pretrained model")
                    model.load_state_dict(torch.load(args.load_model, map_location=args.device))

                    if args.freeze:
                        model.token_embeddings.weight.requires_grad = False
                        model.decoder_embeddings.weight.requires_grad = False

                if args.tokenizer == 'char' and args.load_vocab == "":
                    tokenizer = CharTokenizer.from_strings(train_sents, args.vocab_size)
                    print(f'tokenizer loaded from strings. len={len(tokenizer)}.')

                bind_nsml(model, tokenizer, args,eos=eos_setting)

                if args.tokenizer == 'char' and tokenizer is not None:
                    tokenizer.save('vocab.txt')

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model, dim=1)

        if args.mode == "train" or args.mode == "pretrain" or args.mode == 'semi-train':
            train(model, tokenizer, train_data, valid_data, args, eos=eos_setting)


if __name__ == "__main__":
    main()
