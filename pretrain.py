import argparse
import json
import logging
import os
import torch
from sklearn.model_selection import train_test_split
from tokenizer import CharTokenizer
from data_loader import read_strings
from model import TransformerModel
from train import set_seed, train

logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--val_ratio", type=float, default=0.0005)

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
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--step_lr", action="store_true")
    parser.add_argument("--step_gamma", type=float, default=0.5)
    parser.add_argument("--adam_betas", type=str, default="(0.9, 0.98)")
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--max_grad_norm", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--semi_dataset', type=int, default=0)
    parser.add_argument('--eos_setting', type=bool, default=True)
    parser.add_argument('--eos_multiple', type=float, default=1.5)
    parser.add_argument('--add_noise', action="store_true")
    parser.add_argument('--beamsearch', action="store_true")
    parser.add_argument('--tokenizer', type=str, default="char")

    # pretrain
    parser.add_argument('--mode', type=str, default="pretrain")
    parser.add_argument('--cuda', type=str, default="1")
    parser.add_argument('--model_name', type=str, default="pretrained.pt")
    parser.add_argument('--noisy_file', type=str, default="sejong_noisy.txt")
    parser.add_argument('--clean_file', type=str, default="sejong_clean.txt")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    args.n_gpu = 1

    set_seed(args)

    # Construct tokenizer
    tokenizer = CharTokenizer([])
    tokenizer.load(args.load_vocab)
    args.vocab_size = len(tokenizer)

    logger.info(f"args: {json.dumps(args.__dict__, indent=2, sort_keys=True)}")

    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Construct model
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

    # Load data
    noisy_sents = read_strings(os.path.join('sejong_corpus', args.noisy_file))
    clean_sents = read_strings(os.path.join('sejong_corpus', args.clean_file))
    sents_annotation = ['None'] * len(noisy_sents)

    pairs = [{"noisy": noisy, "clean": clean, "annotation": annot} for noisy, clean, annot in zip(noisy_sents, clean_sents, sents_annotation)]

    # Train-validation split
    train_data, valid_data = train_test_split(pairs, test_size=args.val_ratio, random_state=args.seed)  # test: about 1000
    logger.info(f"# of train data: {len(train_data)}")
    logger.info(f"# of valid data: {len(valid_data)}")

    train(model, tokenizer, train_data, valid_data, args, eos=args.eos_setting)


if __name__ == "__main__":
    main()
