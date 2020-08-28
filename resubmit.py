import argparse
from tokenizer import CharTokenizer
from model import TransformerModel
from train import bind_nsml

parser = argparse.ArgumentParser()
parser.add_argument('--session', type=str, default="")
parser.add_argument('--checkpoint', type=str, default="best")
args = parser.parse_args()
args.tokenizer = 'char'

session = 't0005/rush1-3/' + args.session
checkpoint = args.checkpoint
print(f'session: {session}\ncheckpoint: {checkpoint}')

model = None
tokenizer = CharTokenizer([])

bind_nsml(model, tokenizer)
nsml.load(checkpoint=checkpoint, session=session)

args.vocab_size = len(tokenizer)
print(f'vocab_size: {args.vocab_size}')

model = TransformerModel(
    vocab_size=args.vocab_size,
    hidden_size=args.hidden_size,
    num_attention_heads=args.num_attention_heads,
    num_encoder_layers=args.num_encoder_layers,
    num_decoder_layers=args.num_decoder_layers,
    intermediate_size=args.intermediate_size,
    dropout=args.dropout,
).to(args.device)

bind_nsml(model, tokenizer, args, eos=True)
nsml.load(checkpoint=checkpoint, session=session)
nsml.save('best')
