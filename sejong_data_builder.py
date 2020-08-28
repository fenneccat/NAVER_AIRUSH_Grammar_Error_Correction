from noise_generation import noise
from data_loader import read_strings, write_strings
from model import TransformerModel
from train import get_args, correct
from tokenizer import CharTokenizer
import torch

# args = get_args()
# args.eval_batch_size = 64
# args.vocab_size = 1300
# args.device = 'cuda:0'

# # load model (clean -> noisy)
# model = TransformerModel(
#         vocab_size=args.vocab_size,
#         hidden_size=args.hidden_size,
#         num_attention_heads=args.num_attention_heads,
#         num_encoder_layers=args.num_encoder_layers,
#         num_decoder_layers=args.num_decoder_layers,
#         intermediate_size=args.intermediate_size,
#         dropout=args.dropout,
#     ).to(args.device)
# model.load_state_dict(torch.load('reverse/model.pt', map_location=args.device))

# tokenizer = CharTokenizer([])
# tokenizer.load('reverse/vocab.txt')

clean_sents = read_strings('sejong_corpus/sejong_clean.txt')

# noisy_sents = correct(model, tokenizer, clean_sents, args, eos=args.eos_setting, length_limit=0.1)
noisy_sents = noise(clean_sents)

write_strings('sejong_corpus/sejong_noisy.txt', noisy_sents)
