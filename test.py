from model import TransformerModel
from train import get_args, bind_nsml, evaluate, correct, correct_beam
from tokenizer import CharTokenizer
import torch
import nsml
from torch.utils.data import DataLoader, SequentialSampler
from dataset import TextDataset, collate_fn
from data_loader import read_strings
import os
from evaluation import gleu, gleu_one

from collections import Counter, defaultdict
from statistics import mean, stdev
from tokenization_kobert import KoBertTokenizer

from sklearn.model_selection import train_test_split

args = get_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = CharTokenizer([])


model = TransformerModel(
    vocab_size=args.vocab_size,
    hidden_size=args.hidden_size,
    num_attention_heads=args.num_attention_heads,
    num_encoder_layers=args.num_encoder_layers,
    num_decoder_layers=args.num_decoder_layers,
    intermediate_size=args.intermediate_size,
    dropout=args.dropout,
).to(args.device)

checkpoint = 'best'
sess = 't0005/rush1-3/507'

bind_nsml(model, tokenizer, args)
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

bind_nsml(model, tokenizer, args)
nsml.load(checkpoint=checkpoint, session=sess)

noisy_sents = read_strings(os.path.join(args.data_dir, "train_data", "train_data"))
sents_annotation = read_strings(os.path.join(args.data_dir, "train_data", "train_annotation"))
clean_sents = read_strings(os.path.join(args.data_dir, "train_label"))
pairs = [{"noisy": noisy, "clean": clean, "annotation": annot} for noisy, clean, annot in zip(noisy_sents, clean_sents, sents_annotation)]
train_data, valid_data = pairs[:-1000], pairs[-1000:]
#train_data, valid_data = train_test_split(pairs, test_size=args.val_ratio, random_state=2020)


# 길이가 작은 순으로 sort
valid_data.sort(key=lambda x: len(x['noisy']))
for i in range(0,len(valid_data), 100):
    try:
        print("{}~{}".format(len(valid_data[i]['noisy']), len(valid_data[i+99]['noisy'])))
    except:
        print("last batch: ", i, len(valid_data))
        print("{}~{}".format(len(valid_data[i]['noisy']), len(valid_data[-1]['noisy'])))

valid_dataset = TextDataset(valid_data)
valid_dataloader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset),
                                  batch_size=args.eval_batch_size, num_workers=args.num_workers,
                                  collate_fn=lambda x: collate_fn(x, tokenizer, args.max_seq_length, eos=args.eos_setting, tokenizer_type=args.tokenizer))

(val_loss, val_loss_token), valid_str = evaluate(model, valid_dataloader, args)

valid_noisy = [x['noisy'] for x in valid_data]
valid_clean = [x['clean'] for x in valid_data]
valid_annot = [x['annotation'] for x in valid_data]
prediction = correct_beam(model, tokenizer, valid_noisy, args, eos=args.eos_setting, length_limit=0.15)

print("total GLEU: ", gleu(prediction, valid_clean))

# 길이가 작은 순으로 batch_size마다 GLUE score 찍어보자
batch_size = 100
for i in range((len(valid_noisy) + batch_size - 1) // batch_size):
    val_gleu = gleu(prediction[i*batch_size:(i+1)*batch_size], valid_clean[i*batch_size:(i+1)*batch_size])
    print(f'batch={i}, gleu={val_gleu:5.2f}')

### error type별 갯수 (validation)
error_type_counter = Counter()

for annotation in sents_annotation:
    error_type_counter += Counter(annotation.split(','))

print(error_type_counter)

## 에러별 gleu score를 찍어보자
error_types = ['punctuation','spacing','recommendation','typos','honorific','tense']

gleu_per_error = defaultdict(list)

for noisy, pred, clean, annot_junk in zip(valid_noisy, prediction, valid_clean, valid_annot):
    annots = annot_junk.split(",")
    for annot in annots:
        if annot == 'tense':
            print(pred," ", clean)
        gleu_per_error[annot].append(gleu_one(pred, clean))

for error in error_types:
    if gleu_per_error[error]:
        if len(gleu_per_error[error]) > 1:
            print("Macro avg. GLEU score of {}: mean: {}, std: {}, count: {}"
                  .format(error, mean(gleu_per_error[error]), stdev(gleu_per_error[error]),
                          len(gleu_per_error[error])))
        else:
            print("Macro avg. GLEU score of {}: mean: {}, std: _, count: {}"
                  .format(error, mean(gleu_per_error[error]), len(gleu_per_error[error])))

    else:
        print("there are no such error [{}] in validation set", error)

# 문장 하나씩 출력
for noisy, pred, clean, annot_junk in zip(valid_noisy, prediction, valid_clean, valid_annot):
    print(f'[{noisy}], [{pred}], [{clean}]', annot_junk)
