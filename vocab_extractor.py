import os
import nsml
from nsml import DATASET_PATH
from tokenizer import CharTokenizer
from collections import Counter
from data_loader import read_strings


def bind_nsml(vocab_noisy, vocab_unlabeled, vocab_clean, vocab_total):
    def save(path, **kwargs):
        with open(os.path.join(path, 'vocab_noisy.txt'), 'w') as f:
            for token in vocab_noisy:
                f.write(token + '\n')
        with open(os.path.join(path, 'vocab_unlabeled.txt'), 'w') as f:
            for token in vocab_unlabeled:
                f.write(token + '\n')
        with open(os.path.join(path, 'vocab_clean.txt'), 'w') as f:
            for token in vocab_clean:
                f.write(token + '\n')
        with open(os.path.join(path, 'vocab_total.txt'), 'w') as f:
            for token in vocab_total:
                f.write(token + '\n')
        print("vocab saved!")

    nsml.bind(save)


data_dir = os.path.join(DATASET_PATH, 'train')
noisy_sents = read_strings(os.path.join(data_dir, "train_data", "train_data"))
unlabeled_sents = read_strings(os.path.join(data_dir, "train_data", "train_corpus"))
clean_sents = read_strings(os.path.join(data_dir, "train_label"))

noisy_counter = Counter()
for x in noisy_sents:
    noisy_counter.update(x)

unlabeled_counter = Counter()
for x in unlabeled_sents:
    unlabeled_counter.update(x)

clean_counter = Counter()
for x in clean_sents:
    clean_counter.update(x)

total_counter = noisy_counter + unlabeled_counter + clean_counter

noisy_total = sum(noisy_counter.values())
unlabeled_total = sum(unlabeled_counter.values())
clean_total = sum(clean_counter.values())

vocab_noisy = set()
vocab_unlabeled = set()
vocab_clean = set()

ratio = 0.999

accu_cnt = 0
for char, cnt in sorted(noisy_counter.items(), key=lambda item: item[1], reverse=True):
    vocab_noisy.add(char)
    accu_cnt += cnt
    if accu_cnt / noisy_total >= ratio:
        break

accu_cnt = 0
for char, cnt in sorted(unlabeled_counter.items(), key=lambda item: item[1], reverse=True):
    vocab_unlabeled.add(char)
    accu_cnt += cnt
    if accu_cnt / unlabeled_total >= ratio:
        break

accu_cnt = 0
for char, cnt in sorted(clean_counter.items(), key=lambda item: item[1], reverse=True):
    vocab_clean.add(char)
    accu_cnt += cnt
    if accu_cnt / clean_total >= ratio:
        break

vocab_total = vocab_noisy.union(vocab_unlabeled).union(vocab_clean)
vocab_total = sorted(list(vocab_total), key=lambda x: -total_counter[x])
print(vocab_total)
bind_nsml(vocab_noisy, vocab_unlabeled, vocab_clean, vocab_total)
nsml.save('vocab')
