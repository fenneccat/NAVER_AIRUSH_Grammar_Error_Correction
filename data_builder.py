# save_generated_data test
from noise_generation import save_generated_data, load_generated_data, noise
from data_loader import read_strings
import os
from nsml import DATASET_PATH
import random

data_dir = os.path.join(DATASET_PATH, 'train')

## 데이터 copy를 몇개 만들것인지 정함. train_label clean data에서 noise만든것을 몇번 반복할건지
num_copy = 5

clean_sents = read_strings(os.path.join(data_dir, "train_label"))*num_copy
noisy_sents = noise(clean_sents)

shuffle_idxs = list(range(len(clean_sents)))
random.shuffle(shuffle_idxs)

noise_sents_shuf = []
clean_sents_shuf = []
for i in shuffle_idxs:
     noise_sents_shuf.append(noisy_sents[i])
     clean_sents_shuf.append(clean_sents[i])

save_generated_data(noise_sents_shuf, clean_sents_shuf)

## 현재 돌리는 세션의 이름을 넣으면 됨 't0005/rush1-1/(세션 번호)'. 그 위치에서 만들어진 데이터셋이 잘 로드되는지 확인할 수 있음
checkpoint = 'generated_data'
sess = 't0005/rush1-1/230'
loaded_noisy_sents, loaded_clean_sents = load_generated_data(checkpoint=checkpoint, session=sess)

print(loaded_noisy_sents[:10])
print(loaded_clean_sents[:10])