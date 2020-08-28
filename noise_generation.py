from tqdm import tqdm_notebook

import numpy as np
import re
import shutil
import random
#from g2pk import G2p

kor_begin = 44032
kor_end = 55203
chosung_base = 588
jungsung_base = 28
jaum_begin = 12593
jaum_end = 12622
moum_begin = 12623
moum_end = 12643

chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ',
                'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ',
                 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
                 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ',
                 'ㅡ', 'ㅢ', 'ㅣ']

jongsung_list = [
    ' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ',
    'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ',
    'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ',
    'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jaum_list = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ',
             'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ',
             'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

moum_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
             'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

possible_char = [chosung_list, jungsung_list, jongsung_list]

confusing_dict = {
    '마음': ['맴', '맘'],
    '세다': ['쎄다'],
    '안에는': ['안엔'],
    '밖에는': ['밖엔'],
    '귀여운': ['뽀짝'],
    '이런': ['요런'],
    '시간': ['time', '타임'],
    '주스': ['쥬스'],
    '아': ['애'],
    '돼': ['되'],
    '되': ['돼'],
    '된': ['됀'],
    '안': ['않'],
    '않': ['안'],
    '왜': ['외'],
    '원': ['one'],
    '가격': ['price'],
    '사이즈': ['싸이즈'],
    '주문': ['오더', 'order'],
    '대박': ['대애박', '대애애박', '대애애애박'],
    '피시': ['피', '피씨', 'PC'],
    '바람': ['바램'],
    '너무': ['넘', '늠', '느무', '느무느무'],
    '좋아': ['죠아', '조아', '조아조아'],
    '바탕': ['베이스'],
    '아기': ['애기', '애긔'],
    '엄마': ['맘', '마미'],
    '어머니': ['맘', '마미'],
    '음악': ['뮤직'],
    '갔다': ['가따'],
    '왔다': ['와따'],
    '설레': ['설레이'],
    '버렸어': ['부렀어', '부럿어'],
    '딱': ['뙇'],
    '그저 그렇': ['쏘쏘', '소소'],
    '그저 그러': ['쏘쏘', '소소'],
    '그저 그런': ['쏘쏘', '소소'],
    '이렇': ['요렇', '요롷'],
    '갑니다': ['고고', 'gogo'],
    '가십시오': ['고고', 'gogo', '고고씽', '고고링'],
    '고기': ['꼬기'],
    '좋은': ['굿', '굳'],
    '좋다': ['굿', '굳'],
    '하루': ['데이'],
    '하나도': ['1도'],
    '메뉴': ['menu'],
    ' 대 ': [' vs '],
    '팩스': ['fax'],
    '이메일': ['email', 'e-mail'],
    '진짜': ['실화', '리얼'],
    '놀람': ['놀램'],
    '채팅': ['챗'],
    '최고의': ['베스트'],
    '맛있는 것': ['jmt', '존맛탱', '존맛', '킹맛'],
    '맛있다': ['jmt', '존맛탱', '존맛', '킹맛'],
    '고추냉이': ['와사비'],
    '전망': ['뷰'],
    '시작': ['스타트', '스따또', '스타또', '스타뜨'],
    '이건': ['요건'],
    '어떻게': ['어케'],
    '스테이크': ['steak', '스떼끼', '스테끼'],
    '누구를': ['누굴'],
    '재미있다': ['존잼', '잼'],
    '요가': ['yoga'],
    '때는': ['땐'],
    '바로': ['right', '롸잇'],
    '예약': ['reservation'],
    '되게': ['디게', '디기'],
    '여기': ['요기'],
    '저기': ['죠기', '조기'],
    '구웠': ['굽굽'],
    '고양이': ['냥이', '냥냥이', '냥', '고냥이'],
    '하지만': ['but'],
    '꾸덕꾸덕': ['꾸덕'],
    '잖아': ['쟈나', '쟈냐'],
    '얼른': ['언능', '언넝', '얼릉', '얼렁'],
    '정말': ['겁나', '검나', '존나'],
    '주차': ['파킹', 'parking'],
    '예뻐': ['예뿌', '이뿌'],
    '잘 어울리': ['찰떡'],
    '짭조름': ['짬조롬', '짬쪼롬', '짭자름'],
    '작은': ['쪼꼼', '쪼꼬미'],
    '행사': ['이벤트'],
    '수준': ['퀄리티', '레벨'],
    '후기': ['리뷰'],
    '힘': ['파워풀'],
    '치명적인': ['치명치명한'],
    '기다리': ['웨이팅하'],
    '죠': ['쥬'],
    '최고': ['굿', '굳'],
    '귀엽다': ['귀염', '귀욥', '귀욤'],
    '맛있': ['마싯'],
    '어이': ['어의'],
    '검은': ['검정'],
    '나았': ['낳았'],
    '낫': ['났', '낳'],
    '일부러': ['일부로'],
    '함부로': ['함부러'],
    '가르치': ['가리키'],
    '가리키': ['가르치'],
    '무난': ['문안'],
    '오랜만': ['오랫만'],
    '얘기': ['예기'],
    '금세': ['금새'],
    '웬': ['왠'],
    '왠': ['웬'],
    '며칠': ['몇일', '몇 일'],
    '교향곡': ['교양곡'],
    '훼손': ['회손'],
    '사달': ['사단'],
    '머리말': ['머릿말'],
    '기를': ['길'],
    '어떻': ['어떡'],
    '어떡': ['어떻'],
    '피우': ['피'],
    '불리': ['불리우'],
    '게': ['개'],
    '끗발': ['끝발'],
    '는지': ['런지'],
    '대갚음': ['되갚음'],
    '띄어쓰기': ['띄워쓰기'],
    '목말': ['목마'],
    '범칙': ['벌칙'],
    '희한': ['희안'],
    '감금': ['강금'],
    '꿰': ['꼬'],
    '됐': ['됬'],
    '몹쓸': ['못쓸'],
    '으레': ['으례', '의례', '의레'],
    '이따가': ['있다가'],
    '담백': ['단백'],
    '쩨쩨': ['째째'],
    '굽실': ['굽신'],
    '비비': ['부비'],
    '악천후': ['악천우'],
    '결재': ['결제'],
    '결제': ['결재'],
    '지양': ['지향'],
    '지향': ['지양'],
    '깎': ['깍'],
    '연루': ['연류'],
    '덮밥': ['덥밥'],
    '결딴': ['결단'],
    '늘그막': ['늙으막'],
    '고진감래': ['고진감내'],
    '사면초가': ['사면초과'],
    '환골탈태': ['환골탈퇴'],
    '어폐': ['어패'],
    '바치': ['받치'],
    '해코지': ['해꼬지'],
    '뭐': ['모', '머'],
    '뭘': ['몰', '멀'],
    '뭔': ['먼', '몬'],
    '뭐를': ['멀', '몰'],
    '가게': ['숍', '샵'],
    '감사': ['ㄳ', 'ㄱㅅ'],
    '수고': ['ㅅㄱ']
}

insulting_dict = ['씨발', '시발', 'ㅅㅂ']


def compose(chosung, jungsung, jongsung):
    char = chr(
        kor_begin +
        chosung_base * chosung_list.index(chosung) +
        jungsung_base * jungsung_list.index(jungsung) +
        jongsung_list.index(jongsung)
    )
    return char


def decompose(c):
    if not character_is_korean(c):
        return (c)
    i = ord(c)
    if (jaum_begin <= i <= jaum_end):
        return (c)
    if (moum_begin <= i <= moum_end):
        return (c)

    # decomposition rule
    i -= kor_begin
    cho = i // chosung_base
    jung = (i - cho * chosung_base) // jungsung_base
    jong = (i - cho * chosung_base - jung * jungsung_base)
    return [chosung_list[cho], jungsung_list[jung], jongsung_list[jong]]


def character_is_korean(c):
    i = ord(c)
    return ((kor_begin <= i <= kor_end) or
            (jaum_begin <= i <= jaum_end) or
            (moum_begin <= i <= moum_end))


def separate_jamo(sentence):
    ret = []
    for char in sentence:
        if char == "": continue
        ret.append(decompose(char))

    return ret


def merge_char(parsed_char):
    if len(parsed_char) == 3:
        return compose(*parsed_char)
    else:
        return parsed_char[0]


hangul_error_dict = dict()
## 자음
hangul_error_dict['ㅂ'] = ['ㅃ', 'ㅈ', 'ㅁ']
hangul_error_dict['ㅈ'] = ['ㅉ', 'ㅂ', 'ㄷ', 'ㄴ']
hangul_error_dict['ㄷ'] = ['ㄸ', 'ㅈ', 'ㄱ', 'ㅇ']
hangul_error_dict['ㄱ'] = ['ㄲ', 'ㄷ', 'ㅅ', 'ㄹ']
hangul_error_dict['ㅅ'] = ['ㅆ', 'ㄱ', 'ㅎ']
hangul_error_dict['ㅁ'] = ['ㅂ', 'ㄴ', 'ㅋ']
hangul_error_dict['ㄴ'] = ['ㅁ', 'ㅈ', 'ㅇ', 'ㅌ', 'ㅋ']
hangul_error_dict['ㅇ'] = ['ㄷ', 'ㄴ', 'ㅌ', 'ㅊ', 'ㄹ']
hangul_error_dict['ㄹ'] = ['ㅇ', 'ㄱ', 'ㅎ', 'ㅊ', 'ㅍ']
hangul_error_dict['ㅎ'] = ['ㄹ', 'ㅅ', 'ㅍ']
hangul_error_dict['ㅋ'] = ['ㅁ', 'ㅌ', 'ㄴ']
hangul_error_dict['ㅌ'] = ['ㅋ', 'ㅇ', 'ㅊ']
hangul_error_dict['ㅊ'] = ['ㅌ', 'ㅍ', 'ㄹ']
hangul_error_dict['ㅍ'] = ['ㅊ', 'ㄹ', 'ㅎ']

hangul_error_dict['ㅃ'] = ['ㅂ', 'ㅉ']
hangul_error_dict['ㅉ'] = ['ㅈ', 'ㅃ', 'ㄸ']
hangul_error_dict['ㄸ'] = ['ㄷ', 'ㅉ', 'ㄲ']
hangul_error_dict['ㄲ'] = ['ㄱ', 'ㄸ', 'ㅆ']
hangul_error_dict['ㅆ'] = ['ㅅ', 'ㄱ']

## 모음
hangul_error_dict['ㅛ'] = ['ㅗ', 'ㅕ']
hangul_error_dict['ㅕ'] = ['ㅛ', 'ㅗ', 'ㅓ', 'ㅑ']
hangul_error_dict['ㅑ'] = ['ㅕ', 'ㅏ', 'ㅐ']
hangul_error_dict['ㅐ'] = ['ㅒ', 'ㅑ', 'ㅔ', 'ㅣ']
hangul_error_dict['ㅔ'] = ['ㅖ', 'ㅐ', 'ㅢ']
hangul_error_dict['ㅗ'] = ['ㅛ', 'ㅓ', 'ㅜ']
hangul_error_dict['ㅓ'] = ['ㅕ', 'ㅏ']
hangul_error_dict['ㅏ'] = ['ㅑ', 'ㅣ', 'ㅓ']
hangul_error_dict['ㅣ'] = ['ㅐ']
hangul_error_dict['ㅠ'] = ['ㅜ', 'ㅗ', 'ㅡ']
hangul_error_dict['ㅜ'] = ['ㅠ', 'ㅡ']
hangul_error_dict['ㅡ'] = ['ㅜ']

hangul_error_dict['ㅒ'] = ['ㅐ', 'ㅖ', 'ㅔ']
hangul_error_dict['ㅖ'] = ['ㅔ', 'ㅐ', 'ㅒ']
hangul_error_dict['ㅘ'] = ['ㅚ']
hangul_error_dict['ㅙ'] = ['ㅞ']
hangul_error_dict['ㅚ'] = ['ㅙ', 'ㅘ', 'ㅟ']
hangul_error_dict['ㅝ'] = ['ㅞ']
hangul_error_dict['ㅞ'] = ['ㅙ']
hangul_error_dict['ㅟ'] = ['ㅚ', 'ㅢ']
hangul_error_dict['ㅢ'] = ['ㅟ', 'ㅔ']

## 종성 붙이는 경우
hangul_error_dict[' '] = ['ㅇ', 'ㄴ', 'ㅁ']


class NoiseInjector(object):

    def __init__(self, corpus, shuffle_sigma=0.01,
                 replace_mean=0.05, replace_std=0.03,
                 delete_mean=0.05, delete_std=0.03,
                 add_mean=0.05, add_std=0.03,
                 jamo_typo_mean=0.07, jamo_typo_std=0.03,
                 yuneum_typo_prob=0.07,
                 last_cute_prob=0.15,
                 punctuation_prob=0.1,
                 delete_space_prob=0.3,
                 add_space_prob=0.3,
                 confusing_prob=0.35):
        # READ-ONLY, do not modify
        self.corpus = corpus
        self.shuffle_sigma = shuffle_sigma
        self.replace_a, self.replace_b = self._solve_ab_given_mean_var(replace_mean, replace_std ** 2)
        self.delete_a, self.delete_b = self._solve_ab_given_mean_var(delete_mean, delete_std ** 2)
        self.add_a, self.add_b = self._solve_ab_given_mean_var(add_mean, add_std ** 2)
        self.jamo_typo_a, self.jamo_typo_b = self._solve_ab_given_mean_var(jamo_typo_mean, jamo_typo_std ** 2)
        self.yuneum_typo_prob = yuneum_typo_prob
        self.last_cute_prob = last_cute_prob
        self.punctuation_prob = punctuation_prob
        self.confusing_prob = confusing_prob
        self.delete_space_prob = delete_space_prob
        self.add_space_prob = add_space_prob
        # self.confusing_typo_a, self.confusing_typo_b = self._solve_ab_given_mean_var(confusing_typo_mean, confusing_typo_std**2)

    @staticmethod
    def _solve_ab_given_mean_var(mean, var):
        a = mean * mean * (1. - mean) / var - mean
        b = (1. - mean) * (mean * (1. - mean) / var - 1.)
        return a, b

    def _shuffle_func(self, tgt):
        if self.shuffle_sigma < 1e-6:
            return tgt

        shuffle_key = [i + np.random.normal(loc=0, scale=self.shuffle_sigma) for i in range(len(tgt))]
        new_idx = np.argsort(shuffle_key)
        res = [tgt[i] for i in new_idx]

        return res

    def _replace_func(self, tgt):
        replace_ratio = np.random.beta(self.replace_a, self.replace_b)
        ret = []
        rnd = np.random.random(len(tgt))
        for i, w in enumerate(tgt):
            if rnd[i] < replace_ratio:
                rnd_ex = self.corpus[np.random.randint(len(self.corpus))]
                rnd_word = rnd_ex[np.random.randint(len(rnd_ex))]
                ret.append(rnd_word)
            else:
                ret.append(w)
        return ret

    def _delete_func(self, tgt):
        delete_ratio = np.random.beta(self.delete_a, self.delete_b)
        ret = []
        rnd = np.random.random(len(tgt))
        for i, w in enumerate(tgt):
            if rnd[i] < delete_ratio:
                continue
            ret.append(w)
        return ret

    def _add_func(self, tgt):
        add_ratio = np.random.beta(self.add_a, self.add_b)
        ret = []
        rnd = np.random.random(len(tgt))
        for i, w in enumerate(tgt):
            if rnd[i] < add_ratio:
                rnd_ex = self.corpus[np.random.randint(len(self.corpus))]
                rnd_word = rnd_ex[np.random.randint(len(rnd_ex))]
                ret.append(rnd_word)
            ret.append(w)

        return ret

    def _delete_space_func(self, tgt):
        ret = []
        for i in range(len(tgt)):
            delete_ratio = random.uniform(0, 1)
            if tgt[i] == ' ' and delete_ratio < self.delete_space_prob:
                continue
            else:
                ret.append(tgt[i])

        return ret

    def _add_space_func(self, tgt):
        ret = []
        for i in range(len(tgt)):
            add_ratio = random.uniform(0, 1)
            if add_ratio < self.add_space_prob:
                ret.append(' ')
            ret.append(tgt[i])

        return ret

    def _jamo_typo_func(self, tgt):
        jamo_typo_ratio = np.random.beta(self.jamo_typo_a, self.jamo_typo_b)
        tgt = separate_jamo(tgt)
        ret = []
        rnd = np.random.random(len(tgt))
        for i, w in enumerate(tgt):
            decomposed_char = w
            if rnd[i] < jamo_typo_ratio and len(decomposed_char) == 3:
                if decomposed_char[2] == ' ':
                    typo_loc = np.random.choice(range(2))
                else:
                    typo_loc = np.random.choice(range(3))
                decomposed_char_modified = list(decomposed_char)

                if decomposed_char[typo_loc] in hangul_error_dict:
                    choice_error = np.random.choice(hangul_error_dict[decomposed_char[typo_loc]])
                    if choice_error in possible_char[typo_loc]:
                        decomposed_char_modified[typo_loc] = choice_error

                typo_word = merge_char(decomposed_char_modified)

                ret.append(typo_word)
            else:
                ret.append(merge_char(decomposed_char))

        return ret

    def _yuneum_typo_func(self, tgt):
        tgt = separate_jamo(tgt)

        ret = []
        ids = []
        for i in range(len(tgt) - 1):
            cur_char = tgt[i]
            next_char = tgt[i + 1]

            if len(next_char) == 3 and len(cur_char) == 3 and next_char[0] == 'ㅇ' and cur_char[2] != ' ' and cur_char[
                2] != 'ㅇ' \
                    and cur_char[2] in chosung_list:
                ids.append(i)

        selected_ids = random.sample(ids, int(len(ids) * self.yuneum_typo_prob))

        for i in selected_ids:
            cur_char = tgt[i]
            next_char = tgt[i + 1]

            tgt[i + 1][0] = cur_char[2]
            if tgt[i][2] != 'ㄴ':
                tgt[i][2] = ' '

        for i, p in enumerate(tgt):
            decomposed_char = p
            ret.append(merge_char(decomposed_char))

        return ret

    def _last_cute_func(self, tgt):
        cute_ratio = random.uniform(0, 1)
        if cute_ratio < self.last_cute_prob:
            last_loc = None
            for i in range(len(tgt) - 1, -1, -1):
                if character_is_korean(tgt[i]):
                    last_loc = i
                    break
            if last_loc != None:
                decomposed_char = decompose(tgt[last_loc])
                if len(decomposed_char) == 3 and decomposed_char[2] == ' ':
                    end_char = random.sample(['ㅁ', 'ㄴ', 'ㅇ', 'ㄹ', 'ㅅ', 'ㅋ', 'ㅎ'], 1)
                    decomposed_char[2] = end_char[0]
                    tgt[last_loc] = compose(*decomposed_char)

        return tgt

    def _punctuation_error_func(self, tgt):

        punc_list = ['.', '?', '!']

        for i in range(len(tgt)):
            if tgt[i] in punc_list:
                punc_ratio = random.uniform(0, 1)
                if punc_ratio < self.punctuation_prob:
                    tgt[i] = random.sample(punc_list + [''], 1)[0]

        return tgt

    def _confusing_error_func(self, word):

        confusing_ratio = random.uniform(0, 1)
        random_change_word = word

        changed = False
        if confusing_ratio < self.confusing_prob:
            changed = True
            random_change_word = random.sample(confusing_dict[word], 1)[0]

        return random_change_word, changed

    def _parse(self, pairs):
        align = []
        art = []
        for si in range(len(pairs)):
            ti = pairs[si][0]
            w = pairs[si][1]
            art.append(w)
            if ti >= 0:
                align.append('{}-{}'.format(si, ti))
        return art, align

    def inject_noise(self, tokens):
        ##TODO: last cute 제일 마지막 character만 하도록
        # tgt is a vector of integers

        funcs = [self._add_func, self._shuffle_func, self._replace_func, self._delete_func]
        np.random.shuffle(funcs)
        funcs = [self._jamo_typo_func, self._yuneum_typo_func, self._punctuation_error_func] + funcs
        funcs_last = [self._jamo_typo_func, self._yuneum_typo_func, self._last_cute_func,
                      self._punctuation_error_func] + funcs

        token_parts = []
        token_change = []
        overlap = []
        for confuse_typo in confusing_dict.keys():
            confuse_start = tokens.find(confuse_typo)
            if confuse_start == -1: continue
            confuse_end = confuse_start + len(confuse_typo)
            check = False
            for over in overlap:
                if (over[0] <= confuse_start < over[1]) or (over[0] < confuse_end <= over[1]) or \
                        (confuse_start <= over[0] and over[1] < confuse_end):
                    check = True
            if check: continue
            overlap.append((confuse_start, confuse_end))

        overlap.sort()

        end = 0
        for ovlp in overlap:
            ovlp_st = ovlp[0]
            ovlp_ed = ovlp[1]
            confusing_word = tokens[ovlp_st:ovlp_ed]
            if end != ovlp_st:
                token_parts.append(tokens[end:ovlp_st])
                token_change.append(True)
            rndword, changed = self._confusing_error_func(confusing_word)
            token_parts.append(rndword)
            if changed:
                token_change.append(False)
            else:
                token_change.append(True)
            end = ovlp_ed
        token_parts.append(tokens[end:])
        token_change.append(True)

        final_letters = []
        for idx, tokens in enumerate(token_parts):
            if token_change[idx] == False:
                final_letters.extend([w for w in tokens])
                continue
            letters = [w for w in tokens]
            if idx != len(token_parts) - 1:
                for f in funcs:
                    letters = f(letters)
            else:
                # print("letters", letters)
                for f in funcs_last:
                    letters = f(letters)
            final_letters.extend(letters)

        return ''.join([letter for letter in final_letters])


def add_del_space(lines, add_ratio=0.1, del_ratio=0.3):
    noise_lines = []

    for line in lines:
        noise_line = []

        for token in line:
            if token == ' ':
                if random.uniform(0, 1) > del_ratio:  # not delete space
                    noise_line.append(' ')
            else:
                noise_line.append(token)
                if random.uniform(0, 1) < add_ratio:  # add space
                    noise_line.append(' ')

        noise_lines.append(''.join(noise_line))  # list -> string

    # print(f'[{lines[0]}] -> [{noise_lines[0]}]')

    return noise_lines

def noise(lines):
    tgts = lines
    noise_injector = NoiseInjector(tgts)

    noise_lines = []

    #g2p = G2p()

    end = 0
    cnt = 0
    print("original noisy function")
    for i in tqdm_notebook(range(len(tgts))):
        tgt = tgts[i]
        noise_tgt = noise_injector.inject_noise(tgt)
        noise_lines.append(noise_tgt)
    #         if i != 0 and i % 500000 == 0:
    #             clean_data = lines[end:i+1]
    #             noise_data = noise_lines
    #             noise_lines = []
    #             end = i
    #             print("{}/{}".format(i, len(tgts)))
    #             print(i, noise_tgt)
    #             with open('sejong_clean_noise_function_final_{}.txt'.format(cnt), 'w',encoding='utf-8') as f:
    #                 for line in clean_data:
    #                     f.write("%s\n" % re.sub("\n","",line))
    #             with open('sejong_noisy_noise_function_final_{}.txt'.format(cnt), 'w',encoding='utf-8') as f:
    #                 for line in noise_data:
    #                     f.write("%s\n" % re.sub("\n","",line))
    #             cnt += 1

    #     clean_data = lines[end:]
    #     noise_data = noise_lines
    #     with open('sejong_clean_noise_function_final_{}.txt'.format(cnt), 'w',encoding='utf-8') as f:
    #         for line in clean_data:
    #             f.write("%s\n" % re.sub("\n","",line))
    #     with open('sejong_noisy_noise_function_final_{}.txt'.format(cnt), 'w',encoding='utf-8') as f:
    #         for line in noise_data:
    #             f.write("%s\n" % re.sub("\n","",line))

    #     print("API noisy function")
    #     for i, tgt in tqdm_notebook(enumerate(tgts)):
    #         descriptive_option = random.choice([True, False])
    #         group_vowels_option = random.choice([True, False])
    #         to_syl_option = random.choices([True, False], weights=[0.95,0.05], k=1)
    #         noise_tgt = g2p(tgt, descriptive=descriptive_option, group_vowels=group_vowels_option, to_syl=to_syl_option)
    #         noise_lines.append(noise_tgt)
    #         if i % 50000 == 0:
    #             print("{}/{}".format(i, len(tgts)))
    #             print(i, noise_tgt)

    #     data_total = list(zip(lines*2, noise_lines))

    #     random.shuffle(data_total)

    #     data_clean, data_noise = zip(*data_total)

    return noise_lines

from data_loader import write_strings, read_strings
import os
import nsml


class GeneratedData:
    def __init__(self, noisy_sents=None, clean_sents=None):
        self.noisy_sents = noisy_sents
        self.clean_sents = clean_sents

    def save(self, noisy_sents, clean_sents):
        self.noisy_sents = noisy_sents
        self.clean_sents = clean_sents

    def load(self):
        return self.noisy_sents, self.clean_sents


def bind_generated_data(generated_data):
    def save(dirname, *args):
        noisy_sents, clean_sents = generated_data.load()
        write_strings(os.path.join(dirname, 'generated_data'), noisy_sents)
        write_strings(os.path.join(dirname, 'generated_label'), clean_sents)

    def load(dirname, *args):
        noisy_sents = read_strings(os.path.join(dirname, 'generated_data'))
        clean_sents = read_strings(os.path.join(dirname, 'generated_label'))
        generated_data.save(noisy_sents, clean_sents)

    def infer(raw_data, **kwargs):
        pass

    nsml.bind(save, load, infer)


def save_generated_data(noisy_sents, clean_sents):
    generated_data = GeneratedData(noisy_sents, clean_sents)
    bind_generated_data(generated_data)
    nsml.save('generated_data')


def load_generated_data(checkpoint, session):
    generated_data = GeneratedData()
    bind_generated_data(generated_data)
    nsml.load(checkpoint=checkpoint, session=session)
    noisy_sents, clean_sents = generated_data.load()
    return noisy_sents, clean_sents
