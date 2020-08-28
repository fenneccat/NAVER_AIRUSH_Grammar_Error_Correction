import re

def remove_special_characters(sentence):
    #print("Removing special characters from data")
    no_special_characters = re.sub(r'[^ㄱ-ㅣ가-힣A-Za-z0-9.?! ]+', ' ',sentence)
    return no_special_characters

def multiple_space_to_one(sentence):
    return re.sub(' +',' ',sentence)

def preprocess_sentence(sentence):
    sentence = remove_special_characters(sentence)
    sentence = re.compile("[ㄱ-ㅎ|ㅏ-ㅣ]+").sub('', sentence)
    sentence = multiple_space_to_one(sentence)

    return sentence

def preprocess_noisy_sentence_list(sentence_list):
    new_sentence_list = []
    if type(sentence_list[0]) != dict:
        for sentence in sentence_list:
            new_sentence_list.append(preprocess_sentence(sentence))

    else:
        for sentence_pair in sentence_list:
            sample_dict = {}
            sample_dict['noisy'] = preprocess_sentence(sentence_pair['noisy'])
            sample_dict['clean'] = sentence_pair['clean']
            new_sentence_list.append(sample_dict)

    return new_sentence_list