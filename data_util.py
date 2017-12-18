# -*- coding: utf-8 -*-
# @Time    : 11/20/17 2:38 PM
# @Author  : tyf_aicyber
# @Site    : 
# @File    : leonvo.py
# @Software: PyCharm

import random
import numpy as np
import codecs

flatten = lambda l: [item for sublist in l for item in sublist]  # 二维展成一维
index_seq2tar = lambda s, index2tar: [index2tar[i] for i in s]
index_seq2word = lambda s, index2word: [index2word[i] for i in s]

def readdata():
    train_x = []
    with codecs.open('./data/split/lenovo_train_x', 'r', encoding='utf8') as f:
        for words in f:
            train_x.append(words.strip().split())
    train_y = []
    with codecs.open('./data/split/lenovo_train_y', 'r', encoding='utf8') as f:
        for words in f:
            train_y.append(words.strip().split())
    val_x = []
    with codecs.open('./data/split/lenovo_val_x', 'r', encoding='utf8') as f:
        for words in f:
            val_x.append(words.strip().split())
    val_y = []
    with codecs.open('./data/split/lenovo_val_y', 'r', encoding='utf8') as f:
        for words in f:
            val_y.append(words.strip().split())
    return train_x, train_y, val_x, val_y

def data_pipeline(seq_in, seq_out, length=50):
    sin = []
    sout = []
    # padding，原始序列和标注序列结尾+<EOS>+n×<PAD>
    for i in range(len(seq_in)):
        temp = seq_in[i]
        if len(temp) < length:
            temp.append(u'<EOS>')
            while len(temp) < length:
                temp.append(u'<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = u'<EOS>'
        sin.append(temp)

        temp = seq_out[i]
        if len(temp) < length:
            temp.append(u'<EOS>')
            while len(temp) < length:
                temp.append(u'<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = u'<EOS>'
        sout.append(temp)
    data = list(zip(sin, sout))
    return data

def get_info_from_training_data(data):
    seq_in, seq_out = list(zip(*data))
    vocab = set(flatten(seq_in))
    slot_tag = set(flatten(seq_out))

    # 生成word2index
    word2index = {u'<PAD>': 0, u'<UNK>': 1, u'<SOS>': 2, u'<EOS>': 3}
    for token in vocab:
        if token not in word2index.keys():
            word2index[token] = len(word2index)

    # 生成index2word
    index2word = {v: k for k, v in word2index.items()}

    # 生成tag2index
    tag2index = {u'<PAD>': 0, u'<UNK>': 1, u'<SOS>': 2, u'<EOS>': 3}
    for tag in slot_tag:
        if tag not in tag2index.keys():
            tag2index[tag] = len(tag2index)

    # 生成index2tag
    index2tag = {v: k for k, v in tag2index.items()}

    return word2index, index2word, tag2index, index2tag


def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch


def to_index(train, word2index, slot2index):
    new_train = []
    for sin, sout in train:
        sin_ix = list(map(lambda i: word2index[i] if i in word2index else word2index["<UNK>"],
                          sin))
        true_length = sin.index(u'<EOS>')
        sout_ix = list(map(lambda i: slot2index[i] if i in slot2index else slot2index["<UNK>"],
                           sout))
        true_length_tar = sout.index(u'<EOS>')
        # intent_ix = intent2index[intent] if intent in intent2index else intent2index["<UNK>"]
        new_train.append([sin_ix, true_length, sout_ix, true_length_tar])
    return new_train


if __name__ == '__main__':
    train_x, train_y, val_x, val_y =readdata()
    data_pipeline(val_x, val_y)