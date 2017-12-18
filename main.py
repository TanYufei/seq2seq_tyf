# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import tensorflow as tf
from data_util import *
# from model import Model
from seq2seqModel import Model
# from my_metrics import *
from tensorflow.python import debug as tf_debug
from config import DemoConfig

batch_size = 16
epoch_num = 50


def get_model(mode='train'):
    config = DemoConfig()
    model = Model(config=config, mode=mode)
    # model.build()
    return model

def check_data():
    train_x, train_y, val_x, val_y = readdata()
    train_data_ed = data_pipeline(train_x, train_y)
    test_data_ed = data_pipeline(val_x, val_y)
    word2index, index2word, slot2index, index2slot = get_info_from_training_data(train_data_ed)

    index_train = to_index(train_data_ed, word2index, slot2index)
    index_test = to_index(test_data_ed, word2index, slot2index)

    for i, batch in enumerate(getBatch(batch_size, index_train)):
        print('batch shape ', np.array(batch).shape)

def train(is_debug=False):
    model = get_model()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # sess = tf.Session()
    if is_debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    sess.run(tf.global_variables_initializer())
    # print(tf.trainable_variables())
    # train_data = open("dataset/atis-2.train.w-intent.iob", "r").readlines()
    # test_data = open("dataset/atis-2.dev.w-intent.iob", "r").readlines()
    print('read data')
    train_x, train_y, val_x, val_y = readdata()
    train_data_ed = data_pipeline(train_x, train_y)
    test_data_ed = data_pipeline(val_x, val_y)
    print('data to idx')
    word2index, index2word, tar2index, index2tar = get_info_from_training_data(train_data_ed)
    vocab_size = len(word2index)
    tar_size = len(tar2index)

    index_train = to_index(train_data_ed, word2index, tar2index)
    index_test = to_index(test_data_ed, word2index, tar2index)
    print('====start train====')

    saver = tf.train.Saver()
    for epoch in range(epoch_num):
        mean_loss = 0.0
        train_loss = 0.0
        for i, batch in enumerate(getBatch(batch_size, index_train)):
            # 执行一个batch的训练
            # print('batch shape ',np.array(batch).shape)
            _, loss, decoder_prediction = model.step(sess, "train", batch)
            # if i == 0:
            #     index = 0
            #     print("training debug:")
            #     print("input:", list(zip(*batch))[0][index])
            #     print("length:", list(zip(*batch))[1][index])
            #     print("mask:", mask[index])
            #     print("target:", list(zip(*batch))[2][index])
            #     # print("decoder_targets_one_hot:")
            #     # for one in decoder_targets_one_hot[index]:
            #     #     print(" ".join(map(str, one)))
            #     print("decoder_logits: ")
            #     for one in decoder_logits[index]:
            #         print(" ".join(map(str, one)))
            #     print("slot_W:", slot_W)
            #     print("decoder_prediction:", decoder_prediction[index])
            #     print("intent:", list(zip(*batch))[3][index])
            mean_loss += loss
            train_loss += loss
            if i % 100 == 0:
                if i > 0:
                    mean_loss = mean_loss / 100.0
                print('Average train loss at epoch %d, step %d: %f' % (epoch, i, mean_loss))
                mean_loss = 0
        train_loss /= (i + 1)
        print("[Epoch {}] Average train loss: {}".format(epoch, train_loss))

        # 每训一个epoch，测试一次
        for j, batch in enumerate(getBatch(batch_size, index_test)):
            decoder_prediction = model.step(sess, "test", batch)
            # decoder_prediction = np.transpose(decoder_prediction, [1, 0])
            if j == 0:
                index = random.choice(range(len(batch)))
                # index = 0
                print("Input Sentence        : ", ' '.join(index_seq2word(batch[index][0], index2word)))
                print("Target Truth          : ", ' '.join(index_seq2tar(batch[index][2], index2tar)))
                print("Target Prediction     : ", ' '.join(index_seq2tar(decoder_prediction[index], index2tar)))
                # print("Intent Truth          : ", index2intent[batch[index][3]])
                # print("Intent Prediction     : ", index2intent[intent[index]])
        saver.save('./model/model')

def dtest(model_dir):
    model = get_model(mode='inference')
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    #
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver()
    sess = tf.Session()

    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    # sess.run(tf.global_variables_initializer())

    print('read data')
    train_x, train_y, val_x, val_y = readdata()
    train_data_ed = data_pipeline(train_x, train_y)
    test_data_ed = data_pipeline(val_x, val_y)
    print('data to idx')
    word2index, index2word, tar2index, index2tar = get_info_from_training_data(train_data_ed)
    vocab_size = len(word2index)
    tar_size = len(tar2index)

    index_train = to_index(train_data_ed, word2index, tar2index)
    index_test = to_index(test_data_ed, word2index, tar2index)
    print('====start train====')

    saver = tf.train.Saver()
    for i, batch in enumerate(getBatch(batch_size, index_train)):
        # 执行一个batch的训练
        # print('batch shape ',np.array(batch).shape)
        decoder_prediction = model.step(sess, "inference", batch)

        for j in range(len(batch)):
            print("Input Sentence        : ", ' '.join(index_seq2word(batch[j][0], index2word)))
            print("Target Truth          : ", ' '.join(index_seq2tar(batch[j][2], index2tar)))
            print("Target Prediction     : ", ' '.join(index_seq2tar(decoder_prediction[j], index2tar)))

if __name__ == '__main__':
    # train(is_debug=True)
    # test_data()
    train()
    # check_data()