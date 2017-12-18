# -*- coding: utf-8 -*-
# @Time    : 12/7/17 2:26 PM
# @Author  : tyf_aicyber
# @Site    : 
# @File    : seq2seqModel.py
# @Software: PyCharm

import sys
import tensorflow as tf
from tensorflow.python.layers.core import Dense
# from tensorflow.contrib.seq2seq.python.ops.helper import TrainingHelper
# from tensorflow.contrib.seq2seq.python.ops.basic_decoder import BasicDecoder, BasicDecoderOutput

class Model:
    def __init__(self, config, mode='train'):
        assert mode in ['train', 'test', 'inference']
        self.mode = mode
        self.en_vocab_size = config.en_vocab_size
        self.de_vocab_size = config.de_vocab_size
        self._sos_id = config.tgt_sos
        self._eos_id = config.tgt_eos
        # Data preprocess
        # self.data = data

        # Network
        self.hidden_size = config.hidden_size
        self.encoder_embed_size = config.encoder_embed_size
        self.decoder_embed_size = config.decoder_embed_size

        # Training
        self.optimizer = config.optimizer
        self.n_epoch = config.n_epoch
        self.learning_rate = config.learning_rate

        self.ckpt_dir = config.ckpt_dir

        self._build_graph()

    def _build_graph(self):
        print ('Start building graph...')
        self._init_placeholder()
        self._init_cell()
        self._init_encoder()
        self._init_decoder()
        self._init_training_operator()
        print ('... Graph is built.')

    def _init_placeholder(self):
        print ('Initialize placeholders')
        self.encoder_inputs = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name='input_sequence'
        )
        self.encoder_sequence_length = tf.placeholder(
            dtype=tf.int32,
            shape=[None, ],
            name='input_sequence_length'
        )
        if self.mode == 'train':
            self.decoder_inputs = tf.placeholder(
                dtype=tf.int32,
                shape=[None, None],
                name='target_sequences'
            )

            self.decoder_sequence_length = tf.placeholder(
                dtype=tf.int32,
                shape=[None, ],
                name='target_sequence_length'
            )

    def _init_cell(self):
        self.cell = tf.contrib.rnn.LSTMCell

    def _init_encoder(self):
        print ('Initialize Encoder')
        with tf.variable_scope('Encoder') as scope:
            # Embedding
            self.encoder_embed = tf.Variable(tf.random_uniform([self.en_vocab_size, self.encoder_embed_size],
                                                               -0.1, 0.1), dtype=tf.float32, name="embedding")

            # [Batch_size x encoder_sentence_length x embedding_size]
            encoder_embed_inputs = tf.nn.embedding_lookup(
                self.encoder_embed, self.encoder_inputs, name='embed_inputs'
            )
            encoder_cell = self.cell(self.hidden_size)

            # encoder_outputs: [batch_size x encoder_sentence_length x embedding_size]
            # encoder_last_state: [batch_size x embedding_size]
            encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(
                cell=encoder_cell,
                inputs=encoder_embed_inputs,
                sequence_length=self.encoder_sequence_length,
                time_major=False,
                dtype=tf.float32)

    def _init_bid_encoder(self):
        print ('Initialize Encoder')
        with tf.variable_scope('Encoder') as scope:
            # Embedding
            self.encoder_embed = tf.Variable(tf.random_uniform([self.en_vocab_size, self.encoder_embed_size],
                                                               -0.1, 0.1), dtype=tf.float32, name="embedding")

            # [Batch_size x encoder_sentence_length x embedding_size]
            encoder_embed_inputs = tf.nn.embedding_lookup(
                self.encoder_embed, self.encoder_inputs, name='embed_inputs'
            )

            # 使用单个LSTM cell
            encoder_f_cell = self.cell(self.hidden_size)
            encoder_b_cell = self.cell(self.hidden_size)

            # encoder_outputs: [batch_size x encoder_sentence_length x embedding_size]
            # encoder_last_state: [batch_size x embedding_size]
            (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_f_cell,
                                                cell_bw=encoder_b_cell,
                                                inputs=encoder_embed_inputs,
                                                sequence_length=self.encoder_sequence_length,
                                                dtype=tf.float32, time_major=True)
            encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            encoder_final_state_c = tf.concat(
                (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

            encoder_final_state_h = tf.concat(
                (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

            self.encoder_last_state = tf.contrib.rnn.LSTMStateTuple(
                c=encoder_final_state_c,
                h=encoder_final_state_h
            )

    def _init_decoder(self):
        print ('Initialize Decoder')
        with tf.variable_scope('Decoder') as scope:
            self.decoder_embed = tf.Variable(
                tf.random_uniform([self.de_vocab_size, self.decoder_embed_size], -0.1, 0.1),
                dtype=tf.float32,
                name='dec_embedding'
            )

            decoder_cell = self.cell(self.hidden_size)

            output_layer = Dense(self.de_vocab_size, name='output_projection')

            if self.mode == 'train':

                max_decode_length = tf.reduce_max(self.decoder_sequence_length + 1, name='max_decoder_length')
                # max_decode_length = 50

                decoder_embed_inputs = tf.nn.embedding_lookup(
                    self.decoder_embed, self.decoder_inputs, name='embed_inputs')

                training_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=decoder_embed_inputs,
                    sequence_length=self.decoder_sequence_length + 1,
                    time_major=False,
                    name='train_helper')

                training_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell,
                    helper=training_helper,
                    initial_state=self.encoder_last_state,
                    output_layer=output_layer
                )

                # API:https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode
                training_decoder_outputs, training_decoder_last_state, last_length = tf.contrib.seq2seq.dynamic_decode(
                    training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_decode_length
                )

                # logits: [batch_size x max_dec_len x dec_vocab_size+2]
                logits = tf.identity(training_decoder_outputs.rnn_output, name='logits')
                targets = tf.slice(self.decoder_inputs, [0, 0], [-1, max_decode_length], name='targets')
                # masks: [batch_size x max_dec_len]
                # => ignore outputs after `decoder_senquence_length+1` when calculating loss
                masks = tf.sequence_mask(self.decoder_sequence_length + 1, max_decode_length,
                                         dtype=tf.float32, name='masks'
                                         )

                # API:https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/sequence_loss
                # internal: `tf.nn.sparse_softmax_cross_entropy_with_logits`
                self.batch_loss = tf.contrib.seq2seq.sequence_loss(
                    logits=logits,
                    targets=targets,
                    weights=masks,
                    name='batch_loss'
                )

                # prediction sample for validation
                self.valid_predictions = tf.identity(training_decoder_outputs.sample_id, name='valid_preds')

                # list of trainable weights
                self.training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

            elif self.mode == 'inference':
                batch_size = tf.shape(self.encoder_inputs)[0:1]
                start_tokens = tf.ones(batch_size, dtype=tf.int32) * self._eos_id
                # start_tokens = tf.tile(tf.constant([model_config.GO_ID], dtype=tf.int32), [self.batch_size],
                #								   name='start_tokens')

                inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=self.decoder_embed,
                    start_tokens=start_tokens,
                    end_token=1)

                inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell,
                    helper=inference_helper,
                    initial_state=self.encoder_last_state,
                    output_layer=output_layer)

                infer_dec_outputs, infer_dec_last_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    inference_decoder,
                    output_time_major=False,
                    impute_finished=True)  # ,maximum_iterations=20)

                # [batch_size x dec_sentence_length], tf.int32
                self.predictions = tf.identity(infer_dec_outputs.sample_id, name='predictions')

    def _init_training_operator(self):
        self.train_op = self.optimizer(self.learning_rate, name='train_op').minimize(self.batch_loss)

    def step(self, sess, mode, trarin_batch):
        """ perform each batch"""
        if mode not in ['train', 'test', 'inference']:
            print >> sys.stderr, 'mode is not supported'
            sys.exit(1)
        unziped = list(zip(*trarin_batch))
        # print(np.shape(unziped[0]), np.shape(unziped[1]),
        #       np.shape(unziped[2]), np.shape(unziped[3]))
        if mode == 'train':
            # output_feeds = [self.train_op, self.loss, self.decoder_prediction,
            #                 self.mask, self.slot_W]
            output_feeds = [self.train_op, self.batch_loss, self.valid_predictions]
            # feed_dict = {self.encoder_inputs: np.transpose(unziped[0], [1, 0]),
            #              self.encoder_inputs_actual_length: unziped[1],
            #              self.decoder_targets: unziped[2]}
            feed_dict = {self.encoder_inputs: unziped[0],
                         self.encoder_sequence_length: unziped[1],
                         self.decoder_inputs: unziped[2],
                         self.decoder_sequence_length:unziped[3]
                         }
        if mode in ['test']:
            # output_feeds = [self.decoder_prediction, ]
            # feed_dict = {self.encoder_inputs: np.transpose(unziped[0], [1, 0]),
            #              self.encoder_inputs_actual_length: unziped[1]}
            output_feeds = self.valid_predictions
            # feed_dict = {self.encoder_inputs: unziped[0],
            #              self.encoder_sequence_length: unziped[1]}
            feed_dict = {self.encoder_inputs: unziped[0],
                         self.encoder_sequence_length: unziped[1],
                         self.decoder_inputs: unziped[2],
                         self.decoder_sequence_length: unziped[3]
                         }
        if mode in ['inference']:
            # output_feeds = [self.decoder_prediction, ]
            # feed_dict = {self.encoder_inputs: np.transpose(unziped[0], [1, 0]),
            #              self.encoder_inputs_actual_length: unziped[1]}
            output_feeds = self.valid_predictions
            # feed_dict = {self.encoder_inputs: unziped[0],
            #              self.encoder_sequence_length: unziped[1]}
            feed_dict = {
                self.encoder_inputs: unziped[0],
                self.encoder_sequence_length: unziped[1],
            }

        results = sess.run(output_feeds, feed_dict=feed_dict)
        return results