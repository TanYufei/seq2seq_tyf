import tensorflow as tf

class DemoConfig:

    tgt_sos = 2
    tgt_eos = 3
    # Dataset
    en_vocab_size = 20062
    de_vocab_size = 2060
        
    # Model
    hidden_size = 256
    encoder_embed_size = 200
    decoder_embed_size = 200
    cell = tf.contrib.rnn.BasicLSTMCell
    
    # Training
    optimizer = tf.train.RMSPropOptimizer
    n_epoch = 10
    learning_rate = 0.001

    # Checkpoint Path
    ckpt_dir = './ckpt_dir/'
