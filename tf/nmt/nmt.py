import tensorflow as tf
from inference import inference
from train import train
from hparams import RNNHyperParameters, TemporalCNNHyperParameters
import os




"""
hparams = RNNHyperParameters(num_steps=1000000,
                             train_batch_size=128,
                             infer_batch_size=1,
                             max_training_sequence_length=50,
                             num_units_per_cell=128,
                             num_layers=2,
                             embeddings_size=128,
                             max_gradient_norm=5,
                             initial_learning_rate=1.0,
                             model_name="lstm",
                             src_lang = "en",
                             train_src_dataset_file_name = "/home/lie/nmt/nmt/scripts/iwslt15/train.en",
                             src_vocab_file_name = "/home/lie/nmt/nmt/scripts/iwslt15/vocab.en",
                             tgt_lang = "vi",
                             train_tgt_dataset_file_name = "/home/lie/nmt/nmt/scripts/iwslt15/train.vi",
                             tgt_vocab_file_name = "/home/lie/nmt/nmt/scripts/iwslt15/vocab.vi",
                             model_ckpt_dir="/tmp/nmt_new_tf",
                             infer_src_dataset_file_name="/home/lie/nmt/nmt/scripts/iwslt15/infer3.en")
"""
hparams = TemporalCNNHyperParameters(num_steps=1000000,
                             train_batch_size=32,
                             infer_batch_size=1,
                             max_training_sequence_length=50,
                        filters=128,
                             kernel_size=3,
                             num_layers=2,
                             embeddings_size=128,
                             max_gradient_norm=5,
                             initial_learning_rate=1.0,
                             model_name="temporal",
                             src_lang = "en",
                             train_src_dataset_file_name = "/home/lie/nmt/nmt/scripts/iwslt15/train.en",
                             src_vocab_file_name = "/home/lie/nmt/nmt/scripts/iwslt15/vocab.en",
                             tgt_lang = "vi",
                             train_tgt_dataset_file_name = "/home/lie/nmt/nmt/scripts/iwslt15/train.vi",
                             tgt_vocab_file_name = "/home/lie/nmt/nmt/scripts/iwslt15/vocab.vi",
                             model_ckpt_dir="/tmp/nmt_new_tf",
                             infer_src_dataset_file_name="/home/lie/nmt/nmt/scripts/iwslt15/infer3.en",
                             time_major=False)

#train(hparams)
inference(hparams)
