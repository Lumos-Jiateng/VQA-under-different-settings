import json
import os
from unittest import TestCase
import numpy as np
import heapq
from easydict import EasyDict as edict

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import context
import mindspore.dataset.engine as de
import mindspore.common.dtype as mstype
from mindspore.mindrecord import FileWriter
from mindspore.common.parameter import Parameter
import mindspore.dataset.transforms.c_transforms as deC
from mindspore.common.tensor import Tensor
from mindspore.nn.optim import Adam
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import Callback, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src import tokenization
from src.data_utils import create_training_instance, write_instance_to_file

context.set_context(mode=context.GRAPH_MODE, max_call_depth=20000, device_target="GPU", device_id = 0)

print("in VQA_eval")

vqa_cfg = edict({
    #--------------------------------------nework config-------------------------------------
    'max_length': 512,
    'vocab_file': 'vocab.txt',

    'lr_schedule': edict({
        'learning_rate': 1.0,
        'warmup_steps': 8000,
        'start_decay_step': 16000,
        'min_lr': 0.0,
    }),

    #-----------------------------------save model config-------------------------
    'enable_save_ckpt': False ,        #Enable save checkpointdefault is true.
    'save_checkpoint_steps':5,   #Save checkpoint steps, default is 590.
    'save_checkpoint_num':2,     #Save checkpoint numbers, default is 2.
    'save_checkpoint_path': './checkpoint',    #Save checkpoint file path,default is ./checkpoint/
    'save_checkpoint_name':'vqa',
    'checkpoint_path':'./checkpoint/vqasim_4-30_3.ckpt',     #Checkpoint file path


    #-----------------------------------------------------------
    'train_file_mindrecord': './sim_mindrecord/train.mindrecord',
    'test_file_mindrecord': './sim_testpremindrecord/pre_test.mindrecord',
    'val_file_mindrecord': './sim_mindrecord/val.mindrecord',
    'epoch_size': 30,
    'batch_size': 1,
    'pre_training_ckpt': './ckpt/bert_large_en.ckpt'
})


from vqa.vqa_model import VQASimModel
from resnet.resnet import resnet50

def load_pre_dataset(batch_size=1, data_file=None):
    """
    Load mindrecord dataset
    """
    ds = de.MindDataset(data_file,
                        columns_list=["source_input",
                                      "image_vec", "label"],
                        shuffle=False)
    type_cast_op = deC.TypeCast(mstype.int32)
    type_cast_float = deC.TypeCast(mstype.float32)
    ds = ds.map(input_columns="image_vec", operations=type_cast_float)
    ds = ds.map(input_columns="label", operations=type_cast_float)    
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    ds.channel_name = 'vqa'
    return ds

resnet = resnet50(1024)

total_question = 0
total_right = 0


print("load all answer class")
answer_class = np.load("sim_class.npy", allow_pickle=True).item()
print("possible answer num:", len(answer_class.keys()))

key_num = len(answer_class.keys())
key_list = []
key_token_list = []

for key in answer_class.keys():
    key_list.append(key)
    key_token = key.split()
    key_token_list.append(key_token)

print(key_list[0])
print(key_token_list[0])
print(answer_class[key_list[0]])


""" load bert """
from bert_new.src.bert_model import BertModel, BertConfig

print("load bert")
is_training = False
bert_net_cfg = BertConfig(seq_length=512, hidden_size=1024,num_hidden_layers=24,
                 num_attention_heads=16, vocab_size=30522, type_vocab_size=2, intermediate_size=4096)
use_one_hot_embeddings = False
bert_net = BertModel(bert_net_cfg, is_training, use_one_hot_embeddings)

param_dict = load_checkpoint(vqa_cfg.pre_training_ckpt)
load_param_into_net(bert_net, param_dict)

# 初始化字典 token             
tokenizer = tokenization.WhiteSpaceTokenizer(vocab_file=vqa_cfg.vocab_file) 

def convert_ids_and_mask(input_tokens, tokenizer, max_seq_length):
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    return input_ids, input_mask

def eval(cfg, resnet):
    """
    VQA evaluate
    """
    print("load dataset")
    eval_dataset = load_pre_dataset(cfg.batch_size, data_file=cfg.test_file_mindrecord)

    print("load model")
    VQANet = VQASimModel(resnet, False)
    param_dict = load_checkpoint(cfg.checkpoint_path)
    load_param_into_net(VQANet, param_dict)
    
    print("start evaluate")
    for eval_data in eval_dataset.create_dict_iterator():
        
        # source_input = [batch_size]
        source_input = eval_data['source_input'].asnumpy()[0]
        image_vec = eval_data['image_vec'].asnumpy()
        label = eval_data['label'].asnumpy()[0]

        global total_right
        global total_question
        
        EOS = '[SEP]'
        SOS = '[CLS]'


        """ 首先将输入字符串处理成 bert 的 ids 格式 """
        input_ids = np.zeros((key_num, cfg.max_length))
        input_mask = np.zeros((key_num, cfg.max_length))
        input_token_ids = np.zeros((key_num, cfg.max_length))

        for key_idx in range(key_num):
            source_token = source_input.split()
            key_token = key_token_list[key_idx]
            #print(source_token)
            #print(key_token)

            input_token = [SOS] + source_token + [EOS] + key_token + [EOS]
            #print(input_token)

            input_ids[key_idx], input_mask[key_idx] = convert_ids_and_mask(input_token, tokenizer, cfg.max_length)
            #print(input_ids)
            #print(input_mask)

        
        """
        print(input_ids[0])
        print(input_mask[0])
        print(input_ids.shape)
        print(input_mask.shape)
        print(input_token_ids.shape)
        """
        
            #bert_input[key_idx] = 

        input_ids = Tensor(input_ids, dtype=mstype.int32)
        input_mask = Tensor(input_mask, dtype=mstype.int32)
        input_token_ids = Tensor(input_token_ids, dtype=mstype.int32)
        #print(input_ids)

        # [batch_size, ]
        source_input, _, _ = bert_net(input_ids, input_token_ids, input_mask)
        
        #print(source_input.shape)

        #print(image_vec.shape)
        image_vec = Tensor(np.repeat(image_vec, key_num, axis=0), dtype=mstype.float32)
        #print(image_vec.shape)

        similarity_matrix = VQANet(source_input, image_vec).asnumpy()
        #print(similarity_matrix.shape)
        #print(similarity_matrix)

        sim_list = []
        for dg in range(similarity_matrix.shape[0]):
            sim_list.append(similarity_matrix[dg][dg])
        #print(sim_list)

        ans_index = heapq.nlargest(3, range(len(sim_list)), sim_list.__getitem__)
        #print(ans_index)
        answer = False

        for id in ans_index:
            # 找到了正确答案
            if label[id] == 1:
                answer = True


        if answer:
            total_right += 1


        print("pred answer:", key_list[ans_index[0]])
        for id in range(key_num):
            if label[id] == 1:
                print("true answer:", key_list[id])
                break
        
        total_question += 1
        print("total questions: {0}, total right: {1}".format(total_question, total_right))

eval(vqa_cfg, resnet)
print("total questions: {0}, total right: {1}".format(total_question, total_right))
