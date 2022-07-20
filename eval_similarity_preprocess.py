import os
import numpy as np
from easydict import EasyDict as edict

import mindspore.nn as nn
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

context.set_context(mode=context.GRAPH_MODE, max_call_depth=20000, device_target="GPU")

data_cfg = edict({
        'vocab_file': 'vocab.txt',
        'pre_train_file_mindrecord': './sim_premindrecord/pre_train.mindrecord',
        'pre_test_file_mindrecord': './sim_testpremindrecord/pre_test.mindrecord',
        'pre_val_file_mindrecord': './sim_premindrecord/pre_val.mindrecord',
        'train_file_mindrecord': './sim_mindrecord/train.mindrecord',
        'test_file_mindrecord': './sim_mindrecord/test.mindrecord',
        'val_file_mindrecord': './sim_mindrecord/val.mindrecord',
        'num_splits':4,
        'clip_to_max_len': False,
        'seq_length': 512,
        'class_num': 0,
        'pre_training_ckpt': './ckpt/bert_large_en.ckpt'
})

""" preprocess """

import json
import os
import numpy as np
from PIL import Image

def token_to_ids(token_list, answer_vocab):
    ids = []
    for token in token_list:
        if token in answer_vocab:
            ids.append(answer_vocab[token])
        else :
            answer_vocab[token] = len(answer_vocab)
            ids.append(answer_vocab[token])
    return ids

def data_preprocess(file, answer_vocab, answer_class, class_answer):

    className = dict()

    # 答案的路径
    anno = open(os.path.join('../data/annotations', file + '.json'))
    # 图片的路径
    image_path = os.path.join('../data/images', file)
    # 问题的路径
    ques = open(os.path.join('../data/questions', file + '.json'))

    # 读取答案 json
    annotations = json.load(anno)
    answer_list = annotations['annotations']
    print(answer_list[0])

    # 读取问题 json
    questions = json.load(ques)
    question_list = questions['questions']
    print(question_list[0])

    cf = 0
    num = 0

    prepare_image = []
    prepare_question = []
    prepare_answer_seq = []
    prepare_answer_class = []

    for answer in answer_list:
        print("\rtotal: {0}, can't find: {1}".format(num, cf), end="")

        '''
        找出 question 的 答案
        '''
        # 对 answer 做 classification
        answer_st = answer['multiple_choice_answer']
        if answer_st not in answer_class:
            answer_class[answer_st] = len(answer_class)
        
        label = [0] * len(answer_class)
        label[answer_class[answer_st]] = 1
        answer_label = label
        class_answer[answer_class[answer_st]] = answer_st
        
        # 对 answer 做 seq2seq
        answer_st = answer_st.strip()
        answer_token = []
        if answer_st:
            answer_token = answer_st.split()
            answer_ids = token_to_ids(answer_token, answer_vocab)

        '''
        读取 question 的 图片
        '''
        has_image = False
        if file == "test" or file == "val" :
            answer_picture = os.path.join(image_path, "COCO_val2014_" + str(answer['image_id']).zfill(12) + ".jpg")
        else :
            answer_picture = os.path.join(image_path, "COCO_train2014_" + str(answer['image_id']).zfill(12) + ".jpg")
        image = 0
        try :
            img = Image.open(answer_picture)
            img = img.resize((480, 640))
            image = np.array(img)
            if len(image.shape) == 3:
                has_image = True
        except IOError:
            cf = cf + 1

        '''
        读取 question 的 问题
        '''
        has_question = False
        question_st = ""
        for i in range(len(question_list)):
            if(answer['question_id'] == question_list[i]['question_id']):
                question_st = question_list[i]['question']
                # 需要对结尾 ? 做一点操作
                question_st = question_st.replace('?', " ?")
                has_question = True
                break
            
        if has_question and has_image:
            prepare_image.append(image)
            prepare_question.append(question_st)
            prepare_answer_seq.append(answer_ids)
            prepare_answer_class.append(answer_label)
        
        num = num + 1
        if(num > 50) :
            break
    print("\n")
    return prepare_question, prepare_image, prepare_answer_seq, prepare_answer_class


answer_vocab = dict()
answer_class = dict()
class_answer = dict()
test_question, test_image, test_answer_seq, test_answer_class = data_preprocess('test', answer_vocab, answer_class, class_answer)
train_question, train_image, train_answer_seq, train_answer_class = data_preprocess('train', answer_vocab, answer_class, class_answer)
val_question, val_image, val_answer_seq, val_answer_class = data_preprocess('val', answer_vocab, answer_class, class_answer)

data_cfg.class_num = len(answer_class)
print("class_num:", len(answer_class))

def pre_data_prepare(cfg, dataset, save_file, question_list, image_list, answer_list, answer_class) :
    
    # 初始化字典 token             
    tokenizer = tokenization.WhiteSpaceTokenizer(vocab_file=cfg.vocab_file)  
                
    # 初始化 mindrecord writer
    writer = FileWriter(save_file, 1, overwrite = True)
    
    # 设置 mindrecord 结构
    data_schema = {
                    "source_input"    : {"type": "string"},
                    "image_vec"       : {"type": "int32", "shape": [640, 480, 3]},
                    "label"           : {"type": "int32", "shape": [-1]}
                  }
    
    writer.add_schema(data_schema, dataset)
    
    index = 0
    data = []
                 
    for i in range(len(question_list)):
        # 将每个 question 句子拆成 token
        s_line = question_list[i]
        line = tokenization.convert_to_unicode(s_line).strip()
        line_tokens = tokenizer.tokenize(line)
        
        instance = create_training_instance(line_tokens, answer_list[i], image_list[i], answer_class[i], cfg.seq_length, cfg.class_num)         
        
        # 写入文件缓存区
        features = write_instance_to_file(writer, instance, tokenizer, cfg.seq_length) 

        data = {
            "source_input"   : question_list[i],
            "image_vec"      : np.asarray(features['image_vec'] , dtype=np.int32),
            "label"          : np.asarray(features['label']     , dtype=np.int32)
        }
        
        writer.write_raw_data([data])
        index = index + 1
        print("finish {}/{}".format(index, len(question_list)), end='\r')

    #writer.write_raw_data(data)
    writer.commit()

print("generate test mindrecord")
pre_data_prepare(data_cfg, "test", data_cfg.pre_test_file_mindrecord, test_question, test_image, test_answer_seq, test_answer_class)