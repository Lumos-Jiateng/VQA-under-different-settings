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
        'pre_test_file_mindrecord': './sim_premindrecord/pre_test.mindrecord',
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
            prepare_answer_seq.append(answer_token)
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
np.save("sim_class.npy", answer_class)

def pre_data_prepare(cfg, dataset, save_file, question_list, image_list, answer_list, answer_class) :
    
    # 初始化字典 token             
    tokenizer = tokenization.WhiteSpaceTokenizer(vocab_file=cfg.vocab_file)  
                
    # 初始化 mindrecord writer
    writer = FileWriter(save_file, 1, overwrite = True)
    
    # 设置 mindrecord 结构
    data_schema = {
                    "source_ids"      : {"type": "int32", "shape": [-1]},
                    "source_mask"     : {"type": "int32", "shape": [-1]},
                    "token_type_ids"  : {"type": "int32", "shape": [-1]},
                    "target_ids"      : {"type": "int32", "shape": [-1]},
                    "image_vec"       : {"type": "int32", "shape": [640, 480, 3]},
                    "label"           : {"type": "int32", "shape": [-1]}
                  }
    
    '''
    data_schema = {
                "source_input"    : {"type": "float32", "shape" : [512, 1024]},
                "target_ids"      : {"type": "int32", "shape": [-1]},
                "image_vec"       : {"type": "int32", "shape": [640, 480, 3]},
                "label"           : {"type": "int32", "shape": [-1]}
              }
    '''
    
    writer.add_schema(data_schema, dataset)
    
    index = 0
    data = []
                 
    for i in range(len(question_list)):
        # 将每个 question 句子拆成 token
        s_line = question_list[i]
        line = tokenization.convert_to_unicode(s_line).strip()
        line_tokens = tokenizer.tokenize(line)

        # 把问题的 token 和 答案的 token 连接在一起
        EOS = "[SEP]"
        #print(line_tokens)
        #print(answer_list[i])
        source_tokens = line_tokens + [EOS] + answer_list[i]
        #print(source_tokens)

        
        instance = create_training_instance(source_tokens, [0], image_list[i], answer_class[i], cfg.seq_length, cfg.class_num)         
        
        # 写入文件缓存区
        features = write_instance_to_file(writer, instance, tokenizer, cfg.seq_length) 
        '''
        source_ids = Tensor(np.expand_dims(features['source_ids'], axis = 0), dtype=mstype.int32)
        token_type_ids = Tensor(np.expand_dims(features['token_type_ids'], axis = 0), dtype=mstype.int32)
        source_mask = Tensor(np.expand_dims(features['source_mask'], axis = 0), dtype=mstype.int32)
        sequence_output, pooled_output, embedding_tables = bert_net(source_ids, token_type_ids, source_mask)
        sequence_output = sequence_output.asnumpy()
        data = {
            "source_input"   : sequence_output[0],
            "target_ids"     : np.asarray(features['target_ids'], dtype=np.int32),
            "image_vec"      : np.asarray(features['image_vec'] , dtype=np.int32),
            "label"          : np.asarray(features['label']     , dtype=np.int32)
        }
        '''
        writer.write_raw_data([features])
        index = index + 1
        print("finish {}/{}".format(index, len(question_list)), end='\r')

    #writer.write_raw_data(data)
    writer.commit()

print("generate test mindrecord")
pre_data_prepare(data_cfg, "test", data_cfg.pre_test_file_mindrecord, test_question, test_image, test_answer_seq, test_answer_class)
print("generate train mindrecord")
pre_data_prepare(data_cfg, "train", data_cfg.pre_train_file_mindrecord, train_question, train_image, train_answer_seq, train_answer_class)
print("generate val mindrecord")
pre_data_prepare(data_cfg, "val", data_cfg.pre_val_file_mindrecord, val_question, val_image, val_answer_seq, val_answer_class)


""" process"""

print("load bert")
from bert_new.src.bert_model import BertModel, BertConfig

is_training = False
bert_net_cfg = BertConfig(seq_length=512, hidden_size=1024,num_hidden_layers=24,
                 num_attention_heads=16, vocab_size=30522, type_vocab_size=2, intermediate_size=4096)
use_one_hot_embeddings = False
bert_net = BertModel(bert_net_cfg, is_training, use_one_hot_embeddings)

param_dict = load_checkpoint(data_cfg.pre_training_ckpt)
load_param_into_net(bert_net, param_dict)

print("load success")

print("load pre data")
def load_pre_dataset(batch_size=1, data_file=None):
    """
    Load mindrecord dataset
    """
    ds = de.MindDataset(data_file,
                        columns_list=["source_ids", "source_mask", "token_type_ids",
                                      "image_vec", "label"],
                        shuffle=False)
    type_cast_op = deC.TypeCast(mstype.int32)
    type_cast_float = deC.TypeCast(mstype.float32)
    ds = ds.map(input_columns="source_ids", operations=type_cast_op)
    ds = ds.map(input_columns="source_mask", operations=type_cast_op)
    ds = ds.map(input_columns="token_type_ids", operations=type_cast_op)
    ds = ds.map(input_columns="image_vec", operations=type_cast_float)
    ds = ds.map(input_columns="label", operations=type_cast_float)    
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    ds.channel_name = 'vqa'
    return ds


def data_prepare(cfg, dataset, save_file) :
                
    # 初始化 mindrecord writer
    writer = FileWriter(save_file, 1, overwrite = True)
    
    # 设置 mindrecord 结构
    
    
    data_schema = {
                "source_input"    : {"type": "float32", "shape" : [512, 1024]},
                "image_vec"       : {"type": "int32", "shape": [640, 480, 3]},
                "label"           : {"type": "int32", "shape": [-1]}
              }
    
    
    writer.add_schema(data_schema, "dataset")
    
    index = 0
                 
    for data in dataset.create_dict_iterator():
        source_ids  = data['source_ids']
        source_mask = data['source_mask']
        token_type_ids = data['token_type_ids']
        image_vec = data['image_vec'].asnumpy()
        label = data['label'].asnumpy()
        
        sequence_output, pooled_output, embedding_tables = bert_net(source_ids, token_type_ids, source_mask)
        sequence_output = sequence_output.asnumpy()
        features = []
        for i in range(sequence_output.shape[0]):
            feature = {
                'source_input' : sequence_output[i],
                'image_vec' : image_vec[i],
                'label'     : label[i]
            }
            features.append(feature)
        writer.write_raw_data(features)

        index = index + label.shape[0]
        print("finish {}".format(index) , end='\r')

    #writer.write_raw_data(data)
    writer.commit()

print("start train data prepare")
pre_train_data = load_pre_dataset(batch_size = 16, data_file=data_cfg.pre_train_file_mindrecord)
print("pre_train_data load success")
data_prepare(data_cfg, pre_train_data, data_cfg.train_file_mindrecord)
print("start test data prepare")
pre_test_data = load_pre_dataset(batch_size = 16, data_file=data_cfg.pre_test_file_mindrecord)
print("pre_test_data load success")
data_prepare(data_cfg, pre_test_data, data_cfg.test_file_mindrecord)
print("start val data prepare")
pre_val_data = load_pre_dataset(batch_size = 16, data_file=data_cfg.pre_val_file_mindrecord)
print("pre_val_data load success")
data_prepare(data_cfg, pre_val_data, data_cfg.val_file_mindrecord)