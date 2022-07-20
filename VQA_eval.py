
import json
import os
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
from vqa.vqa_model import VQAModel

context.set_context(mode=context.GRAPH_MODE, max_call_depth=20000, device_target="GPU", device_id = 1)

print("in VQA_eval")

vqa_cfg = edict({
    #--------------------------------------nework config-------------------------------------
    'class_num': 8193,
    'fc_dim': 2048,
    
    'lr_schedule': edict({
        'learning_rate': 1.0,
        'warmup_steps': 8000,
        'start_decay_step': 16000,
        'min_lr': 0.0,
    }),

    #-----------------------------------save model config-------------------------
    'enable_save_ckpt': False ,        #Enable save checkpointdefault is true.
    'save_checkpoint_steps':590,   #Save checkpoint steps, default is 590.
    'save_checkpoint_num':2,     #Save checkpoint numbers, default is 2.
    'save_checkpoint_path': './checkpoint',    #Save checkpoint file path,default is ./checkpoint/
    'save_checkpoint_name':'vqa',
    #'checkpoint_path':'./checkpoint/vqa_nolstm-3_716.ckpt',     #Checkpoint file path
    'checkpoint_path':'./checkpoint/vqa_6-2_716.ckpt',

    #-----------------------------------------------------------
    'train_file_mindrecord': './mindrecordtmp/train.mindrecord',
    'test_file_mindrecord': './val_mindrecord/val.mindrecord',
    'val_file_mindrecord': './mindrecordtmp/train.mindrecord',
    'epoch_size': 30,
    'batch_size': 16,
    'pre_training_ckpt': './ckpt/bert_large_en.ckpt'
})


from vqa.vqa_model import VQAModel
from resnet.resnet import resnet50

def load_dataset(batch_size=1, data_file=None):
    """
    Load mindrecord dataset
    """
    ds = de.MindDataset(data_file,
                        columns_list=["question_type", "answer_type", "multiple_choice_answer",
                                      "source_input", "image_vec", "label"],
                        shuffle=False)
    type_cast_op = deC.TypeCast(mstype.int32)
    type_cast_float = deC.TypeCast(mstype.float32)
    type_cast_string =  deC.TypeCast(mstype.string)
    ds = ds.map(input_columns="source_input", operations=type_cast_float)
    ds = ds.map(input_columns="image_vec", operations=type_cast_float)
    ds = ds.map(input_columns="label", operations=type_cast_float)    
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    ds.channel_name = 'vqa'
    return ds

resnet = resnet50(1024)

total_question = 0
total_right = 0

question_type_dict = dict()
answer_type_dict = dict()


def eval(cfg, resnet):
    """
    VQA evaluate
    """
    print("load dataset")
    eval_dataset = load_dataset(cfg.batch_size, data_file=cfg.test_file_mindrecord)

    print("load model")
    VQANet = VQAModel(resnet, False, 2048, 8193)

    param_dict = load_checkpoint(cfg.checkpoint_path)
    load_param_into_net(VQANet, param_dict)

    print("load annotation")
    # 答案的路径
    anno = open(os.path.join('../data/annotations', "test" + '.json'))
    annotations = json.load(anno)['annotations']
    
    print("start evaluate")
    for eval_data in eval_dataset.create_dict_iterator():
        source_input = eval_data['source_input']
        image_vec = eval_data['image_vec']
        batch_question_type = eval_data['question_type'].asnumpy()
        batch_answer_type = eval_data['answer_type'].asnumpy()
        batch_multi_answer = eval_data['multiple_choice_answer'].asnumpy()
        label = eval_data['label'].asnumpy()

        global total_right
        global total_question

        # pred_class = [batch_size, class_num]
        pred_class = VQANet(source_input, image_vec).asnumpy()

        # select top k prob as answer
        for batch in range(cfg.batch_size):
            ans_index = heapq.nlargest(1, range(len(pred_class[batch])), pred_class[batch].__getitem__)
            #print(ans_index)
            answer = False

            for id in ans_index:
                # 找到了正确答案
                if label[batch][id] == 1:
                    answer = True
            question_index = total_question + batch
            question_type = batch_question_type[batch]

            # 如果是没有出现过的, 添加到字典中
            if not question_type in question_type_dict:
                question_type_dict[question_type] = {
                    'total': 0,
                    'right': 0
                }
            
            question_type_dict[question_type]['total'] += 1

            answer_type = batch_answer_type[batch]

            if not answer_type in answer_type_dict:
                answer_type_dict[answer_type] = {
                    'total': 0,
                    'right': 0
                }

            answer_type_dict[answer_type]['total'] += 1

            if answer:
                question_type_dict[question_type]['right'] += 1
                answer_type_dict[answer_type]['right'] += 1
                total_right += 1

            '''
            print("batch: ", batch)
            print("question_type:", question_type)            
            print("answer_type:", answer_type)
            print("true answer:", batch_multi_answer[batch])
            for id in range(8193):
                if label[batch][id] == 1:
                    print("dict answer:", class_answer.item()[id])
                    break
            '''
        
        total_question += cfg.batch_size
        print("finish {0}, total right: {1}".format(total_question, total_right) , end='\r')

     
eval(vqa_cfg, resnet)
print("total questions: {0}, total right: {1}".format(total_question, total_right))
for answer_type in answer_type_dict:
    print("answer_type: {0}, total questions: {1}, total right: {2}".format(answer_type, answer_type_dict[answer_type]['total'], answer_type_dict[answer_type]['right']))
for question_type in question_type_dict:
    print("question_type: {0}, total questions: {1}, total right: {2}".format(question_type, question_type_dict[question_type]['total'], question_type_dict[question_type]['right']))
np.save("nolstm_answer_type_top10.npy", answer_type_dict)
np.save("nosltm_question_type_top10.npy", question_type_dict)
