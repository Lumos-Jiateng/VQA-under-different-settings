import os
import numpy as np
from easydict import EasyDict as edict

import mindspore as ms
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

#context.set_context(mode=context.PYNATIVE_MODE, max_call_depth=20000, device_target="GPU", device_id=0)
context.set_context(mode=context.GRAPH_MODE, max_call_depth=20000, device_target="GPU", device_id=0)

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
    'enable_save_ckpt': True ,        #Enable save checkpointdefault is true.
    'save_checkpoint_steps': 12,   #Save checkpoint steps, default is 80.
    'save_checkpoint_num':2,     #Save checkpoint numbers, default is 2.
    'save_checkpoint_path': './checkpoint',    #Save checkpoint file path,default is ./checkpoint/
    'save_checkpoint_name':'vqasim',
    'checkpoint_path':'',     #Checkpoint file path


    #-----------------------------------------------------------
    'train_file_mindrecord': './sim_mindrecord/test.mindrecord',
    'test_file_mindrecord': './sim_mindrecord/test.mindrecord',
    'val_file_mindrecord': './sim_mindrecord/val.mindrecord',
    'epoch_size': 30,
    'batch_size': 16,
    'pre_training_ckpt': './ckpt/bert_large_en.ckpt'

})

def load_dataset(batch_size=1, data_file=None):
    """
    Load mindrecord dataset
    """
    ds = de.MindDataset(data_file,
                        columns_list=["source_input", "image_vec", "label"],
                        shuffle=False)
    type_cast_op = deC.TypeCast(mstype.int32)
    type_cast_float = deC.TypeCast(mstype.float32)
    ds = ds.map(input_columns="source_input", operations=type_cast_float)
    ds = ds.map(input_columns="image_vec", operations=type_cast_float)
    ds = ds.map(input_columns="label", operations=type_cast_float)    
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    ds.channel_name = 'vqa'
    return ds

data = next(load_dataset(batch_size = 4, data_file=vqa_cfg.train_file_mindrecord).create_dict_iterator())

source_input = data['source_input']
image_vec = data['image_vec']
label = data['label']

print(source_input.shape)
print(image_vec.shape)
print(label.shape)

from vqa.vqa_model import VQASimModel
from vqa.lr_schedule import create_dynamic_lr
from resnet.resnet import resnet50
from vqa.callback import LossCallBack

resnet = resnet50(1024)

vqa_net = VQASimModel(resnet)
similarity = vqa_net(source_input, image_vec)


from vqa.vqasim_for_train import VQASimTrainOneStepCell, VQASimNetworkWithLoss

vqa_loss = VQASimNetworkWithLoss(resnet, True)


loss = vqa_loss(source_input, image_vec)
print(loss)

print("start testing")
cfg = vqa_cfg
train_dataset = load_dataset(vqa_cfg.batch_size, data_file=vqa_cfg.train_file_mindrecord)
lr = Tensor(create_dynamic_lr(schedule="constant*linear_warmup*rsqrt_decay",
                                  training_steps=train_dataset.get_dataset_size()*cfg.epoch_size,
                                  learning_rate=cfg.lr_schedule.learning_rate,
                                  warmup_steps=cfg.lr_schedule.warmup_steps,
                                  start_decay_step=cfg.lr_schedule.start_decay_step,
                                  min_lr=cfg.lr_schedule.min_lr), mstype.float32)
optimizer = Adam(vqa_loss.trainable_params(), lr)
vqa_train = VQASimTrainOneStepCell(vqa_loss, optimizer)

vqa_train.construct(source_input, image_vec, label)

def train(cfg, resnet):
    """
    VQA training.
    """
    
    train_dataset = load_dataset(cfg.batch_size, data_file=cfg.train_file_mindrecord)

    netwithloss = VQASimNetworkWithLoss(resnet, True)

    if cfg.checkpoint_path:
        parameter_dict = load_checkpoint(cfg.checkpoint_path)
        load_param_into_net(netwithloss, parameter_dict)

    lr = Tensor(create_dynamic_lr(schedule="constant*linear_warmup*rsqrt_decay",
                                  training_steps=train_dataset.get_dataset_size()*cfg.epoch_size,
                                  learning_rate=cfg.lr_schedule.learning_rate,
                                  warmup_steps=cfg.lr_schedule.warmup_steps,
                                  start_decay_step=cfg.lr_schedule.start_decay_step,
                                  min_lr=cfg.lr_schedule.min_lr), mstype.float32)
    optimizer = Adam(netwithloss.trainable_params(), lr)

    loss_result = []

    callbacks = [TimeMonitor(train_dataset.get_dataset_size()), LossCallBack()]
    if cfg.enable_save_ckpt:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                       keep_checkpoint_max=cfg.save_checkpoint_num)
        ckpoint_cb = ModelCheckpoint(prefix=cfg.save_checkpoint_name, directory=cfg.save_checkpoint_path, config=ckpt_config)
        callbacks.append(ckpoint_cb)


    netwithgrads = VQASimTrainOneStepCell(netwithloss, optimizer=optimizer)

    netwithgrads.set_train(True)
    model = Model(netwithgrads)
    print("start training")
    model.train(cfg.epoch_size, train_dataset, callbacks=callbacks)

train(vqa_cfg, resnet)
