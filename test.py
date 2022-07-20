import os
from matplotlib.pyplot import axis
import numpy as np
import heapq
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

a = np.array([[1, 2]])
print(a)
a = np.repeat(a, 3, axis = 0)
print(a)