import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.parameter import Parameter, ParameterTuple

from vqa.vqa_model import VQASimModel

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 5.0

class ClipGradients(nn.Cell):
    """
    Clip gradients.

    Args:
        grads (list): List of gradient tuples.
        clip_type (Tensor): The way to clip, 'value' or 'norm'.
        clip_value (Tensor): Specifies how much to clip.

    Returns:
        List, a list of clipped_grad tuples.
    """
    def __init__(self):
        super(ClipGradients, self).__init__()
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = P.Cast()
        self.dtype = P.DType()

    def construct(self,
                  grads,
                  clip_type,
                  clip_value):
        """Defines the gradients clip."""
        if clip_type != 0 and clip_type != 1:
            return grads

        new_grads = ()
        for grad in grads:
            dt = self.dtype(grad)
            if clip_type == 0:
                t = C.clip_by_value(grad, self.cast(F.tuple_to_array((-clip_value,)), dt),
                                    self.cast(F.tuple_to_array((clip_value,)), dt))
            else:
                t = self.clip_by_norm(grad, self.cast(F.tuple_to_array((clip_value,)), dt))
            new_grads = new_grads + (t,)

        return new_grads

class VQASimTrainingLoss(nn.Cell):
    '''
    Provide VQA training loss

    Returns:
        Tensor, total loss.
    '''
    def __init__(self):
        super(VQASimTrainingLoss, self).__init__(auto_prefix=False)
        self.loss_fn = P.SoftmaxCrossEntropyWithLogits()
        self.reduce_mean = P.ReduceMean()
        self.print = ops.Print()
    
    def construct(self, similarity_matrix):
        '''
        Defines the computation performed.
        similarity_matrix = [batch_size, batch_size]

        '''
        #print("VQASimTrainingLoss input:\n", similarity_matrix)
        # 标准矩阵是一个单位矩阵
        label_matrix = Tensor(np.identity(similarity_matrix.shape[0]), dtype=mstype.float32)


        #print("label matrix:\n", label_matrix)
        #print("similarity matrix:\n", similarity_matrix)
        loss, dlogits = self.loss_fn(similarity_matrix, label_matrix)
        #print(dlogits)
        loss = self.reduce_mean(loss)
        return loss


class VQASimNetworkWithLoss(nn.Cell):
    '''
    Provide VQA loss through network

    Returns:
        Tensor, the loss of the network
    '''

    def __init__(self, resnet, is_training = False):
        super(VQASimNetworkWithLoss, self).__init__(auto_prefix=False)
        self.vqa = VQASimModel(resnet, is_training)
        self.loss = VQASimTrainingLoss()
        self.print = ops.Print()
        self.cast = P.Cast()

    def construct(self, source_input, image_vec):
        """ VQA network with loss """
        similarity_matrix = self.vqa(source_input, image_vec)
        #print("VQASimNetworkWithLoss output:\n", similarity_matrix)
        total_loss = self.loss(similarity_matrix)
        return self.cast(total_loss, mstype.float32)

class VQASimTrainOneStepCell(nn.Cell):
    """
    Encapsulation class of VQASim network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(VQASimTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.clip_gradients = ClipGradients()
        self.cast = P.Cast()

    def set_sens(self, value):
        self.sens = value

    def construct(self, source_input, image_vec, label):
        """Defines the computation performed."""

        weights = self.weights
        loss = self.network(source_input, image_vec)
        
        grads = self.grad(self.network, weights)(source_input, image_vec,
                                                 self.cast(F.tuple_to_array((self.sens,)),
                                                           mstype.float32))
        
        
        grads = self.clip_gradients(grads, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE)

        succ = self.optimizer(grads)
        return F.depend(loss, succ)