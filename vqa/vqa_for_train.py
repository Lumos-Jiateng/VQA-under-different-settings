import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.parameter import Parameter, ParameterTuple

from vqa.vqa_model import VQAModel

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

class VQATrainingLoss(nn.Cell):
    '''
    Provide VQA training loss

    Returns:
        Tensor, total loss.
    '''
    def __init__(self):
        super(VQATrainingLoss, self).__init__(auto_prefix=False)
        self.loss_fn = P.SoftmaxCrossEntropyWithLogits()
        self.reduce_mean = P.ReduceMean()
    
    def construct(self, predict_class, label):
        '''
        Defines the computation performed.
        label = [batch_size, num_class]
        predict_class = [batch_size, num_class]

        '''
        loss, dlogits = self.loss_fn(predict_class, label)
        #print(loss)
        loss = self.reduce_mean(loss)
        return loss


class VQANetworkWithLoss(nn.Cell):
    '''
    Provide VQA loss through network

    Returns:
        Tensor, the loss of the network
    '''

    def __init__(self, resnet, is_training = False, fc_dim = 2048, class_num = 300):
        super(VQANetworkWithLoss, self).__init__(auto_prefix=False)
        self.vqa = VQAModel(resnet, is_training, fc_dim, class_num)
        self.loss = VQATrainingLoss()
        self.cast = P.Cast()

    def construct(self, source_input, image_vec, label):
        """ VQA network with loss """
        predict_class = self.vqa(source_input, image_vec)
        #print("predict_class shape:", predict_class.shape)
        #print("label shape:", label.shape)
        total_loss = self.loss(predict_class, label)
        return self.cast(total_loss, mstype.float32)

class VQATrainOneStepCell(nn.Cell):
    """
    Encapsulation class of VQA network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(VQATrainOneStepCell, self).__init__(auto_prefix=False)
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
        loss = self.network(source_input, image_vec, label)
        
        grads = self.grad(self.network, weights)(source_input, image_vec, label,
                                                 self.cast(F.tuple_to_array((self.sens,)),
                                                           mstype.float32))
        
        
        grads = self.clip_gradients(grads, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE)

        succ = self.optimizer(grads)
        return F.depend(loss, succ)