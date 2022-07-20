import numpy as np

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype


class VQAModel(nn.Cell):
    '''
    VQA with Bert and Restnet on top level
        MLP deal with the classfication task 
    '''
    def __init__(self, resnet, is_training = False, fc_dim = 2048, class_num = 300):
        super(VQAModel, self).__init__()
        self.is_training = is_training
        self.seq2seq = nn.LSTM(1024, 1024, 2, has_bias=True, batch_first=True, bidirectional=True)
        self.reduce = nn.Dense(2048, 1024)
        self.resnet = resnet 
        
        # 利用图片特征对 句长为 512 的 embedding 做 attention
        self.attn_dense = nn.Dense(1024, 512)
        self.projection = nn.Dense(fc_dim, class_num)
        self.softmax = nn.Softmax(axis=1)
        
        self.trans = P.Transpose()
        self.perm = (0, 3, 1, 2)
        self.bmm = P.BatchMatMul()
        self.cast = P.Cast()
        self.concat = P.Concat(axis=1)
        
    def construct(self, source_input, image_vec):
        image_vec = self.trans(image_vec, self.perm)
        
        batch_size = source_input.shape[0]

        # sequence_output = [batch, sequence_length, seq_dim * 2] = [4, 512, 2048]
        #sequence_output, pooled_output, embedding_tables = self.bert(source_ids, token_type_ids, source_mask)
        h0 = Tensor(np.zeros((2 * 2, batch_size, 1024)), dtype=mstype.float32)
        c0 = Tensor(np.zeros((2 * 2, batch_size, 1024)), dtype=mstype.float32)
        sequence_output, _ = self.seq2seq(source_input, (h0, c0))

        #print("sequence_output shape:", sequence_output.shape)
        #sequence_output = [batch, sequence_length, seq_dim] = [4, 512, 1024]
        sequence_output = self.reduce(sequence_output)
        #sequence_output = source_input
        
        # image_output    = [batch, img_dim] = [4, 1024]
        image_output = self.resnet(image_vec)
        #print("image_output shape:", image_output.shape)
        
        # attn_weights = [batch_size, sequence_length]
        attn_weights = self.softmax(self.attn_dense(image_output))
        batch_size = attn_weights.shape[0]
        sequence_length = attn_weights.shape[-1]
        #print("attn_weights shape:", attn_weights.shape)
        
        # attn_weights = [batch_size, 1, sequence_length]
        attn_weights = attn_weights.reshape(batch_size, 1, sequence_length)
        #print("attn_weights shape:", attn_weights.shape)
        
        # sequence_att = [batch_size, fc_dim] = [4, 1024]
        sequence_attn = self.bmm(attn_weights, self.cast(sequence_output, mstype.float32))
        sequence_attn = sequence_attn.reshape(batch_size, -1)
        #print("sequence_attn shape:", sequence_attn.shape)
        
        # output = [batch_size, img_dim + sequence_dim] = [4, 2048]
        output = self.concat((image_output, sequence_attn))
        #print("output shape:", output.shape)
        
        # output_class = [batch_size, class_num]
        output_class = self.projection(output)
        #print("output_class:", output_class.shape)
        
        return output_class

class VQASimModel(nn.Cell):
    '''
    VQA with Bert and Restnet on top level
        MLP deal with the classfication task 
    '''
    def __init__(self, resnet, is_training = False):
        super(VQASimModel, self).__init__()
        self.is_training = is_training
        self.seq2seq = nn.LSTM(1024, 1024, 2, has_bias=True, batch_first=True, bidirectional=True)
        self.reduce = nn.Dense(2048, 1024)
        self.resnet = resnet 
        
        # 利用图片特征对 句长为 512 的 embedding 做 attention
        self.attn_dense = nn.Dense(1024, 512)
        self.softmax = nn.Softmax(axis=1)
        
        self.trans = P.Transpose()
        self.perm = (0, 3, 1, 2)
        self.bmm = P.BatchMatMul()
        self.cast = P.Cast()
        self.concat = P.Concat(axis=1)
        self.matrix_perm = (1, 0)
        self.matrix_mul = ops.MatMul()

        
    def construct(self, source_input, image_vec):
        image_vec = self.trans(image_vec, self.perm)
        
        batch_size = source_input.shape[0]

        # sequence_output = [batch, sequence_length, seq_dim * 2] = [4, 512, 2048]
        #sequence_output, pooled_output, embedding_tables = self.bert(source_ids, token_type_ids, source_mask)
        h0 = Tensor(np.zeros((2 * 2, batch_size, 1024)), dtype=mstype.float32)
        c0 = Tensor(np.zeros((2 * 2, batch_size, 1024)), dtype=mstype.float32)
        sequence_output, _ = self.seq2seq(source_input, (h0, c0))

        #print("sequence_output shape:", sequence_output.shape)
        # sequence_output = [batch, sequence_length, seq_dim] = [4, 512, 1024]
        sequence_output = self.reduce(sequence_output)
        
        # image_output    = [batch, img_dim] = [4, 1024]
        image_output = self.resnet(image_vec)
        #print("image_output shape:", image_output.shape)
        
        # attn_weights = [batch_size, sequence_length]
        attn_weights = self.softmax(self.attn_dense(image_output))
        batch_size = attn_weights.shape[0]
        sequence_length = attn_weights.shape[-1]
        #print("attn_weights shape:", attn_weights.shape)
        
        # attn_weights = [batch_size, 1, sequence_length]
        attn_weights = attn_weights.reshape(batch_size, 1, sequence_length)
        #print("attn_weights shape:", attn_weights.shape)
        
        # sequence_att = [batch_size, fc_dim] = [4, 1024]
        sequence_attn = self.bmm(attn_weights, self.cast(sequence_output, mstype.float32))
        sequence_attn = sequence_attn.reshape(batch_size, -1)
        #print("sequence_attn shape:", sequence_attn.shape)
        
        # similarity_matrix = [batch_size, batch_size]
        image_reverse = self.trans(image_output, self.matrix_perm)
        similarity_matrix = self.matrix_mul(sequence_attn, image_reverse)
        

        return similarity_matrix