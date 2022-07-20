# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Create training instances for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import collections
from easydict import EasyDict as edict
import logging
import numpy as np

from mindspore.mindrecord import FileWriter
import src.tokenization as tokenization

class SampleInstance():
    """A single sample instance (sentence pair)."""

    def __init__(self, source_tokens, target_ids, image_vec, label):
        self.source_tokens = source_tokens
        self.target_ids = target_ids
        self.image_vec = image_vec
        self.label = label

    def __str__(self):
        s = ""
        s += "source tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.source_tokens]))
        #s += "label: %d\n" % (self.label)
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_file(writer, instance, tokenizer, max_seq_length):
    """Create files from `SampleInstance`s."""

    def _convert_ids_and_mask(input_tokens):
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        return input_ids, input_mask

    source_ids, source_mask = _convert_ids_and_mask(instance.source_tokens)
    
    target_ids = instance.target_ids
    while len(target_ids) < max_seq_length:
        target_ids.append(0)
    token_type_ids = [0] * len(source_ids)
    image_vec = instance.image_vec
    label = instance.label
    
    features = {
        "source_ids"     : np.asarray(source_ids,dtype=np.int32),
        "source_mask"    : np.asarray(source_mask,dtype=np.int32),
        "token_type_ids" : np.asarray(token_type_ids, dtype=np.int32),        
        "target_ids"     : np.asarray(target_ids, dtype=np.int32),
        "image_vec"      : np.asarray(image_vec, dtype=np.int32),
        "label"          : np.asarray(label, dtype=np.int32)
    }
    #writer.write_raw_data([features])
    return features

def create_training_instance(source_words, target_ids, image_vec, label, max_seq_length, class_num):
    """Creates `SampleInstance`s for a single sentence pair."""
    SOS = "[CLS]"
    EOS = "[SEP]"

    source_tokens = [SOS] + source_words + [EOS]
    
    while(len(label) < class_num):
        label.append(0)

    instance = SampleInstance(
        source_tokens=source_tokens,
        target_ids = target_ids,
        image_vec=image_vec,
        label=label)
    return instance