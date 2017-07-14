# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 08:40:12 2017

@author: oolabiyi
"""
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.ops.distributions import normal
from tensorflow.python.util import nest

_transpose_batch_time = decoder._transpose_batch_time
def _unstack_ta(inp):
    return tensor_array_ops.TensorArray(
        dtype=inp.dtype, size=array_ops.shape(inp)[0],
        element_shape=inp.get_shape()[1:]).unstack(inp)

def initialize_fn(self):
    finished = math_ops.equal(0, self._sequence_length)
    all_finished = math_ops.reduce_all(finished)
    next_inputs = control_flow_ops.cond(
        all_finished, lambda: self._zero_inputs,
        lambda: nest.map_structure(lambda inp: inp.read(0), self._input_tas))
    return (finished, next_inputs)
    
def sample_fn(self, time, outputs, state):
    sampler = bernoulli.Bernoulli(probs=self._sampling_probability)
    return math_ops.cast(
        sampler.sample(sample_shape=self.batch_size, seed=self._seed),
        dtypes.bool) 
        
def next_inputs_fn(self, time, outputs, state, sample_ids):
    def next_inputs(self, time, outputs, state):
        next_time = time + 1
        finished = (next_time >= self._sequence_length)
        all_finished = math_ops.reduce_all(finished)
        def read_from_ta(inp):
            return inp.read(next_time)
        next_inputs = control_flow_ops.cond(
            all_finished, lambda: self._zero_inputs,
            lambda: nest.map_structure(read_from_ta, self._input_tas))
        return (finished, next_inputs, state)
    (finished, base_next_inputs, state) = next_inputs(self, time, outputs, state)
    def maybe_sample():    
        where_sampling = math_ops.cast(
            array_ops.where(sample_ids), dtypes.int32)
        where_not_sampling = math_ops.cast(
            array_ops.where(math_ops.logical_not(sample_ids)), dtypes.int32)
        outputs_sampling = array_ops.gather_nd(outputs, where_sampling)
        inputs_not_sampling = array_ops.gather_nd(base_next_inputs,
                                                  where_not_sampling)
        base_shape = array_ops.shape(base_next_inputs)
        z_mean, z_logstd = tf.split(outputs_sampling, 2, 1)
        sampled_next_inputs = normal.Normal(z_mean, tf.exp(z_logstd))
        return (array_ops.scatter_nd(indices=where_sampling,
                                         updates=sampled_next_inputs,
                                         shape=base_shape)
                    + array_ops.scatter_nd(indices=where_not_sampling,
                                           updates=inputs_not_sampling,
                                           shape=base_shape))
    all_finished = math_ops.reduce_all(finished)
    no_samples = math_ops.logical_not(math_ops.reduce_any(sample_ids))
    next_inputs = control_flow_ops.cond(
        math_ops.logical_or(all_finished, no_samples),
        lambda: base_next_inputs, maybe_sample)
    return (finished, next_inputs, state)
        
    