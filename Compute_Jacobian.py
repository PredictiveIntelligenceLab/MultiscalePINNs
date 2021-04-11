# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 17:45:07 2020

@author: sifan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gradients_impl as gradient_ops
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.util import nest

def jacobian(output, inputs, use_pfor=True, parallel_iterations=None):
  """Computes jacobian of `output` w.r.t. `inputs`.
  Args:
    output: A tensor.
    inputs: A tensor or a nested structure of tensor objects.
    use_pfor: If true, uses pfor for computing the jacobian. Else uses
      tf.while_loop.
    parallel_iterations: A knob to control how many iterations and dispatched in
      parallel. This knob can be used to control the total memory usage.
  Returns:
    A tensor or a nested structure of tensors with the same structure as
    `inputs`. Each entry is the jacobian of `output` w.r.t. to the corresponding
    value in `inputs`. If output has shape [y_1, ..., y_n] and inputs_i has
    shape [x_1, ..., x_m], the corresponding jacobian has shape
    [y_1, ..., y_n, x_1, ..., x_m]. Note that in cases where the gradient is
    sparse (IndexedSlices), jacobian function currently makes it dense and
    returns a Tensor instead. This may change in the future.
  """
  flat_inputs = nest.flatten(inputs)
  output_tensor_shape = output.shape
  output_shape = array_ops.shape(output)
  output = array_ops.reshape(output, [-1])

  def loop_fn(i):
    y = array_ops.gather(output, i)
    return gradient_ops.gradients(y, flat_inputs,  unconnected_gradients=tf.UnconnectedGradients.ZERO)

  try:
    output_size = int(output.shape[0])
  except TypeError:
    output_size = array_ops.shape(output)[0]

  if use_pfor:
    pfor_outputs = control_flow_ops.pfor(
        loop_fn, output_size, parallel_iterations=parallel_iterations)
  else:
    pfor_outputs = control_flow_ops.for_loop(
        loop_fn,
        [output.dtype] * len(flat_inputs),
        output_size,
        parallel_iterations=parallel_iterations)

  for i, out in enumerate(pfor_outputs):
    if isinstance(out, ops.Tensor):
      new_shape = array_ops.concat(
          [output_shape, array_ops.shape(out)[1:]], axis=0)
      out = array_ops.reshape(out, new_shape)
      out.set_shape(output_tensor_shape.concatenate(flat_inputs[i].shape))
      pfor_outputs[i] = out

  return nest.pack_sequence_as(inputs, pfor_outputs)