
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell, LSTMStateTuple

import tensorflow as tf

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


# @tf_export("nn.rnn_cell.BasicLSTMCell")
class RelationalMemoryCell(LayerRNNCell):
    """Basic LSTM recurrent network cell.

    The implementation is based on: http://arxiv.org/abs/1409.2329.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
    that follows.
    """

    def __init__(self,
                 mem_slots,
                 head_size,
                 num_heads,
                 num_blocks,
                 mlp_layers,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=None,
                 reuse=None,
                 name=None,
                 dtype=None):
        """Initialize the basic LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
          dtype: Default dtype of the layer (default of `None` means use the type
            of the first input). Required when `build` is called before `call`.

          When restoring from CudnnLSTM-trained checkpoints, must use
          `CudnnCompatibleLSTMCell` instead.
        """
        super(RelationalMemoryCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._mem_size = num_heads * head_size
        self._num_units = self._mem_size * mem_slots
        self._mem_slots = mem_slots
        self._head_size = head_size
        self._num_heads = num_heads
        self._num_blocks = num_blocks
        self._mlp_layers = mlp_layers
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def init_state(self, batch_size):
        init_state = tf.eye(self._mem_slots, batch_shape=[batch_size])

        # Pad the matrix with zeros.
        # mem_size = head_size * num_heads
        if self._mem_size > self._mem_slots:
            difference = self._mem_size - self._mem_slots
            pad = tf.zeros((batch_size, self._mem_slots, difference))
            init_state = tf.concat([init_state, pad], -1)
        # Truncation. Take the first `self._mem_size` components.
        elif self._mem_size < self._mem_slots:
            init_state = init_state[:, :, :self._mem_size]

        #Flatten
        flat_init_state = tf.layers.flatten(init_state)
        return flat_init_state, flat_init_state

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth, 2 * self._num_units])
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))  # remove output gate

        self.built = True


    def _multi_head_attention(self, memory, input):
        memory_plus_input = tf.concat([memory, input], axis=1)
        # print(memory_plus_input.get_shape())  #(b, 2* slot, mem_size)

        mi_slots = memory_plus_input.get_shape().as_list()[1]
        q_proj_mat = tf.get_variable("q_projection", [self._mem_size, self._mem_size])
        kv_proj_mat = tf.get_variable("kv_projection",
                                      [self._mem_size,  2 * self._mem_size])

        q = tf.tensordot(memory, q_proj_mat, axes=[[2], [0]])  #(b, slot, mem_size)
        kv = tf.tensordot(memory_plus_input, kv_proj_mat, axes=[[2], [0]])  #(b, 2*slot, 2 * mem_size)

        q_reshape = tf.reshape(q, [-1, self._mem_slots, self._num_heads, self._head_size])  #(b, mem_slot, heads, head_size)
        kv_reshape = tf.reshape(kv, [-1, mi_slots, self._num_heads, self._head_size * 2])  #(b, mem_slot*2, heads, head_size)

        q_transpose = tf.transpose(q_reshape, [0, 2, 1, 3]) #(b, head, slot, mem/head)
        kv_transpose = tf.transpose(kv_reshape, [0, 2, 1, 3])  #(b, head, slot * 2, mem/head)
        k, v = tf.split(kv_transpose, [self._head_size, self._head_size], -1)  #(b, head, slot * 2, head_size)

        qk = tf.nn.softmax(tf.matmul(q_transpose, k, transpose_b=True) / (self._head_size ** 0.5))  # [B, H, N, 2*N]
        qkv = tf.matmul(qk, v)  #(b, heads, slot, head_size)

        qkv = tf.transpose(qkv, [0, 2, 1, 3]) #(b, slot, heads, head_size)
        attended_memory = tf.reshape(qkv, [-1, self._mem_slots, self._mem_size])  #(b, slot, mem_size)

        return attended_memory

    def _feed_forward(self, memory):
        for _ in range(self._mlp_layers):
            memory = tf.layers.dense(memory, self._mem_size)
        return memory

    def _attend_over_memory(self, memory, inputs):
        batch_size = inputs.get_shape().as_list()[0]
        input_proj_mat = tf.get_variable("input_projection", [inputs.get_shape().as_list()[1], self._num_units])
        inputs_proj = tf.matmul(inputs, input_proj_mat)  #(n, dx) -> (n, total_mem)
        input_mat = tf.reshape(inputs_proj, [batch_size, self._mem_slots, -1])  #(n, total_mem) -> #(n, slot, mem_size)
        # print(memory.get_shape(), "memory shape")  #(b, slot, mem_size)

        for _ in range(self._num_blocks):
            attended_memory = self._multi_head_attention(memory, input_mat)
            memory = tf.contrib.layers.layer_norm(attended_memory + memory)
            memory = tf.contrib.layers.layer_norm(self._feed_forward(memory) + memory)

        return memory


    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).

        Args:
          inputs: `2-D` tensor with shape `[batch_size, input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size, num_units]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size, 2 * num_units]`.

        Returns:
          A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)
        mem, _ = state  # state consists of same objects
        # Parameters of gates are concatenated into one multiply for efficiency.
        # if self._state_is_tuple:
        #     c, h = state
        # else:
        #     c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

        # vector -> matrix
        # print(mem.get_shape())  #(b, mem_slot * mem_size)
        mem_mat = tf.reshape(mem, [-1, self._mem_slots, self._mem_size])
        # print(mem_mat.get_shape())  #(b, mem_slot, mem_size)

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, mem], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        att_mem_mat = self._attend_over_memory(mem_mat, inputs)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, f = array_ops.split(
            value=gate_inputs, num_or_size_splits=2, axis=one)

        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = math_ops.add
        multiply = math_ops.multiply
        # new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
        #             multiply(sigmoid(i), self._activation(j)))
        # new_h = multiply(self._activation(new_c), sigmoid(o))

        f_plus_bias_mat = tf.reshape(sigmoid(add(f, forget_bias_tensor)), [-1, self._mem_slots, self._mem_size])
        i_mat = tf.reshape(sigmoid(i), [-1, self._mem_slots, self._mem_size])
        new_mem_mat = multiply(f_plus_bias_mat, mem_mat)
        new_mem_mat = add(new_mem_mat, multiply(i_mat, self._activation(att_mem_mat)))

        # if self._state_is_tuple:
        #     new_state = LSTMStateTuple(new_c, new_h)
        # else:
        #     new_state = array_ops.concat([new_c, new_h], 1)

        new_h = tf.layers.flatten(new_mem_mat)  #(b, num_unit)
        new_state = LSTMStateTuple(new_h, new_h)
        return new_h, new_state


class BasicLSTMCell(LayerRNNCell):
    """Basic LSTM recurrent network cell.

    The implementation is based on: http://arxiv.org/abs/1409.2329.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
    that follows.
    """

    def __init__(self,
                 num_units,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=None,
                 reuse=None,
                 name=None,
                 dtype=None):
        """Initialize the basic LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
          dtype: Default dtype of the layer (default of `None` means use the type
            of the first input). Required when `build` is called before `call`.

          When restoring from CudnnLSTM-trained checkpoints, must use
          `CudnnCompatibleLSTMCell` instead.
        """
        super(BasicLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth, 4 * self._num_units])
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[4 * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).

        Args:
          inputs: `2-D` tensor with shape `[batch_size, input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size, num_units]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size, 2 * num_units]`.

        Returns:
          A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            # print(state) # LSTMStateTuple(c=<tensor..>, h=<tensor>)
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(
            value=gate_inputs, num_or_size_splits=4, axis=one)

        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = math_ops.add
        multiply = math_ops.multiply
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                    multiply(sigmoid(i), self._activation(j)))
        new_h = multiply(self._activation(new_c), sigmoid(o))

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state
