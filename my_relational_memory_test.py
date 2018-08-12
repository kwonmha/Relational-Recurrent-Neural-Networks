
import numpy as np
import relational_memory
import tensorflow as tf

mem_slots = 4
head_size = 32
num_heads = 2
batch_size = 5

input_shape = (batch_size, 3, 3)
mem = relational_memory.RelationalMemory(mem_slots, head_size, num_heads)
inputs = tf.placeholder(tf.float32, input_shape)
init_state = mem.initial_state(batch_size)
out = mem(inputs, init_state, treat_input_as_matrix=True)

with tf.Session() as session:
    tf.global_variables_initializer().run()
    new_out, new_memory = session.run(
        out, feed_dict={inputs: np.zeros(input_shape)}
    )

print(init_state.get_shape().as_list(), new_memory.shape)  # [5, 4, 64] (5, 4, 64)
print(new_out.shape, [batch_size, mem_slots * head_size * num_heads])  #(5, 256) (5, 256)
