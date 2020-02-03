import copy
import tensorflow as tf
from tensorflow.core.framework import graph_pb2

INPUT_GRAPH_DEF_FILE = '../models/test_pb/with_resize3.pb'
OUTPUT_GRAPH_DEF_FILE = '../models/test_pb/wout_phase.pb'

c = tf.constant(False, dtype=bool, shape=[], name='phase_train')


# load our graph
def load_graph(filename):
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
    return graph_def


graph_def = load_graph(INPUT_GRAPH_DEF_FILE)

# Create new graph, and rebuild it from original one
# replacing phase train node def with constant
new_graph_def = graph_pb2.GraphDef()
for node in graph_def.node:
    if node.name == 'phase_train':
        new_graph_def.node.extend([c.op.node_def])
    else:
        new_graph_def.node.extend([copy.deepcopy(node)])

# save new graph
with tf.gfile.GFile(OUTPUT_GRAPH_DEF_FILE, "wb") as f:
    f.write(new_graph_def.SerializeToString())
