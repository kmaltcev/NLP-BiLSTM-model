import tensorflow as tf
from simple_elmo import ElmoModel


class Elmo(ElmoModel):
    def __init__(self):
        super().__init__()
        tf.compat.v1.reset_default_graph()
        super().load("./elmo/", max_batch_size=40)
