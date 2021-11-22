import simple_elmo
import tensorflow as tf
from simple_elmo import ElmoModel


class Elmo(ElmoModel):
    def __init__(self):
        super().__init__()
        tf.compat.v1.reset_default_graph()
        super().load("./elmo/", max_batch_size=40)
        '''
        graph = tf.Graph()
        with graph.as_default() as elmo_graph:
            super(Elmo, self).__init__()
            super(Elmo, self).load("./elmo/", max_batch_size=40)
        with elmo_graph.as_default() as current_graph:
            tf_session = tf.compat.v1.Session(graph=elmo_graph)
            with tf_session.as_default() as self.sess:
                super(Elmo, self).elmo_sentence_input = \
                    simple_elmo.elmo.weight_layers("input", super(Elmo, self).sentence_embeddings_op)
                self.sess.run(tf.compat.v1.global_variables_initializer())
        '''
