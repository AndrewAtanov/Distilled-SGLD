import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('path', '',
                           """ ... """)


reader = tf.train.NewCheckpointReader(FLAGS.path)
var_names_in_checkpoint = reader.get_variable_to_shape_map().keys()
print '\n'.join(var_names_in_checkpoint)
