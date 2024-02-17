import tensorflow as tf

def reset_tpu(training_config):
  if training_config["TPU"]:
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu = 'node-1', zone = 'europe-west4-a', project = 'master2021-313119')
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    print("Running on TPU ", cluster_resolver.master())
    print("REPLICAS: ", strategy.num_replicas_in_sync)
    return strategy
  else:
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    print("Running on GPU.")
    return tf.distribute.MirroredStrategy()