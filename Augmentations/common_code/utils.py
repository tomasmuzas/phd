import tensorflow as tf

def reset_tpu(training_config):
  if training_config["TPU"]:
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    print("Running on TPU ", cluster_resolver.master())
    print("REPLICAS: ", strategy.num_replicas_in_sync)
    return strategy
  else:
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    print("Running on GPU.")
    return tf.distribute.MirroredStrategy()