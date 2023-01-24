import os
from tensorflow.keras import optimizers
from .utils import reset_tpu
from .get_dataset import get_dataset
import tensorflow as tf
import json
from pathlib import Path

def perform_training(models, training_config):
    if not os.path.isdir("gcs"):
        raise Exception("local GCS folder must be mounted!")

    image_size = training_config["IMAGE_SIZE"]
    experiment_path = f"models/{image_size}x{image_size}/experiments/{training_config['EXPERIMENT_DESCRIPTION']}"
    # save configuration in a file
    Path(f"{training_config['LOCAL_GCP_PATH_BASE']}/{experiment_path}").mkdir(parents=True, exist_ok=True)
    f = open(f"{training_config['LOCAL_GCP_PATH_BASE']}/{experiment_path}/training_config.json", "x")
    f.write(json.dumps(training_config))
    f.close()

    for model in models:
        model_name = model['name']
        model_factory = model['func']
        model_starting_fold = model['starting_fold']

        model_path = f"{experiment_path}/{model_name}"
        initial_model_path = f"models/{image_size}x{image_size}/initial_models/{model_name}"
        # Create initial model
        if not os.path.isdir(f"{training_config['LOCAL_GCP_PATH_BASE']}/{initial_model_path}"):
            print("Creating new weights")
            strategy = reset_tpu(training_config)
            with strategy.scope():
                optimizer = optimizers.Adam(learning_rate= training_config["LEARNING_RATE"], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                model = model_factory(training_config)

                model.compile(
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    steps_per_execution = 1,
                    optimizer=optimizer,
                    metrics=[tf.keras.metrics.BinaryAccuracy()])

                print(f"Saving initial model to {training_config['REMOTE_GCP_PATH_BASE']}/{initial_model_path}")
                model.save(f"{training_config['REMOTE_GCP_PATH_BASE']}/{initial_model_path}")

        for i in range(model_starting_fold, 11):
            tf.keras.backend.clear_session()
            strategy = reset_tpu(training_config)
            with strategy.scope():
                # with tf.device("/device:GPU:0"):
                optimizer = optimizers.Adam(learning_rate= training_config["LEARNING_RATE"], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                print(f"{model_name} FOLD {i}")

                best_epoch = 0
                best_loss = 1

                test_dataset = get_dataset(
                    training_config,
                    f"{training_config['REMOTE_GCP_PATH_BASE']}/data/{image_size}x{image_size}/fold_{i}/test",
                    training_config["TEST_BATCH_SIZE"],
                    seed = training_config["SEED"],
                    augment = False,
                    shuffle = False,
                    drop_remainder = True)

                if training_config["TPU"]:
                    test_dataset = test_dataset.cache()

                print(f"Loading model from {training_config['REMOTE_GCP_PATH_BASE']}/{initial_model_path}")
                model = tf.keras.models.load_model(f"{training_config['REMOTE_GCP_PATH_BASE']}/{initial_model_path}")

                model.compile(
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    steps_per_execution = 1,
                    optimizer=optimizer,
                    metrics=[tf.keras.metrics.BinaryAccuracy()])

                for epoch in range(training_config["NUMBER_OF_EPOCHS"]):
                    train_dataset = get_dataset(
                        training_config,
                        f"{training_config['REMOTE_GCP_PATH_BASE']}/data/{image_size}x{image_size}/fold_{i}/train",
                        training_config["TRAIN_BATCH_SIZE"],
                        seed = training_config["SEED"] + epoch,
                        augment = True,
                        shuffle = True,
                        drop_remainder = False)

                    history = model.fit(
                        x= train_dataset,
                        validation_data = test_dataset,
                        epochs = 1,
                        steps_per_epoch = 284672 // training_config["TRAIN_BATCH_SIZE"],
                        validation_steps = 31595 // training_config["TEST_BATCH_SIZE"],
                        verbose = 1,
                        shuffle = False,
                        workers= 32 if training_config["TPU"] else 1)

                    last_loss = history.history['val_loss'][-1]
                    if (last_loss < best_loss):
                        print("Loss improved. Saving model.")
                        best_epoch = epoch
                        best_loss = last_loss
                        model.save(f"{training_config['REMOTE_GCP_PATH_BASE']}/{model_path}/best_loss/fold_{i}")


                    if(epoch - training_config["EARLY_STOPPING_TOLERANCE"] == best_epoch):
                        print("Early stopping")
                        break