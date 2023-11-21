import os
from tensorflow.keras import optimizers
from .utils import reset_tpu
from .get_dataset import get_intial_fold_dataset, shuffle_dataset
import tensorflow as tf
import json
from pathlib import Path
import tensorflow_addons as tfa
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import gc

AUTO = tf.data.AUTOTUNE

def get_dataset_with_objids(path, batch_size):
  dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(path + "/*.tfrec"), num_parallel_reads=AUTO) # if TPU else 20)

  dataset = dataset.map(lambda records: tf.io.parse_single_example(
      records,
      {
          "image": tf.io.FixedLenFeature([], dtype=tf.string),
          "class": tf.io.FixedLenFeature([], dtype=tf.int64),

          "label": tf.io.FixedLenFeature([], dtype=tf.string),
          "objid": tf.io.FixedLenFeature([], dtype=tf.string),
          "one_hot_class": tf.io.VarLenFeature(tf.float32)
      }),
      num_parallel_calls=AUTO)
  dataset = dataset.map(lambda item: (tf.reshape(tf.image.decode_jpeg(item['image'], channels=3), [128, 128, 3]), item['class'], item['objid']), num_parallel_calls=AUTO)
  dataset = dataset.map(lambda x,y,z: (tf.cast(x, tf.float32), y, z), num_parallel_calls=AUTO)
  dataset = dataset.map(lambda x,y,z: (tf.keras.layers.Rescaling(scale=1./255)(x), y, z), num_parallel_calls=AUTO)
  dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(200)
  return dataset


def get_and_log_predictions(model, dataset):
  
#   table = wandb.Artifact("test_predictions" + str(wandb.run.id), type="predictions")
#   predictions_table = wandb.Table(columns=["Id", "Image", "True", "Predicted"])

  galaxy_ids = np.empty([0, 1], dtype=str)
  predictions = np.empty([0, 1], dtype=float)
  true_labels = np.empty([0, 1], dtype=float)

  for images, labels, objids in dataset:
    results = model.predict_on_batch(images)
    
    galaxy_ids = tf.concat([tf.reshape(objids, [-1, 1]), galaxy_ids], axis=0)
    true_labels = tf.concat([tf.reshape(labels, [-1, 1]), true_labels], axis=0)
    predictions = tf.concat([tf.reshape(results, [-1, 1]), predictions], axis=0)

#     for index, image in enumerate(images):
#         predictions_table.add_data(galaxy_ids[index].numpy()[0], wandb.Image(image), int(true_labels[index]), float(predictions[index])) 

#   table.add(predictions_table, "predictions")
#   wandb.run.log_artifact(table)
  return (true_labels, predictions)

def get_and_log_predictions_multiclass(model, dataset):
  
  table = wandb.Artifact("test_predictions" + str(wandb.run.id), type="predictions")
  predictions_table = wandb.Table(columns=["Id", "True", "Predicted"])

  galaxy_ids = np.empty([0, 1], dtype=str)
  predictions = np.empty([0, 1], dtype=float)
  true_labels = np.empty([0, 1], dtype=float)

  for images, labels, objids in dataset:
    results = model.predict_on_batch(images)
    
    galaxy_ids = tf.concat([tf.reshape(objids, [-1, 1]), galaxy_ids], axis=0)
    true_labels = tf.concat([tf.reshape(labels, [-1, 1]), true_labels], axis=0)
    predictions = tf.concat([tf.reshape(tf.argmax(results, axis= 1), [-1, 1]), predictions], axis=0)

    for index, image in enumerate(images):
        predictions_table.add_data(galaxy_ids[index].numpy()[0], int(true_labels[index]), float(predictions[index])) 

  table.add(predictions_table, "predictions")
  wandb.run.log_artifact(table)
  return (true_labels, predictions)

def perform_training(models, training_config):

    binary_mode = training_config["NUMBER_OF_CLASSES"] == None or training_config["NUMBER_OF_CLASSES"] == 2

    if training_config["TPU"] and not os.path.isdir("gcs"):
        raise Exception("local GCS folder must be mounted!")

    if binary_mode:
        print(f"Running in binary classification mode")
    else:
        print("Running in multiclass mode")

    image_size = training_config["IMAGE_SIZE"]
    experiment_path = f"models/{image_size}x{image_size}/experiments/{training_config['EXPERIMENT_DESCRIPTION']}"
    
    os.environ["WANDB_SILENT"] = "true"

    strategy = reset_tpu(training_config)
    with strategy.scope():
        for model in models:
            model_name = model['name']
            model_factory = model['func']
            model_starting_fold = model['starting_fold']

            model_path = f"{experiment_path}/{model_name}"
            initial_model_path = f"models/{image_size}x{image_size}/initial_models/{model_name}"
            # Create initial model
            if not os.path.isdir(f"{training_config['LOCAL_GCP_PATH_BASE']}/{initial_model_path}"):
                print("Creating new weights")
                if(training_config["USE_ADABELIEF_OPTIMIZER"]):
                    print("using AdaBelief optimizer")
                    optimizer = tfa.optimizers.AdaBelief(lr=training_config["LEARNING_RATE"])
                else:
                    print("Using Adam optimizer")
                    optimizer = optimizers.Adam(learning_rate= training_config["LEARNING_RATE"], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                model = model_factory(training_config)

                if binary_mode:
                    model.compile(
                        loss=tf.keras.losses.BinaryCrossentropy(),
                        steps_per_execution = 1,
                        optimizer=optimizer,
                        metrics=[tf.keras.metrics.BinaryAccuracy()])
                else:
                    model.compile(
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        steps_per_execution = 1,
                        optimizer=optimizer,
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

                print(f"Saving initial model to {training_config['REMOTE_GCP_PATH_BASE']}/{initial_model_path}")
                model.save(f"{training_config['REMOTE_GCP_PATH_BASE']}/{initial_model_path}")

            training_config["MODEL_NAME"] = model_name
            strategy = reset_tpu(training_config)
            
            for i in range(model_starting_fold, training_config["FOLDS"] + 1):
                wandb.init(
                    project=f"{training_config['WANDB_PROJECT_NAME']}",
                    name=f"{model_name}/Fold_{i}",
                    config=training_config)

                print("Getting cached train dataset")
                cached_initial_training_dataset = get_intial_fold_dataset(
                    training_config,
                    f"{training_config['REMOTE_GCP_PATH_BASE']}/{training_config['DATASET_PATH']}/fold_{i}/train",
                    training_config["SEED"],
                    shuffle = True)

                if(training_config["USE_ADABELIEF_OPTIMIZER"]):
                    optimizer = tfa.optimizers.AdaBelief(lr=training_config["LEARNING_RATE"])
                else:
                    optimizer = optimizers.Adam(learning_rate= training_config["LEARNING_RATE"], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                print(f"{model_name} FOLD {i}")

                best_epoch = 0
                best_loss = 1

                print("Getting test dataset")
                test_dataset = get_intial_fold_dataset(
                    training_config,
                    f"{training_config['REMOTE_GCP_PATH_BASE']}/{training_config['DATASET_PATH']}/fold_{i}/test",
                    seed = training_config["SEED"],
                    shuffle = False)

                test_dataset = shuffle_dataset(
                    test_dataset,
                    training_config,
                    training_config["TEST_BATCH_SIZE"],
                    seed = training_config["SEED"],
                    augment = False,
                    drop_remainder = training_config["TPU"])

                if training_config["TPU"]:
                    test_dataset = test_dataset.cache()

                print(f"Loading model from {training_config['REMOTE_GCP_PATH_BASE']}/{initial_model_path}")
                model = tf.keras.models.load_model(f"{training_config['REMOTE_GCP_PATH_BASE']}/{initial_model_path}")

                if binary_mode:
                    model.compile(
                        loss=tf.keras.losses.BinaryCrossentropy(),
                        steps_per_execution = training_config['STEPS_PER_EXECUTION'],
                        optimizer=optimizer,
                        metrics=[tf.keras.metrics.BinaryAccuracy()])
                else:
                    model.compile(
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        steps_per_execution = training_config['STEPS_PER_EXECUTION'],
                        optimizer=optimizer,
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

                for epoch in range(training_config["NUMBER_OF_EPOCHS"]):

                    train_dataset = shuffle_dataset(
                        cached_initial_training_dataset,
                        training_config,
                        training_config["TRAIN_BATCH_SIZE"],
                        seed = training_config["SEED"] + epoch,
                        augment = True,
                        drop_remainder = False)

                    history = model.fit(
                        x= train_dataset,
                        validation_data = test_dataset,
                        epochs = 1,
                        steps_per_epoch = training_config["TRAIN_DATASET_SIZE"] // training_config["TRAIN_BATCH_SIZE"],
                        validation_steps = training_config["TEST_DATASET_SIZE"] // training_config["TEST_BATCH_SIZE"],
                        verbose = 1,
                        shuffle = False,
                        workers= 32 if training_config["TPU"] else 1)

                    tf.keras.backend.clear_session()
                    gc.collect()
                    del train_dataset

                    if binary_mode:
                        wandb.log({"acc": history.history['val_binary_accuracy'][-1], "loss": history.history['val_loss'][-1]})
                    else:
                        wandb.log({"acc": history.history['val_sparse_categorical_accuracy'][-1], "loss": history.history['val_loss'][-1]})

                    last_loss = history.history['val_loss'][-1]
                    if (last_loss < best_loss):
                        print("Loss improved. Saving model.")
                        best_epoch = epoch
                        best_loss = last_loss
                        model.save(f"{training_config['REMOTE_GCP_PATH_BASE']}/{model_path}/best_loss/fold_{i}")

                    if(epoch - training_config["EARLY_STOPPING_TOLERANCE"] == best_epoch):
                        print("Early stopping")
                        break
                    
                # best_model = tf.keras.models.load_model(f"{training_config['REMOTE_GCP_PATH_BASE']}/{model_path}/best_loss/fold_{i}")
                # test_dataset_with_ids = get_dataset_with_objids(f"{training_config['REMOTE_GCP_PATH_BASE']}/{training_config['DATASET_PATH']}/fold_{i}/test", training_config["TEST_BATCH_SIZE"])

                # if binary_mode:
                #     true_labels, predictions = get_and_log_predictions(best_model, test_dataset_with_ids)
                #     cm = confusion_matrix(true_labels, np.where(predictions > 0.5, 1, 0))
                #     display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= ['Spiral', 'Elliptical'])
                #     plot = display.plot()
                #     wandb.log({"Confusion matrix": plt})

                #     accuracy = accuracy_score(true_labels, np.where(predictions > 0.5, 1, 0))
                #     precision = precision_score(true_labels, np.where(predictions > 0.5, 1, 0))
                #     recall = recall_score(true_labels, np.where(predictions > 0.5, 1, 0))
                #     f1 = f1_score(true_labels, np.where(predictions > 0.5, 1, 0))
                #     tnr = cm[0][0] / (cm[0][0] + cm[0][1])
                # else:
                #     true_labels, predictions = get_and_log_predictions_multiclass(best_model, test_dataset_with_ids)
                #     cm = confusion_matrix(true_labels, predictions)
                #     display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= ['CertEll', 'UncertEll', 'Unknown', 'UncertSpiral', 'CertSpiral'])
                #     plot = display.plot()
                #     wandb.log({"Confusion matrix": plt})

                #     accuracy = accuracy_score(true_labels, predictions)
                #     precision = precision_score(true_labels, predictions, average = None)
                #     recall = recall_score(true_labels, predictions, average = None)
                #     f1 = f1_score(true_labels, predictions, average = None)
                #     tnr = cm[0][0] / (cm[0][0] + cm[0][1])
                    

                # table = wandb.Table(columns = ['accuracy', 'precision', 'recall', 'f1', 'TNR'], data = [[accuracy, precision, recall, f1, tnr]])
                # wandb.log({"metrics" : table})
                # wandb.log({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'TNR': tnr})

                wandb.finish()

                del cached_initial_training_dataset
                del train_dataset
                del test_dataset

                gc.collect()
                tf.keras.backend.clear_session()