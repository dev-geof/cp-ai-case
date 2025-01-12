import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings

warnings.filterwarnings("ignore")
import datetime
from argparse import ArgumentParser
import pandas as pd
import tensorflow as tf
from termcolor import colored
import logging

from configuration import configure_logging, load_config, create_directory
from model import build_DNN, build_NLP, get_latest_checkpoint
from callbacks import TrainingPlot
from data import (
    load_training_validation_datasets,
    load_training_validation_datasets_NLP,
)


def compile_and_fit(
    model,
    train_tf_dataset,
    val_tf_dataset,
    optimizer,
    learning_rate,
    buffer_size,
    nepochs,
    outputdir,
    verbose,
):
    """
    Compiles and trains a TensorFlow model using specified datasets, hyperparameters, and callbacks.

    Args:
        model (tf.keras.Model): The TensorFlow model to be compiled and trained.
        train_tf_dataset (tf.data.Dataset): The training dataset in TensorFlow Dataset format.
        val_tf_dataset (tf.data.Dataset): The validation dataset in TensorFlow Dataset format.
        learning_rate (float): The learning rate for the AdamW optimizer.
        buffer_size (int): The patience parameter for early stopping and ReduceLROnPlateau callbacks.
        nepochs (int): The number of training epochs.
        outputdir (str): The directory path where checkpoints and training logs will be saved.
        verbose (int): Verbosity mode for model training (e.g., 0, 1, or 2).

    Returns:
        tf.keras.callbacks.History: The history object generated during training, containing details about the training process.

    Notes:
        - The model is compiled with the Adam or AdamW optimizer, Categorical Crossentropy loss, and metrics including accuracy and AUC.
        - If existing checkpoints are found in the output directory, training resumes from the latest checkpoint.
        - Checkpoints, logs, and plots are saved in the `outputdir` directory.
        - Several callbacks are used during training, including:
            - EarlyStopping: Stops training when validation performance stagnates.
            - TensorBoard: Logs training metrics for visualization in TensorBoard.
            - ReduceLROnPlateau: Reduces the learning rate when validation performance stops improving.
            - ModelCheckpoint: Saves the model at each epoch with specified naming conventions.
            - TrainingPlot: Custom callback for plotting training progress (requires implementation of `TrainingPlot`).
        - Ensure the output directory is writable and the datasets are properly preprocessed for model input.
    """
    # Compiling model
    logging.info(colored(f"Compiling model", "light_magenta"))
    if optimizer == "AdamW":
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=learning_rate, weight_decay=1e-4
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate, weight_decay=1e-4
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )

    # Check if there are existing checkpoints
    logging.info(colored(f"Checking existing checkpoints", "light_magenta"))
    checkpoint_dir = f"{outputdir}/training/checkpoints/"

    if os.path.exists(checkpoint_dir):
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)

        if latest_checkpoint:
            logging.info(
                colored(
                    f"Resuming training from checkpoint: {latest_checkpoint}",
                    "yellow",
                )
            )
            model = tf.keras.models.load_model(latest_checkpoint)
    else:
        logging.info(
            colored(
                f"No existing checkpoints found. Starting training from scratch.",
                "yellow",
            )
        )

    # fit the model, input and target to be provided
    log_dir = (
        f"{outputdir}/training/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')} "
    )

    plot_losses = TrainingPlot(f"{outputdir}/training")

    LRMonitor = "val_accuracy"

    class_CPname = (
        outputdir
        + "/training/checkpoints/model.{epoch:02d}-loss-{loss:.5f}-{val_loss:.5f}-acc-{accuracy:.5f}-{val_accuracy:.5f}-auc-{auc:.5f}-{val_auc:.5f}.keras"
    )

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=LRMonitor, factor=0.5, patience=5),
        tf.keras.callbacks.ModelCheckpoint(filepath=class_CPname),
        plot_losses,
    ]

    model_history = model.fit(
        train_tf_dataset,
        epochs=nepochs,
        callbacks=my_callbacks,
        validation_data=val_tf_dataset,
        verbose=verbose,
    )

    logging.info(colored("Training completed", "light_magenta"))


def training(configfile):
    """
    Train a DNN model for classification based on the configuration.

    Parameters:""
    - configfile (str): Path to the YAML configuration file.

    Returns:
    None
    """

    # configure logging module
    configure_logging()

    # Loading configfile
    prime_service = load_config(configfile)
    outputdir = prime_service["general_configuration"]["output_directory"]

    nepochs = prime_service["training_configuration"]["nepochs"]
    optimizer = prime_service["training_configuration"]["optimizer"]
    learning_rate = prime_service["training_configuration"]["learning_rate"]
    # loss = prime_service["training_configuration"]["loss"]
    buffer_size = prime_service["training_configuration"]["buffer_size"]
    verbose = prime_service["training_configuration"]["verbose"]
    batch_size = prime_service["training_configuration"]["batch_size"]

    nDlayers = prime_service["model_parameters"]["nDlayers"]
    vdropout = prime_service["model_parameters"]["vdropout"]
    nodes_per_layer = prime_service["model_parameters"]["nodes_per_layer"]
    act_fn = prime_service["model_parameters"]["act_fn"]
    embedding = prime_service["model_parameters"]["embedding"]
    embedding_dim = prime_service["model_parameters"]["embedding_dim"]

    if prime_service["general_configuration"]["analysis_type"] == "NLP":

        logging.info(colored("NLP TRAINING", "green"))

        # Loading training and validation datasets
        (
            train_tf_dataset,
            val_tf_dataset,
            X_train_title,
            X_train_body,
            X_train_tags,
            _,
            X_val_title,
            X_val_body,
            X_val_tags,
            _,
            _,
            _,
            n_numeric_features,
            nclass,
        ) = load_training_validation_datasets_NLP(configfile)

        # building NLP model
        logging.info(colored("Building classification NLP model", "light_magenta"))
        mha = prime_service["model_parameters"]["mha"]
        nheads = prime_service["model_parameters"]["mha_nheads"]

        model = build_NLP(
            numerical_features_dim=n_numeric_features,
            embedding_dim=embedding_dim,
            mha=mha,
            attention_heads=nheads,
            nDlayers=nDlayers,
            vdropout=vdropout,
            nodes_per_layer=nodes_per_layer,
            act_fn=act_fn,
            nclass=nclass,
            title_voc=X_val_title,
            body_voc=X_val_body,
            tags_voc=X_val_tags,
        )

        # print model summary
        model.summary()

        compile_and_fit(
            model=model,
            train_tf_dataset=train_tf_dataset,
            val_tf_dataset=val_tf_dataset,
            optimizer=optimizer,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            nepochs=nepochs,
            outputdir=outputdir,
            verbose=verbose,
        )

    else:

        logging.info(colored("DNN TRAINING", "green"))

        # Loading training and validation datasets
        logging.info(
            colored("Loading training and validation datasets", "light_magenta")
        )
        train_tf_dataset, val_tf_dataset, _, _, _, _, nclass, nfeatures = (
            load_training_validation_datasets(configfile)
        )

        # building DNN model
        logging.info(colored("Building classification DNN model", "light_magenta"))
        model = build_DNN(
            nfeatures=nfeatures,
            nDlayers=nDlayers,
            vdropout=vdropout,
            nodes_per_layer=nodes_per_layer,
            act_fn=act_fn,
            nclass=nclass,
            embedding=embedding,
            embedding_dim=embedding_dim,
        )

        # print model summary
        model.summary()

        compile_and_fit(
            model=model,
            train_tf_dataset=train_tf_dataset,
            val_tf_dataset=val_tf_dataset,
            optimizer=optimizer,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            nepochs=nepochs,
            outputdir=outputdir,
            verbose=verbose,
        )


def main():
    """
    Entry point for running ML model training.

    Parses command-line arguments and invokes the training function.
    """
    parser = ArgumentParser(description="run DNN training")
    parser.add_argument(
        "--configfile",
        action="store",
        dest="configfile",
        default="config/config.yaml",
        help="Configuration file",
    )
    args = vars(parser.parse_args())
    training(**args)


if __name__ == "__main__":
    main()
