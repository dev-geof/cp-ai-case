import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings

warnings.filterwarnings("ignore")

from argparse import ArgumentParser
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
from termcolor import colored
import logging

from configuration import configure_logging, load_config, create_directory
from model import (
    get_best_checkpoint,
    CustomEmbedding,
)
from plotting import (
    plot_shap_values,
    plot_confusion_matrix,
    plot_class_metrics,
    plot_model_output_probabilities,
    plot_model_log_probability_ratios_and_roc,
)
from data import (
    load_training_validation_datasets,
    load_training_validation_datasets_NLP,
)


def inference(train_dataset, val_dataset, outputdir, batch_size):
    """
    Performs inference using the best-trained model loaded from the specified output directory.

    Args:
        train_dataset (tf.data.Dataset): The dataset to use for generating predictions on the training data.
        val_dataset (tf.data.Dataset): The dataset to use for generating predictions on the validation data.
        outputdir (str): The path to the output directory where the model checkpoints are stored.
        batch_size (int): The batch size to use for generating predictions.

    Returns:
        tuple: A tuple containing:
            - train_predictions (numpy.ndarray): The predictions for the training dataset.
            - val_predictions (numpy.ndarray): The predictions for the validation dataset.
    """

    # Load best model from checkpoint
    with tf.keras.utils.custom_object_scope({"CustomEmbedding": CustomEmbedding}):
        print(f"{outputdir}/training/checkpoints/")
        model = tf.keras.models.load_model(
            get_best_checkpoint(f"{outputdir}/training/checkpoints")
        )

    # Run predictions
    train_predictions = model.predict(train_dataset, batch_size=batch_size)
    val_predictions = model.predict(val_dataset, batch_size=batch_size)

    return train_predictions, val_predictions, model


def performance_assessement(
    train_predictions,
    val_predictions,
    y_train,
    y_val,
    confusion_matrix,
    class_metrics,
    output_prob,
    llr_roc,
    class_names,
    outputdir,
):
    """
    Evaluates and visualizes the performance of a trained model using various metrics and plots.

    Args:
        train_predictions (np.ndarray): Model predictions for the training dataset.
        val_predictions (np.ndarray): Model predictions for the validation dataset.
        y_train (np.ndarray): True labels for the training dataset.
        y_val (np.ndarray): True labels for the validation dataset.
        confusion_matrix (bool): Whether to plot confusion matrices for training and validation sets.
        class_metrics (bool): Whether to compute and plot class-specific metrics (e.g., precision, recall, F1-score).
        output_prob (bool): Whether to plot the distribution of output probabilities for training and validation sets.
        llr_roc (bool): Whether to plot log-likelihood ratios (LLR) and receiver operating characteristic (ROC) curves.
        class_names (list): List of class names for labeling plots and metrics.
        outputdir (str): Directory where the generated plots and metrics should be saved.

    Returns:
        None: The function saves performance plots (confusion matrices, metrics, probability distributions, etc.) to the specified output directory.

    Notes:
        - Confusion matrices visualize true vs. predicted class distributions.
        - Class-specific metrics include precision, recall, and F1-score for each class.
        - Output probability plots analyze the model's confidence distribution for predictions.
        - LLR and ROC plots provide insight into the model's ability to distinguish between classes.
    """

    if confusion_matrix == True:

        # Predictions for train, validation, and test sets
        logging.info(
            colored("Evaluating and plotting confusion matrices", "light_magenta")
        )
        # Plot confusion matrices
        plot_confusion_matrix(
            y_train,
            train_predictions,
            "Training",
            class_names,
            f"{outputdir}/validation",
        )
        plot_confusion_matrix(
            y_val, val_predictions, "Validation", class_names, f"{outputdir}/validation"
        )

    if class_metrics == True:
        # Class metrics for train, validation, and test sets
        logging.info(colored("Evaluating and plotting class metrics", "light_magenta"))
        plot_class_metrics(
            y_train,
            train_predictions,
            "Training",
            class_names,
            f"{outputdir}/validation",
        )
        plot_class_metrics(
            y_val, val_predictions, "Validation", class_names, f"{outputdir}/validation"
        )
    if output_prob == True:
        # Output probabilities distribution
        logging.info(
            colored("Plotting output probability distributions", "light_magenta")
        )
        plot_model_output_probabilities(
            train_predictions,
            y_train,
            "Training",
            class_names,
            f"{outputdir}/validation",
        )
        plot_model_output_probabilities(
            val_predictions,
            y_val,
            "Validation",
            class_names,
            f"{outputdir}/validation",
        )
    if llr_roc == True:
        # Output probabilities distribution
        logging.info(
            colored("Plotting log probability ratios and roc curves", "light_magenta")
        )
        plot_model_log_probability_ratios_and_roc(
            train_predictions,
            y_train,
            "Training",
            class_names,
            f"{outputdir}/validation",
        )
        plot_model_log_probability_ratios_and_roc(
            val_predictions,
            y_val,
            "Validation",
            class_names,
            f"{outputdir}/validation",
        )


def validation(
    configfile: str,
):
    """
    Validate ML model for classification based on the configuration.

    Parameters:""
    - configfile (str): Path to the YAML configuration file.

    Returns:
    None
    """

    # configure logging module
    configure_logging()
    logging.info(colored("VALIDATION", "green"))

    # Loading configfile
    prime_service = load_config(configfile)

    outputdir = prime_service["general_configuration"]["output_directory"]
    create_directory(f"{outputdir}/validation")

    analysis_title = prime_service["general_configuration"]["analysis_title"]
    class_names = prime_service["validation_configuration"]["output_classes"]
    batch_size = prime_service["training_configuration"]["batch_size"]

    confusion_matrix = prime_service["validation_configuration"][
        "plot_confusion_matrix"
    ]
    class_metrics = prime_service["validation_configuration"]["plot_class_metrics"]
    output_prob = prime_service["validation_configuration"]["plot_output_probabilities"]
    llr_roc = prime_service["validation_configuration"]["plot_output_probabilities"]

    if prime_service["general_configuration"]["analysis_type"] == "NLP":
        logging.info(colored("NLP analysis", "green"))

        # Loading training and validation datasets
        logging.info(
            colored("Loading training and validation datasets", "light_magenta")
        )
        (
            _,
            _,
            X_train_title,
            X_train_body,
            X_train_tags,
            X_train_num,
            X_val_title,
            X_val_body,
            X_val_tags,
            X_val_num,
            y_train,
            y_val,
            n_numeric_features,
            nclass,
        ) = load_training_validation_datasets_NLP(configfile)

        # Loading best model and run inference on training and validation datasets
        logging.info(colored("Loading best model and run inference", "light_magenta"))
        train_predictions, val_predictions, _ = inference(
            train_dataset=[X_train_title, X_train_body, X_train_tags, X_train_num],
            val_dataset=[X_val_title, X_val_body, X_val_tags, X_val_num],
            outputdir=outputdir,
            batch_size=batch_size,
        )

        performance_assessement(
            train_predictions=train_predictions,
            val_predictions=val_predictions,
            y_train=y_train,
            y_val=y_val,
            confusion_matrix=confusion_matrix,
            class_metrics=class_metrics,
            output_prob=output_prob,
            llr_roc=llr_roc,
            class_names=class_names,
            outputdir=outputdir,
        )

    else:

        logging.info(colored("DNN validation", "green"))

        # Loading training and validation datasets
        logging.info(
            colored("Loading training and validation datasets", "light_magenta")
        )
        _, _, X_train, X_val, y_train, y_val, nclass, nfeatures = (
            load_training_validation_datasets(configfile)
        )

        # Loading best model and run inference on training and validation datasets
        logging.info(colored("Loading best model and run inference", "light_magenta"))
        train_predictions, val_predictions, model = inference(
            train_dataset=X_train,
            val_dataset=X_val,
            outputdir=outputdir,
            batch_size=batch_size,
        )

        performance_assessement(
            train_predictions=train_predictions,
            val_predictions=val_predictions,
            y_train=y_train,
            y_val=y_val,
            confusion_matrix=confusion_matrix,
            class_metrics=class_metrics,
            output_prob=output_prob,
            llr_roc=llr_roc,
            class_names=class_names,
            outputdir=outputdir,
        )


def main():
    """
    Entry point for running ML model validation.

    Parses command-line arguments and invokes the transformer_training function.
    """
    parser = ArgumentParser(description="run ML model validation")
    parser.add_argument(
        "--configfile",
        action="store",
        dest="configfile",
        default="config/config.yaml",
        help="Configuration file",
    )
    args = vars(parser.parse_args())
    validation(**args)


if __name__ == "__main__":
    main()
