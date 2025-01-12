import re
import sys
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from termcolor import colored
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import time
import spacy

from configuration import load_config

# Loading spaCy model globally
nlp = spacy.load("en_core_web_sm")


def load_dataset(path):
    """
    Load a dataset from a CSV file.

    Parameters
    ----------
    path : str
        The file path to the CSV file containing the dataset.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the loaded dataset.
    """
    # return pd.read_csv(path).head(100)
    return pd.read_csv(path)


def dataset_overview(df):
    """
    Provide an overview of the dataset.

    Displays basic dataset information including:
    - Data types and non-null counts
    - The first 5 rows of the dataset
    - Basic statistical summaries (mean, standard deviation, min, max, etc.)

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze.

    Returns
    -------
    None
    """
    logging.info(colored("Dataset Info", "yellow"))
    print(df.info())
    logging.info(colored("First 5 Rows", "yellow"))
    print(df.head())
    logging.info(colored("Basic Statistics", "yellow"))
    print(df.describe())


def missing_data_summary(df):
    """
    Summarize missing data in the dataset.

    Generates a summary of missing values including:
    - The total count of missing values for each column
    - The percentage of missing data for each column

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze.

    Returns
    -------
    None
    """
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame(
        {"Missing Values": missing_data, "Percentage": missing_percentage}
    ).sort_values(by="Missing Values", ascending=False)
    logging.info(colored("Missing Data:", "yellow"))
    print(missing_df)


def label_encoding(df, obj_features):
    """
    Apply label encoding to categorical features.

    Converts categorical columns into numeric representations using label encoding.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to process.
    obj_features : list of str
        List of column names containing categorical features to encode.

    Returns
    -------
    None
    """
    le = LabelEncoder()
    for f in obj_features:
        df[f] = le.fit_transform(df[f])


def standardization(df, column):
    """
    Standardizes a specified column in a DataFrame using z-score normalization.

    Args:
        df (pd.DataFrame): The input DataFrame containing the column to standardize.
        column (str): The name of the column in the DataFrame to be standardized.

    Returns:
        pd.DataFrame: The DataFrame with the specified column standardized in place.

    Notes:
        - Standardization scales the column values to have a mean of 0 and a standard deviation of 1.
        - Uses `StandardScaler` from `sklearn.preprocessing` for standardization.
    """
    scaler = StandardScaler()
    df[column] = scaler.fit_transform(df[column])


def ordinal_categorical_encoding(df, column, order):
    """
    Encodes an ordinal categorical column in a DataFrame using a specified order of categories.

    Args:
        df (pd.DataFrame): The input DataFrame containing the column to encode.
        column (str): The name of the column in the DataFrame to be encoded.
        order (list): A list specifying the order of the categories for the ordinal encoding.

    Returns:
        pd.DataFrame: The DataFrame with the specified column encoded as integers based on the provided order.

    Notes:
        - The function first converts the column to an ordered categorical type using the specified `order`.
        - The categories are then encoded into integers, where the smallest integer corresponds to the first category in `order`.
        - If a value in the column is not in the provided `order`, it will result in a missing value (`-1`).
    """
    # Ordered categorical type
    df[column] = pd.Categorical(df[column], categories=order, ordered=True)

    # Encode categories into integers
    df[column] = df[column].cat.codes


def fill_missing_data(df, mis_data_features):
    """
    Fill missing values with the mean for specified features.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to process.
    mis_data_features : list of str
        List of column names where missing values should be filled.

    Returns
    -------
    None
    """
    for f in mis_data_features:
        df[f] = df[f].fillna(df[f].mean())


def load_training_validation_datasets(configfile):
    """
    Loads, preprocesses, and splits a dataset into training, validation, and test sets for machine learning tasks.

    Args:
        configfile (str): Path to the configuration file containing dataset paths, training parameters, and preprocessing options.

    Returns:
        tuple: A tuple containing:
            - train_tf_dataset (tf.data.Dataset): Training dataset in TensorFlow Dataset format.
            - val_tf_dataset (tf.data.Dataset): Validation dataset in TensorFlow Dataset format.
            - test_tf_dataset (tf.data.Dataset): Test dataset in TensorFlow Dataset format.
            - X_train (np.ndarray): Training input features.
            - X_val (np.ndarray): Validation input features.
            - X_test (np.ndarray): Test input features.
            - y_train (np.ndarray): Training target values (one-hot encoded if `resampling` is applied).
            - y_val (np.ndarray): Validation target values (one-hot encoded if `resampling` is applied).
            - y_test (np.ndarray): Test target values (one-hot encoded if `resampling` is applied).
            - nclass (int): Number of unique target classes.
            - nfeatures (int): Number of input features.

    Notes:
        - The function reads training configuration from the provided `configfile`.
        - Applies SMOTE (Synthetic Minority Oversampling Technique) for dataset resampling if enabled in the configuration.
        - Splits the data into training (70%), validation (15%), and test (15%) sets using stratified sampling.
        - Converts the splits into TensorFlow Datasets with shuffling and batching applied.
        - Ensure the configuration file contains required fields, including dataset path, target feature name, batch size, and resampling flag.
    """

    # Loading configfile
    prime_service = load_config(configfile)

    dataset_path = prime_service["training_configuration"]["train_dataset"]
    target_feature = prime_service["training_configuration"]["target_feature"]
    batch_size = prime_service["training_configuration"]["batch_size"]
    resampling = prime_service["training_configuration"]["resampling"]

    # Loading config file and dataset
    logging.info(colored("Loading curated dataset", "light_magenta"))
    df = load_dataset(dataset_path)

    # Splitting dataset into input features and target to prepare for training and test
    logging.info(
        colored("Splitting dataset into input features and target", "light_magenta")
    )
    X = df.drop(columns=target_feature)
    y = df[target_feature]
    nclass = len(y.unique())

    # Apply SMOTE for resampling datasets
    if resampling == True:
        logging.info(colored("Apply SMOTE for resampling datasets", "light_magenta"))
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        unique_classes, counts = np.unique(y_resampled, return_counts=True)
        class_fractions = counts / len(y_resampled)
        # Print results
        for cls, count, fraction in zip(unique_classes, counts, class_fractions):
            print(f"Class {cls}: Count = {count}, Fraction = {fraction:.4f}")

        # one-hot encoded format
        y = tf.keras.utils.to_categorical(y_resampled, num_classes=nclass)
        X = X_resampled

    # Split into 75% train, 25% validation with stratified sampling
    logging.info(
        colored(
            "Splitting datasets into training, validation and test", "light_magenta"
        )
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # Convert training, validation and test datasets to tf.data.Dataset
    logging.info(
        colored(
            "Convert training, validation and test datasets to tf.data.Dataset",
            "light_magenta",
        )
    )
    datasets = [X_train, X_val, y_train, y_val]
    datasets = [ds.astype(np.float32) for ds in datasets]
    X_train, X_val, y_train, y_val = datasets
    nfeatures = X_train.shape[1]

    train_tf_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_tf_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    # Shuffle and batch the training data
    train_tf_dataset = (
        train_tf_dataset.shuffle(buffer_size=1000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_tf_dataset = val_tf_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return (
        train_tf_dataset,
        val_tf_dataset,
        X_train,
        X_val,
        y_train,
        y_val,
        nclass,
        nfeatures,
    )


def load_training_validation_datasets_NLP(configfile):
    """
    Loads, preprocesses, and splits datasets for Natural Language Processing (NLP) tasks into training and validation sets.

    Args:
        configfile (str): Path to the configuration file containing dataset paths, training parameters, and feature specifications.

    Returns:
        tuple: A tuple containing:
            - train_tf_dataset (tf.data.Dataset): Training dataset in TensorFlow Dataset format.
            - val_tf_dataset (tf.data.Dataset): Validation dataset in TensorFlow Dataset format.
            - X_train_title (tf.Tensor): Training titles as string tensors.
            - X_train_body (tf.Tensor): Training bodies as string tensors.
            - X_train_tags (tf.Tensor): Training tags as string tensors.
            - X_train_num (tf.Tensor): Training numerical features as float tensors.
            - X_val_title (tf.Tensor): Validation titles as string tensors.
            - X_val_body (tf.Tensor): Validation bodies as string tensors.
            - X_val_tags (tf.Tensor): Validation tags as string tensors.
            - X_val_num (tf.Tensor): Validation numerical features as float tensors.
            - y_train (np.ndarray): Training target values (one-hot encoded).
            - y_val (np.ndarray): Validation target values (one-hot encoded).
            - n_numeric_features (int): Number of numerical features.
            - nclass (int): Number of unique target classes.

    Notes:
        - The function reads training configuration from the provided `configfile`.
        - Splits training and validation datasets into input features (`title`, `body`, `tags`, and numerical) and target values.
        - Converts string and numerical features to TensorFlow tensors for model input.
        - One-hot encodes the target feature (`y_train` and `y_val`) for multi-class classification.
        - Applies shuffling and batching to the training dataset, with prefetching for optimization.
        - Ensure the configuration file includes paths to training and validation datasets, feature names, and batch size.
    """

    # Loading configfile
    prime_service = load_config(configfile)

    dataset_train_path = prime_service["training_configuration"]["train_dataset"]
    dataset_val_path = prime_service["training_configuration"]["val_dataset"]
    target_feature = prime_service["training_configuration"]["target_feature"]
    title_feature = prime_service["training_configuration"]["title_feature"]
    body_feature = prime_service["training_configuration"]["body_feature"]
    tags_feature = prime_service["training_configuration"]["tags_feature"]
    num_features = prime_service["training_configuration"]["num_features"]
    batch_size = prime_service["training_configuration"]["batch_size"]

    # Loading config file and dataset
    logging.info(
        colored("Loading curated training and validation dataset", "light_magenta")
    )
    df_train = load_dataset(dataset_train_path)
    df_val = load_dataset(dataset_val_path)

    # Splitting dataset into input features and target to prepare for training and test
    logging.info(
        colored(
            "Splitting dataset into input features and target for training and validation",
            "light_magenta",
        )
    )
    X_train = df_train.drop(columns=target_feature)
    nclass = len(df_train[target_feature].unique())
    y_train = tf.keras.utils.to_categorical(
        df_train[target_feature], num_classes=nclass
    )
    X_val = df_val.drop(columns=target_feature)
    y_val = tf.keras.utils.to_categorical(df_val[target_feature], num_classes=nclass)

    X_train_title = df_train[title_feature]
    X_train_body = df_train[body_feature]
    X_train_tags = df_train[tags_feature]
    X_train_num = df_train[num_features]

    X_val_title = df_val[title_feature]
    X_val_body = df_val[body_feature]
    X_val_tags = df_val[tags_feature]
    X_val_num = df_val[num_features]

    # convertion to proper tf dtypes
    X_train_title = X_train_title.astype(str)
    X_train_body = X_train_body.astype(str)
    X_train_tags = X_train_tags.astype(str)
    X_train_num = X_train_num.astype(float)
    y_train = y_train.astype(int)

    X_val_title = X_val_title.astype(str)
    X_val_body = X_val_body.astype(str)
    X_val_tags = X_val_tags.astype(str)
    X_val_num = X_val_num.astype(float)
    y_val = y_val.astype(int)

    X_train_title = tf.convert_to_tensor(X_train_title, dtype=tf.string)
    X_train_body = tf.convert_to_tensor(X_train_body, dtype=tf.string)
    X_train_tags = tf.convert_to_tensor(X_train_tags, dtype=tf.string)
    X_train_num = tf.convert_to_tensor(X_train_num, dtype=np.float32)

    X_val_title = tf.convert_to_tensor(X_val_title, dtype=tf.string)
    X_val_body = tf.convert_to_tensor(X_val_body, dtype=tf.string)
    X_val_tags = tf.convert_to_tensor(X_val_tags, dtype=tf.string)
    X_val_num = tf.convert_to_tensor(X_val_num, dtype=np.float32)

    train_tf_dataset = tf.data.Dataset.from_tensor_slices(
        ((X_train_title, X_train_body, X_train_tags, X_train_num), y_train)
    )
    val_tf_dataset = tf.data.Dataset.from_tensor_slices(
        ((X_val_title, X_val_body, X_val_tags, X_val_num), y_val)
    )

    # Shuffle and batch the training data
    train_tf_dataset = (
        train_tf_dataset.shuffle(buffer_size=1000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_tf_dataset = val_tf_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    n_numeric_features = X_train_num.shape[1]

    return (
        train_tf_dataset,
        val_tf_dataset,
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
    )
