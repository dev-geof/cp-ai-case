import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from termcolor import colored
import logging
import sys
from itertools import chain

from configuration import (
    configure_logging,
    load_config,
    create_directory,
)
from data import (
    load_dataset,
    dataset_overview,
    missing_data_summary,
    fill_missing_data,
    label_encoding,
    ordinal_categorical_encoding,
    standardization,
)
from nlp import (
    clean_text_tokens,
    apply_cleaning_with_progress,
    temporal_extraction,
)
from plotting import (
    plot_numerical_distribution,
    plot_non_numerical_distribution,
    plot_correlation_matrix,
    plot_word_frequency,
    plot_sentence_length_distribution
)

def preparation(
    configfile: str,
):
    """
    Training Dataset Extraction

    Parameters
    ----------
    configfile : str
        configuration file path

    Returns
    -------

    """

    # configure logging module
    configure_logging()
    logging.info(colored("DATASET PREPARATION", "green"))

    # loading configfile
    prime_service = load_config(configfile)

    # general configuration
    outputdir = prime_service["general_configuration"]["output_directory"]
    analysis_title = prime_service["general_configuration"]["analysis_title"]
    analysis_type = prime_service["general_configuration"]["analysis_type"]
    raw_datasets = prime_service["general_configuration"]["raw_datasets"]
    curated_datasets = prime_service["general_configuration"]["curated_datasets"]

    data_mining = prime_service["preparation_configuration"]["data_mining"]
    preprocessing_regular = prime_service["preparation_configuration"]["preprocessing_regular"]["enabled"]
    preprocessing_nlp = prime_service["preparation_configuration"]["preprocessing_nlp"]["enabled"]

    # Loop over raw datasets
    for d in raw_datasets:
        logging.info(colored(f"Processing: {d}", "yellow"))
        df = load_dataset(d)

        # perform quick data mining
        if data_mining == True:
            logging.info(colored("Data Mining", "light_magenta"))
            dataset_overview(df)
            missing_data_summary(df)

        # perform regular preprocessing
        if preprocessing_regular == True:
            logging.info(colored("Regular Preprocessing", "light_magenta"))

            # (0) Identify numerical, non-numerical columns and columns with missing data
            numerical_columns = df.select_dtypes(include=[np.number]).columns
            non_numerical_columns = df.select_dtypes(include=["object"]).columns
            # Plotting raw feature distributions
            logging.info(colored("Plotting raw feature distributions", "light_magenta"))
            plot_numerical_distribution(df, numerical_columns, analysis_title, "raw", outputdir)
            plot_non_numerical_distribution(df, non_numerical_columns, analysis_title, "raw", outputdir)

            # (1) Drop useless features
            logging.info(colored("Dropping useless features", "light_magenta"))
            to_be_dropped = prime_service["preparation_configuration"]["preprocessing_regular"]["options"]["to_be_dropped"]
            logging.info(colored(to_be_dropped, "yellow"))
            df = df.drop(columns=to_be_dropped)

            # (2) Numerical features standardization
            logging.info(colored("Numerical features standardization", "light_magenta"))
            num_features = prime_service["preparation_configuration"]["preprocessing_regular"]["options"]["num_features"]
            logging.info(colored(num_features, "yellow"))
            standardization(df, num_features)
            plot_numerical_distribution(df, num_features, analysis_title, "standard", outputdir)

            # (3) # Processing ordinal feature with label encoding
            logging.info(colored("Processing ordinal features ", "light_magenta"))
            for item in prime_service["preparation_configuration"]["preprocessing_regular"]["options"]["ordinal_features"]:
                feature_name = item["feature"]
                order = item["order"]

                # Process ordering
                df[feature_name] = pd.Categorical(df[feature_name], categories=order, ordered=True)
                plot_non_numerical_distribution(df, [feature_name], analysis_title, "ordered", outputdir)
                # Process rebinning
                if "rebin" in item and item["rebin"]:
                    # Initialize mapping for rebinning
                    mapping = {}
                    for rebin_item in item["rebin"]:
                        group = rebin_item["group"]
                        label = rebin_item["label"]
                        # Map each group value to its corresponding label
                        for value in group:
                            mapping[value] = label
                    # Apply rebinning to the feature
                    df[feature_name] = df[feature_name].map(mapping)
                    plot_non_numerical_distribution(df, [feature_name], analysis_title, "rebinned", outputdir)

                # Process numerical labelling
                df[feature_name] = pd.Categorical(df[feature_name])
                df[feature_name] = df[feature_name].cat.codes
                plot_non_numerical_distribution(df, [feature_name], analysis_title, "labelled", outputdir)
                logging.info(colored(feature_name, "yellow"))

            # (4) # Processing remaining categorical features with label encoding
            logging.info(colored("Processing ordinal features ", "light_magenta"))
            for f in prime_service["preparation_configuration"]["preprocessing_regular"]["options"]["other_categorical_features"]:
                logging.info(colored(f, "yellow"))
                df[f] = pd.Categorical(df[f])
                df[f] = df[f].cat.codes
                plot_non_numerical_distribution(df, [f], analysis_title, "labelled", outputdir)

            # (5) Fill missing data with corresponding column mean value
            logging.info(colored("Fill missing data with corresponding column mean value","light_magenta"))
            missing_data = df.isnull().sum()
            missing_data_columns = pd.Index(missing_data[missing_data > 0].index)
            fill_missing_data(df, missing_data_columns)
            missing_data_summary(df)

            # (6) Plot correlation matrix
            logging.info(colored("Plotting correlation matrix for numerical fetaures","light_magenta",))
            plot_correlation_matrix(df, analysis_title, outputdir)

            # (7) Save curated dataset
            logging.info(colored("Save curated datasets", "light_magenta"))
            df.to_csv(f"{os.path.splitext(os.path.basename(d))[0]}_curated.csv", index=False)


        # perform regular preprocessing
        if preprocessing_nlp == True:
            logging.info(colored("Regular Preprocessing", "light_magenta"))

            # (1) Drop useless features
            logging.info(colored("Dropping useless features", "light_magenta"))
            to_be_dropped = prime_service["preparation_configuration"]["preprocessing_nlp"]["options"]["to_be_dropped"]
            logging.info(colored(to_be_dropped, "yellow"))
            df = df.drop(columns=to_be_dropped)

            # cleaning textual features
            logging.info(colored("Cleaning textural features", "light_magenta"))
            for f in prime_service["preparation_configuration"]["preprocessing_nlp"]["options"]["text_features"]:
                logging.info(colored(f"- {f}", "yellow"))
                plot_sentence_length_distribution(df, f, analysis_title, "before_cleaning", outputdir)
                apply_cleaning_with_progress(df, column=f, new_column=f)
                plot_word_frequency(df, f, analysis_title, outputdir)
                plot_sentence_length_distribution(df, f, analysis_title, "after_cleaning", outputdir)

            # Preprocess tags
            logging.info(colored("Cleaning tags features", "light_magenta"))
            for f in prime_service["preparation_configuration"]["preprocessing_nlp"]["options"]["tags_features"]:
                logging.info(colored(f"- {f}", "yellow"))
                df[f] = df[f].apply(lambda x: x.replace('><', ' ').replace('<', '').replace('>', '') if isinstance(x, str) else x)
                plot_sentence_length_distribution(df, f, analysis_title, "before_cleaning", outputdir)
                plot_word_frequency(df, f, analysis_title, outputdir)
                plot_sentence_length_distribution(df, f, analysis_title, "after_cleaning", outputdir)

            # Extract temporal features
            logging.info(colored("Extract temporal features", "light_magenta"))
            for f in prime_service["preparation_configuration"]["preprocessing_nlp"]["options"]["temporal_features"]:
                logging.info(colored(f"- {f}", "yellow"))
                temporal_extraction(df, f)

            # Target encoding
            logging.info(colored("Target encoding", "light_magenta"))
            for item in prime_service["preparation_configuration"]["preprocessing_nlp"]["options"]["ordinal_features"]:
                feature_name = item["feature"]
                order = item["order"]

                # Process ordering
                df[feature_name] = pd.Categorical(df[feature_name], categories=order, ordered=True)
                plot_non_numerical_distribution(df, [feature_name], analysis_title, "ordered", outputdir)
                df[feature_name] = df[feature_name].cat.codes
                plot_non_numerical_distribution(df, [feature_name], analysis_title, "labelled", outputdir)
                logging.info(colored(feature_name, "yellow"))

            # Save curated dataset
            logging.info(colored("Save curated datasets", "light_magenta"))
            df.to_csv(f"{os.path.splitext(os.path.basename(d))[0]}_curated.csv", index=False)





    """
    # perform regular preprocessing
    if preprocessing_regular == True:
        logging.info(colored("Regular Preprocessing", "light_magenta"))
        # loop over the raw datasets
        for d in raw_datasets:
            logging.info(colored(f"Processing: {d}", "yellow"))
            df = load_dataset(d)

            # (0) Identify numerical, non-numerical columns and columns with missing data
            numerical_columns = df.select_dtypes(include=[np.number]).columns
            non_numerical_columns = df.select_dtypes(include=["object"]).columns
            # Plotting raw feature distributions
            logging.info(colored("Plotting raw feature distributions", "light_magenta"))
            plot_numerical_distribution(
                df, numerical_columns, analysis_title, "raw", outputdir
            )
            plot_non_numerical_distribution(
                df, non_numerical_columns, analysis_title, "raw", outputdir
            )

            # (1) Drop useless features
            logging.info(colored("Dropping useless features", "light_magenta"))
            to_be_dropped = prime_service["preparation_configuration"][
                "preprocessing_regular"
            ]["options"]["to_be_dropped"]
            logging.info(colored(to_be_dropped, "yellow"))
            df = df.drop(columns=to_be_dropped)

            # (2) Numerical features standardization
            logging.info(colored("Numerical features standardization", "light_magenta"))
            num_features = prime_service["preparation_configuration"][
                "preprocessing_regular"
            ]["options"]["num_features"]
            logging.info(colored(num_features, "yellow"))
            standardization(df, num_features)
            plot_numerical_distribution(
                df, num_features, analysis_title, "standard", outputdir
            )

            # (3) # Processing ordinal feature with label encoding
            logging.info(colored("Processing ordinal features ", "light_magenta"))
            for item in prime_service["preparation_configuration"][
                "preprocessing_regular"
            ]["options"]["ordinal_features"]:
                feature_name = item["feature"]
                order = item["order"]

                # Process ordering
                df[feature_name] = pd.Categorical(
                    df[feature_name], categories=order, ordered=True
                )
                plot_non_numerical_distribution(
                    df, [feature_name], analysis_title, "ordered", outputdir
                )
                # Process rebinning
                if "rebin" in item and item["rebin"]:
                    # Initialize mapping for rebinning
                    mapping = {}
                    for rebin_item in item["rebin"]:
                        group = rebin_item["group"]
                        label = rebin_item["label"]
                        # Map each group value to its corresponding label
                        for value in group:
                            mapping[value] = label
                    # Apply rebinning to the feature
                    df[feature_name] = df[feature_name].map(mapping)
                    plot_non_numerical_distribution(
                        df, [feature_name], analysis_title, "rebinned", outputdir
                    )

                # Process numerical labelling
                df[feature_name] = pd.Categorical(df[feature_name])
                df[feature_name] = df[feature_name].cat.codes
                plot_non_numerical_distribution(
                    df, [feature_name], analysis_title, "labelled", outputdir
                )
                logging.info(colored(feature_name, "yellow"))

            # (4) # Processing remaining categorical features with label encoding
            logging.info(colored("Processing ordinal features ", "light_magenta"))
            for f in prime_service["preparation_configuration"][
                "preprocessing_regular"
            ]["options"]["other_categorical_features"]:
                logging.info(colored(f, "yellow"))
                df[f] = pd.Categorical(df[f])
                df[f] = df[f].cat.codes
                plot_non_numerical_distribution(
                    df, [f], analysis_title, "labelled", outputdir
                )

            # Fill missing data with corresponding column mean value
            logging.info(
                colored(
                    "Fill missing data with corresponding column mean value",
                    "light_magenta",
                )
            )
            missing_data = df.isnull().sum()
            missing_data_columns = pd.Index(missing_data[missing_data > 0].index)
            fill_missing_data(df, missing_data_columns)
            missing_data_summary(df)

            # Plot correlation matrix
            logging.info(
                colored(
                    "Plotting correlation matrix for numerical fetaures",
                    "light_magenta",
                )
            )
            plot_correlation_matrix(df, analysis_title, outputdir)

            # (6) Save curated dataset
            logging.info(colored("Save curated datasets", "light_magenta"))
            df.to_csv(
                f"{os.path.splitext(os.path.basename(d))[0]}_curated.csv", index=False
            )

    # perform nlp preprocessing
    if preprocessing_nlp == True:
        logging.info(colored("NLP preprocessing", "light_magenta"))

        for d in raw_datasets:
            logging.info(colored(f"Processing {d}", "yellow"))
            df = load_dataset(d)

            # drop useless features
            logging.info(colored("Dropping useless features", "light_magenta"))
            to_be_dropped = prime_service["preparation_configuration"][
                "preprocessing_nlp"
            ]["options"]["to_be_dropped"]
            logging.info(colored(to_be_dropped, "yellow"))
            df = df.drop(columns=to_be_dropped)

            # cleaning textual features
            logging.info(colored("Cleaning textural features", "light_magenta"))
            for f in prime_service["preparation_configuration"]["preprocessing_nlp"][
                "options"
            ]["text_features"]:
                logging.info(colored(f"- {f}", "yellow"))
                plot_sentence_length_distribution(df, f, analysis_title, "before_cleaning", outputdir)
                apply_cleaning_with_progress(df, column=f, new_column=f)
                plot_word_frequency(df, f, analysis_title, outputdir)
                plot_sentence_length_distribution(df, f, analysis_title, "after_cleaning", outputdir)

            # Preprocess tags
            logging.info(colored("Cleaning tags features", "light_magenta"))
            for f in prime_service["preparation_configuration"]["preprocessing_nlp"]["options"]["tags_features"]:
                logging.info(colored(f"- {f}", "yellow"))
                df[f] = df[f].apply(lambda x: x.replace('><', ' ').replace('<', '').replace('>', '') if isinstance(x, str) else x)
                plot_sentence_length_distribution(df, f, analysis_title, "before_cleaning", outputdir)
                plot_word_frequency(df, f, analysis_title, outputdir)
                plot_sentence_length_distribution(df, f, analysis_title, "after_cleaning", outputdir)

            # Extract temporal features
            logging.info(colored("Extract temporal features", "light_magenta"))
            for f in prime_service["preparation_configuration"]["preprocessing_nlp"][
                "options"
            ]["temporal_features"]:
                logging.info(colored(f"- {f}", "yellow"))
                temporal_extraction(df, f)

            # Target encoding
            logging.info(colored("Target encoding", "light_magenta"))
            for item in prime_service["preparation_configuration"]["preprocessing_nlp"][
                "options"
            ]["ordinal_features"]:
                feature_name = item["feature"]
                order = item["order"]

                # Process ordering
                df[feature_name] = pd.Categorical(df[feature_name], categories=order, ordered=True)
                plot_non_numerical_distribution(df, [feature_name], analysis_title, "ordered", outputdir)
                df[feature_name] = df[feature_name].cat.codes
                plot_non_numerical_distribution(df, [feature_name], analysis_title, "labelled", outputdir)
                logging.info(colored(feature_name, "yellow"))

            # Save curated dataset
            logging.info(colored("Save curated datasets", "light_magenta"))
            df.to_csv(
                f"{os.path.splitext(os.path.basename(d))[0]}_curated.csv", index=False
            )

        """


def main():
    """
    Entry point for running training data preparation.

    Parses command-line arguments and invokes the training_data_extraction function.
    """
    parser = ArgumentParser(
        description="Run dataset mining & preparation for AI case two"
    )
    parser.add_argument(
        "--configfile",
        action="store",
        dest="configfile",
        default="config/config.yaml",
        help="Configuration file path",
    )
    args = vars(parser.parse_args())
    preparation(**args)


if __name__ == "__main__":
    main()
