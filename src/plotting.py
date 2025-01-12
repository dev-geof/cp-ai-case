import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from configuration import create_directory
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
)
import seaborn as sns
from collections import Counter


def plot_numerical_distribution(df, columns, analysis_title, type, outputdir):
    """
    Plot the distribution of numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the numerical columns to analyze.
    columns : list of str
        A list of column names to plot.
    analysis_title : str
        Title to display on the plots.
    outputdir : str
        Directory path where the plots will be saved.

    Returns
    -------
    None
    """
    for col in columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], fill=True, bins=40, element="step")
        plt.ylabel("Number of Entries")
        plt.yscale("log")
        plt.title(analysis_title)
        plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
        create_directory(f"{outputdir}/Input_Features")
        plt.savefig(
            f"{outputdir}/Input_Features/Num_{col}_{type}.pdf", transparent=True
        )
        plt.tight_layout()
        plt.close()
        print(f"Num_{col}_{type}.pdf has been created")


def plot_non_numerical_distribution(df, columns, analysis_title, type, outputdir):
    """
    Plot the distribution of non-numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the categorical columns to analyze.
    columns : list of str
        A list of column names to plot.
    analysis_title : str
        Title to display on the plots.
    outputdir : str
        Directory path where the plots will be saved.

    Returns
    -------
    None
    """
    for col in columns:
        plt.figure(figsize=(10, 8))
        sns.countplot(data=df, x=col)
        plt.xticks(rotation=30)
        plt.ylabel("Number of Entries")
        # plt.yscale('log')
        plt.title(analysis_title)
        plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
        create_directory(f"{outputdir}/Input_Features")
        plt.savefig(
            f"{outputdir}/Input_Features/Obj_{col}_{type}.pdf", transparent=True
        )
        plt.tight_layout()
        plt.close()
        print(f"Obj_{col}_{type}.pdf has been created")


def plot_boxplot(df, columns, target, bins, analysis_title, outputdir):
    """
    Plot boxplots of numerical features against the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the columns to analyze.
    columns : list of str
        A list of numerical column names to plot.
    target : str
        The target column for comparison.
    bins : int
        The number of bins to rebin the target variable if it is numerical.
    analysis_title : str
        Title to display on the plots.
    outputdir : str
        Directory path where the plots will be saved.

    Returns
    -------
    None
    """
    # Rebin target column if it is numerical, otherwise use as-is
    if df[target].dtype != "object":
        # Create labeled bins with ranges for the target variable
        df["rebinned_target"] = pd.cut(df[target], bins=bins)
        target_to_plot = "rebinned_target"
    else:
        target_to_plot = target

    for col in columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[target], y=df[col], palette="Set3")
        plt.title(f"Boxplot of {col} by {target_to_plot} (Rebinned into {bins} bins)")
        plt.xlabel(f"{target} (Rebinned)" if df[target].dtype != "object" else target)
        plt.xticks(rotation=30)  # Rotate x-axis labels if they are long
        plt.savefig(f"{outputdir}/Boxplots/Num_{col}_box.pdf", transparent=True)
        print(f"Num_{col}_box.pdf has been created")
        plt.tight_layout()
        plt.close()


def plot_correlation_matrix(df, analysis_title, outputdir):
    """
    Plot the correlation matrix for numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing numerical features.
    analysis_title : str
        Title to display on the plot.
    outputdir : str
        Directory path where the plot will be saved.

    Returns
    -------
    None
    """
    corr_matrix = df.corr()
    plt.figure(figsize=(25, 15))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
    plt.title(f"{analysis_title} - Correlation Matrix")
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    plt.xticks(rotation=90)  # Rotate x-axis labels if they are long
    plt.savefig(f"{outputdir}/Correlation.pdf", transparent=True)
    plt.tight_layout()
    plt.close()


def plot_shap_values(columns, shap_values, outputdir):
    """
    Plot mean absolute SHAP values for features.

    Parameters
    ----------
    columns : list of str
        Feature names corresponding to the SHAP values.
    shap_values : np.ndarray
        Array of SHAP values for each feature.
    outputdir : str
        Directory path where the plot will be saved.

    Returns
    -------
    None
    """
    plt.figure(figsize=(12, 5))
    plt.barh(columns, shap_values.flatten())
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title("Average SHAP Values for Input Features")
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.savefig(f"{outputdir}/SHAP.pdf", transparent=True)
    plt.tight_layout()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title, class_names, outputdir):
    """
    Plot and save a normalized confusion matrix as percentages.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels (one-hot encoded).
    y_pred : np.ndarray
        Predicted class labels (one-hot encoded).
    title : str
        Title to display on the plot.
    outputdir : str
        Directory path where the plot will be saved.

    Returns
    -------
    None
    """
    cm = confusion_matrix(
        y_true.argmax(axis=1), y_pred.argmax(axis=1), normalize="true"
    )  # Row-normalized matrix
    cm_percentage = cm * 100  # Convert to percentages
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_percentage, display_labels=class_names
    )
    disp.plot(cmap=plt.cm.Blues, values_format=".1f")
    plt.title(f"{title} Confusion Matrix (Normalized)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    # plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.savefig(f"{outputdir}/{title}_confusion_matrix.pdf", transparent=True)
    plt.close()


def plot_class_metrics(y_true, y_pred, title, class_names, outputdir):
    """
    Plot precision, recall, and F1-score for each class.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels (one-hot encoded).
    y_pred : np.ndarray
        Predicted class labels (one-hot encoded).
    outputdir : str
        Directory path where the plot will be saved.

    Returns
    -------
    None
    """
    report = classification_report(
        y_true.argmax(axis=1), y_pred.argmax(axis=1), output_dict=True
    )
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df = metrics_df.loc[
        ~metrics_df.index.isin(["accuracy", "macro avg", "weighted avg"])
    ]

    # Plot bar chart
    metrics_df[["precision", "recall", "f1-score"]].plot(kind="bar", figsize=(10, 6))
    plt.title(f"Class-Wise Metrics - {title}")
    plt.ylabel("Score")
    plt.xlabel("Classes")
    x = np.arange(len(class_names))  # the label locations
    # width = 0.05  # the width of the bars
    plt.xticks(x, np.array(class_names), rotation=30)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{outputdir}/{title}_class_metrics.pdf", transparent=True)
    plt.close()


def plot_model_output_probabilities(
    predicted_probabilities, y_test, title, class_names, output_dir
):
    """
    Plots the output probability distributions for each class, grouped by the true labels, using matplotlib.

    Parameters:
    ----------
    predicted_probabilities : np.ndarray
        Precomputed predicted probabilities for each class (shape: [n_samples, n_classes]).
    y_test : np.ndarray
        True labels for the test set in one-hot encoded format (shape: [n_samples, n_classes]).
    class_names : list
        List of class names corresponding to the output probabilities.
    output_dir : str
        Directory to save the plots.
    """
    # Decode one-hot encoded y_test into true class indices
    true_labels = np.argmax(y_test, axis=1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot probability distributions for each class
    num_classes = len(class_names)
    for class_idx, class_name in enumerate(class_names):
        plt.figure(figsize=(10, 6))

        for label in range(num_classes):
            # Extract probabilities for the current class where the true label matches
            class_probs = predicted_probabilities[true_labels == label, class_idx]
            plt.hist(
                class_probs,
                histtype="step",
                bins=40,
                range=(0, 1),
                # alpha=0.6,
                label=f"True Label: {class_names[label]}",
                # edgecolor="k",
                linewidth="2",
            )

        # Customize the plot
        plt.title(f"Probability Distribution for Class {class_name} - {title}")
        plt.xlabel(f"Predicted Probability of Class {class_name}")
        plt.ylabel("Number of Entries")
        plt.legend(title="True Label", loc="upper right")
        plt.yscale("log")
        # plt.grid(visible=True, alpha=0.5)

        # Save the plot
        plt.savefig(
            f"{output_dir}/{title}_probability_distribution_{class_name}.pdf",
            transparent=True,
        )
        plt.close()


def plot_model_log_probability_ratios_and_roc(
    predicted_probabilities, y_test, title, class_names, output_dir
):
    """
    Plots the log probability ratio distributions and ROC curves for each class, grouped by the true labels, using matplotlib.
    The ROC curve has Efficiency on the x-axis and Rejection on the y-axis.

    Parameters:
    ----------
    predicted_probabilities : np.ndarray
        Precomputed predicted probabilities for each class (shape: [n_samples, n_classes]).
    y_test : np.ndarray
        True labels for the test set in one-hot encoded format (shape: [n_samples, n_classes]).
    class_names : list
        List of class names corresponding to the output probabilities.
    output_dir : str
        Directory to save the plots.
    """
    # Decode one-hot encoded y_test into true class indices
    true_labels = np.argmax(y_test, axis=1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a figure with two rows and num_classes columns
    num_classes = len(class_names)
    plt.figure(figsize=(12, 6 * num_classes))

    for class_idx, class_name in enumerate(class_names):
        # --- Plot Log Probability Ratio Distributions ---
        plt.subplot(num_classes, 2, 2 * class_idx + 1)
        for label in range(num_classes):
            # Calculate log probability ratio for the current class
            class_probs = predicted_probabilities[:, class_idx]
            other_probs_sum = np.sum(predicted_probabilities, axis=1) - class_probs
            log_prob_ratios = np.log(
                class_probs / (other_probs_sum + 1e-12)
            )  # Avoid division by zero

            # Filter log probability ratios where the true label matches
            log_ratios_filtered = log_prob_ratios[true_labels == label]

            plt.hist(
                log_ratios_filtered,
                histtype="step",
                bins=40,
                range=(-5, 5),
                linewidth=2,
                label=f"True Label: {class_names[label]}",
            )

        # Customize the log probability ratio plot
        plt.title(f"Log Probability Ratio for Class {class_name} - {title}")
        plt.xlabel(f"Log Ratio of Probability (Class {class_name} / Sum(Others))")
        plt.ylabel("Number of Entries")
        plt.legend(title="True Label", loc="upper right")
        plt.yscale("log")

        # --- Plot ROC Curve for each true label (for this class) ---
        plt.subplot(num_classes, 2, 2 * class_idx + 2)

        # Compute ROC for this class vs. all others for each true label
        for label in range(num_classes):
            binary_true_labels = (true_labels == label).astype(
                int
            )  # 1 if true label is this class, 0 otherwise
            scores = predicted_probabilities[:, class_idx] / (
                np.sum(predicted_probabilities, axis=1) + 1e-12
            )  # Ratio

            fpr, tpr, _ = roc_curve(binary_true_labels, scores)

            # Rejection is 1 - FPR
            rejection = 1 - fpr

            # Plot ROC for this true label
            plt.plot(
                tpr, rejection, label=f"True Label: {class_names[label]}", linewidth=2
            )

        # Add diagonal reference line
        plt.plot([1, 0], [0, 1], linestyle="--", color="gray", linewidth=1)

        # Customize the ROC plot
        plt.title(f"ROC Curves for Class {class_name} - {title}")
        plt.xlabel("Efficiency (True Positive Rate)")
        plt.ylabel("Rejection (False Positive Rate)")
        plt.legend(loc="lower right")
        plt.grid(visible=True, alpha=0.5)

    # Save the combined plot
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/{title}_log_ratio_and_roc_{class_name}.pdf", transparent=True
    )
    plt.close()


def plot_word_frequency(df, column, analysis_title, outputdir):
    """
    Plots the frequency of words across all rows in the specified column, displaying the top 50 most frequent words.

    Args:
        df (pd.DataFrame): The input DataFrame containing the text data to analyze.
        column (str): The name of the column in the DataFrame that contains the text (sentence/word data).
        analysis_title (str): The title for the word frequency plot.
        outputdir (str): The directory where the plot will be saved.

    Returns:
        None: The function saves the word frequency plot as a PDF in the specified output directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(f"{outputdir}/Input_Features", exist_ok=True)

    # Split all the text in the specified column into individual words and flatten them into a single list
    all_words = []
    for text in df[column]:
        all_words.extend(text.split())  # Split each row into words and add to the list

    # Count the frequency of each word
    word_counts = Counter(all_words)

    # Get the 40 most common words
    most_common_words = word_counts.most_common(50)

    # Separate words and their counts
    words, counts = zip(*most_common_words)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.title(analysis_title)
    plt.xlabel(f"Words in {column}")
    plt.ylabel("Number of Entries")
    plt.xticks(rotation=90)  # Rotate x-axis labels to fit longer words
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

    # Save the plot as a PDF
    plt.savefig(
        f"{outputdir}/Input_Features/Word_frequency_{column}.pdf", transparent=True
    )
    plt.tight_layout()
    plt.close()

    print(f"Word_frequency_{column}.pdf has been created")


def plot_sentence_length_distribution(df, column, analysis_title, type, outputdir):
    """
    Plots the distribution of sentence lengths (in terms of word count) for a specified column in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the sentences to analyze.
        column (str): The name of the column in the DataFrame that contains the sentences.
        analysis_title (str): The title for the plot that will be displayed.
        type (str): A label or identifier (e.g., "train", "val") to distinguish the type of dataset.
        outputdir (str): Directory where the plot will be saved.

    Returns:
        None: The function saves the sentence length distribution plot as a PDF in the specified output directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(f"{outputdir}/Input_Features", exist_ok=True)

    # Calculate the length of each sentence (in terms of the number of words)
    sentence_lengths = [len(text.split()) for text in df[column]]

    # Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(sentence_lengths, fill=True, bins=40, element="step")
    plt.title(analysis_title)
    plt.xlabel(f"Sentence Length (Number of Words) - {column}")
    plt.ylabel("Number of Entries")
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.savefig(
        f"{outputdir}/Input_Features/Sentence_Length_Distribution_{column}_{type}.pdf",
        transparent=True,
    )
    plt.tight_layout()
    plt.close()

    print(f"Sentence_Length_Distribution_{column}_{type}.pdf has been created")
