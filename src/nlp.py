import time
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from itertools import chain
from tqdm import tqdm
import spacy
nlp = spacy.load("en_core_web_sm") # Loading spaCy model globally


def clean_text_tokens(text):
    """
    Clean and preprocess text data, returning a list of tokens.

    - Removes HTML tags.
    - Converts text to lowercase.
    - Removes stopwords and non-alphabetic tokens using spaCy.

    Parameters
    ----------
    text : str
        The input text to clean.

    Returns
    -------
    list
        A list of cleaned and preprocessed tokens.
    """

    if not isinstance(text, str):  # Handle cases where text might be NaN or non-string
        return ""

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Process text with spaCy
    doc = nlp(text.lower())

    filtered_tokens = [
        token.text for token in doc 
        if not (token.is_stop or not token.is_alpha or token.like_url or token.like_email)
    ]

    # Recombine the filtered tokens into a single text
    recombined_text = ' '.join(filtered_tokens)

    return recombined_text

def apply_cleaning_with_progress(df, column, new_column):
    """
    Apply text cleaning to a DataFrame column with progress tracking.

    Uses `tqdm` to show a progress bar while cleaning text data.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing the column to clean.
    column : str
        The name of the column to clean.
    new_column : str
        The name of the new column to store the cleaned text.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with the cleaned text added as a new column.
    """

    start_time = time.time()

    # Initialize tqdm with column name in the description
    tqdm.pandas(desc=f"Processing '{column}'")

    # Apply the cleaning function with tqdm progress bar
    df[new_column] = df[column].progress_apply(clean_text_tokens)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(
        f"Processing of column '{column}' complete. Total time taken: {elapsed_time:.2f} seconds."
    )

    return df

def temporal_extraction(df, column):
    """
    Extracts temporal features from a datetime column in a DataFrame, including cyclic encodings,
    and adds them as new columns.

    Args:
        df (pd.DataFrame): The input DataFrame containing the column to process.
        column (str): The name of the column in the DataFrame that contains datetime data.

    Returns:
        pd.DataFrame: The modified DataFrame with new temporal feature columns:
            - "Year": The year extracted from the datetime column.
            - "Month": The month extracted from the datetime column.
            - "Day_of_week": The day of the week extracted from the datetime column (0=Monday, 6=Sunday).
            - "Hour_of_day": The hour of the day extracted from the datetime column (0-23).
            - "Hour_of_week": The hour of the week, computed as (`Day_of_week` * 24 + `Hour_of_day`).
            - Sine and cosine encodings for "Hour_of_day", "Day_of_week", and "Month" for cyclic representation.
    """
    df[column] = pd.to_datetime(df[column])

    # Extract basic temporal features
    df["Year"] = df[column].dt.year
    df["Month"] = df[column].dt.month
    df["Day_of_week"] = df[column].dt.dayofweek
    df["Hour_of_day"] = df[column].dt.hour
    df["Hour_of_week"] = df["Day_of_week"] * 24 + df["Hour_of_day"]

    # Cyclic encodings
    df["Hour_of_day_sin"] = np.sin(2 * np.pi * df["Hour_of_day"] / 24)
    df["Hour_of_day_cos"] = np.cos(2 * np.pi * df["Hour_of_day"] / 24)
    df["Day_of_week_sin"] = np.sin(2 * np.pi * df["Day_of_week"] / 7)
    df["Day_of_week_cos"] = np.cos(2 * np.pi * df["Day_of_week"] / 7)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    # Drop the original datetime column (if no longer needed)
    df = df.drop(columns=column)


