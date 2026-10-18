#!/usr/bin/env python3


def normalize_columns(df):
    """
    Normalize dataframe column names by:
    - Converting to lowercase
    - Replacing whitespace and special characters with underscores

    Args:
        df: Input dataframe (pandas or polars)

    Returns:
        Dataframe with normalized column names
    """
    # Create a mapping of old column names to normalized names
    normalized_names = {}
    for col in df.columns:
        # Convert to lowercase and replace whitespace with underscore
        normalized = col.lower().replace(" ", "_")
        normalized_names[col] = normalized

    # Rename columns based on the mapping
    return df.rename(columns=normalized_names)

