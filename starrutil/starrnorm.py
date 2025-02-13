import pandas as pd
from typing import Callable, Optional

def normalize_by_lib(
        df: pd.DataFrame,
        counts_per: int = 1e6,
        dna_prefix: str = "input",
        rna_prefix: str = "output",
        rep_suffix: str = "_rep",
        rename_func: Optional[Callable[[str], str]] = lambda x: x + "_norm") -> pd.DataFrame:
    """
    Normalize the DNA and RNA count columns based on library sizes.

    The library sizes are approximated as the column sums, so this
    may be inaccurate if the columns are not the full datasets or a
    reasonably large randomly sampled subset.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be normalized.
    counts_per : int, optional
        A scaling factor post-normalization. Default is 1e6, for counts per million.
    dna_prefix : str, optional
        The prefix for DNA columns. Default is "input".
    rna_prefix : str, optional
        The prefix for RNA columns. Default is "output".
    rep_suffix : str, optional
        The suffix for replicate columns. Default is "_rep".
    rename_func : callable, optional
        A function to rename the normalized columns. Default is appending "_norm" to the column names.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the normalized columns.
    """
    col_regex = "^(" + dna_prefix + "|" + rna_prefix + ").*" + rep_suffix
    libsizes = df.filter(regex=col_regex).sum(axis=0)
    dfnorm = df.filter(regex=col_regex) / libsizes * counts_per
    if rename_func is None:
        rename_func = lambda x: x
    return dfnorm.rename(columns=rename_func)

def mean_fc(df: pd.DataFrame,
            pseudocount: int = 0, 
            dna_prefix: str = "input",
            rna_prefix: str = "output",
            rep_suffix: str = "_rep") -> pd.Series:
    """
    Calculate the mean fold change between RNA and DNA columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    pseudocount : int, optional
        A small constant added to avoid division by zero. Default is 0.
    dna_prefix : str, optional
        The prefix for DNA columns. Default is "input".
    rna_prefix : str, optional
        The prefix for RNA columns. Default is "output".
    rep_suffix : str, optional
        The suffix for replicate columns. Default is "_rep".

    Returns
    -------
    pd.Series
        A Series containing the mean fold change for each row.
    """
    dna_cols = "^" + dna_prefix + ".*" + rep_suffix
    rna_cols = "^" + rna_prefix + ".*" + rep_suffix
    return ((df.filter(regex=rna_cols).mean(axis=1) + pseudocount) /
            (df.filter(regex=dna_cols).mean(axis=1) + pseudocount))