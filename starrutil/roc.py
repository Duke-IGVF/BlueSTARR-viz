import pandas as pd
from sklearn.metrics import RocCurveDisplay, roc_curve
import matplotlib.pyplot as plt

def topN_dataset(df: pd.DataFrame,
                 neg_label : str,
                 N: int=1000,
                 effect_col: str='mpra_max_log2effect',
                 class_col: str='label',
                 **sample_kwargs) -> pd.DataFrame:
    """
    Create a dataset containing the top N variants with the highest absolute effect sizes 
    and a random sample of N negative class variants.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be sampled. Must include columns for 
        effect sizes and class labels.
    neg_label : str
        The label for the negative class in the `class_col` column.
    N : int, optional
        The number of top variants and negative class variants to include in the output 
        dataset (default is 1000).
    effect_col : str, optional
        The name of the column containing the effect sizes (default is 'mpra_max_log2effect').
    class_col : str, optional
        The name of the column containing the class labels (default is 'label').
    **sample_kwargs : dict, optional
        Additional keyword arguments to pass to the `pandas.DataFrame.sample` method 
        when sampling negative class variants.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing:
        - The top N variants with the highest absolute effect sizes.
        - A random sample of N variants from the negative class.

    Notes
    -----
    - The function ensures that the top N variants are selected based on the absolute 
      values in the `effect_col` column.
    - The negative class variants are sampled randomly from the rows where the `class_col` 
      column matches the `neg_label`.

    """
    topN = df[effect_col].abs().argsort()[-N:]
    return pd.concat([df.iloc[topN],
                      df.loc[df[class_col] == neg_label].sample(N, **sample_kwargs)],
                      ignore_index=True)

def roc_topN(df: pd.DataFrame,
             neg_label: str,
             pos_label: str,
             Ns: list[int] = [1000, 500, 200],
             topN_effect_col: str = 'mpra_max_log2effect',
             pred_effect_col: str = 'log2FC',
             class_col: str = 'label',
             ax: plt.Axes = None,
             include_full: bool = True,
             print_precise: bool = True,
             **sample_kwargs) -> tuple[dict, plt.Axes]:
    """
    Generate ROC curves for the top N variants with the highest absolute effect sizes 
    and optionally for the full dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be analyzed. Must include columns 
        for effect sizes, predictions, and class labels.
    neg_label : str
        The label for the negative class in the `class_col` column.
    pos_label : str
        The label for the positive class in the `class_col` column.
    Ns : list of int, optional
        A list of values for N, representing the number of top variants to include 
        in the ROC analysis (default is [1000, 500, 200]).
    topN_effect_col : str, optional
        The name of the column containing the effect sizes used to select the top N 
        variants (default is 'mpra_max_log2effect').
    pred_effect_col : str, optional
        The name of the column containing the predicted effect sizes (default is 'log2FC').
    class_col : str, optional
        The name of the column containing the class labels (default is 'label').
    ax : matplotlib.axes.Axes, optional
        The matplotlib Axes object to plot the ROC curves on. If None, a new Axes 
        object is created (default is None).
    include_full : bool, optional
        Whether to include a ROC curve for the full dataset (default is True).
    print_precise : bool, optional
        Whether to print precise AUC values for each ROC curve (default is True).
    **sample_kwargs : dict, optional
        Additional keyword arguments to pass to the `topN_dataset` function when 
        sampling negative class variants.

    Returns
    -------
    tuple[dict, matplotlib.axes.Axes]
        A tuple containing:
        - A dictionary where keys are N values (or 'All' for the full dataset) and 
          values are `RocCurveDisplay` objects for the corresponding ROC curves.
        - The matplotlib Axes object containing the plotted ROC curves.

    Notes
    -----
    - The function uses the `topN_dataset` function to create datasets for the top N 
      variants and randomly sampled negative class variants.
    - The ROC curves are generated using `RocCurveDisplay.from_predictions`.
    - The AUC values for each ROC curve are printed if `print_precise` is True.

    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    rocs = {}
    if include_full:
        roc = RocCurveDisplay.from_predictions(
            df[class_col],
            df[pred_effect_col].abs(), pos_label=pos_label,
            name='All', ax=ax)
        rocs['All'] = roc
    df_topNs = {N: topN_dataset(df,
                                neg_label=neg_label, N=N,
                                effect_col=topN_effect_col,
                                class_col=class_col,
                                **sample_kwargs) for N in Ns}
    for N, df_topN in df_topNs.items():
        chance_level_args = dict(
            plot_chance_level=True,
            chance_level_kw={'linestyle': 'dotted', 'linewidth': 1.5, 'color': 'gray'}
        ) if N == Ns[-1] else {}
        roc = RocCurveDisplay.from_predictions(
            df_topN[class_col],
            df_topN[pred_effect_col].abs(), pos_label=pos_label,
            name=f'MPRA Top-{N}', ax=ax,
            **chance_level_args)
        rocs[N] = roc
    if print_precise:
        print(f"  AUCs:")
        for N, roc in rocs.items():
            if type(N) == int:
                print(f"    Top {N:4} = {roc.roc_auc:.3f}")
            else:
                print(f"    {N} data = {roc.roc_auc:.3f}")
    return rocs, ax
