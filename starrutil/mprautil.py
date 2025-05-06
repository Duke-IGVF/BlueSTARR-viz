import polars as pl
import pandas as pd
import duckdb
from pathlib import Path
from collections.abc import Iterable
from Bio.Align import PairwiseAligner

def read_kircher_mpra_data(data_dir: str | Path,
                           file_path_col: str | None='file_path') -> pl.LazyFrame:
    """
    Reads MPRA (Massively Parallel Reporter Assay) data from a specified directory 
    and processes it into a Polars DataFrame.

    The columns in the files are the following (according to, and taken verbatim from, the [wiki accompanying the file deposition](https://osf.io/75b2m/wiki/Files/)):
    1. **Chromosome** - Chromosome of the variant.
    2. **Position** - Chromosomal position (GRCh38 or GRCh37) of the variant. _(We are using the GRCh38 coordinates)_
    3. **Ref** - Reference allele of the variant (A, T, G, or C).
    4. **Alt** - Alternative allele of the variant (A, T, G, or C). One base-pair deletions are represented as -.
    5. **Tags** - Number of unique tags associated with the variant.
    6. **DNA** - Count of DNA sequences that contain the variant (used for fitting the linear model).
    7. **RNA** - Count of RNA sequences that contain the variant (used for fitting the linear model).
    8. **Value** - Log2 variant expression effect derived from the fit of the linear model (coefficient).
    9. **P-Value** - P-value of the coefficient.

    Parameters
    ----------
    data_dir : str or Path
        The directory containing the MPRA data files. Can be a string or a Path object.
    file_path_col : str or None, optional
        The name of the column to include file paths in the resulting DataFrame. 
        If None, file paths will not be included. Default is 'file_path'.

    Returns
    -------
    pl.LazyFrame
        A Polars LazyFrame containing the processed MPRA data with the following columns:
        - `allele_pos`: Adjusted position of the allele (Position - 1).
        - `chrom`: Chromosome name prefixed with 'chr'.
        - `ref_allele`: Reference allele.
        - `alt_allele`: Alternate allele.
        - `p_value`: P-value from the data.
        - `experiment`: Extracted experiment name from the file path.
        - `region_type`: Region type extracted from the file path.
        - `region`: Region name extracted from the experiment column.

    Notes
    -----
    - The function filters rows where the `Ref` and `Alt` columns have a length of 1 byte 
      and `Alt` is not '-' (i.e., drops deletions).
    - Null values are dropped from the resulting DataFrame.
    - The function assumes input files are in TSV format with a header and uses "NA" 
      to represent null values.
    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    if file_path_col:
        data_dir = data_dir / "*"
    mut_data = pl.scan_csv(data_dir / "*.tsv", separator="\t",
                           has_header=True, null_values=["NA"], include_file_paths=file_path_col)
    mut_data = mut_data.filter(
        pl.col('Ref').str.len_bytes() == 1,
        pl.col('Alt').str.len_bytes() == 1,
        pl.col('Alt') != '-',
    ).with_columns(
        pl.col('Position').add(-1).alias('allele_pos'),
        (pl.lit('chr') + pl.col('Chromosome')).alias('chrom'),
        pl.col('Ref').alias('ref_allele'),
        pl.col('Alt').alias('alt_allele'),
        pl.col('P-Value').alias('p_value'),
        pl.col(file_path_col).str.extract(r'GRCh38_(.*).tsv$').alias('experiment'),
        pl.col(file_path_col).str.split('/').list.get(-2).alias('region_type')
    ).with_columns(
        pl.col('experiment').str.extract(r'^([^-.]+)').alias('region'),
    ).drop(['Ref', 'Alt', 'Position', 'Chromosome']).drop_nulls()    
    return mut_data

def get_kircher_mpra_regions(mpra_df: pl.DataFrame | pd.DataFrame=None,
                             data_dir: str | Path=None):
    """
    Function to get MPRA regions from the provided DataFrame or directory.
    
    Parameters
    ----------
    mpra_df : pl.DataFrame or pd.DataFrame, optional
        The DataFrame containing MPRA data. If None, data will be read from the directory.
    data_dir : str or Path, optional
        The directory containing the MPRA data files. Can be a string or a Path object.
    
    Returns
    -------
    pl.DataFrame or pd.DataFrame
        A DataFrame containing the MPRA regions. The DataFrame will have the following columns:
        - `region_type`: Type of the region (Enhancers, Promoters).
        - `region`: Name of the region.
        - `chrom`: Chromosome name.
        - `start_pos`: Start position of the region (zero-based).
        - `end_pos`: End position of the region (zero-based).
        - `region_length`: Length of the region.
        - `experiments`: List of experiments associated with the region.
    
    Note that start_pos and end_pos are both zero-based coordinates, not
    half-open interval coordinates. Hence, the length of the region is
    end_pos - start_pos + 1.
    """
    if mpra_df is None:
        mpra_df = read_kircher_mpra_data(data_dir)
    regions = duckdb.sql(
        "select region_type, region, chrom, "
        "min(allele_pos) as start_pos, max(allele_pos) as end_pos, "
        "max(allele_pos) - min(allele_pos) + 1 as region_length, "
        "group_concat(distinct experiment, ',') as experiments "
        "from mpra_df group by region_type, region, chrom "
        "order by region_type, region, chrom").df()
    return regions

def read_kircher_gksvm_data(data_dir: str | Path,
                            file_path_col: str | None='file_path') -> pl.LazyFrame:
    """
    Read and process gkSVM data from a directory or its subdirectories into a Polars LazyFrame.

    Parameters
    ----------
    data_dir : str or Path
        The directory containing the gkSVM data files. Can be a string or a Path object.
    file_path_col : str or None, optional
        The name of the column to include file paths in the resulting DataFrame. 
        If None, file paths will not be included. Default is 'file_path'.

    Returns
    -------
    pl.LazyFrame
        A Polars LazyFrame containing the processed gkSVM data with the following columns:
        - `rel_pos`: Relative position of the allele (zero-based).
        - `ref_allele`: Reference allele.
        - `alt_allele`: Alternate allele.
        - `log2fc`: Log2 fold change values.
        - `region_type`: Region type extracted from the file path, if file paths are included.
        - `region`: Region name extracted from the file basename, if file paths are included.
        - `cell_line`: Cell line extracted from the file basename, if file paths are included.
        - `experiment`: Experiment name extracted from the file basename, if file paths are included.

    Notes
    -----
    - The function assumes input files are in TSV format with no header and uses "NA" 
      to represent null values.
    - The file paths are parsed to extract metadata such as region type, region name, 
      cell line, and experiment name.
    - The resulting LazyFrame drops null values and unnecessary columns.

    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    if file_path_col:
        data_dir = data_dir / "*"
    gksvm_raw = pl.scan_csv(data_dir / "*.tsv", separator="\t",
                            has_header=False,
                            new_columns=['rel_pos', 'ref_allele', 'alt_allele', 'log2fc'],
                            null_values=["NA"], include_file_paths="file_path")
    gksvm_proc = gksvm_raw.with_columns(
        pl.col('rel_pos').add(-1),
        pl.col('file_path').str.split('/').list.get(-2).alias('region_type'),
        pl.col('file_path').str.split('/').list.get(-1).alias('basename'),
    ).with_columns(
        pl.col('basename').str.extract(r'^([^-.]+)').alias('region'),
        pl.col('basename').str.extract(r'-([^-.]+).tsv$').alias('cell_line'),
        pl.col('basename').str.extract(r'-(.+)-').alias('experiment'),
    ).drop(['basename']).drop_nulls()
    return gksvm_proc

def add_class_labels(df: pd.DataFrame,
                     Pval_pos: float,
                     Pval_neg: float,
                     abs_effect_neg: float = 0.05,
                     pos_label: str | None = None,
                     neg_label: str | None = None,
                     min_tags: int = 10,
                     label_col: str = 'label') -> pd.DataFrame:
    """
    Add class labels to a DataFrame based on MPRA p-values, effect sizes, and tag counts.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing MPRA data with columns for p-values, effect sizes, and tag counts.
    Pval_pos : float
        The p-value threshold for labeling positive mutations (e.g., p < 1e-5).
    Pval_neg : float
        The p-value threshold for labeling negative mutations (e.g., p > 0.01).
    abs_effect_neg : float, optional
        The maximum absolute effect size for labeling negative mutations (default is 0.05).
    pos_label : str or None, optional
        The label to assign to positive mutations. If None, a default label based on `Pval_pos` is used.
    neg_label : str or None, optional
        The label to assign to negative mutations. If None, a default label based on `Pval_neg` is used.
    min_tags : int, optional
        The minimum number of tags required to label a mutation (default is 10).
    label_col : str, optional
        The name of the column to store the labels (default is 'label').

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with an additional column containing the assigned labels.

    Notes
    -----
    - Mutations with p-values below `Pval_pos` and tag counts greater than or equal to `min_tags` are labeled as positive.
    - Mutations with p-values above `Pval_neg`, absolute effect sizes below `abs_effect_neg`, and tag counts greater than or equal to `min_tags` are labeled as negative.
    - Mutations that do not meet the criteria for either positive or negative labels are left unlabeled (i.e., `None`).
    """
    df[label_col] = None
    if pos_label is None:
        pos_label = f"MPRA p<{Pval_pos}"
    if neg_label is None:
        neg_label = f"MPRA p>{Pval_neg}"
    df.loc[(df['mpra_p_value'] < Pval_pos) &
           (df['mpra_tags'] >= min_tags), label_col] = pos_label 
    df.loc[(df['mpra_p_value'] > Pval_neg) &
           (df['mpra_max_log2effect'].abs() < abs_effect_neg) &
           (df['mpra_tags'] >= min_tags), label_col] = neg_label    
    return df

def revcomp(seq: str | Iterable[str]) -> str | list[str]:
    """
    Reverse complement of a DNA sequence.

    Parameters
    ----------
    seq : str or Iterable[str]
        A single DNA sequence as a string or an iterable of DNA bases.

    Returns
    -------
    str or list[str]
        The reverse complement of the input sequence. If the input is an iterable 
        of bases, a list of reverse complement bases is returned.

    Notes
    -----
    - The function handles both uppercase and lowercase nucleotide bases.
    - Non-standard bases are returned unchanged in the reverse complement.

    """
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'a': 't', 't': 'a', 'c': 'g', 'g': 'c'}
    rc = [comp.get(base, base) for base in reversed(seq)]
    return ''.join(rc) if isinstance(seq, str) else rc

def align_refs(target: str, query: str) -> tuple[int, list[tuple[int, int]]]:
    """
    Align a query sequence to a target sequence using local alignment.

    Parameters
    ----------
    target : str
        The target DNA sequence to which the query will be aligned.
    query : str
        The query DNA sequence to be aligned to the target.

    Returns
    -------
    tuple[int, list[tuple[int, int]]]
        A tuple containing:
        - An integer indicating the orientation of the alignment:
          1 for forward orientation, -1 for reverse complement orientation.
        - A list of tuples representing the aligned blocks. Each tuple contains
          the start and end positions of the alignment in the target and query sequences.

    Notes
    -----
    - The function uses local alignment (using blastn scoring parameters) to align the query to the target.
    - If the alignment score is less than half the length of the target, the function
      attempts to align the reverse complement of the query.

    """
    orientation = 1
    aligner = PairwiseAligner('blastn')
    aligner.mode = 'local'
    alignments = aligner.align(target, query)
    if (alignments[0].score < len(target) / 2):
        # If the score is low, try aligning the reverse complement
        query = revcomp(query)
        alignments = aligner.align(target, query)
        orientation = -1
    return orientation, alignments[0].aligned

def seq_from_df(df: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
                region: str, pos_col: str='allele_pos', ref_col: str='ref_allele') -> str:
    """
    Extract a DNA sequence from a DataFrame based on a specified region.

    Parameters
    ----------
    df : pandas.DataFrame, polars.DataFrame, or polars.LazyFrame
        The input DataFrame containing DNA sequence data. Must include columns for region, 
        position, and reference allele.
    region : str
        The name of the region for which the sequence will be extracted.
    pos_col : str, optional
        The name of the column containing the positions of the alleles (default is 'allele_pos').
    ref_col : str, optional
        The name of the column containing the reference alleles (default is 'ref_allele').

    Returns
    -------
    str
        The concatenated DNA sequence for the specified region.

    Raises
    ------
    ValueError
        If the input `df` is not a pandas.DataFrame, polars.DataFrame, or polars.LazyFrame.

    Notes
    -----
    - The function ensures that the data is sorted by region and position before extracting the sequence.
    - Redundant positions are properly handled.

    """
    if isinstance(df, pd.DataFrame):
        df = df[['region', pos_col, ref_col]].drop_duplicates().sort_values(['region', pos_col])
    elif isinstance(df, pl.DataFrame) or isinstance(df, pl.LazyFrame):
        df = (df.select(['region', pos_col, ref_col])
              .unique(subset=['region', pos_col, ref_col])
              .sort(['region', pos_col]))
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        df = df.to_pandas()
    else:
        raise ValueError("Input must be a DataFrame or LazyFrame.")
    return ''.join(df.loc[df['region'] == region, ref_col].values)

def map_to_mpra_region(gksvm_data: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
                       mpra_data: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
                       regions: list[str] | None = None,
                       ref_col: str = 'ref_allele',
                       alt_col: str = 'alt_allele',
                       pos_col: str = 'rel_pos',
                       effect_col: str = 'log2fc',
                       verbose: bool = False) -> pd.DataFrame:
    """
    Map relative positions and reference and alt alleles from gksvm data
    to MPRA regions by aligning sequences and adjusting positions.

    Parameters
    ----------
    gksvm_data : pandas.DataFrame, polars.DataFrame, or polars.LazyFrame
        The input DataFrame containing gksvm data. Must include columns for regions, 
        positions, reference alleles, and alternate alleles. *Note that if a
        pandas.DataFrame is provided, it will be updated in place.*
    mpra_data : pandas.DataFrame, polars.DataFrame, or polars.LazyFrame
        The input DataFrame containing MPRA data. Must include columns for regions, 
        positions, and reference alleles.
    regions : list of str or None, optional
        A list of regions to map. If None, all unique regions in `gksvm_data` will be used.
    ref_col : str, optional
        The name of the column containing reference alleles (default is 'ref_allele').
    alt_col : str, optional
        The name of the column containing alternate alleles (default is 'alt_allele').
    pos_col : str, optional
        The name of the column containing relative positions (default is 'rel_pos').
    effect_col : str, optional
        The name of the column containing effect sizes (default is 'log2fc').
    verbose : bool, optional
        If True, prints detailed information about the mapping process (default is False).

    Returns
    -------
    pandas.DataFrame
        The gksvm data with updated positions and alleles according to results of mapping
        (i.e., aligning) to the MPRA regions.

    Notes
    -----
    - The function aligns the gksvm sequences to the MPRA reference sequences using local alignment.
    - If the alignment indicates reverse orientation, the reference and alternate alleles are reverse-complemented.
    - Gaps in the alignment are removed, and relative positions are adjusted accordingly.

    """
    if isinstance(gksvm_data, pl.LazyFrame):
        gksvm_data = gksvm_data.collect()
    if isinstance(gksvm_data, pl.DataFrame):
        gksvm_data = gksvm_data.to_pandas()
    if regions is None:
        regions = gksvm_data['region'].unique()
    for region in regions:
        # Get the reference sequence for the region
        mpra_seq = seq_from_df(mpra_data, region)
        # Get the reference sequence for the gksvm data
        gksvm_seq = seq_from_df(gksvm_data, region, pos_col=pos_col, ref_col=ref_col)
        # Align the gksvm sequence to the reference sequence
        orientation, align_blocks = align_refs(mpra_seq, gksvm_seq)

        # If the orientation is -1, reverse complement the gksvm ref and alt alleles
        region_mask = gksvm_data['region'] == region
        if orientation == -1:
            if verbose:
                print(f"Region '{region}' found in reverse orientation, reverse-complenting.")
            gksvm_data.loc[region_mask, ref_col] = revcomp(gksvm_data.loc[region_mask, ref_col].values)
            gksvm_data.loc[region_mask, alt_col] = revcomp(gksvm_data.loc[region_mask, alt_col].values)
            gksvm_data.loc[region_mask, effect_col] = gksvm_data.loc[region_mask, effect_col].iloc[::-1]
        elif verbose:
            print(f"Region '{region}' found in forward orientation.")
        # if there are any gaps in the alignment, remove them
        pos_offset = 0
        if align_blocks.shape[1] > 1:
            if verbose:
                print(f"Region '{region}' has gaps in the alignment.")
            for i in range(align_blocks.shape[1]-1):
                ins_start, ins_end = (align_blocks[1,i,1], align_blocks[1,i+1,0])
                if verbose:
                    print(f"\tRemoving gap {ins_start}:{ins_end} ({ins_end-ins_start} base(s)).")
                gksvm_data = gksvm_data[~(region_mask &
                                          (gksvm_data[pos_col]+pos_offset >= ins_start) & 
                                          (gksvm_data[pos_col]+pos_offset < ins_end))]
                if verbose:
                    print(f"\tRe-enumerating relative positions >= {ins_end}.")
                gksvm_data.loc[(region_mask & (gksvm_data[pos_col] >= ins_end)), pos_col] -= (ins_end - ins_start)
                pos_offset += ins_end - ins_start
        elif verbose:
            print(f"Region '{region}' has no gaps in alignment.")
        # Add initial offset to the relative positions
        if verbose:
            print(f"Region '{region}' has initial offset {align_blocks[1,0,0]}, adjusting relative positions.")
        gksvm_data.loc[region_mask, pos_col] -= align_blocks[1,0,0]
    return gksvm_data