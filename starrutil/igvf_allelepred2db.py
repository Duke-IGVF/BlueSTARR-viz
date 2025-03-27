import numpy as np
import pandas as pd
import polars as pl
import duckdb
from pathlib import Path
from typing import Callable, Optional
import argparse

def read_pred_table(pred_file: Path | str,
                    convFC: Optional[Callable[[float], float]] = lambda x: x/np.log(2),
                    FC_name: str = "log2FC") -> pd.DataFrame:
    """
    Reads a prediction table from a file and processes it. The file is expected to
    have the following format:

    ```
    chr4:156105483-156105721 chr4:156105452-156105752 pos=156105483 ref=G G -0.32947034
    ```

    The columns contain the following information:
    - cCRE region in `chr#:start-end` genome region notation
    - sequence bin used for prediction; centered on cCRE and length based on what model was trained with (same genome region notation)
    - `pos=NNNNNNN`: chromosome position being mutated
    - `ref=X`: reference genome allele (=X) at this position
    - `Y`: the allele being tested for the sequence bin
    - the activation value predicted for the sequence bin with allele `Y` in the position being mutated

    Parameters
    ----------
    pred_file : Path or str
        Path to the input file containing allele predictions.
    convFC : Callable[[float], float], optional
        Function to convert the fold change values (default is
        lambda x: x/np.log(2), to convert values from log to log2).
    FC_name : str, optional
        Name of the fold change column (default is "log2FC").

    Returns
    -------
    pd.DataFrame
        Processed prediction table with columns for chromosome, start, end, allele
        position, reference allele, allele, and fold change values.
    """
    # last column is FC, which can be neg and thus have a minus sign
    pred_vals = pd.read_csv(pred_file, sep="\t", index_col=False, header=None,
                            usecols=[5], names=[FC_name])
    # convert as desired
    if convFC is not None:
        pred_vals[FC_name] = convFC(pred_vals[FC_name])
    # get the remaining columns and merge
    cols = ["cre_chrom", "cre_start", "cre_end",
            "bin_chrom", "bin_start", "bin_end",
            "allele_pos", "ref_allele", "allele"]
    pred_table = pd.read_csv(pred_file, sep="\t|:|=|-", index_col=False, header=None,
                             usecols=[0, 1, 2, 3, 4, 5, 7, 9, 10], names=cols,
                             engine="python")
    pred_table = pred_table.merge(pred_vals, left_index=True, right_index=True)
    # compute relative activation of allele to ref
    refFC = pred_table.loc[pred_table["ref_allele"] == pred_table["allele"], [FC_name]]
    refFC = refFC.loc[refFC.index.repeat(4)].reset_index(drop=True)
    pred_table["FC_to_ref"] = pred_table[FC_name] - refFC[FC_name]
    # done
    return pred_table

def read_pred_table2(source: Path | str,
                     sep: str = "\t",
                     glob: bool = True) -> pl.LazyFrame:
    """
    Reads prediction table(s) from file(s) and processes it. The file(s) is/are expected to
    be in the "processed" format, with the following columns (order does not matter):
    
    - cre_chrom: chromosome of the cCRE region (renamed to chrom in output
      unless named differently)
    - cre_start: start position of the cCRE region
    - cre_end: end position of the cCRE region
    - SPDI: the SPDI notation of the mutation
    - effect: the predicted fold change value of the mutation compared to reference
      (renamed to log2FC in output unless named differently)

    Parameters
    ----------
    source : Path or str
        Path to the input file(s) containing allele predictions. This can be a single file or a glob pattern.
    sep : str, optional
        The delimiter to use for parsing the input file(s) (default is "\t").
    glob : bool, optional
        If True, treats the source as a glob pattern (default is True).

    Returns
    -------
    pl.LazyFrame
        Processed prediction table with columns for chromosome, cre_start, cre_end, allele
        position, reference allele, allele, and fold change values.
    """
    pred_table = pl.scan_csv(source, separator=sep, glob=glob)
    return pred_table.with_columns(
        pl.col('SPDI').str.split(':')
        .list.slice(1)
        .list.to_struct(fields=['allele_pos','ref_allele','allele'])
    ).unnest('SPDI').rename({'effect': 'log2FC', 'cre_chrom': 'chrom'}).with_columns(
        pl.col('allele_pos').cast(pl.UInt32)
    )

def allele_preds2db(source: Path | str,
                    db_file: Path | str,
                    sep: str = "\t",
                    column_map: dict[str, str] = {
                        "chrom": "cre_chrom", 
                        "cre_start": "cre_start",
                        "cre_end": "cre_end",
                        "log2FC": "effect",
                        "SPDI": [None, "allele_pos", "ref_allele", "allele"]
                    },
                    compression: str = "LZ4",
                    partition_by: str = "chrom",
                    verbose: bool = False, **kwargs) -> None:
    """
    Converts a file with allele predictions to a Parquet database. The database
    will be partitioned by chromosome.

    The file(s) is/are expected to be in the "processed" format, with the following
    columns (order does not matter, but names must match the values of the column_map):
    
    - cre_chrom: chromosome of the cCRE region
    - cre_start: start position of the cCRE region
    - cre_end: end position of the cCRE region
    - SPDI: the SPDI notation of the mutation (name must match the last key of column_map)
    - effect: the predicted fold change value of the mutation compared to reference

    Parameters
    ----------
    source : Path or str
        Path to the input file(s) containing allele predictions in the "processed" format.
        This can be a single file or a glob pattern.
    db_file : Path or str
        Path to the output Parquet database file. An already existing database will
        be deleted rather than appended to.
    sep : str, optional
        The delimiter to use for parsing the input file(s) (default is "\t").
    column_map : dict[str, str], optional
        A mapping of database column names to input file column names Parquet
        database, except for the last key, which is the name of the SPDI column
        in the input files, and as value has an array of database column names
        for the components.
    compression : str, optional
        The compression algorithm to use for the Parquet database (default is "LZ4").
    partition_by : str, optional
        The column to partition the database by (default is "chrom").
    verbose : bool, optional
        If True, prints progress information (default is False).
    **kwargs : dict
        Keyword arguments to replace individual keys in the column map.

    Returns
    -------
    None
    """
    if verbose:
        duckdb.execute("PRAGMA enable_progress_bar;")
        print(f"Parsing/loading input files: {source}")

    if kwargs:
        bad_keys = set(kwargs.keys()) - set(column_map.keys())
        if bad_keys:
            raise ValueError(f"Invalid keys in kwargs: {bad_keys}")
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        column_map.update(kwargs)
        if verbose:
            print(f"Updated column map: {column_map}")
    SPDI = list(column_map.keys())[-1]
    preds = duckdb.sql(
        f"select {column_map['chrom']} as chrom, " +
        f"{column_map['cre_start']} as cre_start, {column_map['cre_end']} as cre_end, " +
        f"{column_map['log2FC']} as log2FC, " +
        f"cast(string_split({SPDI}, ':')[2] as int64) as {column_map[SPDI][1]}, " +
        f"string_split({SPDI}, ':')[3] as {column_map[SPDI][2]}, " +
        f"string_split({SPDI}, ':')[4] as {column_map[SPDI][3]} " +
        f"from read_csv('{source}', delim='{sep}')"
    )
    if verbose:
        print("Collecting input data and writing to Parquet database")
    partition_spec = ""
    if partition_by:
        if verbose:
            print(f"Partitioning by column {partition_by}")
        partition_spec = f"PARTITION BY ({partition_by}), FILENAME_PATTERN '{uuid}-part{i}', "
    elif verbose:
        print("No partitioning by column")
    duckdb.sql(
        f"COPY preds TO '{db_file}' " +
        f"(FORMAT parquet, {partition_spec}" +
        f"OVERWRITE, COMPRESSION '{compression}')"
    )

def _main():
    parser = argparse.ArgumentParser(
        description="""
        Converts file(s) with mutation effect predictions to a Parquet database as output.
        The file(s) is/are expected to be in the "processed" format, with columns
        cre_chrom (chromosome of the cCRE region), cre_start (start position of the cCRE region),
        cre_end (end position of the cCRE region), SPDI (the SPDI notation of the mutation),
        and effect (the predicted log2 fold change value of the mutation compared to reference).
        Columns can be in any order.
        """
    )
    parser.add_argument('--input', '-i', required=True, help='Path to the input file(s), can be a glob pattern')
    parser.add_argument('--db', required=True, help='Path to the Parquet database (directory if partitioned)')
    parser.add_argument('--sep', default='\t', help='Delimiter for the input file (default is tab)')
    parser.add_argument('--compression', default='LZ4', help='Compression algorithm for the Parquet database (default is LZ4)')
    parser.add_argument('--partition-by', default='chrom', help='Column to partition the database by (default is chrom)')
    parser.add_argument('--chrom', default=None, help='Name of the chromosome column in the input file if different from cre_chrom')
    parser.add_argument('--cre-start', default=None, help='Name of the cCRE start column in the input file if different from cre_start')
    parser.add_argument('--cre-end', default=None, help='Name of the cCRE end column in the input file if different from cre_end')
    parser.add_argument('--effect', default=None, help='Name of the effect column in the input file if different from effect')
    parser.add_argument('--verbose', '-v', action='store_true', help='Prints progress information')
    args = parser.parse_args()
    
    allele_preds2db(args.input, args.db,
                    sep=args.sep, 
                    compression=args.compression,
                    partition_by=args.partition_by,
                    verbose=args.verbose,
                    chrom=args.chrom,
                    cre_start=args.cre_start,
                    cre_end=args.cre_end,
                    log2FC=args.effect)

if __name__ == "__main__":
    _main()
