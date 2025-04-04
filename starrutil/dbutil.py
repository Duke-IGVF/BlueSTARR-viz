import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import duckdb

def relation_to_pq(rel: pd.DataFrame | pa.Table | duckdb.DuckDBPyRelation,
                   path: str,
                   partition_cols: list[str] = None,
                   existing_data_behavior: str = 'overwrite_or_ignore',
                   **kwargs) -> None:
    """
    Write a DuckDB relation or Arrow Table to a Parquet file or partitioned dataset.

    Parameters
    ----------
    rel : pd.DataFrame or pa.Table or duckdb.DuckDBPyRelation
        The input relation, either a DuckDB relation or an Arrow Table.
    path : str
        The file path or directory where the Parquet file or dataset will be written.
    partition_cols : list of str, optional
        A list of columns to partition the dataset by. Default is None, which results in a single Parquet file.
    existing_data_behavior : str, optional
        Behavior when data already exists at the path. Default is 'overwrite_or_ignore'.
    **kwargs : 
        Additional keyword arguments passed to the Parquet writer.

    Returns
    -------
    None
    """
    if isinstance(rel, pd.DataFrame):
        rel = pa.Table.from_pandas(rel)
    elif not isinstance(rel, pa.Table):
        rel = rel.to_arrow_table()
    if partition_cols is None:
        pq.write_table(rel, path, **kwargs)
    else:
        pq.write_to_dataset(rel, 
                            root_path=path, 
                            partition_cols=partition_cols, 
                            existing_data_behavior=existing_data_behavior,
                            **kwargs)
        
def motif_hits_to_dataset(starr_data: duckdb.DuckDBPyRelation,
                          motifs_db: duckdb.DuckDBPyRelation,
                          verbose: bool = False) -> duckdb.DuckDBPyRelation:
    """
    Generate a DuckDB relation of hits from a database to motifs to a
    STARR-seq dataset. The hits are generated by joining the starr_data relation
    with the motifs_db relation.

    Compared to a single SQL query, this function is much more efficient by
    processing the join for one chromosome at a time. The efficiency gain
    assumes that at least the starr_data relation is partitioned
    by chromosome.

    Parameters
    ----------
    starr_data : duckdb.DuckDBPyRelation
        The DuckDB relation containing the data with sequence bins and their activations.
    motifs_db : duckdb.DuckDBPyRelation
        The DuckDB relation containing the motif match data from a genome-wide scan.
    verbose : bool, optional
        If True, prints progress information. Default is False.

    Returns
    -------
    duckdb.DuckDBPyRelation
        A DuckDB relation containing the resulting database of motif hits.
    """
    aliases = ('d', 'm')
    project_expr = (
        'd.chrom, d.start as seq_start, d.end as seq_end, '
        'm.motif_id, m.motif_alt_id, '
        'regexp_extract(m.motif_alt_id, \'\.([^\.]+)$\', 1) as motif_name, '
        'm.start as mot_start, m.stop as mot_stop, d.log2FC'
    )
    join_expr = 'd.start <= m.start and d.end >= m.stop'
    return partition_join(starr_data, motifs_db,
                          partition_col='chrom',
                          join_expr=join_expr,
                          aliases=aliases,
                          project_expr=project_expr,
                          verbose=verbose)


def partition_join(rel1: duckdb.DuckDBPyRelation,
                   rel2: duckdb.DuckDBPyRelation,
                   partition_col: str,
                   join_expr: str,
                   aliases: tuple[str, str] = None,
                   project_expr: str = '*',
                   verbose: bool = False) -> duckdb.DuckDBPyRelation:
    """
    Perform a join of two relations by performing it one partition at a time,
    followed by concatenating the results. This can be substantially more
    efficient than a regular join when at least one of the relations is very
    large and partitioned by the column to partition the join by.

    If the second relation contains the partition column, it
    is filtered concordantly, meaning the partition column effectively becomes
    a join column, whether explicitly included in the join expression or not.

    Parameters
    ----------
    rel1 : duckdb.DuckDBPyRelation
        The first relation to join. Expected to contain all unique values of
        the specified partition column.
    rel2 : duckdb.DuckDBPyRelation
        The second relation to join. If it, too, contains the partition column,
        the partition column effectively becomes an additional join column.
    partition_col : str
        The column to partition the join by.
    join_expr : str
        The join expression. If it uses aliases for the relations different
        from their default aliases, they must be specified in the aliases parameter.
    aliases : tuple[str, str], optional
        The aliases for the first and second relation, respectively, that are
        being used in the join and/or project expression(s). Default is the
        default aliases of the input relations.
    project_expr : str, optional
        The projection expression (in essence, the SELECT clause). Default is '*'.
    verbose : bool, optional
        If True, prints progress information. Default is False.

    Returns
    -------
    duckdb.DuckDBPyRelation
        A DuckDB relation containing the result of the partitioned join.
    """
    partition_values = rel1.unique(partition_col).df().values.squeeze()
    rel = None
    for partition_value in partition_values:
        if verbose:
            print(f"Running query for {partition_col}='{partition_value}'")
        if aliases is None:
            aliases = (rel1.alias, rel2.alias)
        else:
            rel1 = rel1.set_alias(aliases[0])
            rel2 = rel2.set_alias(aliases[1])
        sql = (
            "SELECT " + project_expr + 
            f" FROM rel1 AS {aliases[0]} JOIN rel2 AS {aliases[1]} ON ({join_expr})" +
            f" WHERE {aliases[0]}.{partition_col} = ?"
        )
        params = [partition_value]
        if partition_col in rel2.columns:
            sql = sql + f" AND {aliases[1]}.{partition_col} = ?"
            params.append(partition_value)
        r = duckdb.sql(sql, params=params)
        if rel is None:
            rel = r
        else:
            rel = rel.union(r)
    return rel
