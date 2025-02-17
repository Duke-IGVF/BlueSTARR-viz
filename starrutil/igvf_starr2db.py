import argparse
from pathlib import Path
import pandas as pd
from dbutil import relation_to_pq

def starr2db(input_file: Path | str, db_file: Path | str, verbose: bool = False) -> None:
    if verbose:
        print(f"Parsing/loading input file: {input_file}")
    if isinstance(input_file, Path):
        input_file = str(input_file)
    compression = 'gzip' if input_file.endswith('.gz') else None
    df = pd.read_csv(input_file, compression=compression, delimiter="\t")
    if verbose:
        print("Processing input file (column names, index, etc)")
    df.columns = (
        df.columns
        .str.replace('^(?!input).*_rep', 'output_rep', regex=True)
        .str.replace('^.*_log2FC', 'log2FC', regex=True)
    )
    # keep the index as it may be used as identifier
    df.reset_index(inplace=True, drop=False)
    if verbose:
        print(f"Writing to Parquet database: {db_file}")
    relation_to_pq(df, db_file, partition_cols=['chrom'], existing_data_behavior='delete_matching')

def main():
    parser = argparse.ArgumentParser(
        description='Converts a file with STARRseq sequence bins and their activations as input file to a Parquet database as output.')
    parser.add_argument('--input', '-i', required=True, help='Path to the input file')
    parser.add_argument('--db', required=True, help='Path to the Parquet database (directory)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Prints progress information')
    
    args = parser.parse_args()
    
    starr2db(args.input, args.db, args.verbose)

if __name__ == "__main__":
    main()
