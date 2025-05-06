#!/usr/bin/env python3

import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert a Parquet database to a delimited text file (default: TSV).")
    parser.add_argument("--input", "-i", help="Path to the input Parquet file")
    parser.add_argument("--output", "-o", help="Path to the output file")
    parser.add_argument("--delimiter", "-d", default="\t", help="Field delimiter (default: tab)")

    args = parser.parse_args()

    if not args.input or not args.output:
        parser.error("Input and output file paths are required.")
    
    df = pd.read_parquet(args.input)
    df.to_csv(args.output, sep=args.delimiter, index=False)

if __name__ == "__main__":
    main()