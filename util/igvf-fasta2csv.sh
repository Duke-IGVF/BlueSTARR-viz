#!/bin/bash

## Takes as input fasta sequence file where the identifier line looks like this:
## >1234 /chr22:12345678-87654321
## where the identifier is an index into some master data table with a row for this
## sequence bin as identified by chromosome, start and end
##
## Outputs CSV with 4 columns (ID, chromosome, start pos, end pos)
##
## Input has to be on stdin

grep "^>" | sed -e 's/>\([0-9]*\).*\(chr[0-9XYM]*\):\([0-9]*\)-\([0-9]*\)/\1,\2,\3,\4/'
