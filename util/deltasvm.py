# Adapted from the following script:
# https://www.beerlab.org/deltasvm_models/downloads/score_snp_seq.py
#
# If you use this code, please follow the citation instructions here:
# https://www.beerlab.org/deltasvm_models/
#
# Specifically, cite the following papers:
# - Shigaki D, Adato O, Adhikar A, Dong S, Hawkins-Hooker A, Inoue F,
#   Juven-Gershon T, Kenlay H, Martin B, Patra A, Penzar D, Schubach M,
#   Xiong C, Yan Z, Boyle A, Kreimer A, Kulakovskiy IV, Reid J, Unger R,
#   Yosef N, Shendure J, Ahituv N, Kircher M, and Beer MA.
#   Human Mutation 2019. doi:10.1002/humu.23797
# - Lee D, Gorkin DU, Baker M, Strober BJ, Asoni AL, McCallion AS, Beer MA. 2015.
#   A method to predict the impact of regulatory variants from DNA sequence.
#   Nat Genet. 2015 doi:10.1038/ng.3331
# 
import sys
from collections.abc import Iterable
from collections import namedtuple
import pandas as pd
import argparse

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


def invert(str):
    str = str.upper()
    return revcomp(str)

def load_pretrained(wtfile: str) -> dict:
    """
    Load a pre-trained gkSVM model from a file.

    Parameters
    ----------
    wtfile : str
        Path to the file containing the pre-trained model weights.

    Returns
    -------
    dict
        A dictionary with k-mers as keys and their weights as values.
        The k-mer is stored in both its original and reverse complement forms.

    """
    infile = open(wtfile, "r")
    wt = {}
    for line in infile:
        f = line.strip().split()
        wt[f[0]] = float(f[1])
        wt[invert(f[0])] = float(f[1])
    infile.close()
    return wt

def score_seq(seq: str, model: dict, k: int=0) -> float:
    """
    Score a sequence using a pre-trained model.

    Parameters
    ----------
    seq : str
        The DNA sequence to score.
    model : dict
        A dictionary containing the pre-trained model weights, with k-mers as keys
        and their corresponding scores as values.
    k : int, optional
        The length of the k-mers to be used for scoring. If not provided or set to 0,
        it will be inferred from the model.

    Returns
    -------
    float
        The score for the input sequence.

    """
    if not k:
        k = len(next(iter(model)))
    score = 0

    for i in range(0, len(seq) - k + 1):
        a = seq[i:i + k]
        score += model[a]

    return score

def sliding_subseqs(seq: str, k: int):
    """
    Generate sliding windows of size k*2-1 over a string sequence.

    Parameters
    ----------
    seq : str
        The input string to process.
    k : int
        The k-mer size of the model. The actual window size will be k*2-1,
        so that sliding k-mers over the window will each include the center
        (at position k, 0-based).

    Yields
    ------
    str
        The current window of size k*2-1.

    """
    window_size = k * 2 - 1

    if len(seq) < window_size:
        raise ValueError(f"Sequence length must be at least {window_size} characters.")
    
    # Loop over all possible starting positions
    for i in range(len(seq) - window_size + 1):
        # Extract and yield the current window
        window = seq[i:i + window_size]
        yield window

def deltaSVM_saturated_mutations(seq: str,
                                 model: dict, k: int=0,
                                 start_pos: int=0,
                                 pos_col: str='pos',
                                 ref_col: str='ref',
                                 alt_col: str='alt',
                                 score_col: str='score') -> Iterable[tuple]:
    """
    Generate all possible single-nucleotide mutations for a given sequence,
    using a sliding window of size k*2-1, and calculate the delta SVM scores.

    Note that the first k-1 bases and the last k-1 bases of the sequence
    will not be scored, because the scored mutation must be at the center of
    a window of size k*2-1.

    Parameters
    ----------
    seq : str
        The input DNA sequence.
    model : dict
        A dictionary containing the pre-trained model weights, with k-mers as keys
        and their corresponding scores as values.
    k : int
        The k-mer size for the model. If not provided or set to 0,
        it will be inferred from the model.
    start_pos : int
        The starting position for the mutations. Default is 0, i.e., positions
        in the generated records are relative to the start of the sequence.
    pos_col : str
        The name of the column for the position in the output. Default is 'pos'.
    ref_col : str
        The name of the column for the reference base in the output. Default is 'ref'.
    alt_col : str
        The name of the column for the alternate base in the output. Default is 'alt'.
    score_col : str
        The name of the column for the delta SVM score in the output. Default is 'score'.

    Yields
    ------
    tuple
        A named tuple containing the position, reference base, alternate base,
        and deltaSVM score for each mutation.

    """
    if not k:
        k = len(next(iter(model)))
    rowtuple = namedtuple('rowtuple', [pos_col, ref_col, alt_col, score_col])
    for pos, subseq in enumerate(sliding_subseqs(seq, k), start=start_pos):
        for base in "ACGT":
            if subseq[k-1] != base:
                mutated = subseq[:k-1] + base + subseq[k:]
                deltaSVM = score_seq(mutated, model, k) - score_seq(subseq, model, k)
                yield rowtuple(pos+k-1, subseq[k-1], base, deltaSVM)

def main():
    parser = argparse.ArgumentParser(description="Calculate deltaSVM scores for all mutations of a sequence.")
    parser.add_argument("--seq", "-s", help="DNA sequence")
    parser.add_argument("--model", "-m", help="Path to the pre-trained model weights file")

    args = parser.parse_args()
    seq = args.seq
    model = load_pretrained(args.model)
    k = len(next(iter(model)))

    if len(seq)<2*k-1 :
        raise ValueError(f"seq should be at least {2*k-1}bp to accurately score with {k}-mer weights")

    df = pd.DataFrame(deltaSVM_saturated_mutations(seq, model, k))
    df.to_csv(sys.stdout, sep="\t", index=False, header=True)

if __name__ == '__main__':
    main()
