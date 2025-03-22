import duckdb
import numpy as np
from scipy import interpolate

def bonferroni_approx_cutoff(rel, alpha: float = 0.05) -> float:
    """
    Calculate the approximate Bonferroni correction cutoff for significance.

    The number of tests is approximated as the number of rows in the relation.
    The returned p-value is an approximate cutoff because it is truncated
    (not rounded) to the nearest power of 10.

    Parameters
    ----------
    rel : 
        A DuckDB-queryable relation containing the data.
    alpha : float, optional
        The significance level. Default is 0.05.

    Returns
    -------
    float
        The minimum p-value needed for significance after Bonferroni correction.
    """
    # obtain the number of rows in the relation as a number
    num_matches = duckdb.sql(f'select count(*) from rel').fetchall()[0][0]

    if num_matches == 0:
        raise ValueError("The relation contains no rows.")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Significance level 'alpha' must be between 0 and 1.")
        
    # obtain the minimum p-value needed for significance after Bonferroni correction
    p_min = alpha / num_matches
    p_min = 10**int(np.log10(p_min))
    return p_min

def qvalue_estimate(pv: np.ndarray, 
                    m: int = None,
                    verbose: bool = False,
                    lowmem: bool = False,
                    pi0: float = None) -> np.ndarray:
    """
    Estimates q-values from p-values. Returns a vector of q-values of the
    same shape as the input p-values.

    Original implementation is from Nicolò Fusi in 2012 (https://github.com/nfusi/qvalue),
    under a BSD-3-Clause license. This version is a Python 3 adaptation by Ryan Neff in 2018
    (https://gist.github.com/ryananeff/0232597b04ec1e5947de2ad8b9292d6e).

    Parameters
    ----------
    pv : np.ndarray
        The p-values for which to estimate q-values.
    m : int, optional
        Number of tests. If not specified, m = pv.size.
    verbose : bool, optional
        Print verbose messages? Default is False.
    lowmem : bool, optional
        Use memory-efficient in-place algorithm. Default is False.
    pi0 : float, optional
        If None, it's estimated as suggested in Storey and Tibshirani, 2003.
         For most GWAS this is not necessary, since pi0 is extremely likely to be 1.
    
    Returns
    -------
    np.ndarray
        q-values of the same shape as the input p-values.
    """
    assert(pv.min() >= 0 and pv.max() <= 1), "p-values should be between 0 and 1"

    original_shape = pv.shape
    pv = pv.ravel()  # flattens the array in place, more efficient than flatten()

    if m is None:
        m = float(len(pv))
    else:
        # the user has supplied an m
        m *= 1.0

    # if the number of hypotheses is small, just set pi0 to 1
    if len(pv) < 100 and pi0 is None:
        pi0 = 1.0
    elif pi0 is not None:
        pi0 = pi0
    else:
        # evaluate pi0 for different lambdas
        pi0 = []
        lam = np.arange(0, 0.95, 0.01)
        counts = np.array([(pv > i).sum() for i in np.arange(0, 0.95, 0.01)])
        for l in range(0,len(lam)):
            pi0.append(counts[l]/(m*(1-lam[l])))

        pi0 = np.array(pi0)

        # fit natural cubic spline
        tck = interpolate.splrep(lam, pi0, k=3)
        pi0 = interpolate.splev(lam[-1], tck)
        if verbose:
            print("qvalues pi0=%.3f, estimated proportion of null features " % pi0)

        if pi0 > 1:
            if verbose:
                print("got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0)
            pi0 = 1.0

    assert(pi0 >= 0 and pi0 <= 1), "pi0 is not between 0 and 1: %f" % pi0

    if lowmem:
        # low memory version, only uses 1 pv and 1 qv matrices
        qv = np.zeros((len(pv),))
        last_pv = pv.argmax()
        qv[last_pv] = (pi0*pv[last_pv]*m)/float(m)
        pv[last_pv] = -np.inf
        prev_qv = last_pv
        for i in range(int(len(pv))-2, -1, -1):
            cur_max = pv.argmax()
            qv_i = (pi0*m*pv[cur_max]/float(i+1))
            pv[cur_max] = -np.inf
            qv_i1 = prev_qv
            qv[cur_max] = min(qv_i, qv_i1)
            prev_qv = qv[cur_max]

    else:
        p_ordered = np.argsort(pv)
        pv = pv[p_ordered]
        qv = pi0 * m/len(pv) * pv
        qv[-1] = min(qv[-1], 1.0)

        for i in range(len(pv)-2, -1, -1):
            qv[i] = min(pi0*m*pv[i]/(i+1.0), qv[i+1])

        # reorder qvalues
        qv_temp = qv.copy()
        qv = np.zeros_like(qv)
        qv[p_ordered] = qv_temp

    # reshape qvalues
    qv = qv.reshape(original_shape)

    return qv
