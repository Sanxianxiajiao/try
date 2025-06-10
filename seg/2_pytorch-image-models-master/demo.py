from scipy.special import comb
from scipy.stats import kendalltau
import numpy as np

np.arange(10)

def calc_worst_case_kdt(N):
    return (comb(N-1,2)-(N-1))/(comb(N,2))

def calc_worst_case_kdt_V2(N):
    return 1-4/N