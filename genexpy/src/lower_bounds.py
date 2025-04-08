import numpy as np

def sample_mean_embedding_lowerbound(eps: float, n: int, kbar: float, v: float) -> float:
    """
    Lower bound on the distance in the RKHS between the sample mean embedding and the real mean embedding [1, Thm 2.1].
    [1] Wolfer, Geoffrey, and Pierre Alquier. "Variance-Aware Estimation of Kernel Mean Embedding."
        arXiv preprint arXiv:2210.06672 (2022).

    :param eps: how close the sample mean embedding and the real mean embedding should be
    :type eps: float
    :param n: sample size
    :type n: int
    :param kbar: superior of the kernel
    :type kbar: float
    :param v: variance of k in the RKHS
    :type v: float
    :return: the lower bound
    :rtype: float
    """
    return 1 - np.exp(- n / (8/3 * np.sqrt(kbar))**2 * (np.sqrt(2*v + 16/3 * eps * np.sqrt(kbar)) - np.sqrt(2*v))**2)


def generalizability_lowerbound_base(eps: float, n: int, kbar: float, v: float) -> float:
    """
    Lower bound on the distance in the RKHS between the sample mean embedding and the real mean embedding [1, Thm 2.1].
    [1] Wolfer, Geoffrey, and Pierre Alquier. "Variance-Aware Estimation of Kernel Mean Embedding."
        arXiv preprint arXiv:2210.06672 (2022).

    :param eps: how close the sample mean embedding and the real mean embedding should be
    :type eps: float
    :param n: sample size
    :type n: int
    :param kbar: superior of the kernel
    :type kbar: float
    :param v: variance of k in the RKHS
    :type v: float
    :return: the lower bound
    :rtype: float
    """
    return 1 - np.exp(- n / (16/3 * np.sqrt(kbar))**2 * (np.sqrt(8*v - 32/3 * eps * np.sqrt(kbar)) - np.sqrt(8*v))**2)


# TODO check what happens for eps > 3/4 * v/sqrt(kbar)
# TODO check if the variance is the correct one
def generalizability_lowerbound(eps: float, n: int, kbar: float, v: float) -> float:
    """
    Lower bound on the distance in the RKHS between the sample mean embedding and the real mean embedding [1, Thm 2.1].
    [1] Wolfer, Geoffrey, and Pierre Alquier. "Variance-Aware Estimation of Kernel Mean Embedding."
        arXiv preprint arXiv:2210.06672 (2022).

    :param eps: how close the sample mean embedding and the real mean embedding should be
    :type eps: float
    :param n: sample size
    :type n: int
    :param kbar: superior of the kernel
    :type kbar: float
    :param v: variance of k in the RKHS
    :type v: float
    :return: the lower bound
    :rtype: float
    """

    return 1 - np.exp(- 9*n / (16*kbar) * (v - 2/3*np.sqrt(kbar)*eps - np.sqrt(v)*np.sqrt(v-4/3*np.sqrt(kbar)*eps)))


def mmd_lowerbound(eps: float, n: int, kbar: float):
    """
    Calculate the lower bound probability for the absolute difference between
    empirical MMD and true MMD to be within specified bounds.

    Parameters:
    n (int): Sample size
    K (float): Upper bound for the kernel function
    epsilon (float): Tolerance level for deviation

    Returns:
    float: Lower bound for the difference between MMD and empirical MMD
    """
    return 1 - np.exp(- (eps - (2 * kbar / n) ** 0.5)**2 * n / (4*kbar)) #if eps > (2*kbar / n)**0.5 else np.nan


