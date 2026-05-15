import numpy as np

"""
Can be used to penalize false correlations while preserving true systemic signals
to balance the covariance matrix of a portfolio 
"""

# NOTE WIP needs to calculate optimal Frobenious norm distance (compute_optimal_penalty)
def leodoit_wolf_shrinkage(X):
    T, N = X.shape
    sample_cov = np.cov(X, rowvar=False)
    target = np.diag(np.diag(sample_cov)) # Diagonal prior
    delta = compute_optimal_penalty(X, sample_cov)
    sigma_star = delta * target + (1 - delta) * sample_cov
    return sigma_star









