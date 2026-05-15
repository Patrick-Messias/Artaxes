"""
https://www.youtube.com/watch?v=7zVG8w4u8tQ
Statistical Arbitrage via Principal Component Analysis (PCA)


"""


import numpy as np


# Can be used to penalize false correlations while preserving true systemic signals
#to balance the covariance matrix of a portfolio 

# NOTE WIP needs to calculate optimal Frobenious norm distance (compute_optimal_penalty)
def leodoit_wolf_shrinkage(X):
    T, N = X.shape
    sample_cov = np.cov(X, rowvar=False)
    target = np.diag(np.diag(sample_cov)) # Diagonal prior
    delta = compute_optimal_penalty(X, sample_cov)
    sigma_star = delta * target + (1 - delta) * sample_cov
    return sigma_star


# Lasgest eigenvals represent the principal components
def extract_principal_components(sigma_star):
    #Optimized Eigen-Decomposition for symmetric matrices
    eigenvals, eigenvecs = np.linalg.eigh(sigma_star)

    # Mathematical vectors are not sorted by default
    idx_desc = np.argsort(eigenvals)[::-1]

    sorted_eigenvals = eigenvals[idx_desc]
    sorted_eigenvecs = eigenvecs[:,idx_desc]

    return sorted_eigenvals, sorted_eigenvecs


# Divide the exposure, garanteeing absolute dollar-neutrality and capping leverage
def build_eigen_portfolios(eigenvecs, vols):
    # Element-wise division by asset vol
    raw_weights = eigenvecs / vols[:,np.newaxis]

    # Normalize to 1.0 gross exposure (Dollar Neutrality)
    gross_exposure = np.sum(np.abs(raw_weights), axis=0)
    eigen_portfolios = raw_weights / gross_exposure

    return eigen_portfolios

# Filters the sp500 matrix into pure idiosyncratic signals
def extract_residuals(X, eigenvecs, k_components):
    # Isolate the top K dominant macro factors
    V_k = eigenvecs[:, :k_components]

    # Compute systemic reconstruction (Subspace projection)
    X_sys = X @ V_k @ V_k.t

    # Subtract systemic drift to isolate idiosyncrasies
    epsilon = X - X_sys

    return epsilon

# Z-Score Normalization
def normalize_zscore(epsilon_matrix):
    # Calculates historical parameters along the time axis
    mu = np.mean(epsilon_matrix, axis=0)
    sigma = np.std(epsilon_matrix, axis=0)

    # Executes normalized signal generation
    z_scores = (epsilon_matrix - mu) / sigma

    return z_scores

# Rolling
def execute_rolling_pca(X, window, k_factors):
    T, N = X.shape
    z_scores = np.zeros_like(X)

    # Iterates point-in-time execution
    for t in range(window, T):
        X_local = X[t-window:t, :] # slice historical window
        cov_t = leodoit_wolf_shrinkage(X_local)
        vals, vecs = np.linalg.eigh(cov_t)
        idx = np.argsort(vals)[::-1]
        V_k = vecs[:, idx][:, :k_factors]



















