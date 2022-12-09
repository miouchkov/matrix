import numpy as np
import pandas as pd

from algorithms import get_svd_terms_full_ranks, spectral_initialization, perform_gradient_descent_iteration


def load_movie_data(ml_100k_fpath='data/ml-100k/u.data'):
    """
    Load the MovieLens 100K dataset.
    :param ml_100k_fpath:
    :return: a dataframe containing the MovieLens data
    """
    data = np.fromfile(ml_100k_fpath, sep='\t').reshape((100000, 4))
    df = pd.DataFrame(data, columns=['user_id', 'movie_id', 'rating', 'timestamp'])
    df['user_id'] -= 1
    df['movie_id'] -= 1
    return df


def create_small_incomplete_matrix(log=True):
    """
    Create a small incomplete matrix to be used for the convex optimization experiment.
    :param log: boolean indicating whether to print the matrix details
    :return: and incomplete matrix m and its complete counterpart m_true.
    """
    n1, n2 = 5, 5
    m = np.empty((n1, n2))
    for i in range(n1):
        for j in range(n2):
            m[i, j] = i * j

    m_true = m.copy()
    m[(0, 0)] = np.nan
    m[(3, 0)] = np.nan
    m[(2, 2)] = np.nan
    m[(3, 3)] = np.nan
    m[(1, 1)] = np.nan
    m[(3, 1)] = np.nan
    m[(2, 1)] = np.nan

    if log:
        omega = list(zip(*np.where(~np.isnan(m))))
        print('fraction complete: ', len(omega) / (n1 * n2))
        for i in range(n1):
            print_str = ''
            for j in range(n2):
                val = m[(i, j)]
                print_str += ('?' if np.isnan(val) else str(val)) + '\t'
            print(print_str)
    return m, m_true


def get_metrics_dict(z, m, m_test=None):
    """
    Compute and return a dictionary with the estimated complete matrix and train/test error.
    :param z: the estimated complete matrix (aka M_hat, or X.dot(Y.T) where X,Y are the left and right matrix factors).
    :param m: the incomplete matrix
    :param m_test: matrix to be used for test performance evaluation
    :return:
    """
    rank_results = {'M_hat': z}
    if m_test is not None:
        train_error = np.linalg.norm(np.nan_to_num(z - m), 'fro')
        test_error = np.linalg.norm(np.nan_to_num(z - m_test), 'fro')
        train_norm = np.linalg.norm(np.nan_to_num(m), 'fro')
        test_norm = np.linalg.norm(np.nan_to_num(m_test), 'fro')
        normalized_error_train = train_error / train_norm
        normalized_error_test = test_error / test_norm
        rank_results['train_error'] = normalized_error_train
        rank_results['test_error'] = normalized_error_test
    return rank_results


def matrix_factorization_multiple_ranks(m, ranks, m_test=None, alpha=0.001, max_iters=200):
    """
    Run the matrix factorization via gradient descent algorithm across several ranks.
    We initialize the matrix factors x,y via spectral initialization.
    We use a constant learning rate and a fixed number of iterations.
    :param m: the matrix to be completed
    :param ranks: list of ranks
    :param m_test: an optional test matrix to evaluate test performance
    :param alpha: learning rate
    :param max_iters: maximum iterations of gradient descent
    :return: a dictionary of information regarding the experiment
    """
    n1, n2 = m.shape
    omega = list(zip(*np.where(~np.isnan(m))))
    p = len(omega) / (n1 * n2)
    svd_terms = get_svd_terms_full_ranks(m)

    results_dict = {}
    for r in ranks:
        x, y = spectral_initialization(r, svd_terms)
        for iteration in range(max_iters):
            x, y = perform_gradient_descent_iteration(x, y, m, omega, p, alpha)
        z = x.dot(y.T)
        results_dict[r] = get_metrics_dict(z, m, m_test)
    return results_dict
