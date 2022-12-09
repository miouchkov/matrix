import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.vstack import vstack


def mean_squared_error(x, y):
    # Mean Squared Error (MSE)
    if isinstance(x, cp.expressions.variable.Variable):
        mse = np.sum((y - x.value) ** 2)
    else:
        mse = np.sum((y - x) ** 2)
    return mse


def nuclear_norm_convex_optimization(m):
    """
    Given an incomplete matrix m (a sparse matrix where most entries are nan), run SDP that is equivalent to a nuclear
    norm convex optimization to find the completed matrix x. Return the relevant problem variables.
    :param m: the matrix to be completed
    :return:
    problem: the cvxpy problem
    x: the optimized complete matrix
    w1, w2: the two matrices used in formulating the SDP
    """
    n1, n2 = m.shape
    omega = list(zip(*np.where(~np.isnan(m))))
    x = cp.Variable((n1, n2))
    w1 = cp.Variable((n1, n1))
    w2 = cp.Variable((n2, n2))

    constraints = []
    for (user_id, movie_id) in omega:
        # add constraint that X_ij == M_ij where i,j is an observed entry of M
        constraints.append(x[user_id, movie_id] == m[(user_id, movie_id)])

    # add constraint that matrix [[W1, X], [X, W2]] is positive semi-definite
    row1 = hstack([w1, x])
    row2 = hstack([cp.transpose(x), w2])
    matrix_constraint = vstack([row1, row2]) >> cp.Constant(0)
    constraints.append(matrix_constraint)

    # Objective function: choose W1,W2 to minimize 1/2 * (trace(W1) + trace(W2))
    objective = cp.Minimize(cp.Constant(0.5) * (cp.trace(w1) + cp.trace(w2)))
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return problem, x, w1, w2


def get_nanfilled_matrix(m):
    """
    Fill in each nan entry of m, using the mean of the entries row as the fill-in value.
    :param m: a matrix where some entries are nan values.
    :return: m_nanfilled: The matrix m with nan entries filled.
    """
    # Replace each nan entry with the mean of its row
    m_nanfilled = m.copy()
    row_means = np.nanmean(m, axis=1)
    for i in range(m.shape[0]):
        m_nanfilled[i, :] = np.nan_to_num(m[i, :], nan=row_means[i])
    return m_nanfilled


def get_svd_terms_full_ranks(m):
    """
    Find the full SVD of m. Some entries of m may be nans, so we first fill them in as described in get_nanfilled_matrix
    :param m: a matrix (with potential nan values)
    :return: u_full, s_full, vh_full: the full SVD of m (after nan-filling)
    """
    # Create SVD decomposition. Fill nans before performing SVD
    m_nanfilled = get_nanfilled_matrix(m)
    # Get full rank SVD
    u_full, s_full, vh_full = np.linalg.svd(m_nanfilled)
    return u_full, s_full, vh_full


def spectral_initialization(r, svd_terms=None, m=None):
    """
    Given a rank r, return spectral initialization of x,y via r-SVD. Recomputes the SVD if svd_terms not provided.
    :param r: rank of the SVD
    :param svd_terms: optional SVD terms (recomputed if none provided)
    :param m: optional matrix m to be used for SVD computation
    :return: x_0, y_0 matrices create via spectral initialization
    """
    if svd_terms is None:
        if m is None:
            raise Exception('parameter `m` is not given, cannot perform svd')
        u_full, s_full, vh_full = get_svd_terms_full_ranks(m)
    else:
        u_full, s_full, vh_full = svd_terms
    # Use r-SVD for spectral initialization of X_0, Y_0
    u, d, vh = u_full[:, :r], np.diag(s_full[:r]), vh_full[:r, :]
    x_0 = u.dot(d ** 0.5)
    y_0 = vh.T.dot(d ** 0.5)
    return x_0, y_0


def perform_gradient_descent_iteration(x, y, m, omega, p, alpha):
    """
    We perform a single gradient descent step of the matrix factorization approach to matrix completion. We use a
    constant learning rate alpha.
    Mathematical details: Let f(X, Y) = 1/2 * norm(P_omega(X.dot(Y.T) - M))^2 where P_omega is selection operator.
    Our objective function is F(X,Y) = f(X, Y) plus a regularization term 1/16*norm(X.T.dot(X) + Y.T.dot(Y))^2.
    The gradient of F wrt X and Y are computed by hand as grad_x and grad_y in the code.
    :param x: left matrix of factorization
    :param y: right matrix of factorization
    :param m: matrix to be completed
    :param omega: indices which are present in m
    :param p: proportion of indices present in m
    :param alpha: learning rate
    :return: x,y the left and right matrices of the factorization
    """
    prediction_loss_term = x.dot(y.T) - m
    select_operator = np.zeros(prediction_loss_term.shape)
    for i, j in omega:
        select_operator[i, j] = prediction_loss_term[i, j]

    # Get gradients of F wrt X and Y
    regularization_term = x.T.dot(x) - y.T.dot(y)
    grad_x = 1 / (2 * p) * select_operator.dot(y) + 1 / 4 * x.dot(regularization_term)
    grad_y = 1 / (2 * p) * select_operator.T.dot(x) - 1 / 4 * y.dot(regularization_term)

    # Constant step gradient descent step
    x -= alpha * grad_x
    y -= alpha * grad_y
    return x, y
