import autograd.numpy as np
from autograd import grad

from utils import fast_matrix_det, fast_matrix_slogdet, fast_quadratic_form, woodbury, rbf

# ==============================================================================
# Functions for Titsias' Sparse GP
# ==============================================================================

def get_predictive_variational_gp(X_m, X_n, Y, sigma_noise, sigma_f, length_scale):
    """
    Closure that gives back the predictive mean and covariance function for a
    sparse GP.

    X_m          - M x H matrix: M inducing points of dimension H
    X_n          - N x H matrix: N input data points of dimension H
    Y            - N x D matrix: N output data points of dimension D
    sigma_noise  - scalar: assumed measurement noise on each output dimension
    sigma_f      - scalar: scale factor of RBF kernel
    length_scale - scalar: length scale of RBF kernel

    RBF kernel:

    k(x_i, x_j) = sigma_f^2 * exp( -1. / (2 * length_scale^2) * (x_i - x_j)'(x_i - x_j))

    """
    # ======================================================================
    # Ensure the input has the correct size
    # ======================================================================
    if len(X_m.shape) != 2:
        raise IncompatibleShapeError(
            "X_m rank of shape must be 2 not: {}".format(len(X_m.shape)))

    if len(X_n.shape) != 2:
        raise IncompatibleShapeError(
            "X_n rank of shape must be 2 not: {}".format(len(X_n.shape)))

    if len(Y.shape) != 2:
        raise IncompatibleShapeError(
            "Y rank of shape must be 2 not: {}".format(len(Y.shape)))

    m, h = X_m.shape
    n, d = Y.shape
    n_, h_ = X_n.shape

    if h != h_:
        raise IncompatibleShapeError(
            "X_m and X_n must share the same 2nd dimension: {} and {} must match".format(h, h_))

    if n != n_:
        raise IncompatibleShapeError(
            "Y and X_n must share the same 1nd dimension: {} and {} must match".format(n, n_))

    # ======================================================================
    # Do stuff
    # ======================================================================

    # Joint covariance of q(f_n, f_m)
    K = rbf(np.concatenate((X_n, X_m), axis=0), sigma_f, length_scale)

    K_nn = K[:n, :n]
    K_mm = K[n:, n:]
    K_nm = K[:n, n:]
    K_mm_inv = np.linalg.inv(K_mm)

    Sigma = np.linalg.inv(K_mm + sigma_noise ** -2. * np.dot(K_nm.T, K_nm))
    mu = sigma_noise ** -2. * np.dot(K_mm, np.dot(np.dot(Sigma, K_nm.T), Y.reshape(n, -1)))


    # Produce predictive mean and covariance functions

    def mu_fn(x_preds):
        """
        x_preds - P x H matrix: P predictive input points of dimension H
        """

        if len(x_preds.shape) != 2:
            raise IncompatibleShapeError(
                "x_preds rank of shape must be 2 not: {}".format(len(x_preds.shape)))

        p, h__ = x_preds.shape

        if h != h__:
            raise IncompatibleShapeError(
                "x_preds and X_m must share the same 2nd dimension: {} and {} must match".format(h, h__))

        K_ = rbf(np.concatenate((x_preds, X_m), axis=0), sigma_f, length_scale)
        K_xm = K_[:p, p:]

        return np.dot(np.dot(K_xm, K_mm_inv), mu)

    def cov_fn(x_preds):
        """
        x_preds - P x H matrix: P predictive input points of dimension H
        """

        if len(x_preds.shape) != 2:
            raise IncompatibleShapeError(
                "x_preds rank of shape must be 2 not: {}".format(len(x_preds.shape)))

        p, h__ = x_preds.shape

        if h != h__:
            raise IncompatibleShapeError(
                "x_preds and X_m must share the same 2nd dimension: {} and {} must match".format(h, h__))


        K_ = rbf(np.concatenate((x_preds, X_m), axis=0), sigma_f, length_scale)
        K_xx = K_[:p, :p]
        K_xm = K_[:p, p:]

        return K_xx + sigma_noise ** 2. - np.dot(np.dot(K_xm, K_mm_inv), K_xm.T) \
                + np.dot(np.dot(K_xm, Sigma), K_xm.T)

    return mu_fn, cov_fn


def gp_log_prob(X, Y, sigma_noise, sigma_f, length_scale, verbose=False):
    """
    Log probability of an RBF-kernel GP. If the output is vector-valued, then
    the individual dimensions are assumed independent.

    X            - N x H matrix: N input data points of dimension H
    Y            - N x D matrix: N output data points of dimension D
    sigma_noise  - scalar: assumed measurement noise on each output dimension
    sigma_f      - scalar: scale factor of RBF kernel
    length_scale - scalar: length scale of RBF kernel

    RBF kernel:

    k(x_i, x_j) = sigma_f^2 * exp( -1. / (2 * length_scale^2) * (x_i - x_j)'(x_i - x_j))

    """
    # ======================================================================
    # Ensure the input has the correct size
    # ======================================================================
    if len(X.shape) != 2:
        raise IncompatibleShapeError(
            "X rank of shape must be 2 not: {}".format(len(X.shape)))

    if len(Y.shape) != 2:
        raise IncompatibleShapeError(
            "Y rank of shape must be 2 not: {}".format(len(Y.shape)))

    n, d = Y.shape
    n_, h_ = X.shape

    if n != n_:
        raise IncompatibleShapeError(
            "Y and X must share the same 1nd dimension: {} and {} must match".format(n, n_))

    # ======================================================================
    # Do stuff
    # ======================================================================

    K = rbf(X, sigma_f, length_scale)
    noise_cov = (sigma_noise ** 2.) * np.eye(n)

    Sigma = noise_cov + K

    log_det_gp_cov, _ = np.linalg.slogdet(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)

    log_prob = -n / 2. * np.log(2 * np.pi) \
                -0.5 * log_det_gp_cov \
                -0.5 * np.trace(np.dot(np.dot(Y.T, Sigma_inv), Y))

    if verbose:
        print("\n==================")
        print(det_gp_cov)
        print(log_prob)
        print("\n==================")

    return log_prob


def sparse_gp_log_prob(X_n, X_m, Y, sigma_noise, sigma_f, length_scale, verbose=False):
    # ======================================================================
    # Ensure the input has the correct size
    # ======================================================================
    if len(X_m.shape) != 2:
        raise IncompatibleShapeError(
            "X_m rank of shape must be 2 not: {}".format(len(X_m.shape)))

    if len(X_n.shape) != 2:
        raise IncompatibleShapeError(
            "X_n rank of shape must be 2 not: {}".format(len(X_n.shape)))

    if len(Y.shape) != 2:
        raise IncompatibleShapeError(
            "Y rank of shape must be 2 not: {}".format(len(Y.shape)))

    m, h = X_m.shape
    n, d = Y.shape
    n_, h_ = X_n.shape

    if h != h_:
        raise IncompatibleShapeError(
            "X_m and X_n must share the same 2nd dimension: {} and {} must match".format(h, h_))

    if n != n_:
        raise IncompatibleShapeError(
            "Y and X_n must share the same 1nd dimension: {} and {} must match".format(n, n_))
    
    # ======================================================================
    # Ensure the input has the correct size
    # ======================================================================
    
    # Joint covariance of q(f_n, f_m)
    K = rbf(np.concatenate((X_n, X_m), axis=0), sigma_f, length_scale)

    K_nn = K[:n, :n]
    K_mm = K[n:, n:]
    K_nm = K[:n, n:]
    K_mm_inv = np.linalg.inv(K_mm)
    
    # Sigma from Titsias' paper
    Sigma = np.linalg.inv(K_mm + sigma_noise ** -2. * np.dot(K_nm.T, K_nm))
    
    # Sigma as in the covariance of the predictive distribution
    trace_term = 0
    log_det_gp_cov = 0
    
    for i in range(n):
        cov = K_nn[i, i] - np.dot(np.dot(K_nm[i, :], K_mm_inv - Sigma), K_nm[i, :].T)
        
        trace_term = trace_term + 1./cov * np.dot(Y[i, :], Y[i, :].T)
        log_det_gp_cov = log_det_gp_cov + np.log(cov)
    
    mu = sigma_noise ** -2. * np.dot(K_nm, np.dot(np.dot(Sigma, K_nm.T), Y))
    
    log_prob = -n / 2. * np.log(2 * np.pi) \
                -0.5 * log_det_gp_cov \
                -0.5 * trace_term
    
    if verbose:
        print("\n==================")
        print(log_det_gp_cov)
        print(log_prob)
        print("\n==================")
    
    return log_prob
    
    
def free_energy(X_n, X_m, Y, sigma_noise, sigma_f, length_scale, verbose=False):
    """
    Free energy of a sparse variational GP. If the output is vector-valued, then
    the individual dimensions are assumed independent. 

    X_m          - M x H matrix: M inducing points of dimension H
    X_n          - N x H matrix: N input data points of dimension H
    Y            - N x D matrix: N output data points of dimension D
    sigma_noise  - scalar: assumed measurement noise on each output dimension
    sigma_f      - scalar: scale factor of RBF kernel
    length_scale - scalar: length scale of RBF kernel

    RBF kernel:

    k(x_i, x_j) = sigma_f^2 * exp( -1. / (2 * length_scale^2) * (x_i - x_j)'(x_i - x_j))

    """
    # ======================================================================
    # Ensure the input has the correct size
    # ======================================================================
    if len(X_m.shape) != 2:
        raise IncompatibleShapeError(
            "X_m rank of shape must be 2 not: {}".format(len(X_m.shape)))

    if len(X_n.shape) != 2:
        raise IncompatibleShapeError(
            "X_n rank of shape must be 2 not: {}".format(len(X_n.shape)))

    if len(Y.shape) != 2:
        raise IncompatibleShapeError(
            "Y rank of shape must be 2 not: {}".format(len(Y.shape)))

    m, h = X_m.shape
    n, d = Y.shape
    n_, h_ = X_n.shape

    if h != h_:
        raise IncompatibleShapeError(
            "X_m and X_n must share the same 2nd dimension: {} and {} must match".format(h, h_))

    if n != n_:
        raise IncompatibleShapeError(
            "Y and X_n must share the same 1nd dimension: {} and {} must match".format(n, n_))

    # ======================================================================
    # Do stuff
    # ======================================================================

    K = rbf(np.concatenate((X_n, X_m), axis=0), sigma_f, length_scale)
    
    K_nn = K[:n, :n]
    K_mm = K[n:, n:]
    K_nm = K[:n, n:]
    K_mm_inv = np.linalg.inv(K_mm + 1e-6 * np.eye(K_mm.shape[0]))

    noise_cov_diag = (sigma_noise ** 2.) * np.ones(n)
    noise_cov = (sigma_noise ** 2.) * np.eye(n)

    gp_cov = np.dot(np.dot(K_nm, K_mm_inv), K_nm.T)
    log_det_gp_cov, _ = fast_matrix_slogdet(noise_cov_diag, K_nm, K_nm.T, K_mm)
    gp_QF = fast_quadratic_form(noise_cov_diag, K_nm, K_nm.T, K_mm, Y)

    log_prob_gaussian = -n / 2. * np.log(2 * np.pi) \
                        -0.5 * log_det_gp_cov \
                        -0.5 * np.trace(gp_QF)

    regularising_term = float(d) / (2 * sigma_noise ** 2.) * (np.trace(K_nn) - np.trace(gp_cov))

    if verbose:
        print("\n==================")
        print("Noise Covariance Det: \t{}: ".format(np.linalg.slogdet(noise_cov)[0]))
        print("K inverse Det: \t\t{}".format(np.linalg.det(K_mm_inv)))
        print("GP covariance Det: \t{}".format(log_det_gp_cov))
        print("Quadratic Form Trace: \t{}".format(np.trace(gp_QF)))
        print("Regularising Term: \t{}".format(regularising_term))
        print("Log probability: \t{}".format(log_prob_gaussian))
        print("\n==================")

    return log_prob_gaussian - regularising_term


def fit_inducing_points(X_n, X_m, Y, log_sigma_noise=1., log_sigma_f=0., log_length_scale=-1., learn_rate=1e-1,
            num_iter=5000, early_stopping=1e-3, verbose=True, log_every=100):

    F_lambda = lambda X_m_, log_sigma_noise_, log_sigma_f_, log_length_scale_: \
                   free_energy(X_n, 
                               X_m_, 
                               Y, 
                               np.exp(log_sigma_noise_), 
                               np.exp(log_sigma_f_), 
                               np.exp(log_length_scale_), 
                               verbose=verbose)

    theta = [X_m, log_sigma_noise, log_sigma_f, log_length_scale]
    dF_dtheta = [grad(F_lambda, i) for i in range(len(theta))]

    # Keep track of the loss
    prev_F = F_lambda(*theta)

    for i in range(num_iter):

        grads = [dF_dtheta[j](*theta) for j in range(len(theta))]
        theta = [theta[j] + learn_rate * gradient / len(Y) for j, gradient in enumerate(grads)]

        current_F = F_lambda(*theta)

        if abs(prev_F - current_F) <= early_stopping:
            print("Early convergence at iteration {}!".format(i))
            return tuple(theta)
        else:
            prev_F = current_F
            yield tuple(theta)

            
def fit_latents(X,
                X_m,
                Y,
                sigma_noise, 
                sigma_f, 
                length_scale, 
                learn_rate=1e-6, 
                early_stopping=1e-3,
                num_iter=10000, 
                verbose=True, 
                log_every=50):

    K = rbf(X, sigma_noise, sigma_f, length_scale)
    
    loglik_lambda = lambda X_: sparse_gp_log_prob(X_, X_m, Y, sigma_noise, sigma_f, length_scale, verbose=False)
    
    dloglik_dX = grad(loglik_lambda, 0)

    prev_loglik = loglik_lambda(X)
    
    for i in range(num_iter):
        
        X = X + learn_rate * dloglik_dX(X)
        
        current_loglik = loglik_lambda(X)
        
        if abs(prev_loglik - current_loglik) <= early_stopping:
            print("Early convergence at iteration {}!".format(i))
            return X
        else:
            prev_loglik = current_loglik
            yield X
        
        if verbose and i % log_every == 0: 
            print("Log-likelihood (iteration {}): {:.3f}".format(i + 1, current_loglik))
    
    if verbose:
        print("Final log-likelihood: {:.3f}".format(current_loglik))
    return X
