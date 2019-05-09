import autograd.numpy as np

def rbf(X, sigma_f, length_scale):

    num_points = X.shape[0]

    cov = np.dot(X, X.T)
    diag = np.diag(cov)

    # (x_n - x_m)' (x_n - x_m) = x_n'x_n + x_m'x_m - 2x_n'x_m
    cov_ = diag.reshape((num_points, 1)) + diag.reshape((1, num_points)) - 2 * cov

    return (sigma_f ** 2.) * np.exp(-1. / (2 * length_scale ** 2.) * cov_)


def woodbury(A_diag, B, C, D):
    """
    Calculates the inverse of a corrected diagonal matrix in O(N x M^2) time.

    A_diag - Vector representation of an N x N diagonal matrix
    B - N x M matrix
    C - M x N matrix
    D - M x M invertible matrix

    (A + B(D_inv)C)_inv = A_inv - A_inv B(D + CA_inv B)_inv CA_inv

    E = (D + CA_inv B)_inv

    https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    """
    # Ensure A is a vector
    if len(A_diag.shape) != 1:
        raise Exception("A has to be a vector!")

    n = A_diag.shape[0]
    m = D.shape[0]

    # Ensure conformity of matrices
    if B.shape != (n, m) or C.shape != (m, n) or D.shape != (m, m):
        raise Exception("A {}, B {}, C {} and D {} have to be conformal!".format(
            A_diag.shape*2, B.shape, C.shape, D.shape))

    A_diag_inv = 1. / A_diag

    E = np.linalg.inv(D + np.dot(C * A_diag_inv.reshape((1, -1)), B))

    inv = -np.dot(np.dot(A_diag_inv.reshape((-1, 1)) * B, E), C * A_diag_inv.reshape((1, -1)))

    inv[np.diag_indices_from(inv)] += A_diag_inv.squeeze()

    return inv

def fast_matrix_det(A_diag, B, C, D):
    """
    Calculates the determinant of a corrected diagonal matrix in O(N x M^2) time.

    A_diag - Vector representation of an N x N diagonal matrix
    B - N x M matrix
    C - M x N matrix
    D - M x M invertible matrix

    | A + B D_inv C | = | D + C A_inv B | x | D_inv | x | A |

    https://en.wikipedia.org/wiki/Matrix_determinant_lemma#Generalization
    """
    # Ensure A is a vector
    if len(A_diag.shape) != 1:
        raise Exception("A has to be a vector!")

    n = A_diag.shape[0]
    m = D.shape[0]

    # Ensure conformity of matrices
    if B.shape != (n, m) or C.shape != (m, n) or D.shape != (m, m):
        raise Exception("A {}, B {}, C {} and D {} have to be conformal!".format(
            A_diag.shape*2, B.shape, C.shape, D.shape))


    A_diag_inv = 1. / A_diag

    return np.linalg.det(D + np.dot(C * A_diag_inv.reshape((1, -1)), B)) * \
            np.linalg.det( np.linalg.inv(D) ) * np.prod(A_diag)


def fast_quadratic_form(A_diag, B, C, D, ys):
    """
    Calculates the quadratic form

        y' (A + B(D_inv)C)_inv y

    in O(N x M^2) time.

    A_diag - Vector representation of an N x N diagonal matrix
    B - N x M matrix
    C - M x N matrix
    D - M x M invertible matrix

        (A + B(D_inv)C)_inv = A_inv - A_inv B(D + CA_inv B)_inv CA_inv

        E = (D + CA_inv B)_inv

    https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    """
    # Ensure A is a vector
    if len(A_diag.shape) != 1: 
        raise Exception("A has to be a vector!")

    # Ensure ys is a vector
    if len(ys.shape) != 1: 
        raise Exception("ys has to be a vector!")

    n = A_diag.shape[0]
    m = D.shape[0]

    # Ensure conformity of ys with the matrices
    if len(ys) != n:
        raise Exception("ys (length {}) has to match the dimensions of the matrix ({}, {})!".format(
            len(ys), n, n))

    # Ensure conformity of matrices
    if B.shape != (n, m) or C.shape != (m, n) or D.shape != (m, m):
        raise Exception("A {}, B {}, C {} and D {} have to be conformal!".format(
            A_diag.shape*2, B.shape, C.shape, D.shape))

    A_diag_inv = 1. / A_diag

    A_inv_ys = A_diag_inv * ys

    ys_A_inv_ys = np.dot(ys, A_inv_ys)

    E = np.linalg.inv(D + np.dot(C * A_diag_inv.reshape((1, -1)), B))

    ys_A_inv_B = np.dot(A_inv_ys.reshape((1, -1)), B)
    C_A_inv_ys = np.dot(C, A_inv_ys.reshape((-1, 1)))

    return ys_A_inv_ys - np.dot(np.dot(ys_A_inv_B, E), C_A_inv_ys)
