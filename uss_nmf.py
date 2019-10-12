import time
import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition.nmf import _initialize_nmf
from numpy.linalg import norm
from os import path

eps = np.finfo(np.float).eps
RESULT_DICT = {'Q': None, 'U': None, 'H': None, 'i': 0}

def refine_factor_matrix(X):
    """ This function takes matrix as input and checks for underflow and
    replaces entries with lowest possible float-value eps to correct underflow.
    In case of nan entries, it replaces them by corresponding rowmean. """
    X[X < eps] = eps
    row_mean = np.nanmean(X, axis=1)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(row_mean, inds[0])
    return X

def sparse_to_matrix(X) :
    """ This function takes sparse matrix and converts it to dense. """
    if issparse(X):
        X = X.toarray()
    else:
        X = np.array(X)
    return X

def initialize_factor_matrices(S, Y, W, init, dtype, logger, config):
    """ This function initializes factor matrices based on the choice of initialization method
    either 'random' or 'nndsvd', random seed can be set based on user input. """
    if config.FIXED_SEED == 'Y':
        np.random.seed(int(config.SEED_VALUE))
    logger.debug('Initializing factor matrices')
    if init == 'random':
        U = np.array(np.random.rand(int(config.N), int(config.L_COMPONENTS)), dtype=dtype)
        M = np.array(np.random.rand(int(config.L_COMPONENTS), int(config.N)), dtype=dtype)
        Q = np.array(np.random.rand(int(config.Q), int(config.L_COMPONENTS)), dtype=dtype)
    elif init == 'nndsvd':
        U, M = _initialize_nmf(S, int(config.L_COMPONENTS), 'nndsvd')
        Q, _ = _initialize_nmf(W*Y, int(config.L_COMPONENTS), 'nndsvd')
    else:
        raise('Unknown init option ("%s")' % init)
    U = sparse_to_matrix(U)
    M = sparse_to_matrix(M)
    Q = sparse_to_matrix(Q)
    H = np.array(np.random.rand(int(config.K), int(config.N)), dtype=dtype)
    C = np.array(np.random.rand(int(config.K), int(config.L_COMPONENTS)), dtype=dtype)
    logger.debug('Initialization completed')
    return M, U, C, H, Q

def __LS_updateM_L2(S, M, U, alpha, lmbda):
    """ Multiplicative update equation for M """
    UtU = np.dot(U.T, U)
    numerator = alpha * np.dot(U.T, S)
    denominator = alpha * np.dot(UtU, M) + lmbda * M
    denominator[denominator == 0] = eps
    M = M * (numerator / denominator)
    M = refine_factor_matrix(M)
    return M

def __LS_updateQ_L2(Y, U, W, Q, S, theta, delta, lmbda):
    """ Multiplicative update equation for Q """
    mod_Y = W * Y
    QUt = np.dot(Q, U.T)
    d = np.diag(np.sum(S, axis=1))
    numerator = theta * np.dot(mod_Y, U) + delta * np.dot(np.dot(QUt, S), U)
    denominator = theta * np.dot((W * QUt), U) + lmbda * Q + delta * np.dot(np.dot(QUt, d), U)
    denominator[denominator == 0] = eps
    Q = Q * (numerator / denominator)
    Q = refine_factor_matrix(Q)
    return Q

def __LS_updateC_L2(H, U, C, beta, lmbda):
    """ Multiplicative update equation for C """
    UtU = np.dot(U.T, U)
    numerator = beta * np.dot(H, U)
    denominator = beta * np.dot(C, UtU) + lmbda * C
    denominator[denominator == 0] = eps
    C = C * (numerator / denominator)
    C = refine_factor_matrix(C)
    return C

def __LS_updateH_L2(H, U, C, A, W, Y, beta, zeta, lmbda, phi):
    """ Multiplicative update equation for H """
    E = np.dot((W * Y).T, W * Y)
    d = np.diag(np.sum(E, axis=1))
    numerator = beta * 2 * np.dot(C, U.T) + zeta * 4 * H + phi * 2 * np.dot(H, E)
    denominator = zeta * 4 * np.dot(H, A) + beta * 2 * H + lmbda * 2 * H + phi * 2 * np.dot(H, d)
    numerator[numerator <= 0] = eps
    denominator[denominator <= 0] = eps
    H = H * np.sqrt(np.sqrt(numerator / denominator))
    H = refine_factor_matrix(H)
    return H

def __LS_updateU_L2(S, M, U, H, C, W, Y, Q, alpha, beta, theta, delta, lmbda):
    """ Multiplicative update equation for U """
    Ut = U.T
    nominator = np.zeros((Ut.shape), dtype=U.dtype)
    dnominator = np.zeros((Ut.shape), dtype=U.dtype)
    MMt = np.dot(M, M.T)
    CtC = np.dot(C.T, C)
    d = np.diag(np.sum(S, axis=1))
    nominator += alpha * np.dot(M, S.T)
    dnominator += alpha * np.dot(MMt, Ut) + lmbda * Ut
    nominator += beta * np.dot(C.T, H)
    dnominator += beta * np.dot(CtC, Ut)
    nominator += theta * np.dot(Q.T, (W * Y))
    dnominator += theta * np.dot(Q.T, (W * np.dot(Q, Ut)))
    nominator += delta * np.dot(np.dot(Q.T, np.dot(Q, Ut)), S)
    dnominator += delta * np.dot(np.dot(Q.T, np.dot(Q, Ut)), d)
    dnominator[dnominator == 0] = eps
    Ut = Ut * (nominator / dnominator)
    U = Ut.T

    U = refine_factor_matrix(U)
    return U

def __LS_compute_fit(S, M, U, H, C, W, Y, Q, alpha, beta, zeta, theta, lmbda, phi, delta):
    E = np.dot((W * Y).T, W * Y)
    d1 = np.diag(np.sum(E, axis=1))
    d2 = np.diag(np.sum(S, axis=1))
    UQt = np.dot(U, Q.T)
    UM = np.dot(U, M)
    fitSMU = norm(S - UM) ** 2

    mod_Y = W * Y
    QUt = np.dot(Q, U.T)
    mod_QUt = W * QUt
    fitWYQUt = norm(mod_Y - mod_QUt) ** 2

    CUt = np.dot(C, U.T)
    fitHCUt = norm(H - CUt) ** 2

    HHt = np.dot(H, H.T)
    fitHHtI = norm(HHt - np.eye(H.shape[0])) ** 2

    z1 = np.trace(np.dot(np.dot(H, d1 - E), H.T))
    z2 = np.trace(np.dot(np.dot(QUt, d2 - S), UQt))
    l2_reg = norm(U) ** 2 + norm(Q) ** 2 + norm(C) ** 2 + norm(M) ** 2 + norm(H) ** 2
    return (alpha*fitSMU + beta*fitHCUt + zeta*fitHHtI + theta*fitWYQUt + phi*z1 + delta*z2 + lmbda*l2_reg)

def factorize(config, S, B, Y, Y_train, train_ids, val_ids, test_ids, logger):
    # ---------- Get the parameter values-----------------------------------------
    alpha = float(config.ALPHA)
    beta = float(config.BETA)
    theta = float(config.THETA)
    phi = float(config.PHI)
    delta = float(config.DELTA)
    lmbda = float(config.LAMBDA)
    zeta = float(config.ZETA)
    init = config.INIT
    maxIter = int(config.MAX_ITER)
    costF = config.COST_F
    stop_index = int(config.STOP_INDEX)
    early_stopping = int(config.EARLY_STOPPING)
    if costF == 'LS':
        conv = float(config.CONV_LS)
    n = np.shape(S)[0]
    config.N = n
    q, _ = Y.shape
    config.Q = q
    dtype = np.float32
    mult = 1
    delta = 1.0 #theta
    phi = 1.0 #beta
    # Creation of class prior vector from training data
    train_label_dist = np.sum(Y_train.T, axis=0) / np.sum(Y_train)

    # Creation of penalty matrix from label matrix and training data
    W = np.copy(Y.T)
    unlabelled_ids = np.logical_not(train_ids)
    n_unlabelled = np.count_nonzero(unlabelled_ids)
    W[unlabelled_ids, :] = np.zeros((n_unlabelled, q))
    W[train_ids, :] = np.ones((n - n_unlabelled, q))
    W = W.T

    # ---------- Initialize factor matrices-----------------------------------------
    M, U, C, H, Q = initialize_factor_matrices(S, Y, W, init, dtype, logger, config)
    exectimes = []
    best_lr_result = RESULT_DICT
    max_lr_accu = -1
    conv_list = list()
    #  ------ compute factorization -------------------------------------------
    fit = 0

    for iter in range(maxIter):
        tic = time.time()
        fitold = fit
        if costF == 'LS':
            A = np.dot(H.T, H)
            M = __LS_updateM_L2(S, M, U, alpha, lmbda)
            if theta != 0:
                Q = __LS_updateQ_L2(Y, U, W, Q, S, theta, delta, lmbda)
            if beta != 0:
                C = __LS_updateC_L2(H, U, C, beta, lmbda)
                H = __LS_updateH_L2(H, U, C, A, W, Y, beta, zeta, lmbda, phi)
            U = __LS_updateU_L2(S, M, U, H, C, W, Y, Q, alpha, beta, theta, delta, lmbda)
            fit = __LS_compute_fit(S, M, U, H, C, W, Y, Q, alpha, beta, zeta, theta, lmbda, phi, delta)

            if ((iter % (config.STEP * mult)) == 0):
                from main_algo import get_perf_metrics
                # Can be set test_ids during testing or val_ids during hyper-param search using validation
                val_lr_accu = get_perf_metrics(config, U, Q, Y.T, train_ids, test_ids, 'lr')
                if val_lr_accu["micro_f1"] > max_lr_accu:
                    best_lr_result = {'Q': Q, 'U': U, 'H': H, 'i': iter}
                    max_lr_accu = val_lr_accu["micro_f1"]
                    stop_index = 0
                else :
                    stop_index = stop_index + 1

        if iter <= 1 :
            fitchange = abs(fitold - fit)
        else :
            fitchange = abs(fitold - fit) / fitold
        conv_list.append(fit)
        toc = time.time()
        exectimes.append(toc - tic)
        logger.debug('::: [%3d] fit: %0.5f | delta: %7.1e | secs: %.5f' % (iter, fit, fitchange, exectimes[-1]))

        if iter > maxIter or fitchange < conv:
            break
        if stop_index > early_stopping:
            logger.debug("Early stopping")
            break

    return best_lr_result
