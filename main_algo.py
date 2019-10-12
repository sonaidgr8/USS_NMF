def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import argparse
import logging
import collections
from os import path, listdir
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
from scipy.io import loadmat
import scipy.sparse as sp
from scipy.sparse import issparse
import data_utils as du
import utils as ut
import os
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pickle as pkl

algo_fn = {"1":"uss_nmf"}

def get_ArgumentParser():
    parser = argparse.ArgumentParser()
    # Experiment configuratio
    parser.add_argument("--DATA_DIR", default="cora")
    parser.add_argument("--DATA_PATH", default="Datasets/")
    parser.add_argument("--LOG_DIR", default="emb/")
    parser.add_argument("--FOLDER_SUFFIX", default="emb.npy")
    parser.add_argument("--LOG_FNAME", default="_ussnmf.log")
    # Weights for different components in the objective functions
    parser.add_argument("--ETA", default=1.0, help="First and Second order proximity mixing parameters")
    parser.add_argument("--ALPHA", default=1.0, help="Similarity matrix factorization weight")
    parser.add_argument("--BETA", default=0.1, help="Community Indicator matrix factorization weight")
    parser.add_argument("--THETA", default=0.5, help="Label matrix factorization weight")
    parser.add_argument("--PHI", default=10.0, help="Cluster Learning regularization weight")
    parser.add_argument("--GAMMA", default=0.0, help="Modularity Maximization weight")
    parser.add_argument("--DELTA", default=0.5, help="Label propogation (LL) weight")
    parser.add_argument("--LAMBDA", default=1.0, help="L2 regularization weight for factor matrices")
    parser.add_argument("--ZETA", default=1e+4, help="L2 regularization weight for orthogonality constraint")
    # Factorization parameters
    parser.add_argument("--MAX_ITER", default=500)
    parser.add_argument("--L_COMPONENTS", default=128)
    parser.add_argument("--K", default=5)
    parser.add_argument("--INIT", default="random")
    parser.add_argument("--PROJ", default=True)
    parser.add_argument("--COST_F", default='LS')
    parser.add_argument("--CONV_LS", default=5e-11)
    parser.add_argument("--CONV_KL", default=5e-11)
    parser.add_argument("--CONV_MUL", default=5e-11)
    parser.add_argument("--MULTI_LABEL", default=False)
    parser.add_argument("--TYPE", default='1')
    parser.add_argument("--FIXED_SEED", default='Y')
    parser.add_argument("--SEED_VALUE", default=0)
    parser.add_argument("--STEP", default=1)
    parser.add_argument("--STOP_INDEX", default=0)
    parser.add_argument("--EARLY_STOPPING", default=300)
    parser.add_argument("--SAVE_EMB", default=True)
    parser.add_argument("--HP_SEARCH", default=False)

    return parser

def init(config, dir_name, file_name):
    ut.check_n_create(dir_name)
    ut.check_n_create("tmp_output_files")
    file_name = dir_name + "_" + file_name
    logging.basicConfig(filename=file_name, filemode='w', level=logging.DEBUG)
    return logging.getLogger(algo_fn[config.TYPE])

def load_dataset(config, dir_name):
    # Load data :  all are in (nxn), (nxq), (nxm) format
    relation = loadmat(path.join(dir_name, config.DATA_DIR.lower()+".mat"))['network']
    truth = loadmat(path.join(dir_name, config.DATA_DIR.lower() + ".mat"))['group']
    if issparse(relation):
        relation = relation.toarray().astype(np.float64)
    if issparse(truth):
        truth = truth.toarray().astype(np.int8)

    n_ids, n_labels = np.shape(truth)
    all_label_dist = du.get_all_label_distribution(truth)
    multilabel = False
    if np.sum(truth) > n_ids:
        multilabel = True
    elif np.sum(truth) < n_ids and truth.shape[1] > 1:
        raise ValueError('Some nodes have no labels!')
    expt_sets = np.array(
        [f for f in listdir(path.join(dir_name, 'index')) if not path.isfile(path.join(dir_name, 'index', f))], dtype=np.int32)
    expt_sets = np.sort(expt_sets)
    n_folds = len([1 for f in listdir(path.join(dir_name, 'index', str(expt_sets[0]))) if
                   not path.isfile(path.join(dir_name, 'index', f))])
    print("================ Dataset Details : Start ================")
    print("Dataset: %s" % config.DATA_DIR.capitalize())
    print("Sets: %s" % expt_sets)
    print("N_Folds: %d" % n_folds)
    print("Number of nodes : %d" % n_ids)
    print("Number of labels : %d" % n_labels)
    print("All label distribution : %s" % all_label_dist)


    datasets_template = collections.namedtuple('Datasets_template', ['relations', 'truth', 'expt_sets', \
                                                                     'n_folds', 'n_ids', 'n_labels', \
                                                                     'all_label_dist',
                                                                     'multilabel'])
    dataset = datasets_template(relations=relation, truth=truth, expt_sets=expt_sets, \
                                n_folds=n_folds, n_ids=n_ids, n_labels=n_labels,\
                                all_label_dist=all_label_dist, multilabel=multilabel)
    print("================ Dataset Details : End ================")
    return dataset

def construct_indicator(y_score, y):
    # rank the labels by the scores directly
    num_label = np.sum(y, axis=1, dtype=np.int)
    y_sort = np.fliplr(np.argsort(y_score, axis=1))
    y_pred = np.zeros_like(y, dtype=np.int)
    for i in range(y.shape[0]):
        for j in range(num_label[i]):
            y_pred[i, y_sort[i, j]] = 1
    return y_pred

def get_perf_metrics(config, entity_embedding, Q, labels, train_ids, test_ids, choice):
    pred_ids = test_ids
    labelled_ids = train_ids
    if choice == 'lr' :
        clf = OneVsRestClassifier(LogisticRegression())
        clf.fit(entity_embedding[labelled_ids, :], labels[labelled_ids, :])
        predictions = clf.predict_proba(entity_embedding[pred_ids, :])
        clf_norm = OneVsRestClassifier(LogisticRegression())
        entity_embedding_norm = normalize(entity_embedding, axis=1, norm='l2')
        clf_norm.fit(entity_embedding_norm[labelled_ids, :], labels[labelled_ids, :])
        predictions_norm = clf_norm.predict_proba(entity_embedding_norm[pred_ids, :])
    elif choice == 'svm':
        clf = OneVsRestClassifier(LinearSVC(random_state=0))
        clf.fit(entity_embedding[labelled_ids, :], labels[labelled_ids, :])
        predictions = clf.decision_function(entity_embedding[pred_ids, :])
        clf_norm = OneVsRestClassifier(LinearSVC(random_state=0))
        entity_embedding_norm = normalize(entity_embedding, axis=1, norm='l2')
        clf_norm.fit(entity_embedding_norm[labelled_ids, :], labels[labelled_ids, :])
        predictions_norm = clf_norm.decision_function(entity_embedding_norm[pred_ids, :])
    elif choice =='n' :
        Y_hat = np.dot(Q, entity_embedding.T)
        Y_hat = Y_hat.T
        predictions = Y_hat[pred_ids, :]
        Y_hat_norm = normalize(Y_hat, axis=1, norm='l2')
        predictions_norm = Y_hat_norm[pred_ids, :]

    y_pred = construct_indicator(predictions, labels[pred_ids, :])
    mi = f1_score(labels[pred_ids, :], y_pred, average="micro")
    ma = f1_score(labels[pred_ids, :], y_pred, average="macro")
    acc = accuracy_score(labels[pred_ids, :], y_pred)

    y_pred_norm = construct_indicator(predictions_norm, labels[pred_ids, :])
    mi_norm = f1_score(labels[pred_ids, :], y_pred_norm, average="micro")
    ma_norm = f1_score(labels[pred_ids, :], y_pred_norm, average="macro")
    acc_norm = accuracy_score(labels[pred_ids, :], y_pred_norm)

    if mi >= mi_norm:
        performances = {'accuracy': acc, 'micro_f1': mi, 'macro_f1': ma}
        return collections.Counter(performances)
    else:
        performances_norm = {'accuracy': acc_norm, 'micro_f1': mi_norm, 'macro_f1': ma_norm}
        return collections.Counter(performances_norm)


def main():
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Matrix Factorization~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    config = get_ArgumentParser().parse_args()
    config.LOG_FNAME = algo_fn[config.TYPE]+".log"
    logger = init(config, config.LOG_DIR, config.LOG_FNAME)
    dataset = load_dataset(config, path.join(config.DATA_PATH, config.DATA_DIR.lower()))
    config.MULTI_LABEL = dataset.multilabel
    print("Config: %s" % (config))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    S = du.get_proximity_matrix(dataset.relations, float(config.ETA))
    B = du.get_modularity_matrix(dataset.relations)

    perc_data = dataset.expt_sets
    for a in perc_data :
        temp1 = {}
        temp2 = {}
        temp3 = {}
        print("% of randomly sampled training data ---- ", a)
        avg_lr_acc = {'micro_f1' : 0.0, 'macro_f1' : 0.0, 'accuracy' : 0.0}
        avg_svm_acc = {'micro_f1' : 0.0, 'macro_f1' : 0.0, 'accuracy' : 0.0}
        avg_n_acc = {'micro_f1' : 0.0, 'macro_f1' : 0.0, 'accuracy' : 0.0}
        itr = 0
        # for b in range(1, dataset.n_folds + 1): # You can choose to run for all five folds
        for b in range(1, 1 + 1): # Running for only one fold
            data_dir = path.join(config.DATA_PATH, config.DATA_DIR.lower(), 'index', str(a), str(b))
            train_ids = np.load(path.join(data_dir, 'train_ids.npy')).astype(dtype=bool)
            val_ids = np.load(path.join(data_dir, 'val_ids.npy')).astype(dtype=bool)
            train_ids = np.logical_or(train_ids, val_ids)
            test_ids = np.load(path.join(data_dir, 'test_ids.npy')).astype(dtype=bool)
            labelled_ids = train_ids
            unlabelled_ids = np.logical_not(labelled_ids)
            n_unlabelled = np.count_nonzero(unlabelled_ids)
            labels = np.copy(dataset.truth)
            labels[unlabelled_ids, :] = np.zeros((n_unlabelled, dataset.n_labels))
            Y = dataset.truth
            Y_train = labels

            if not os.path.isfile(path.join(config.LOG_DIR, config.FOLDER_SUFFIX+"_U" + str(b)+".npy")) :
                module = __import__(algo_fn[config.TYPE])
                best_result_lr = getattr(module, 'factorize')(config, S, B, Y.T, Y_train.T, train_ids, val_ids, test_ids, logger)
            else :
                U = np.load(path.join(config.LOG_DIR, config.FOLDER_SUFFIX+"_U" + str(b)+".npy"))
                Q = np.load(path.join(config.LOG_DIR, config.FOLDER_SUFFIX+"_Q" + str(b)+".npy"))
                best_result_lr = {'Q': Q, 'U': U, 'H': None, 'i': 0}

            best_lr_accu = get_perf_metrics(config, best_result_lr['U'], best_result_lr['Q'], Y, train_ids, test_ids, 'lr')
            best_svm_accu = get_perf_metrics(config, best_result_lr['U'], best_result_lr['Q'], Y, train_ids, test_ids, 'svm')
            if config.TYPE in ["1", "12", "18", "21", "22", "23", "25"]:
                best_n_accu = best_lr_accu
            else :
                best_n_accu = get_perf_metrics(config, best_result_lr['U'], best_result_lr['Q'], Y, train_ids, test_ids, 'n')
            for k, v in avg_lr_acc.items() :
                avg_lr_acc[k] = avg_lr_acc[k] + best_lr_accu[k]
                avg_svm_acc[k] = avg_svm_acc[k] + best_svm_accu[k]
                avg_n_acc[k] = avg_n_acc[k] + best_n_accu[k]
            logger.debug("Iter# {} LR_Micro_F1: {} SVM_Micro_F1: {} N_Micro_F1: {}".format(best_result_lr['i'], best_lr_accu['micro_f1'], best_svm_accu['micro_f1'], best_n_accu["micro_f1"]))
            itr += 1
            if config.SAVE_EMB:
                logger.info("Save embedding to %s", config.LOG_DIR)
                np.save(path.join(config.LOG_DIR, config.FOLDER_SUFFIX + "_U" + str(a) +"_"+ str(b)+".npy"), best_result_lr['U'], allow_pickle=False)
                np.save(path.join(config.LOG_DIR, config.FOLDER_SUFFIX + "_Q" + str(a) +"_"+ str(b) + ".npy"), best_result_lr['Q'], allow_pickle=False)
        avg_lr_acc = {k: v / itr for k, v in avg_lr_acc.items()}
        avg_svm_acc = {k: v / itr for k, v in avg_svm_acc.items()}
        avg_n_acc = {k: v / itr for k, v in avg_n_acc.items()}
        for k, v in {"50_MI":'micro_f1', "50_MA":'macro_f1', "50_AC":'accuracy'}.items() :
            temp1[k] = avg_lr_acc[v]
            temp2[k] = avg_svm_acc[v]
            temp3[k] = avg_n_acc[v]
        with open("tmp_output_files/"+str(config.DATA_DIR) + "_" + config.TYPE+ "_" + str(a) + "_" + "best_params_nc.txt", 'wb') as fp:
            pkl.dump({"LR":temp1, "SVM":temp2, "N":temp3}, fp)

if __name__ == "__main__":
    main()
