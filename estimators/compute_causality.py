import multiprocessing
import numpy as np
from joblib import Parallel, delayed
from sklearn import preprocessing
from tqdm import tqdm
from methods.ivy import Ivy
from estimators.wald_estimator import WaldEstimatorStats

def ComputeCausalitySingle(
    X, 
    Y,
    IVs, 
    IV_Model_list, 
    estimator_list, 
    is_soft_label=True, 
    ablation_train=1, 
    ablation_test=1, 
    Z_true=None,
    deps = [],
    random_seed = None,
    chrome_id_list = [],
    anchor = None,
    use_canonical=False,
    use_forward=False,
    return_predictive_score=False):

    if random_seed is not None:
        np.random.seed(random_seed)

    # train test split
    train_index, test_index = SplitDataset(Y, ablation_train, ablation_test)

    # X trian and test split need to be seperately specified
    X_train = X[train_index]
    X_test = X[test_index]

    IVs_train = IVs[train_index,:]
    IVs_test = IVs[test_index,:]

    # determine if X is binary
    if(len(set(X))>2):
        # continuous, scale X
        scaler = preprocessing.StandardScaler().fit(X_train.reshape(-1, 1))
        X_train = scaler.transform(X_train.reshape(-1, 1)).reshape(-1)
        X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(-1)
        mode = "cxby"
    else:
        # binary
        mode = "bxby"

    causality_list = []
    for estimator in estimator_list:

        # if estimator.__name__ in ["MedianWaldEstimator", "WeightedMedianWaldEstimator"]:
        #     causality = estimator(X_test, Y[test_index], IVs_test, mode=mode)
        #     # update causality_list
        #     if isinstance(causality,list):
        #         causality_list.extend(causality)
        #     else:
        #         causality_list.append(causality)
        #     continue

        for IV_Model in IV_Model_list:

            # train the IV_model
            iv_model = IV_Model()
            iv_model.train(IVs_train,X_train,deps=deps,anchor=anchor,
            chrome_id_list=chrome_id_list,lr=1e-4,n_epochs=10000,log_train_every=20000,
            verbose=False,use_canonical=use_canonical,use_forward=use_forward)

            # compute synthesized IV on IVs_test
            if is_soft_label is True:
                # for Ivy, when X is continuous, use ad-hoc score for synthesis
                if (IV_Model.__name__ in ["Ivy","IvyMetal","IvyMultiple"]) and \
                    (mode is "cxby"):
                    Z_test = iv_model.predict_proba(IVs_test,is_ad_hoc=True)
                elif IV_Model.__name__ is "ObservationalAssociation":
                    Z_test = iv_model.predict_proba(IVs_test,X_test)
                # otherwise just use regular probability
                else:
                    Z_test = iv_model.predict_proba(IVs_test)
            elif is_soft_label is False:
                Z_test = iv_model.predict(IVs_test)
            
            # compute Wald estimator
            if estimator.__name__ in ["WeightedMedianWaldEstimator"]:
                if type(iv_model).__name__ is "WeightedMajorityVote":
                       results = [WaldEstimatorStats(X_train, Y[train_index], 
                       IVs_train[:,i], mode=mode, detail=True) for i in 
                       range(IVs_train.shape[1])]
                       weights = np.array([(x['beta_x_z']/x['se_y_z'])**2 for x in results])
                else:
                    weights = iv_model.get_weights()
                causality = estimator(X_test, Y[test_index], IVs_test, mode=mode, weights=weights)
                # update causality_list
                if isinstance(causality,list):
                    causality_list.extend(causality)
                else:
                    causality_list.append(causality)
                continue
            else:
                causality = estimator(
                    X_test, 
                    Y[test_index], 
                    Z_test, 
                    mode=mode, 
                    return_predictive_score=return_predictive_score)

            # update causality_list
            if isinstance(causality,list):
                causality_list.extend(causality)
            else:
                causality_list.append(causality)

        if Z_true is not None:
            causality = estimator(X_test, Y[test_index], Z_true[test_index], mode=mode)
            # update causality_list
            if(len(set(X))>2):
                causality_list.extend(causality)
            else:
                causality_list.append(causality)

    return causality_list

def ComputeCausality(
    X, 
    Y, 
    IVs, 
    IV_Model_list, 
    estimator_list, 
    is_soft_label=True, 
    ablation_train=1, 
    ablation_test=1, 
    Z_true=None, 
    deps=[],
    chrome_id_list = [],
    n_trial=100,
    num_cores=None, 
    random_seed=None,
    anchor = None,
    use_canonical=False,
    use_forward=False,
    return_predictive_score=False):

    if num_cores is None:
        num_cores = multiprocessing.cpu_count()

    if random_seed is None:
        random_seed = [None]*n_trial

    # preprocessing 
    if "Ivy" in [x.__name__ for x in IV_Model_list]:
        pass
    
    result = Parallel(n_jobs=num_cores)(
        delayed(ComputeCausalitySingle)(
            X, 
            Y, 
            IVs, 
            IV_Model_list,
            estimator_list, 
            is_soft_label = is_soft_label, 
            ablation_train = ablation_train, 
            ablation_test = ablation_test, 
            Z_true = Z_true, 
            random_seed = random_seed[i], 
            deps = deps,
            chrome_id_list = chrome_id_list,
            anchor = anchor,
            use_canonical = use_canonical,
            use_forward=use_forward,
            return_predictive_score=return_predictive_score) 
        for i in tqdm(range(n_trial)))

    return result

def SplitDataset(Y, ablation_train=1, ablation_test=1):

    n = Y.shape[0]
    while True:
        # split train and test in halves    
        train_index = np.random.choice(n,n//2,replace=False)
        test_index = np.setdiff1d(range(n),train_index)
        # ablation in training set
        train_index = np.random.choice(train_index,round(n/2*ablation_train)-1,replace=False)
        # ablation in test set
        test_index = np.random.choice(test_index,round(n/2*ablation_test)-1,replace=False)
        # make sure that the split includes both calsses
        if ( len(set(Y[train_index]))>1 and len(set(Y[test_index]))>1):
            break

    return np.sort(train_index), np.sort(test_index)